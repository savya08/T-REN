import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append('..')
from model import FeatureExtractor, RegionEncoder, TextEncoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.getLogger().setLevel(logging.WARNING)


class TemporalTokenAggregator(nn.Module):
    def __init__(self, merging_threshold=0.85):
        super().__init__()
        self.merging_threshold = merging_threshold
        self.reset()

    def reset(self):
        self.track_pred_tokens = []
        self.track_text_aligned_tokens = []
        self.track_region_masks = []
        self.track_counts = []
        self.track_last_frame = []
        self.track_members = []

    @torch.inference_mode()
    def update(self, curr_pred_tokens, curr_text_aligned_tokens, curr_region_masks, frame_id):
        if curr_pred_tokens.numel() == 0:
            return
        
        # If this is the first frame, start one track per region
        num_regions = curr_pred_tokens.shape[0]
        if len(self.track_pred_tokens) == 0:
            for region_idx in range(num_regions):
                self.track_pred_tokens.append(curr_pred_tokens[region_idx].clone())
                self.track_text_aligned_tokens.append(curr_text_aligned_tokens[region_idx].clone())
                self.track_region_masks.append([curr_region_masks[region_idx].clone()])
                self.track_counts.append(1)
                self.track_last_frame.append(frame_id)
                self.track_members.append([(frame_id, region_idx)])
            return

        # Fetch track indices that ended at previous frame
        active_track_idxs = [k for k, f in enumerate(self.track_last_frame) if f == frame_id - 1]

        # If there are no active tracks, start new ones
        if len(active_track_idxs) == 0:
            for region_idx in range(num_regions):
                self.track_pred_tokens.append(curr_pred_tokens[region_idx].clone())
                self.track_text_aligned_tokens.append(curr_text_aligned_tokens[region_idx].clone())
                self.track_region_masks.append([curr_region_masks[region_idx].clone()])
                self.track_counts.append(1)
                self.track_last_frame.append(frame_id)
                self.track_members.append([(frame_id, region_idx)])
            return

        # If there are active tracks, fetch the active track tokens
        active_track_pred_tokens = [self.track_pred_tokens[k] for k in active_track_idxs]
        active_track_pred_tokens = torch.stack(active_track_pred_tokens, dim=0)

        # Compute similarity between active track tokens and current frame tokens
        curr_pred_tokens_norm = F.normalize(curr_pred_tokens, p=2, dim=-1)
        active_track_pred_tokens_norm = F.normalize(active_track_pred_tokens, p=2, dim=-1)
        similarity = torch.mm(active_track_pred_tokens_norm, curr_pred_tokens_norm.t())

        # Find aggregation candidates
        above = (similarity >= self.merging_threshold).nonzero(as_tuple=False)
        if above.numel() == 0:
            aggregation_candidates = []
        else:
            active_idx_v = above[:, 0]
            region_idx_v = above[:, 1]
            scores = similarity[active_idx_v, region_idx_v]
            order = scores.argsort(descending=True).cpu().tolist()
            aggregation_candidates = [(scores[i].item(), active_idx_v[i].item(), region_idx_v[i].item()) for i in order]

        # Aggregate the current frame tokens into the active tracks
        used_active, used_regions = set(), set()
        for similarity_score, active_idx, region_idx in aggregation_candidates:
            if active_idx in used_active or region_idx in used_regions:
                continue
            used_active.add(active_idx)
            used_regions.add(region_idx)
            track_idx = active_track_idxs[active_idx]
            track_count = self.track_counts[track_idx]
            
            # Update the track token and count using running mean update
            self.track_pred_tokens[track_idx] = \
                (self.track_pred_tokens[track_idx] * track_count + curr_pred_tokens[region_idx]) / (track_count + 1)
            self.track_text_aligned_tokens[track_idx] = \
                (self.track_text_aligned_tokens[track_idx] * track_count + curr_text_aligned_tokens[region_idx]) / (track_count + 1)
            self.track_region_masks[track_idx].append(curr_region_masks[region_idx].clone())
            self.track_counts[track_idx] = track_count + 1
            self.track_last_frame[track_idx] = frame_id
            self.track_members[track_idx].append((frame_id, region_idx))

        # Start a new track for unmatched regions
        for region_idx in range(num_regions):
            if region_idx in used_regions:
                continue
            self.track_pred_tokens.append(curr_pred_tokens[region_idx].clone())
            self.track_text_aligned_tokens.append(curr_text_aligned_tokens[region_idx].clone())
            self.track_region_masks.append([curr_region_masks[region_idx].clone()])
            self.track_counts.append(1)
            self.track_last_frame.append(frame_id)
            self.track_members.append([(frame_id, region_idx)])

    @torch.inference_mode()
    def get_result(self):
        if len(self.track_pred_tokens) == 0:
            return torch.empty(0), []
        track_pred_tokens = torch.stack(self.track_pred_tokens, dim=0)
        track_text_aligned_tokens = torch.stack(self.track_text_aligned_tokens, dim=0)
        return {
            'track_pred_tokens': track_pred_tokens,
            'track_text_aligned_tokens': track_text_aligned_tokens,
            'track_region_masks': self.track_region_masks,
            'track_members': self.track_members,
        }


class VideoREN(nn.Module):
    def __init__(self, config):
        super(VideoREN, self).__init__()
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        
        # Create the models
        self.tren_image_encoder = FeatureExtractor(config, device=device)
        self.tren_region_encoder = RegionEncoder(config).to(device).eval()
        self.tren_text_encoder = TextEncoder(config, device=device)
        self.temporal_token_aggregator = TemporalTokenAggregator()

        # Load the checkpoint
        self.checkpoint_path = os.path.join(self.exp_dir, 'tren_region_encoder.pth')
        self.load_checkpoint()

        # Image preprocessing transforms
        self.image_resolution = config['parameters']['image_resolution']
        self.patch_size = config['architecture']['patch_size']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_resolution, self.image_resolution), antialias=True),
        ])
        
        # Grid points for querying region encoder
        grid_size = self.image_resolution // self.patch_size
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            if 'ren_region_encoder_state' in checkpoint:
                self.tren_region_encoder.load_state_dict(checkpoint['ren_region_encoder_state'])
                print('T-REN region encoder loaded from checkpoint')
            if 'ren_image_encoder_state' in checkpoint:
                self.tren_image_encoder.load_state_dict(checkpoint['ren_image_encoder_state'])
                print('T-REN image encoder loaded from checkpoint')
            if 'ren_text_encoder_state' in checkpoint:
                self.tren_text_encoder.load_state_dict(checkpoint['ren_text_encoder_state'])
                print('T-REN text encoder loaded from checkpoint')
        else:
            print('No checkpoint found, exiting.')
            exit()

    def forward(self, frames, batch_size=32, aggregate_tokens=True):
        T = len(frames)
        transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        self.temporal_token_aggregator.reset()

        token_count_with_patch_features = 0
        token_count_without_temporal_aggregation = 0
        with torch.inference_mode():
            for start in range(0, T, batch_size):
                end = min(T, start + batch_size)
                frame_batch = transformed_frames[start:end].to(device)
                feature_maps = self.tren_image_encoder(frame_batch)['feature_maps']
                grid_points = [self.grid_points for _ in range(frame_batch.shape[0])]

                tren_outputs = self.tren_region_encoder(feature_maps, grid_points, aggregate_tokens=aggregate_tokens)
                pred_tokens = tren_outputs['pred_tokens']
                text_aligned_tokens = tren_outputs['text_aligned_tokens']
                region_masks = tren_outputs['region_masks']

                # Update the temporal token aggregator
                for frame_idx in range(frame_batch.shape[0]):
                    frame_id = start + frame_idx
                    
                    # Count tokens for this frame
                    token_count_without_temporal_aggregation += pred_tokens[frame_idx].shape[0]
                    token_count_with_patch_features += (self.image_resolution // self.patch_size) ** 2
                    
                    # Update aggregator
                    self.temporal_token_aggregator.update(pred_tokens[frame_idx], text_aligned_tokens[frame_idx], 
                                                          region_masks[frame_idx], frame_id)

        track_results = self.temporal_token_aggregator.get_result()
        token_count_with_temporal_aggregation = track_results['track_pred_tokens'].shape[0]
        compression = {
            'from_patches': token_count_with_patch_features / token_count_with_temporal_aggregation,
            'from_regions': token_count_without_temporal_aggregation / token_count_with_temporal_aggregation,
        }
        return track_results, compression