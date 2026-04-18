import os
import sys
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append('..')
from model import FeatureExtractor, RegionEncoder, TextEncoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.getLogger().setLevel(logging.WARNING)


def mask_to_bbox(mask):
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask)
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return None
    x_min = np.min(cols)
    y_min = np.min(rows)
    x_max = np.max(cols)
    y_max = np.max(rows)
    return np.array([x_min, y_min, x_max - x_min, y_max - y_min])


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)
    intersection_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou


class TemporalTokenAggregator(nn.Module):
    def __init__(self, merging_threshold=0.65):
        super().__init__()
        self.merging_threshold = merging_threshold
        self.reset()

    def reset(self):
        self.track_pred_tokens = []
        self.track_text_aligned_tokens = []
        self.track_region_points = []
        self.track_counts = []
        self.track_last_frame = []
        self.track_members = []

    def get_region_point(self, region_mask, frame_resolution):
        flat_idx = region_mask.reshape(-1).argmax().item()
        h, w = region_mask.shape
        y_mask = flat_idx // w
        x_mask = flat_idx % w
        frame_h, frame_w = frame_resolution
        if h > 1:
            y = int(round(y_mask * (frame_h - 1) / (h - 1)))
        else:
            y = 0
        if w > 1:
            x = int(round(x_mask * (frame_w - 1) / (w - 1)))
        else:
            x = 0
        y = min(max(y, 0), frame_h - 1)
        x = min(max(x, 0), frame_w - 1)
        return (y, x)

    @torch.inference_mode()
    def update(self, curr_pred_tokens, curr_text_aligned_tokens, curr_region_masks, frame_id, frame_resolution):
        if curr_pred_tokens.numel() == 0:
            return
        
        # If this is the first frame, start one track per region
        num_regions = curr_pred_tokens.shape[0]
        if len(self.track_pred_tokens) == 0:
            for region_idx in range(num_regions):
                self.track_pred_tokens.append(curr_pred_tokens[region_idx].clone())
                self.track_text_aligned_tokens.append(curr_text_aligned_tokens[region_idx].clone())
                self.track_region_points.append([self.get_region_point(curr_region_masks[region_idx], frame_resolution)])
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
                self.track_region_points.append([self.get_region_point(curr_region_masks[region_idx], frame_resolution)])
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
            self.track_region_points[track_idx].append(self.get_region_point(curr_region_masks[region_idx], frame_resolution))
            self.track_counts[track_idx] = track_count + 1
            self.track_last_frame[track_idx] = frame_id
            self.track_members[track_idx].append((frame_id, region_idx))

        # Start a new track for unmatched regions
        for region_idx in range(num_regions):
            if region_idx in used_regions:
                continue
            self.track_pred_tokens.append(curr_pred_tokens[region_idx].clone())
            self.track_text_aligned_tokens.append(curr_text_aligned_tokens[region_idx].clone())
            self.track_region_points.append([self.get_region_point(curr_region_masks[region_idx], frame_resolution)])
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
            'track_region_points': self.track_region_points,
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
        self.temporal_token_aggregator = TemporalTokenAggregator(config['parameters']['temporal_merging_threshold'])

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

    def forward(self, frames, batch_size=32):
        T = len(frames)
        frame_resolution = frames[0].shape[:2]
        transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        self.temporal_token_aggregator.reset()

        token_count_with_patch_features = 0
        token_count_without_temporal_aggregation = 0
        with torch.inference_mode():
            for start in tqdm(range(0, T, batch_size), desc='Processing image frames'):
                end = min(T, start + batch_size)
                frame_batch = transformed_frames[start:end].to(device)
                feature_maps = self.tren_image_encoder(frame_batch)['feature_maps']
                grid_points = [self.grid_points for _ in range(frame_batch.shape[0])]

                tren_outputs = self.tren_region_encoder(feature_maps, grid_points, aggregate_tokens=True)
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
                                                          region_masks[frame_idx], frame_id, frame_resolution)

        track_results = self.temporal_token_aggregator.get_result()
        token_count_with_temporal_aggregation = track_results['track_pred_tokens'].shape[0]
        compression = {
            'from_patches': token_count_with_patch_features / token_count_with_temporal_aggregation,
            'from_regions': token_count_without_temporal_aggregation / token_count_with_temporal_aggregation,
        }
        return track_results, compression


class TextQueryEncoder(nn.Module):
    def __init__(self, config):
        super(TextQueryEncoder, self).__init__()
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

        # Create the model
        self.tren_text_encoder = TextEncoder(config, device=device)

        # Load the checkpoint
        self.checkpoint_path = os.path.join(self.exp_dir, 'tren_region_encoder.pth')
        self.load_checkpoint()
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            if 'ren_text_encoder_state' in checkpoint:
                self.tren_text_encoder.load_state_dict(checkpoint['ren_text_encoder_state'])
                print('T-REN text encoder loaded from checkpoint')
        else:
            print('No checkpoint found, exiting.')
            exit()

    def forward(self, query_text):
        text_tokens = self.tren_text_encoder([query_text])
        return text_tokens


class VisualQueryEncoder(nn.Module):
    def __init__(self, config):
        super(VisualQueryEncoder, self).__init__()
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

        # Create the model
        self.tren_image_encoder = FeatureExtractor(config, device=device)
        self.tren_region_encoder = RegionEncoder(config).to(device).eval()

        # Load the checkpoint
        self.checkpoint_path = os.path.join(self.exp_dir, 'tren_region_encoder.pth')
        self.load_checkpoint()

        # Image preprocessing transforms
        image_resolution = config['parameters']['image_resolution']
        patch_size = config['architecture']['patch_size']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])
        
        # Grid points for querying region encoder
        grid_size = image_resolution // patch_size
        x_coords = np.linspace(patch_size // 2, image_resolution - patch_size // 2, grid_size, dtype=int)
        y_coords = np.linspace(patch_size // 2, image_resolution - patch_size // 2, grid_size, dtype=int)
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
        else:
            print('No checkpoint found, exiting.')
            exit()
    
    def select_best_overlapping_region(self, region_masks, query_bbox, frame_resolution):
        best_match_idx = None
        best_match_iou = 0

        # Resize region masks to frame resolution
        if len(region_masks.shape) == 3:
            region_masks = region_masks.unsqueeze(0)
        region_masks_resized = F.interpolate(region_masks, size=frame_resolution, mode='bilinear', align_corners=False)
        if region_masks_resized.shape[0] == 1:
            region_masks_resized = region_masks_resized[0]
        region_masks_resized = region_masks_resized.cpu().numpy()
        
        # Find the best overlapping region
        for region_idx in range(len(region_masks_resized)):
            region_mask = region_masks_resized[region_idx]
            mask_max = region_mask.max()
            if mask_max > 0:
                threshold = min(0.6, 0.8 * mask_max)
                region_mask_binary = region_mask >= threshold
            else:
                continue
            
            region_bbox = mask_to_bbox(region_mask_binary)
            if region_bbox is None:
                continue
            region_iou = iou(query_bbox, region_bbox)
            if region_iou > best_match_iou:
                best_match_iou = region_iou
                best_match_idx = region_idx
        return best_match_idx
    
    def forward(self, frame, bbox):
        frame_resolution = frame.shape[:2]
        transformed_frames = self.transform(frame)[None].to(device)

        with torch.inference_mode():
            feature_maps = self.tren_image_encoder(transformed_frames)['feature_maps']
            grid_points = [self.grid_points for _ in range(transformed_frames.shape[0])]

            tren_outputs = self.tren_region_encoder(feature_maps, grid_points, aggregate_tokens=True)
            pred_tokens = tren_outputs['pred_tokens'][0]
            region_masks = tren_outputs['region_masks'][0]

            best_match_idx = self.select_best_overlapping_region(region_masks, bbox, frame_resolution)
            if best_match_idx is None:
                query_tokens = pred_tokens.mean(dim=0)
            else:
                query_tokens = pred_tokens[best_match_idx]
        return query_tokens[None]


class PatchBasedQuerySearch(nn.Module):
    def __init__(self, config):
        super(PatchBasedQuerySearch, self).__init__()
        self.patch_encoder = torch.hub.load('facebookresearch/dinov3', 
                                            'dinov3_vitl16_dinotxt_tet1280d20h24l')[0].to(device).eval()
        self.tren_image_encoder = FeatureExtractor(config['tren'], device=device)
        self.text_query_encoder = TextQueryEncoder(config['tren'])
        self.visual_query_encoder = VisualQueryEncoder(config['tren'])
        self.similarity_threshold = config['parameters']['similarity_threshold']

        # Image preprocessing transforms
        image_resolution = config['tren']['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

    def find_similarity(self, video_tokens, query_tokens):
        num_frames, num_patches, embedding_dim = video_tokens.shape
        video_token_norm = F.normalize(video_tokens, p=2, dim=-1).view(num_frames * num_patches, embedding_dim)
        query_token_norm = F.normalize(query_tokens, p=2, dim=-1).reshape(-1, embedding_dim)
        similarity = torch.mm(video_token_norm, query_token_norm.t())
        similarity = similarity.view(num_frames, num_patches, -1).squeeze(-1)
        return similarity

    def find_matching_frames(self, frames, text_query_tokens, visual_query_tokens, batch_size=32):
        T = len(frames)
        transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        frame_similarities = []
        with torch.inference_mode():
            for start in tqdm(range(0, T, batch_size), desc='Processing image frames'):
                end = min(T, start + batch_size)
                frame_batch = transformed_frames[start:end].to(device)

                # Get the patch tokens for the current batch of frames
                feature_maps = self.tren_image_encoder(frame_batch)['feature_maps']
                patch_tokens = feature_maps.flatten(-2).permute(0, 2, 1)
                text_aligned_patch_tokens = self.patch_encoder.visual_model(frame_batch)[1]

                # Find the most similar patch tokens for each frame in the batch
                text_level_similarity = self.find_similarity(text_aligned_patch_tokens, text_query_tokens)
                visual_level_similarity = self.find_similarity(patch_tokens, visual_query_tokens)
                similarity = text_level_similarity.max(dim=-1).values * visual_level_similarity.max(dim=-1).values
                frame_similarities.append(similarity)
                del frame_batch, feature_maps, patch_tokens, text_aligned_patch_tokens
            torch.cuda.empty_cache()

        frame_similarities = torch.cat(frame_similarities, dim=0)
        selected_frame_ids = torch.where(frame_similarities >= self.similarity_threshold)[0]
        return selected_frame_ids.tolist()

    def forward(self, record_file, annotation):
        with open(record_file, 'rb') as f:
            record = pickle.load(f)
        frames = record['frames']
        
        # Generate the text query tokens
        query_text = annotation['object_title']
        text_query_tokens = self.text_query_encoder(query_text)

        # Generate the visual query tokens
        visual_crop = annotation['visual_crop']
        query_frame = frames[visual_crop['frame_number']]
        x = int((visual_crop['x'] / visual_crop['original_width']) * frames.shape[2])
        y = int((visual_crop['y'] / visual_crop['original_height']) * frames.shape[1])
        height = int((visual_crop['height'] / visual_crop['original_height']) * frames.shape[1])
        width = int((visual_crop['width'] / visual_crop['original_width']) * frames.shape[2])
        query_bbox = [x, y, width, height]
        visual_query_tokens = self.visual_query_encoder(query_frame, query_bbox)

        # Find frames matching the text query
        queried_frames = frames[:annotation['query_frame']]
        selected_frame_ids = self.find_matching_frames(queried_frames, text_query_tokens, visual_query_tokens)
        return selected_frame_ids, {'from_patches': -1.0, 'from_regions': -1.0}


class QuerySearch(nn.Module):
    def __init__(self, config):
        super(QuerySearch, self).__init__()
        self.video_ren = VideoREN(config['tren'])
        self.text_query_encoder = TextQueryEncoder(config['tren'])
        self.visual_query_encoder = VisualQueryEncoder(config['tren'])
        self.similarity_threshold = config['parameters']['similarity_threshold']

    def visualize_tracks(self, frames, track_members, track_region_points, track_similarity, save_dir='vis'):
        os.makedirs(save_dir, exist_ok=True)
        from matplotlib import pyplot as plt
        for track_id in range(len(track_members)):
            os.makedirs(f'{save_dir}/track-{track_id}', exist_ok=True)
            for (frame_id, region_idx), region_point in zip(track_members[track_id], track_region_points[track_id]):
                plt.imshow(frames[frame_id])
                plt.scatter(region_point[1], region_point[0], color='red', marker='x')
                plt.title(f'Similarity: {track_similarity[track_id]}')
                plt.axis('off')
                plt.savefig(f'{save_dir}/track-{track_id}/frame-{frame_id}.png')
                plt.close()

    def find_similarity(self, video_tokens, query_tokens):
        video_tokens_norm = F.normalize(video_tokens, p=2, dim=-1)
        query_tokens_norm = F.normalize(query_tokens, p=2, dim=-1)
        similarity = torch.mm(video_tokens_norm, query_tokens_norm.t())
        return similarity

    def forward(self, record_file, annotation, visualize_selected_tracks=False):
        with open(record_file, 'rb') as f:
            record = pickle.load(f)
        frames = record['frames']
        
        # Generate the video region tokens
        queried_frames = frames[:annotation['query_frame']]
        video_tokens, compression = self.video_ren(queried_frames)
        pred_tokens = video_tokens['track_pred_tokens']
        text_aligned_tokens = video_tokens['track_text_aligned_tokens']
        track_region_points = video_tokens['track_region_points']
        track_members = video_tokens['track_members']
        
        # Generate the text query tokens
        query_text = annotation['object_title']
        text_query_tokens = self.text_query_encoder(query_text)

        # Generate the visual query tokens
        visual_crop = annotation['visual_crop']
        query_frame = frames[visual_crop['frame_number']]
        x = int((visual_crop['x'] / visual_crop['original_width']) * frames.shape[2])
        y = int((visual_crop['y'] / visual_crop['original_height']) * frames.shape[1])
        height = int((visual_crop['height'] / visual_crop['original_height']) * frames.shape[1])
        width = int((visual_crop['width'] / visual_crop['original_width']) * frames.shape[2])
        query_bbox = [x, y, width, height]
        visual_query_tokens = self.visual_query_encoder(query_frame, query_bbox)
        
        # Find similarities between the video tokens and the query tokens
        text_level_similarity = self.find_similarity(text_aligned_tokens, text_query_tokens)
        visual_level_similarity = self.find_similarity(pred_tokens, visual_query_tokens)
        similarity = text_level_similarity * visual_level_similarity
        
        # Find the object tracks that match the query
        similar_track_idxs = torch.where(similarity >= self.similarity_threshold)[0]
        selected_region_points = [track_region_points[idx] for idx in similar_track_idxs]
        selected_track_members = [track_members[idx] for idx in similar_track_idxs]
        selected_track_similarity = [similarity[idx].max() for idx in similar_track_idxs]
        if len(selected_track_members) == 0:
            flat_idx = similarity.argmax()
            most_similar_track_idx = flat_idx // similarity.shape[1]
            selected_track_members = [track_members[most_similar_track_idx]]
            selected_region_points = [track_region_points[most_similar_track_idx]]
            selected_track_similarity = [similarity[most_similar_track_idx][0]]

        # Consolidate the frame ids for the selected tracks
        selected_frame_ids = []
        for track_members in selected_track_members:
            for frame_id, _ in track_members:
                selected_frame_ids.append(frame_id)
        selected_frame_ids = list(set(selected_frame_ids))

        # Visualize the selected tracks
        if visualize_selected_tracks:
            self.visualize_tracks(frames, selected_track_members, selected_region_points, selected_track_similarity, 
                                  save_dir=f'selected_tracks/{query_text}')
        return selected_frame_ids, compression