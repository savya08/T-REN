import os
import yaml
import json
import random
import math
import logging
from tqdm import tqdm
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from scipy.optimize import linear_sum_assignment
from dataloader import RENDataset, collate_fn
from model import FeatureExtractor, RegionTokenGenerator, TextEncoder, RegionEncoder
from task_utils import print_log


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 7
use_wandb = os.environ.get('USE_WANDB', '').lower() in ('1', 'true', 'yes')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger().setLevel(logging.WARNING)


class Trainer:
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print_log(f'Configs: {config}', self.exp_dir)

        # Instantiate the dataloaders
        train_dataset = RENDataset(config, split='train')
        self.train_loader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                                       sampler=train_dataset.get_weighted_sampler(), collate_fn=collate_fn,
                                       num_workers=config['parameters']['num_workers'], pin_memory=True)
        val_dataset = RENDataset(config, split='val')
        self.val_loader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'],
                                     sampler=val_dataset.get_weighted_sampler(), collate_fn=collate_fn,
                                     num_workers=config['parameters']['num_workers'], pin_memory=True)
        
        # Pre-compute category mappings for efficiency
        self.category_to_idx = json.load(open('aux_files/cat_to_idx.json', 'r'))

        # Set training parameters
        self.num_epochs = config['parameters']['num_epochs']
        self.total_steps = self.num_epochs * len(self.train_loader)
        self.accumulation_steps = config['parameters']['accumulation_steps']
        self.warmup_steps = config['parameters']['warmup_steps']
        self.logging_steps = config['parameters']['logging_steps']
        self.max_grad_norm = config['parameters']['max_grad_norm']
        self.scaler = GradScaler()
        self.config = config
        
        # Create the models
        self.org_image_encoder = FeatureExtractor(config, device=device)
        self.org_region_encoder = RegionTokenGenerator(device=device)
        self.tren_image_encoder = FeatureExtractor(config, device=device)
        self.tren_region_encoder = RegionEncoder(config).to(device)
        self.tren_text_encoder = TextEncoder(config, device=device)

        # Freeze teacher models
        self.org_image_encoder.eval()
        self.org_region_encoder.eval()
        for param in self.org_image_encoder.parameters():
            param.requires_grad = False
        for param in self.org_region_encoder.parameters():
            param.requires_grad = False

        # Define the trainable parameters
        param_groups = []
        def add_param_group(model, lr, name):
            if lr != 0.0:
                param_groups.append({'params': model.parameters(), 'lr': lr, 'name': name})
            else:
                for param in model.parameters():
                    param.requires_grad = False
        add_param_group(self.tren_region_encoder, config['parameters']['tren_region_encoder_lr'], 'tren_region_encoder')
        add_param_group(self.tren_image_encoder, config['parameters']['tren_image_encoder_lr'], 'tren_image_encoder')
        add_param_group(self.tren_text_encoder, config['parameters']['tren_text_encoder_lr'], 'tren_text_encoder')

        # Define the optimizer and loss function
        self.optimizer = optim.AdamW(param_groups)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Initialize training state
        self.start_epoch = 0
        self.start_iter = 0
        self.checkpoint_path = os.path.join(self.exp_dir, 'checkpoint.pth')
        self.best_val_loss = float('inf')

        # Load checkpoint if it exists
        self.load_checkpoint()

    def save_checkpoint(self, epoch, iter_count, val_loss, mode='latest'):
        checkpoint = {
            'epoch': epoch,
            'iter_count': iter_count,
            'best_val_loss': val_loss,
            'optimizer_state': self.optimizer.state_dict()
        }
        if self.config['parameters']['tren_region_encoder_lr'] != 0.0:
            print_log('Saving T-REN region encoder state', self.exp_dir)
            checkpoint['ren_region_encoder_state'] = self.tren_region_encoder.state_dict()
        if self.config['parameters']['tren_image_encoder_lr'] != 0.0:
            print_log('Saving T-REN image encoder state', self.exp_dir)
            checkpoint['ren_image_encoder_state'] = self.tren_image_encoder.state_dict()
        if self.config['parameters']['tren_text_encoder_lr'] != 0.0:
            print_log('Saving T-REN text encoder state', self.exp_dir)
            checkpoint['ren_text_encoder_state'] = self.tren_text_encoder.state_dict()
        if mode == 'best':
            torch.save(checkpoint, os.path.join(self.exp_dir, 'best_checkpoint.pth'))
        else:
            torch.save(checkpoint, self.checkpoint_path)
        print_log(f'Saved checkpoint with val_loss {val_loss.item():.4f}', self.exp_dir)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            if 'ren_region_encoder_state' in checkpoint:
                self.tren_region_encoder.load_state_dict(checkpoint['ren_region_encoder_state'])
                print_log('T-REN region encoder loaded from checkpoint', self.exp_dir)
            if 'ren_image_encoder_state' in checkpoint:
                self.tren_image_encoder.load_state_dict(checkpoint['ren_image_encoder_state'])
                print_log('T-REN image encoder loaded from checkpoint', self.exp_dir)
            if 'ren_text_encoder_state' in checkpoint:
                self.tren_text_encoder.load_state_dict(checkpoint['ren_text_encoder_state'])
                print_log('T-REN text encoder loaded from checkpoint', self.exp_dir)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                print_log('Optimizer loaded from checkpoint', self.exp_dir)
            except ValueError as e:
                print_log(f'Could not load optimizer state: {e}', self.exp_dir)
                print_log('Starting with fresh optimizer state', self.exp_dir)
            print_log(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}', self.exp_dir)
        else:
            print_log('No checkpoint found, starting training from scratch.', self.exp_dir)
    
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    def feature_similarity_loss(self, pred_tokens_v1, pred_tokens_v2, targets_v1, targets_v2, loss_mask_v1, loss_mask_v2):
        cos_loss_v1 = 1 - F.cosine_similarity(pred_tokens_v1, targets_v1, dim=-1)
        cos_loss_v1 = (cos_loss_v1 * loss_mask_v1).sum() / (loss_mask_v1.sum() + 1e-8)
        cos_loss_v2 = 1 - F.cosine_similarity(pred_tokens_v2, targets_v2, dim=-1)
        cos_loss_v2 = (cos_loss_v2 * loss_mask_v2).sum() / (loss_mask_v2.sum() + 1e-8)
        cos_loss = cos_loss_v1 + cos_loss_v2
        
        hidden_dim = pred_tokens_v1.shape[-1]
        pred_a = F.normalize(pred_tokens_v1.view(-1, hidden_dim), p=2, dim=-1)
        pred_b = F.normalize(pred_tokens_v2.view(-1, hidden_dim), p=2, dim=-1)
        tgt_a = F.normalize(targets_v1.view(-1, hidden_dim), p=2, dim=-1)
        tgt_b = F.normalize(targets_v2.view(-1, hidden_dim), p=2, dim=-1)
        pred_sim = torch.matmul(pred_a, pred_b.T)
        tgt_sim = torch.matmul(tgt_a, tgt_b.T)
        loss_mask = torch.matmul(loss_mask_v1.view(-1, 1).float(), loss_mask_v2.view(-1, 1).float().T)
        sim_loss = F.l1_loss(pred_sim, tgt_sim, reduction='none')
        sim_loss = (sim_loss * loss_mask).sum() / loss_mask.sum()
        return (cos_loss + sim_loss) / 2

    def region_region_contrastive_loss(self, pred_tokens_v1, pred_tokens_v2, region_ids_v1, region_ids_v2, temp=0.1):
        # Concatenate all tokens and ids for vectorized processing
        all_tokens = torch.cat([pred_tokens_v1, pred_tokens_v2], dim=1)
        all_tokens = all_tokens.view(-1, all_tokens.shape[-1])
        all_ids = torch.cat([region_ids_v1, region_ids_v2], dim=1)
        all_ids = all_ids.view(-1)
        
        # Compute similarity matrix
        all_tokens = F.normalize(all_tokens, p=2, dim=1)
        sim_matrix = torch.matmul(all_tokens, all_tokens.T) / temp
        
        # Create batch mask to avoid cross-batch comparisons
        tokens_per_batch = pred_tokens_v1.shape[1] * pred_tokens_v1.shape[2] * 2
        batch_indices = torch.arange(pred_tokens_v1.shape[0], device=all_tokens.device)
        batch_indices = batch_indices.repeat_interleave(tokens_per_batch)
        batch_mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)
        
        # Apply batch mask to similarity matrix to set cross-batch similarities to -inf
        sim_matrix = sim_matrix.masked_fill(~batch_mask, float('-inf'))
        
        # Create positive mask for same region IDs within the same batch
        pos_mask = (all_ids.unsqueeze(0) == all_ids.unsqueeze(1)) & batch_mask
        pos_mask.fill_diagonal_(False)
        
        # Compute logits with numerical stability
        logits_max = torch.max(sim_matrix, dim=1, keepdim=True)[0]
        sim_matrix_stable = sim_matrix - logits_max
        
        # Compute numerator
        exp_sim = torch.exp(sim_matrix_stable)
        numerator = (exp_sim * pos_mask).sum(dim=1)
        
        # Compute denominator
        eye_mask = torch.eye(sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.bool)
        denominator = (exp_sim * ~eye_mask).sum(dim=1)
        
        # Compute loss only for tokens with positive pairs
        valid_tokens = pos_mask.sum(dim=1) > 0
        if valid_tokens.sum() == 0:
            return torch.tensor(0.0, device=all_tokens.device, requires_grad=True)
        
        # Compute losses
        losses = -torch.log(numerator[valid_tokens] / (denominator[valid_tokens] + 1e-8))
        return losses.mean()
    
    def region_text_contrastive_loss(self, pred_tokens_v1, pred_tokens_v2, categories_v1, categories_v2, temp=0.1):
        def flatten_and_filter(pred_tokens, categories):
            batch_size, num_points, num_scales, hidden_dim = pred_tokens.shape
            flat_tokens = pred_tokens.view(-1, hidden_dim)
            flat_categories = []
            flat_idxs = []
            for batch_idx in range(batch_size):
                for point_idx in range(num_points):
                    for scale_idx in range(num_scales):
                        c = categories[batch_idx][point_idx][scale_idx]
                        if c != 'none':
                            flat_categories.append(c)
                            flat_idxs.append(batch_idx * num_points * num_scales + point_idx * num_scales + scale_idx)
            if len(flat_categories) == 0:
                return None, None
            flat_idxs = torch.tensor(flat_idxs, device=pred_tokens.device, dtype=torch.long)
            valid_tokens = flat_tokens[flat_idxs]
            return valid_tokens, flat_categories

        # Flatten and filter out 'none'
        pred_tokens_v1_flat, categories_v1_flat = flatten_and_filter(pred_tokens_v1, categories_v1)
        pred_tokens_v2_flat, categories_v2_flat = flatten_and_filter(pred_tokens_v2, categories_v2)

        # Early exit if no valid tokens
        if (pred_tokens_v1_flat is None) and (pred_tokens_v2_flat is None):
            return torch.tensor(0.0, device=pred_tokens_v1.device)

        # Encode text for valid categories
        text_encoding_v1 = self.tren_text_encoder(categories_v1_flat) if pred_tokens_v1_flat is not None else None
        text_encoding_v2 = self.tren_text_encoder(categories_v2_flat) if pred_tokens_v2_flat is not None else None

        # Normalize encodings
        if pred_tokens_v1_flat is not None:
            pred_tokens_v1_flat = F.normalize(pred_tokens_v1_flat, dim=-1)
            text_encoding_v1 = F.normalize(text_encoding_v1, dim=-1)
        if pred_tokens_v2_flat is not None:
            pred_tokens_v2_flat = F.normalize(pred_tokens_v2_flat, dim=-1)
            text_encoding_v2 = F.normalize(text_encoding_v2, dim=-1)

        # Build label indices
        def build_indices(flat_categories, device):
            idxs = torch.full((len(flat_categories),), -1, device=device, dtype=torch.long)
            for i, c in enumerate(flat_categories):
                idxs[i] = self.category_to_idx[c]
            return idxs
        idxs_v1 = build_indices(categories_v1_flat, pred_tokens_v1.device) if pred_tokens_v1_flat is not None else None
        idxs_v2 = build_indices(categories_v2_flat, pred_tokens_v2.device) if pred_tokens_v2_flat is not None else None

        # Vectorized loss computation on reduced sets
        def compute_vectorized_loss(pred_tokens, text_enc, idxs):
            if pred_tokens is None or pred_tokens.shape[0] == 0:
                return torch.tensor(0.0, device=pred_tokens_v1.device)

            sim_img2text = torch.matmul(pred_tokens, text_enc.T) / temp
            sim_text2img = sim_img2text.T

            pos_mask = (idxs.unsqueeze(0) == idxs.unsqueeze(1))

            def compute_one_direction(sim_matrix, pos_mask_dir):
                if sim_matrix.numel() == 0:
                    return torch.tensor(0.0, device=sim_matrix.device)

                # Numerical stability
                sim_max = torch.max(sim_matrix, dim=1, keepdim=True)[0]
                sim_stable = sim_matrix - sim_max
                exp_sim = torch.exp(sim_stable)

                numerator = (exp_sim * pos_mask_dir).sum(dim=1)
                denominator = exp_sim.sum(dim=1)

                valid_with_pos = pos_mask_dir.sum(dim=1) > 0
                if valid_with_pos.sum() == 0:
                    return torch.tensor(0.0, device=sim_matrix.device)

                losses = -torch.log(numerator[valid_with_pos] / (denominator[valid_with_pos] + 1e-8))
                return losses.mean()

            loss_img2text = compute_one_direction(sim_img2text, pos_mask)
            loss_text2img = compute_one_direction(sim_text2img, pos_mask.T)
            return (loss_img2text + loss_text2img) / 2

        loss_v1 = compute_vectorized_loss(pred_tokens_v1_flat, text_encoding_v1, idxs_v1)
        loss_v2 = compute_vectorized_loss(pred_tokens_v2_flat, text_encoding_v2, idxs_v2)
        return (loss_v1 + loss_v2) / 2

    def region_mask_loss(self, pred_masks_v1, pred_masks_v2, target_masks_v1, target_masks_v2, loss_mask_v1, loss_mask_v2):
        pred_masks_v1 = pred_masks_v1.flatten(-2)
        pred_masks_v2 = pred_masks_v2.flatten(-2)
        target_masks_v1 = target_masks_v1.flatten(-2)
        target_masks_v2 = target_masks_v2.flatten(-2)

        def focal_loss_with_probs(probs, targets, alpha=0.25, gamma=2.0):
            probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)
            logits = torch.logit(probs)
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** gamma
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
            return focal_loss

        bce_loss_a = focal_loss_with_probs(pred_masks_v1, target_masks_v1.float())
        bce_loss_a = (bce_loss_a.mean(dim=-1) * loss_mask_v1).sum() / (loss_mask_v1.sum() + 1e-8)
        intersection_a = (pred_masks_v1 * target_masks_v1).sum(dim=-1)
        union_a = pred_masks_v1.sum(dim=-1) + target_masks_v1.sum(dim=-1)
        dice_score_a = (2 * intersection_a + 1e-6) / (union_a + 1e-6)
        dice_loss_a = 1 - (dice_score_a * loss_mask_v1).sum() / (loss_mask_v1.sum() + 1e-8)
        loss_a = (bce_loss_a + dice_loss_a) / 2

        bce_loss_b = focal_loss_with_probs(pred_masks_v2, target_masks_v2.float())
        bce_loss_b = (bce_loss_b.mean(dim=-1) * loss_mask_v2).sum() / (loss_mask_v2.sum() + 1e-8)
        intersection_b = (pred_masks_v2 * target_masks_v2).sum(dim=-1)
        union_b = pred_masks_v2.sum(dim=-1) + target_masks_v2.sum(dim=-1)
        dice_score_b = (2 * intersection_b + 1e-6) / (union_b + 1e-6)
        dice_loss_b = 1 - (dice_score_b * loss_mask_v2).sum() / (loss_mask_v2.sum() + 1e-8)
        loss_b = (bce_loss_b + dice_loss_b) / 2

        loss = (loss_a + loss_b) / 2
        return loss
    
    def hungarian_reordering(self, ren_region_tokens, org_region_tokens):
        pred_tokens = ren_region_tokens['pred_tokens']
        target_tokens = org_region_tokens['pooled_tokens']
        text_aligned_tokens = ren_region_tokens['text_aligned_tokens']
        region_masks = ren_region_tokens['region_masks']
        
        batch_size, num_points, num_scales, _ = pred_tokens.shape
        reordered_pred_tokens = pred_tokens.clone()
        reordered_text_aligned_tokens = text_aligned_tokens.clone()
        reordered_region_masks = region_masks.clone()

        # Process each (batch_idx, point_idx) pair
        for batch_idx in range(batch_size):
            for point_idx in range(num_points):
                # Get the predicted and target tokens for the current point
                target_tokens_at_point = target_tokens[batch_idx, point_idx]
                pred_tokens_at_point = pred_tokens[batch_idx, point_idx]
                region_masks_at_point = region_masks[batch_idx, point_idx]
                text_aligned_tokens_at_point = text_aligned_tokens[batch_idx, point_idx]

                # Compute cosine similarity between predicted and target tokens
                target_tokens_at_point = F.normalize(target_tokens_at_point, p=2, dim=1)
                pred_tokens_at_point = F.normalize(pred_tokens_at_point, p=2, dim=1)
                similarity_matrix = torch.matmul(pred_tokens_at_point, target_tokens_at_point.t())
                
                # Convert to cost matrix (negative similarity for minimization)
                cost_matrix = -similarity_matrix.detach().cpu().float().numpy()
                
                # Solve assignment problem using Hungarian algorithm
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Reorder the tokens at the current point
                reorder_indices = torch.zeros(num_scales, dtype=torch.long, device=pred_tokens.device)
                for j, target_idx in enumerate(col_indices):
                    pred_idx = row_indices[j]
                    reorder_indices[target_idx] = pred_idx
                
                reordered_pred_tokens[batch_idx, point_idx] = pred_tokens_at_point[reorder_indices]
                reordered_text_aligned_tokens[batch_idx, point_idx] = text_aligned_tokens_at_point[reorder_indices]
                reordered_region_masks[batch_idx, point_idx] = region_masks_at_point[reorder_indices]
        
        return {
            'pred_tokens': reordered_pred_tokens,
            'region_masks': reordered_region_masks,
            'text_aligned_tokens': reordered_text_aligned_tokens,
        }

    def step(self, v1, v2, dataset_type):
        if v1['images'].shape[0] == 0 or v2['images'].shape[0] == 0:
            return {
                'loss_feat': 0.0,
                'loss_mask': 0.0,
                'loss_cont': 0.0,
                'loss_text': 0.0,
                'loss': 0.0,
            }
        
        v1['images'] = v1['images'].to(device)
        v1['region_masks'] = v1['region_masks'].to(device)
        v1['region_ids'] = v1['region_ids'].to(device)
        v1['attn_masks'] = v1['attn_masks'].to(device)
        v1['loss_mask'] = v1['loss_mask'].to(device)
        v1['grid_points'] = v1['grid_points'].to(device)
        v2['images'] = v2['images'].to(device)
        v2['region_masks'] = v2['region_masks'].to(device)
        v2['region_ids'] = v2['region_ids'].to(device)
        v2['attn_masks'] = v2['attn_masks'].to(device)
        v2['loss_mask'] = v2['loss_mask'].to(device)
        v2['grid_points'] = v2['grid_points'].to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            # Compute predictions and targets for v1
            tren_image_encoder_outputs_v1 = self.tren_image_encoder(v1['images'])
            tren_region_tokens_v1 = self.tren_region_encoder(tren_image_encoder_outputs_v1['feature_maps'], v1['grid_points'])
            org_image_encoder_outputs_v1 = self.org_image_encoder(v1['images'])
            org_region_tokens_v1 = self.org_region_encoder(v1['attn_masks'], org_image_encoder_outputs_v1['class_tokens'], 
                                                           org_image_encoder_outputs_v1['feature_maps'], 
                                                           org_image_encoder_outputs_v1['register_tokens'])

            # Compute predictions and targets for v2
            tren_image_encoder_outputs_v2 = self.tren_image_encoder(v2['images'])
            tren_region_tokens_v2 = self.tren_region_encoder(tren_image_encoder_outputs_v2['feature_maps'], v2['grid_points'])
            org_image_encoder_outputs_v2 = self.org_image_encoder(v2['images'])
            org_region_tokens_v2 = self.org_region_encoder(v2['attn_masks'], org_image_encoder_outputs_v2['class_tokens'], 
                                                           org_image_encoder_outputs_v2['feature_maps'], 
                                                           org_image_encoder_outputs_v2['register_tokens'])

            # Reorder the region tokens using Hungarian algorithm
            tren_region_tokens_v1 = self.hungarian_reordering(tren_region_tokens_v1, org_region_tokens_v1)
            tren_region_tokens_v2 = self.hungarian_reordering(tren_region_tokens_v2, org_region_tokens_v2)

            # Compute loss
            loss_feat = self.feature_similarity_loss(pred_tokens_v1=tren_region_tokens_v1['pred_tokens'],
                                                     pred_tokens_v2=tren_region_tokens_v2['pred_tokens'],
                                                     targets_v1=org_region_tokens_v1['pooled_tokens'],
                                                     targets_v2=org_region_tokens_v2['pooled_tokens'],
                                                     loss_mask_v1=v1['loss_mask'], loss_mask_v2=v2['loss_mask'])
            loss_mask = self.region_mask_loss(pred_masks_v1=tren_region_tokens_v1['region_masks'],
                                              pred_masks_v2=tren_region_tokens_v2['region_masks'],
                                              target_masks_v1=v1['region_masks'], target_masks_v2=v2['region_masks'],
                                              loss_mask_v1=v1['loss_mask'], loss_mask_v2=v2['loss_mask'])
            loss_text = self.feature_similarity_loss(pred_tokens_v1=tren_region_tokens_v1['text_aligned_tokens'],
                                                     pred_tokens_v2=tren_region_tokens_v2['text_aligned_tokens'],
                                                     targets_v1=org_region_tokens_v1['text_aligned_tokens'],
                                                     targets_v2=org_region_tokens_v2['text_aligned_tokens'],
                                                     loss_mask_v1=v1['loss_mask'], loss_mask_v2=v2['loss_mask'])
            loss_cont = 0.0
            
            if dataset_type == 'region_region':
                loss_cont = self.region_region_contrastive_loss(pred_tokens_v1=tren_region_tokens_v1['pred_tokens'],
                                                                pred_tokens_v2=tren_region_tokens_v2['pred_tokens'],
                                                                region_ids_v1=v1['region_ids'], region_ids_v2=v2['region_ids'])

            if dataset_type == 'region_text':
                loss_text += self.region_text_contrastive_loss(pred_tokens_v1=tren_region_tokens_v1['text_aligned_tokens'],
                                                               pred_tokens_v2=tren_region_tokens_v2['text_aligned_tokens'],
                                                               categories_v1=v1['categories'], categories_v2=v2['categories'])

        loss = loss_feat + loss_mask + loss_cont + loss_text
        return {
            'loss_feat': loss_feat,
            'loss_mask': loss_mask,
            'loss_cont': loss_cont,
            'loss_text': loss_text,
            'loss': loss,
        }
    
    def validate(self, num_batches=50):
        self.tren_region_encoder.eval()
        self.tren_image_encoder.eval()
        self.tren_text_encoder.eval()

        loss_feat, loss_mask, loss_cont, loss_text, loss = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx == num_batches:
                    break
                
                region_region_v1, region_region_v2, region_text_v1, region_text_v2 = batch
                region_region_outputs = self.step(region_region_v1, region_region_v2, 'region_region')
                region_text_outputs = self.step(region_text_v1, region_text_v2, 'region_text')

                loss += region_region_outputs['loss'] + region_text_outputs['loss']
                loss_feat += region_region_outputs['loss_feat'] + region_text_outputs['loss_feat']
                loss_mask += region_region_outputs['loss_mask'] + region_text_outputs['loss_mask']
                loss_cont += region_region_outputs['loss_cont'] + region_text_outputs['loss_cont']
                loss_text += region_region_outputs['loss_text'] + region_text_outputs['loss_text']

        self.tren_region_encoder.train()
        self.tren_image_encoder.train()
        self.tren_text_encoder.train()

        loss_feat = loss_feat / num_batches
        loss_mask = loss_mask / num_batches
        loss_cont = loss_cont / num_batches
        loss_text = loss_text / num_batches
        loss = loss / num_batches
        return {
            'loss_feat': loss_feat,
            'loss_mask': loss_mask,
            'loss_cont': loss_cont,
            'loss_text': loss_text,
            'loss': loss,
        }
    
    def train(self):
        if self.config['parameters']['tren_region_encoder_lr'] != 0.0:
            self.tren_region_encoder.train()
        if self.config['parameters']['tren_image_encoder_lr'] != 0.0:
            self.tren_image_encoder.train()
        if self.config['parameters']['tren_text_encoder_lr'] != 0.0:
            self.tren_text_encoder.train()
        
        iter_count = self.start_iter
        self.optimizer.zero_grad()
        for epoch in range(self.start_epoch, self.num_epochs):
            for batch in tqdm(self.train_loader, desc=f'Running epoch {epoch}'):
                # Forward pass
                region_region_v1, region_region_v2, region_text_v1, region_text_v2 = batch
                region_region_outputs = self.step(region_region_v1, region_region_v2, 'region_region')
                region_text_outputs = self.step(region_text_v1, region_text_v2, 'region_text')
                
                train_outputs = {
                    'loss': region_region_outputs['loss'] + region_text_outputs['loss'],
                    'loss_feat': region_region_outputs['loss_feat'] + region_text_outputs['loss_feat'],
                    'loss_mask': region_region_outputs['loss_mask'] + region_text_outputs['loss_mask'],
                    'loss_cont': region_region_outputs['loss_cont'] + region_text_outputs['loss_cont'],
                    'loss_text': region_region_outputs['loss_text'] + region_text_outputs['loss_text'],
                }
                train_loss_feat = train_outputs['loss_feat']
                train_loss_mask = train_outputs['loss_mask']
                train_loss_cont = train_outputs['loss_cont']
                train_loss_text = train_outputs['loss_text']
                train_loss = train_outputs['loss']

                # Backward pass
                self.scaler.scale(train_loss).backward()
                torch.nn.utils.clip_grad_norm_([p for group in self.optimizer.param_groups for p in group['params']], 
                                               max_norm=self.max_grad_norm)
                if (iter_count + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Log progress
                if (iter_count + 1) % self.logging_steps == 0:
                    val_outputs = self.validate()
                    val_loss_feat = val_outputs['loss_feat']
                    val_loss_mask = val_outputs['loss_mask']
                    val_loss_cont = val_outputs['loss_cont']
                    val_loss_text = val_outputs['loss_text']
                    val_loss = val_outputs['loss']
                    self.save_checkpoint(epoch, iter_count, val_loss)
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, iter_count, val_loss, 'best')
                    if use_wandb:
                        wandb.log({
                            'train_loss': train_loss,
                            'train_loss_feat': train_loss_feat,
                            'train_loss_mask': train_loss_mask,
                            'train_loss_cont': train_loss_cont,
                            'train_loss_text': train_loss_text,
                            'val_loss': val_loss,
                            'val_loss_feat': val_loss_feat,
                            'val_loss_mask': val_loss_mask,
                            'val_loss_cont': val_loss_cont,
                            'val_loss_text': val_loss_text,
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                        })
                iter_count += 1
        

if __name__ == '__main__':
    if use_wandb:
        wandb.init(project='ren')

    with open(f'configs/train_dinov3_vitl16.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()