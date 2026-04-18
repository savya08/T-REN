import os
import math
import logging
import numpy as np
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from task_utils import CenterPadding, upsample_features


logging.getLogger().setLevel(logging.WARNING)


class FeatureExtractor(nn.Module):
    def __init__(self, config, device, return_class_token=False):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = config['pretrained']['feature_extractor']
        self.patch_size = config['architecture']['patch_size']
        self.backbone_ckpt_path = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'], 
                                               'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        self.head_ckpt_path = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'],
                                           'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth')
        self.return_class_token = return_class_token
        self.device = device

        if self.feature_extractor == 'dinov3_vitl16':
            self.backbone, _ = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16_dinotxt_tet1280d20h24l', 
                                              backbone_weights=self.backbone_ckpt_path, weights=self.head_ckpt_path)
            del self.backbone.text_model
            self.model = self.backbone.visual_model.backbone.to(device)
            self.head = self.backbone.visual_model.head.to(device)
        else:
            raise ValueError(f'Feature extractor {self.feature_extractor} not supported.')
    
    def extract_dinov3(self, images, batch_size=1024, patch_length=16, layers=[23]):
        transform = kornia.augmentation.AugmentationSequential(
            CenterPadding(multiple=patch_length),
            kornia.augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        transformed_images = transform(images)
        
        class_tokens, text_aligned_class_tokens, patch_tokens, register_tokens, feature_maps = [], [], [], [], []
        for i in range(0, transformed_images.shape[0], batch_size):
            image_batch = transformed_images[i:(i + batch_size)].to(device=self.device)
            with torch.inference_mode():
                features_out = self.model.get_intermediate_layers(image_batch, return_class_token=True,
                                                                  return_extra_tokens=True, n=layers)
                class_tokens.append(features_out[-1][1])
                patch_tokens.append(features_out[0][0])
                register_tokens.append(features_out[0][2])

                head_inputs = torch.cat([class_tokens[-1].unsqueeze(1), register_tokens[-1], patch_tokens[-1]], dim=1)
                text_aligned_class_token = self.head(head_inputs)[0][0:1]
                text_aligned_class_tokens.append(text_aligned_class_token)

                B, _, C = patch_tokens[-1].size()
                H, W = image_batch.shape[2], image_batch.shape[3]
                patch_H, patch_W = math.ceil(H / patch_length), math.ceil(W / patch_length)
                feature_maps.append(patch_tokens[-1].permute(0, 2, 1).view(B, C, patch_H, patch_W))

        class_tokens = torch.cat(class_tokens, dim=0)
        text_aligned_class_tokens = torch.cat(text_aligned_class_tokens, dim=0)
        patch_tokens = torch.cat(patch_tokens, dim=0)
        register_tokens = torch.cat(register_tokens, dim=0)
        feature_maps = torch.cat(feature_maps, dim=0)
        return class_tokens, text_aligned_class_tokens, patch_tokens, register_tokens, feature_maps
    
    def forward(self, images, resize=False):
        if self.feature_extractor == 'dinov3_vitl16':
            class_tokens, text_aligned_class_tokens, patch_tokens, register_tokens, feature_maps = self.extract_dinov3(images)
        
        if resize:
            image_height, image_width = images.shape[2], images.shape[3]
            padded_height = math.ceil(image_height / self.patch_size) * self.patch_size
            padded_width = math.ceil(image_width / self.patch_size) * self.patch_size
            resized_feature_maps = []
            chunk_size = 32
            for i in range(0, len(feature_maps), chunk_size):
                resized_feature_maps.append(upsample_features(feature_maps[i:i + chunk_size], image_height,
                                                              image_width, padded_height, padded_width))
            feature_maps = torch.cat(resized_feature_maps)
        
        return {
            'class_tokens': class_tokens,
            'text_aligned_class_tokens': text_aligned_class_tokens,
            'patch_tokens': patch_tokens,
            'register_tokens': register_tokens,
            'feature_maps': feature_maps
        }


class RegionTokenGenerator(nn.Module):
    def __init__(self, pooling_method='average', device='cuda'):
        super(RegionTokenGenerator, self).__init__()
        self.model, _ = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16_dinotxt_tet1280d20h24l')
        self.model = self.model.visual_model.head.to(device)
        self.pooling_method = pooling_method
        self.device = device

    def forward(self, regions, class_tokens, feature_maps, register_tokens):
        pooled_tokens, text_aligned_tokens = [], []
        for scale_idx in range(regions.shape[2]):
            scale_pooled_tokens, scale_text_aligned_tokens = [], []
            for batch_idx in range(len(regions)):
                image_regions = regions[batch_idx, :, scale_idx]
                image_class_tokens = class_tokens[batch_idx]
                image_feature_maps = feature_maps[batch_idx]
                image_register_tokens = register_tokens[batch_idx]
                if image_regions.numel() == 0:
                    scale_pooled_tokens.append(torch.zeros((0, image_feature_maps.shape[0]), device=self.device))
                    scale_text_aligned_tokens.append(torch.zeros((0, image_feature_maps.shape[0]), device=self.device))
                    continue

                # Get the features that pertain to the regions
                region_features = torch.einsum('rhw,chw->rc', image_regions.float(), image_feature_maps)
                text_alignment_inputs = torch.cat([image_class_tokens[None], image_register_tokens, region_features], dim=0)
                text_aligned_region_features = self.model(text_alignment_inputs[None])[0][image_register_tokens.shape[0] + 1 :]
                
                # Pool the region features
                if self.pooling_method == 'average':
                    valid_elements = image_regions.sum(dim=(1, 2), dtype=torch.float32).clamp(min=1).unsqueeze(1)
                    region_features = region_features / valid_elements
                    text_aligned_region_features = text_aligned_region_features / valid_elements
                else:
                    raise ValueError(f'Pooling method {self.pooling_method} not supported.')
                scale_pooled_tokens.append(region_features)
                scale_text_aligned_tokens.append(text_aligned_region_features)
            scale_pooled_tokens = torch.stack(scale_pooled_tokens)
            scale_text_aligned_tokens = torch.stack(scale_text_aligned_tokens)
            pooled_tokens.append(scale_pooled_tokens.unsqueeze(2))
            text_aligned_tokens.append(scale_text_aligned_tokens.unsqueeze(2))
        pooled_tokens = torch.cat(pooled_tokens, dim=2)
        text_aligned_tokens = torch.cat(text_aligned_tokens, dim=2)
        return {
            'pooled_tokens': pooled_tokens,
            'text_aligned_tokens': text_aligned_tokens
        }


class TextEncoder(nn.Module):
    def __init__(self, config, device, prompt='photo of a '):
        super(TextEncoder, self).__init__()
        self.ckpt_path = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'],
                                      'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth')
        self.model, self.tokenizer = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16_dinotxt_tet1280d20h24l', 
                                                    weights=self.ckpt_path)
        del self.model.visual_model
        self.model = self.model.to(device)
        self.prompt = prompt
        self.device = device

    def forward(self, texts):
        texts = [self.prompt + t for t in texts]
        text_tokens = self.tokenizer.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings[:, text_embeddings.shape[1] // 2 :]
        return text_embeddings


class PositionalEmbedding2D(nn.Module):
    def __init__(self, embedding_dim=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        generator = torch.Generator()
        generator.manual_seed(42)
        self.register_buffer("positional_encoding_gaussian_matrix", 
                             scale * torch.randn((2, embedding_dim // 2), generator=generator))

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class AttentionLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads=8, dropout=0.1, use_bias=False, use_v_proj=True, use_out_proj=True):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'Hidden dimension must be a multiple of the number of heads.'
        self.head_dim = hidden_dim // num_heads
        if not use_v_proj:
            assert kv_dim == hidden_dim, 'Key and value dimensions must be the same as the hidden dimension if not using v_proj.'

        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=use_bias)
        nn.init.kaiming_normal_(self.q_proj.weight, mode='fan_in', nonlinearity='linear')
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=use_bias)
        nn.init.kaiming_normal_(self.k_proj.weight, mode='fan_in', nonlinearity='linear')
        if use_v_proj:
            self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=use_bias)
            nn.init.kaiming_normal_(self.v_proj.weight, mode='fan_in', nonlinearity='linear')
        else:
            self.v_proj = nn.Identity()
        if use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            if use_v_proj:
                nn.init.zeros_(self.v_proj.bias)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)
        if use_out_proj:
            self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
            nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')
            if use_bias:
                nn.init.zeros_(self.out_proj.bias)
        else:
            self.out_proj = nn.Identity()

        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None, attn_threshold=None):
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        query = self.q_proj(q).view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        key = self.k_proj(k).view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        value = self.v_proj(v).view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        if attn_threshold is not None:
            max_attn_scores, _ = attn_scores.max(dim=-1, keepdim=True)
            thresholding_mask = attn_scores >= (attn_threshold * max_attn_scores)
            attn_scores = attn_scores.masked_fill(thresholding_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, value)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_dim)

        out = self.out_proj(attn_out)
        return out, attn_weights


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.linear1(x)
        z = self.gelu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        return z


class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, mlp_dim, num_heads, dropout, use_bias):
        super(CrossAttentionBlock, self).__init__()
        self.query_norm = nn.LayerNorm(q_dim)
        self.cross_attn = AttentionLayer(q_dim, kv_dim, hidden_dim, num_heads, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context, mask=None):
        x = self.query_norm(query)
        x, attn_scores = self.cross_attn(q=x, k=context, v=context, mask=mask)
        x = self.dropout(x)
        x = x + query

        y = self.mlp_norm(x)
        y = self.mlp(y)
        out = self.out_norm(y) + x
        return out, attn_scores


class TextAlignmentBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, output_dim, dropout=0.1):
        super(TextAlignmentBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.linear1(x)
        z = self.gelu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        return z


class TokenAggregator(nn.Module):
    def __init__(self, config):
        super(TokenAggregator, self).__init__()
        self.merging_iou_threshold = config['parameters']['merging_iou_threshold']
        self.merging_similarity_threshold = config['parameters']['merging_similarity_threshold']
        self.binarization_threshold = config['parameters'].get('binarization_threshold', 0.5)

    def _compute_binary_masks(self, region_masks):
        num_masks = region_masks.shape[0]
        return (region_masks.reshape(num_masks, -1) > self.binarization_threshold).float()

    def _compute_iou_matrix(self, binary_masks):
        num_masks = binary_masks.shape[0]
        if num_masks == 0:
            return torch.zeros(0, 0, device=binary_masks.device)
        intersection = torch.mm(binary_masks, binary_masks.t())
        areas = binary_masks.sum(dim=1)
        union = areas.unsqueeze(1) + areas.unsqueeze(0) - intersection
        return intersection / torch.clamp(union, min=1.0)

    def _find_connected_components(self, adjacency):
        n = adjacency.shape[0]
        if n == 0:
            return []
        
        # Initialize labels
        labels = torch.arange(n, device=adjacency.device)
        
        # Iterative label propagation
        for _ in range(int(np.ceil(np.log2(n + 1))) + 1):
            neighbor_labels = torch.where(adjacency, labels.unsqueeze(0).expand(n, -1), labels.unsqueeze(1).expand(-1, n))
            new_labels = neighbor_labels.min(dim=1)[0]
            new_labels = torch.minimum(new_labels, labels)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
        
        # Convert to groups
        labels_cpu = labels.cpu().tolist()
        label_to_group = {}
        for idx, label in enumerate(labels_cpu):
            if label not in label_to_group:
                label_to_group[label] = []
            label_to_group[label].append(idx)
        return list(label_to_group.values())

    def _compute_token_similarity_matrix(self, pred_tokens):
        num_tokens = pred_tokens.shape[0]
        if num_tokens == 0:
            return torch.zeros(0, 0, device=pred_tokens.device)
        pred_tokens = F.normalize(pred_tokens, p=2, dim=-1)
        return torch.mm(pred_tokens, pred_tokens.t())

    def group_predictions(self, region_masks, pred_tokens=None):
        num_masks = region_masks.shape[0]
        if num_masks == 0:
            return []
        binary_masks = self._compute_binary_masks(region_masks)
        iou_matrix = self._compute_iou_matrix(binary_masks)
        mask_adjacency = iou_matrix > self.merging_iou_threshold
        if pred_tokens is not None and pred_tokens.shape[0] == num_masks:
            token_sim = self._compute_token_similarity_matrix(pred_tokens)
            token_adjacency = token_sim > self.merging_similarity_threshold
            adjacency = mask_adjacency | token_adjacency
        else:
            adjacency = mask_adjacency
        return self._find_connected_components(adjacency)

    def forward(self, ren_outputs, remove_singleton_groups=True):
        pred_tokens = ren_outputs['pred_tokens']
        region_masks = ren_outputs['region_masks']
        text_aligned_tokens = ren_outputs['text_aligned_tokens']

        pred_tokens = torch.flatten(pred_tokens, 1, 2)
        region_masks = torch.flatten(region_masks, 1, 2)
        text_aligned_tokens = torch.flatten(text_aligned_tokens, 1, 2)

        aggregated_outputs = {'pred_tokens': [], 'region_masks': [], 'text_aligned_tokens': []}
        for batch_idx in range(pred_tokens.shape[0]):
            batch_pred_tokens = pred_tokens[batch_idx]
            batch_region_masks = region_masks[batch_idx]
            batch_text_aligned_tokens = text_aligned_tokens[batch_idx]

            groups = self.group_predictions(batch_region_masks, batch_pred_tokens)

            kept_groups = []
            for local_group_idxs in groups:
                if remove_singleton_groups and len(local_group_idxs) == 1:
                    continue
                global_idxs = torch.tensor(local_group_idxs, device=batch_region_masks.device)
                group_mean_mask = batch_region_masks[global_idxs].mean(dim=0)
                kept_groups.append({'global_idxs': global_idxs, 'mean_mask': group_mean_mask})

            if len(kept_groups) == 0:
                mask_areas = batch_region_masks.sum(dim=(-2, -1))
                best_idx = mask_areas.argmax()
                kept_groups.append({
                    'global_idxs': best_idx.unsqueeze(0),
                    'mean_mask': batch_region_masks[best_idx],
                })

            new_pred_tokens, new_region_masks, new_text_aligned_tokens = [], [], []
            for gd in kept_groups:
                global_idxs = gd['global_idxs']
                new_pred_tokens.append(pred_tokens[batch_idx][global_idxs].mean(dim=0))
                new_region_masks.append(gd['mean_mask'])
                new_text_aligned_tokens.append(text_aligned_tokens[batch_idx][global_idxs].mean(dim=0))

            aggregated_outputs['pred_tokens'].append(torch.stack(new_pred_tokens, dim=0))
            aggregated_outputs['region_masks'].append(torch.stack(new_region_masks, dim=0))
            aggregated_outputs['text_aligned_tokens'].append(torch.stack(new_text_aligned_tokens, dim=0))
        return aggregated_outputs


class RegionEncoder(nn.Module):
    def __init__(self, config):
        super(RegionEncoder, self).__init__()
        hidden_dim = config['architecture']['hidden_dim']
        text_embed_dim = config['architecture']['text_embed_dim']
        image_resolution = config['parameters']['image_resolution']
        patch_size = config['architecture']['patch_size']
        feature_map_resolution = image_resolution // patch_size
        self.feature_map_resolution = feature_map_resolution
        self.image_resolution = image_resolution

        # Create position embeddings for the prompts and feature maps
        position_embedder = PositionalEmbedding2D(hidden_dim)
        location_embeddings = position_embedder((image_resolution, image_resolution))
        feature_embeddings = position_embedder((feature_map_resolution, feature_map_resolution)).flatten(-2).permute(1, 0)
        self.register_buffer('location_embeddings', location_embeddings)
        self.register_buffer('feature_embeddings', feature_embeddings)

        # Define scale embeddings for multiscale region tokens
        self.num_multiscale_regions = config['parameters']['num_multiscale_regions']
        self.scale_embeddings = nn.Embedding(self.num_multiscale_regions, hidden_dim)
        nn.init.normal_(self.scale_embeddings.weight, std=0.02)

        # Instantiate the prompt and region attention layers
        self.num_decoder_layers = config['architecture']['num_decoder_layers']
        self.num_attention_heads = config['architecture']['num_attention_heads']
        self.prompt_attention_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_heads=self.num_attention_heads)
            for _ in range(self.num_decoder_layers)
        ])
        self.prompt_attention_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_decoder_layers)])
        self.region_attention_layers = nn.ModuleList([
            CrossAttentionBlock(q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, mlp_dim=2 * hidden_dim,
                                num_heads=self.num_attention_heads, dropout=0.1, use_bias=False) 
            for _ in range(self.num_decoder_layers)
        ])
        self.region_attention_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_decoder_layers)])

        # Instantiate the region token prediction head
        self.token_prediction_head = AttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_heads=1, dropout=0.0, 
                                                    use_v_proj=False, use_out_proj=False)

        # Instantiate the text alignment head
        self.text_alignment_block = TextAlignmentBlock(hidden_dim, 2 * hidden_dim, text_embed_dim)

        # Instantiate the token aggregator
        self.token_aggregator = TokenAggregator(config)

    def load_state_dict_resolution_agnostic(self, state_dict, strict=False):
        model_state = self.state_dict()
        new_state = dict(state_dict)

        # Interpolate location_embeddings if spatial size differs
        if 'location_embeddings' in new_state and new_state['location_embeddings'].shape != model_state['location_embeddings'].shape:
            old = new_state['location_embeddings']
            target_shape = model_state['location_embeddings'].shape
            if old.shape[0] == target_shape[0]:
                resized = F.interpolate(old.unsqueeze(0), size=(target_shape[1], target_shape[2]), mode='bilinear', align_corners=False)
                new_state['location_embeddings'] = resized.squeeze(0)
            else:
                new_state['location_embeddings'] = model_state['location_embeddings'].clone()

        # Interpolate feature_embeddings if spatial size differs
        if 'feature_embeddings' in new_state and new_state['feature_embeddings'].shape != model_state['feature_embeddings'].shape:
            old = new_state['feature_embeddings']
            target = model_state['feature_embeddings']
            if old.shape[1] == target.shape[1]:
                num_pos_old, C = old.shape
                num_pos_new = target.shape[0]
                h_old = int(round(num_pos_old ** 0.5))
                w_old = num_pos_old // h_old
                h_new = int(round(num_pos_new ** 0.5))
                w_new = num_pos_new // h_new
                old_2d = old.view(h_old, w_old, C).permute(2, 0, 1).unsqueeze(0)
                resized = F.interpolate(old_2d, size=(h_new, w_new), mode='bilinear', align_corners=False)
                new_state['feature_embeddings'] = resized.squeeze(0).permute(1, 2, 0).reshape(-1, C)
            else:
                new_state['feature_embeddings'] = model_state['feature_embeddings'].clone()

        return self.load_state_dict(new_state, strict=strict)

    def forward(self, feature_maps, grid_points, aggregate_tokens=False, remove_singleton_groups=True):
        if isinstance(grid_points, list):
            grid_points = torch.stack([gp.to(feature_maps.device) for gp in grid_points])
        batch_size, num_prompts, _ = grid_points.shape

        # Create scale prompt embeddings for multiscale region tokens
        scale_prompt_embeddings = self.scale_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        scale_prompt_embeddings = scale_prompt_embeddings.unsqueeze(1).repeat(1, num_prompts, 1, 1)

        # Create spatial prompt embeddings to encode the location of the point prompts
        spatial_prompt_embeddings = self.location_embeddings[:, grid_points[..., 0], grid_points[..., 1]]
        spatial_prompt_embeddings = spatial_prompt_embeddings.permute(1, 2, 0).unsqueeze(2)
        spatial_prompt_embeddings = spatial_prompt_embeddings.repeat(1, 1, self.num_multiscale_regions, 1)

        # Create the query tokens
        q = scale_prompt_embeddings

        # Get the key and value tokens for the region attention layers
        kv = feature_maps.flatten(-2).permute(0, 2, 1)
        kv = kv + self.feature_embeddings[None]

        # Apply the region attention layers and the prompt attention layers
        for layer_idx in range(self.num_decoder_layers):
            q += spatial_prompt_embeddings

            # Apply the region attention layer
            q = q.reshape(batch_size, num_prompts * self.num_multiscale_regions, -1)
            q, _ = self.region_attention_layers[layer_idx](q, kv)
            q = q.reshape(batch_size, num_prompts, self.num_multiscale_regions, -1)
            q = self.region_attention_norms[layer_idx](q)

            # Apply the prompt attention layer
            q = q.reshape(batch_size * num_prompts, self.num_multiscale_regions, -1)
            q, _ = self.prompt_attention_layers[layer_idx](q, q, q)
            q = self.prompt_attention_norms[layer_idx](q)
            q = q.reshape(batch_size, num_prompts, self.num_multiscale_regions, -1)
        prompt_tokens = q

        # Get the region tokens
        q = prompt_tokens.reshape(batch_size, num_prompts * self.num_multiscale_regions, -1)
        k = kv
        v = kv - self.feature_embeddings[None]
        pred_tokens, attn_weights = self.token_prediction_head(q, k, v)
        pred_tokens = pred_tokens.reshape(batch_size, num_prompts, self.num_multiscale_regions, -1)
        attn_weights = attn_weights.reshape(batch_size, num_prompts, self.num_multiscale_regions, -1)

        # Get the region masks
        region_masks = attn_weights / attn_weights.max(dim=-1, keepdim=True)[0]
        region_masks = region_masks.reshape(batch_size, num_prompts, self.num_multiscale_regions, 
                                            self.feature_map_resolution, self.feature_map_resolution)

        # Get text aligned tokens
        text_aligned_tokens = self.text_alignment_block(pred_tokens)

        outputs = {
            'pred_tokens': pred_tokens,
            'region_masks': region_masks,
            'text_aligned_tokens': text_aligned_tokens,
        }
        if aggregate_tokens:
            outputs = self.token_aggregator(outputs, remove_singleton_groups=remove_singleton_groups)
        return outputs


if __name__ == '__main__':
    import yaml
    from tqdm import tqdm
    from dataloader import COCOStuffDataset

    # Load the config
    with open('configs/train_dinov3_vitl16.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    dataset = COCOStuffDataset(config, 'val')

    # Generate the grid points
    image_resolution = config['parameters']['image_resolution']
    patch_size = config['architecture']['patch_size']
    grid_size = image_resolution // patch_size
    x_coords = np.linspace(patch_size // 2, image_resolution - patch_size // 2, grid_size, dtype=int)
    y_coords = np.linspace(patch_size // 2, image_resolution - patch_size // 2, grid_size, dtype=int)
    grid_points = np.array([(y, x) for y in y_coords for x in x_coords])
    grid_points = torch.tensor(grid_points)[None]
    
    # Get the models
    region_encoder = RegionEncoder(config).to(device)
    feature_extractor = FeatureExtractor(config, device)

    # Generate the region tokens
    for item in tqdm(dataset):
        image = item[0].to(device)
        feature_maps = feature_extractor(image[None])['feature_maps']
        ren_outputs = region_encoder(feature_maps, grid_points, aggregate_tokens=True)
        print(ren_outputs['pred_tokens'][0].shape)
        print(ren_outputs['region_masks'][0].shape)
        print(ren_outputs['text_aligned_tokens'][0].shape)
        break