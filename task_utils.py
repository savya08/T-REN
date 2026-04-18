import os
import math
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def upsample_features(image_features, new_h, new_w, padded_h, padded_w, upsampling_method='bilinear'):
    if upsampling_method == 'bilinear':
        upsampled_feature = torch.nn.functional.interpolate(image_features, 
                                                            size=[padded_h, padded_w], mode='bilinear')
        upsampled_feature = T.CenterCrop((new_h, new_w))(upsampled_feature)
    else:
        raise ValueError(f'{upsampling_method} is not a valid upsampling method.')
    return upsampled_feature


def visualize_features(features, image, save_path):
    image_height, image_width = image.shape[1], image.shape[2]

    pca = PCA(n_components=3)
    reshaped_features = features.permute(1, 2, 0).reshape(image_height * image_width, -1).float().numpy()
    pca.fit(reshaped_features)
    pca_features = pca.transform(reshaped_features)
    pca_features = (pca_features - pca_features.min(axis = -1)[..., None]) / \
        (pca_features.max(axis = -1)[..., None] - pca_features.min(axis = -1)[..., None])
    vis_features = pca_features.reshape(image_height, image_width, 3)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(vis_features)
    plt.axis('off')
    plt.savefig(save_path)
    plt.clf()


def visualize_cosine_similarity(features, images, save_dir, grid_size=64):
    os.makedirs(save_dir, exist_ok=True)
    features = F.normalize(features, p=2, dim=1).flatten(-2)
    batch_size, _, num_tokens = features.shape
    
    for batch_idx in range(batch_size):
        similarity_map = features[batch_idx].t().mm(features[batch_idx])
        for token_idx in range(num_tokens):
            token_similarity_map = similarity_map[token_idx]
            token_similarity_map = token_similarity_map.reshape(grid_size, grid_size)

            row = token_idx // grid_size
            col = token_idx % grid_size

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(images[batch_idx].cpu().permute(1, 2, 0).float().numpy())
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(token_similarity_map.float().detach().cpu().numpy())
            plt.plot(col, row, 'rx', markersize=3, markeredgewidth=2, label='Query token')
            plt.axis('off')
            os.makedirs(f'{save_dir}/batch-{batch_idx}', exist_ok=True)
            plt.savefig(f'{save_dir}/batch-{batch_idx}/token-{token_idx}.jpg')
            plt.clf()
            plt.close()


def visualize_regions(regions, image, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, mask in enumerate(regions):
        plt.imshow(mask[:, :, None] * image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{idx}.jpg'))
        plt.clf()
    
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'image.jpg'))
    plt.clf()


def visualize_attn_weights(attn_weights, images, patch_size, grid_points=None, attn_aggregation='max', save_dir='attn_vis'):
    batch_size, num_heads, num_q, _ = attn_weights.shape
    h, w = images.shape[-2:]

    for batch_idx in range(images.shape[0]):
        batch_dir = f'{save_dir}/batch-{batch_idx}'
        os.makedirs(batch_dir, exist_ok=True)
        plt.imshow(images[batch_idx].permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')
        plt.savefig(f'{batch_dir}/image.jpg')
        plt.clf()

        attn_weights = attn_weights.view(batch_size, num_heads, num_q, h // patch_size, w // patch_size)

        for q_idx in range(num_q):
            attn_map = F.sigmoid(attn_weights[batch_idx, :, q_idx]).detach().cpu().numpy()
            if attn_aggregation == 'max':
                combined_attn_map = np.max(attn_map, axis=0)
            elif attn_aggregation == 'mean':
                combined_attn_map = np.mean(attn_map, axis=0)
            plt.imshow(combined_attn_map)
            plt.axis('off')
            if grid_points is not None:
                plt.scatter([grid_points[batch_idx][q_idx][1] / patch_size], [grid_points[batch_idx][q_idx][0] / patch_size],
                            marker='o', s=20, c='red')
            plt.savefig(f'{batch_dir}/query-{q_idx}.jpg')
            plt.close()


def pad_or_truncate_tokens(tokens, pad_length, pad_value):
    current_length, dim_size = tokens.shape
    
    if current_length > pad_length:
        return tokens[:pad_length]
    
    if current_length < pad_length:
        padding = torch.full((pad_length - current_length, dim_size), pad_value, 
                              dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, padding], dim=0)


def print_log(log_str, save_dir=None):
    print(log_str)
    if save_dir is not None:
        log_file = os.path.join(save_dir, 'log.txt')
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')