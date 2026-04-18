import os
import io
import yaml
import random
import requests
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from model import FeatureExtractor, RegionEncoder, TextEncoder


seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TREN(nn.Module):
    def __init__(self, config):
        super(TREN, self).__init__()
        
        # Create the models
        self.tren_image_encoder = FeatureExtractor(config, device=device).eval()
        self.tren_region_encoder = RegionEncoder(config).to(device).eval()
        self.tren_text_encoder = TextEncoder(config, device=device).eval()

        # Load the checkpoint
        self.checkpoint_path = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'], 'tren_region_encoder.pth')
        self.load_checkpoint()
        
        # Grid points for region tokens
        image_resolution = config['parameters']['image_resolution']
        patch_size = config['architecture']['patch_size']
        self.grid_size = image_resolution // patch_size
        x_coords = np.linspace(1, image_resolution - 2, self.grid_size, dtype=int)
        y_coords = np.linspace(1, image_resolution - 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            if 'ren_region_encoder_state' in checkpoint:
                self.tren_region_encoder.load_state_dict_resolution_agnostic(checkpoint['ren_region_encoder_state'])
                print('T-REN region encoder loaded from checkpoint')
            if 'ren_image_encoder_state' in checkpoint:
                self.tren_image_encoder.load_state_dict(checkpoint['ren_image_encoder_state'])
                print('T-REN image encoder loaded from checkpoint')
            if 'ren_text_encoder_state' in checkpoint:
                self.tren_text_encoder.load_state_dict(checkpoint['ren_text_encoder_state'])
                print('T-REN text encoder loaded from checkpoint')
        else:
            print('No T-REN checkpoint found, exiting.')
            exit()
    
    def visualize_regions(self, region_masks, images, mask_labels=None, save_dir='region_vis'):
        for batch_idx in range(len(region_masks)):
            os.makedirs(f'{save_dir}/batch-{batch_idx}', exist_ok=True)
            for region_idx in range(region_masks[batch_idx].shape[0]):
                plt.subplot(1, 2, 1)
                plt.imshow(images[batch_idx].permute(1, 2, 0).detach().cpu().numpy())
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(region_masks[batch_idx][region_idx].detach().cpu().numpy(), cmap='magma')
                plt.title(mask_labels[batch_idx][region_idx])
                plt.axis('off')

                plt.tight_layout(pad=0.5)
                plt.savefig(f'{save_dir}/batch-{batch_idx}/region-{region_idx}.png', bbox_inches='tight', pad_inches=0.2)
                plt.close()

    def forward(self, images, texts=None, aggregate_tokens=True):
        prompts = [self.grid_points for _ in range(images.shape[0])]
        with torch.no_grad():
            backbone_outputs = self.tren_image_encoder(images)
            feature_maps = backbone_outputs['feature_maps']
            class_tokens = backbone_outputs['text_aligned_class_tokens']
            tren_outputs = self.tren_region_encoder(feature_maps, prompts, aggregate_tokens=aggregate_tokens)
            tren_outputs['class_tokens'] = class_tokens
        if texts is not None:
            text_encodings = self.tren_text_encoder(texts)
            tren_outputs['text_encodings'] = text_encodings
        return tren_outputs


def test_tren():
    with open(f'configs/train_dinov3_vitl16.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load and process image
    url = 'https://storage.googleapis.com/labelbox-datasets/facebook_sam_1b/data/sa_100372.jpg'
    image = requests.get(url).content
    image = Image.open(io.BytesIO(image)).convert('RGB')
    transforms = T.Compose([T.Resize((config['parameters']['image_resolution'], config['parameters']['image_resolution'])),
                            T.ToTensor()])
    image = transforms(image)
    image = image.unsqueeze(0).to(device)

    # Define the text categories
    text = ['sky', 'lamp', 'clock tower', 'clock', 'car', 'wall', 'ground', 'road', 'dog']
    
    # Load TREN
    tren = TREN(config)

    # Process the image
    tren_outputs = tren(image, texts=text)
    text_aligned_tokens = tren_outputs['text_aligned_tokens'][0]
    region_masks = tren_outputs['region_masks']
    text_encodings = tren_outputs['text_encodings']

    # For each region, find the most similar text label
    similarity = F.normalize(text_aligned_tokens, dim=-1) @ F.normalize(text_encodings, dim=-1).T
    best_idxs = similarity.argmax(dim=-1)
    predicted_categories = [[text[i] for i in best_idxs]]

    # Visualize the regions
    num_regions = text_aligned_tokens.shape[0]
    save_dir = 'region_vis/'
    tren.visualize_regions(region_masks, image, mask_labels=predicted_categories, save_dir=save_dir)
    print()
    print(f'Number of tokens needed to encode the image: {num_regions}.')
    print(f'Saved the cross-attention maps along with the region labels to `{save_dir}`.')


if __name__ == '__main__':
    test_tren()