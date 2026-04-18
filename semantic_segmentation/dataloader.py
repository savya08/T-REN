import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class ImageMaskTransform:
    def __init__(self, image_resolution=224):
        self.image_resolution = image_resolution

    def __call__(self, image, mask):
        image = T.functional.resize(image, (self.image_resolution, self.image_resolution), interpolation=T.InterpolationMode.BICUBIC)
        image = T.functional.to_tensor(image)
        mask = T.functional.resize(mask, (self.image_resolution, self.image_resolution), interpolation=T.InterpolationMode.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)
        return image, mask


class ADESegmentation(Dataset):
    def __init__(self, config, split='val'):
        self.root_dir = config['data']['ade_root_dir']
        self.split = split
        
        self.image_dir = os.path.join(self.root_dir, 'images', self.split)
        self.mask_dir = os.path.join(self.root_dir, 'annotations', self.split)
        
        self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

        image_resolution = config['tren']['parameters']['image_resolution']
        self.transform = ImageMaskTransform(image_resolution)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Load image
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Load segmentation mask
        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask_width, mask_height = mask.size

        # Apply transformations to the image and the mask
        image, mask = self.transform(image, mask)
        mask -= 1
        return {
            'image': image,
            'mask': mask,
            'image_id': image_name[:-4],
            'mask_shape': (mask_height, mask_width),
        }


class CityscapesSegmentation(Dataset):
    def __init__(self, config, split='val'):
        self.root_dir = config['data']['cityscapes_root_dir']
        self.split = split
        
        image_paths = []
        image_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
        for dir in os.listdir(image_dir):
            for file in os.listdir(os.path.join(image_dir, dir)):
                if file.endswith('.png'):
                    image_paths.append(os.path.join(image_dir, dir, file))
        self.image_paths = sorted(image_paths)
        
        mask_paths = []
        mask_dir = os.path.join(self.root_dir, 'gtFine', self.split)
        for dir in os.listdir(mask_dir):
            for file in os.listdir(os.path.join(mask_dir, dir)):
                if file.endswith('.png') and 'labelIds' in file:
                    mask_paths.append(os.path.join(mask_dir, dir, file))
        self.mask_paths = sorted(mask_paths)
        
        image_resolution = config['tren']['parameters']['image_resolution']
        self.transform = ImageMaskTransform(image_resolution)

        id_map = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3,
            13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
            26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255,
        }
        self.id_lookup = torch.full((256,), 255, dtype=torch.long)
        for old_id, new_id in id_map.items():
            if old_id >= 0:
                self.id_lookup[old_id] = new_id
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load segmentation mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        mask_width, mask_height = mask.size

        # Apply transformations to the image and the mask
        image, mask = self.transform(image, mask)

        # Map mask values to train IDs
        mask = mask.long()
        mask = torch.where(mask == -1, torch.tensor(255, dtype=torch.long), mask)
        mask = torch.clamp(mask, 0, 255)
        mask_shape = mask.shape
        mask_flat = mask.flatten()
        mask_flat = self.id_lookup[mask_flat]
        mask = mask_flat.reshape(mask_shape)
        return {
            'image': image,
            'mask': mask,
            'image_id': image_path.split('/')[-1].split('.')[0],
            'mask_shape': (mask_height, mask_width),
        }