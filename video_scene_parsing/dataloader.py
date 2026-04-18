import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class VSPWSegmentation(Dataset):
    def __init__(self, config, split='val'):
        self.root_dir = config['data']['vspw_root_dir']
        self.split = split
        self.num_classes = config['data']['vspw_num_classes']

        data_dir = os.path.join(self.root_dir, 'data')
        split_file = os.path.join(self.root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]

        self.data = {}
        for video_id in video_ids:
            origin_dir = os.path.join(data_dir, video_id, 'origin')
            mask_dir = os.path.join(data_dir, video_id, 'mask')
            if not os.path.isdir(origin_dir) or not os.path.isdir(mask_dir):
                continue
            frames = [f for f in os.listdir(origin_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._')]
            frames = sorted(frames)
            valid = []
            for frame in frames:
                base, _ = os.path.splitext(frame)
                if os.path.exists(os.path.join(mask_dir, base + '.png')):
                    valid.append(frame)
            if len(valid) >= 1:
                self.data[video_id] = valid
        self.video_ids = list(self.data.keys())
        self.data_dir = os.path.join(self.root_dir, 'data')

    def preprocess_mask(self, label):
        label = np.asarray(label, dtype=np.int64).copy()
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
        return label

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = self.data[video_id]

        images, masks = [], []
        mask_height, mask_width = None, None
        for frame_id in frames:
            base, _ = os.path.splitext(frame_id)
            image_path = os.path.join(self.data_dir, video_id, 'origin', frame_id)
            mask_path = os.path.join(self.data_dir, video_id, 'mask', base + '.png')

            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask_height, mask_width = mask.shape[0], mask.shape[1]
            mask = self.preprocess_mask(mask)

            images.append(image)
            masks.append(mask)

        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)
        base_first = os.path.splitext(frames[0])[0]
        base_last = os.path.splitext(frames[-1])[0]
        image_id = f"{video_id}_{base_first}_to_{base_last}"
        return {
            'images': images,
            'masks': masks,
            'image_id': image_id,
            'mask_shape': (mask_height, mask_width),
        }