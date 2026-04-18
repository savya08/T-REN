import os
import json
import cv2
import mmap
import pickle
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
from pycocotools import mask
from concurrent.futures import ProcessPoolExecutor, as_completed


CATEGORIES = set()


def collate_fn(batch):
    # Separate region-region and region-text samples
    region_region_items, region_text_items = [], []
    for item in batch:
        if item[0]['dataset_type'] == 'region_region':
            region_region_items.append(item)
        else:
            region_text_items.append(item)
    
    # Helper function to create batch from items
    def create_batch_from_items(items):
        if not items:
            v1 = {
                'images': torch.empty(0, 3, 224, 224),
                'region_masks': [],
                'attn_masks': [],
                'region_ids': torch.empty(0, dtype=torch.long),
                'region_types': [],
                'categories': [],
                'descriptions': [],
                'loss_mask': torch.empty(0, dtype=torch.bool),
                'grid_points': []
            }
            v2 = {
                'images': torch.empty(0, 3, 224, 224),
                'region_masks': [],
                'attn_masks': [],
                'region_ids': torch.empty(0, dtype=torch.long),
                'region_types': [],
                'categories': [],
                'descriptions': [],
                'loss_mask': torch.empty(0, dtype=torch.bool),
                'grid_points': []
            }
            return v1, v2
        
        v1 = {}
        v1['images'] = torch.stack([item[0]['image'] for item in items])
        v1['region_masks'] = torch.stack([torch.tensor(np.array(item[0]['region_masks'])) for item in items])
        v1['attn_masks'] = torch.stack([torch.tensor(np.array(item[0]['attn_masks'])) for item in items])
        v1['region_ids'] = torch.stack([torch.tensor(np.array(item[0]['region_ids'])) for item in items])
        v1['categories'] = [item[0]['categories'] for item in items]
        v1['descriptions'] = [item[0]['descriptions'] for item in items]
        v1['loss_mask'] = torch.stack([torch.tensor(np.array(item[0]['loss_mask'])) for item in items])
        v1['grid_points'] = torch.stack([torch.tensor(np.array(item[0]['grid_points'])) for item in items])
        
        v2 = {}
        v2['images'] = torch.stack([item[1]['image'] for item in items])
        v2['region_masks'] = torch.stack([torch.tensor(np.array(item[1]['region_masks'])) for item in items])
        v2['attn_masks'] = torch.stack([torch.tensor(np.array(item[1]['attn_masks'])) for item in items])
        v2['region_ids'] = torch.stack([torch.tensor(np.array(item[1]['region_ids'])) for item in items])
        v2['categories'] = [item[1]['categories'] for item in items]
        v2['descriptions'] = [item[1]['descriptions'] for item in items]
        v2['loss_mask'] = torch.stack([torch.tensor(np.array(item[1]['loss_mask'])) for item in items])
        v2['grid_points'] = torch.stack([torch.tensor(np.array(item[1]['grid_points'])) for item in items])
        
        return v1, v2
    
    # Create the four separate batches
    region_region_v1, region_region_v2 = create_batch_from_items(region_region_items)
    region_text_v1, region_text_v2 = create_batch_from_items(region_text_items)
    return region_region_v1, region_region_v2, region_text_v1, region_text_v2


def _process_single_annotation_sa1b(args):
    idx, annotation_path = args
    masks = []
    mask_array = None
    save_resolution = 256
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            if 'annotations' in data and isinstance(data['annotations'], list):
                for annotation in data['annotations']:
                    if 'segmentation' in annotation:
                        segmentation = annotation['segmentation']
                        mask_array = mask.decode(segmentation)
                        mask_array = cv2.resize(mask_array, (save_resolution, save_resolution), interpolation=cv2.INTER_NEAREST)
                        
                        # Pack mask and convert to bytes
                        packed_mask = np.packbits(mask_array.astype(np.uint8))
                        mask_bytes = packed_mask.tobytes()
                        masks.append(mask_bytes)
    except Exception as e:
        print(f"Error processing {annotation_path}: {e}")
    return idx, masks, mask_array.shape if mask_array is not None else None


class SA1BDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        images_dir = config['data'][f'sa1b_{split}_images_dir']
        annotations_dir = config['data'][f'sa1b_{split}_annotations_dir']
        self.images_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))]
        self.annotations_paths = [os.path.join(annotations_dir, f) for f in sorted(os.listdir(annotations_dir))]
        self.binary_cache_dir = Path(config['data'][f'sa1b_regions_binary_cache_dir'])
        self.split = split

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Cache regions masks as binaries for faster processing
        os.makedirs(self.binary_cache_dir, exist_ok=True)
        self.cache_metadata_path = os.path.join(self.binary_cache_dir, f'{split}_cache_metadata.pkl')
        self.binary_cache_path = os.path.join(self.binary_cache_dir, f'{split}_binary_cache.bin')
        if not (os.path.exists(self.cache_metadata_path) and os.path.exists(self.binary_cache_path)):
            self._cache_mask_binaries()
        
        # Initialize the binary cache by loading the meta-data and performing memory mapping
        with open(self.cache_metadata_path, 'rb') as f:
            self.cache_metadata = pickle.load(f)
        self.binary_file = open(self.binary_cache_path, 'rb')
        self.mm = mmap.mmap(self.binary_file.fileno(), 0, access=mmap.ACCESS_READ)

    def _cache_mask_binaries(self):
        metadata = {}
        offset = 0
        num_workers = min(os.cpu_count() or 4, 32)
        batch_size = num_workers * 32
        
        # Open binary file once for writing
        with open(self.binary_cache_path, 'wb') as binary_file:
            # Process annotations in batches to avoid memory issues with large datasets
            total_files = len(self.annotations_paths)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for batch_start in tqdm(range(0, total_files, batch_size), 
                                       desc=f'Caching SA-1B masks for {self.split}'):
                    batch_end = min(batch_start + batch_size, total_files)
                    batch_indices = range(batch_start, batch_end)
                    
                    # Submit batch of tasks
                    future_to_idx = {
                        executor.submit(_process_single_annotation_sa1b, (idx, self.annotations_paths[idx])): idx
                        for idx in batch_indices
                    }
                    
                    # Collect results for this batch as they complete
                    batch_results = {}
                    for future in as_completed(future_to_idx):
                        idx, masks, shape = future.result()
                        batch_results[idx] = (masks, shape)
                    
                    # Write batch results to binary file in order
                    for idx in batch_indices:
                        masks, shape = batch_results[idx]
                        
                        # Save metadata for the masks
                        if masks and shape is not None:
                            metadata[idx] = {
                                'offset': offset,
                                'sizes': [len(m) for m in masks],
                                'shape': shape
                            }
                            
                            # Write masks to binary file
                            for mask_bytes in masks:
                                binary_file.write(mask_bytes)
                                offset += len(mask_bytes)
                        else:
                            # No masks for this annotation
                            metadata[idx] = {
                                'offset': offset,
                                'sizes': [],
                                'shape': None
                            }
        
        # Save metadata
        with open(self.cache_metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Read masks from the memory-mapped file
        regions = []
        metadata = self.cache_metadata[idx]
        
        if metadata['sizes']:
            offset = metadata['offset']
            shape = metadata['shape']
            for size in metadata['sizes']:
                self.mm.seek(offset)
                mask_bytes = self.mm.read(size)
                packed_mask = np.frombuffer(mask_bytes, dtype=np.uint8)
                unpacked_mask = np.unpackbits(packed_mask)[:np.prod(shape)].reshape(shape)
                resized_mask = cv2.resize(unpacked_mask.astype(np.float32), 
                                         (image.shape[1], image.shape[2]), 
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                regions.append(resized_mask)
                offset += size

        # Remove regions that are too small
        filtered_regions = []
        for region in regions:
            if region is not None and np.sum(region) > 512:
                filtered_regions.append(region)
        regions = filtered_regions
        
        # Add region categories and descriptions (adjust based on your JSON structure)
        categories = ['none'] * len(regions)
        descriptions = ['none'] * len(regions)
        return image, regions, categories, descriptions
    
    def __del__(self):
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'binary_file'):
            self.binary_file.close()


class COCOStuffDataset(Dataset):
    def __init__(self, config, split):
        image_dir = config['data'][f'cocostuff_{split}_images_dir']
        annotations_path = config['data'][f'cocostuff_{split}_annotations_path']
        labels_path = config['data'][f'cocostuff_labels_path']
        self.image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        self.annotations_paths = [os.path.join(annotations_path, f) for f in sorted(os.listdir(annotations_path))]

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Load the object categories
        self.categories = []
        with open(labels_path, 'r') as f:
            for line in f.read().splitlines():
                category_name = line.split(': ')[-1]
                self.categories.append(category_name)
                CATEGORIES.add(category_name)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        annotations_path = self.annotations_paths[idx]
        annotations = Image.open(annotations_path).convert('RGB')
        annotations = np.array(annotations)
        regions, categories, descriptions = [], [], []
        for category_id in np.unique(annotations):
            if category_id == 0 or category_id == 255:
                continue
            region = (annotations == category_id).astype(np.uint8)
            region = cv2.resize(region, (image.shape[1], image.shape[2]))
            regions.append(region[:, :, 0])
            categories.append(self.categories[category_id + 1])
            descriptions.append('none')
        return image, regions, categories, descriptions


class OpenImagesDataset(Dataset):
    def __init__(self, config, split):
        self.image_dir = config['data'][f'openimages_{split}_images_dir']
        self.masks_dir = config['data'][f'openimages_{split}_masks_dir']
        annotations_path = config['data'][f'openimages_{split}_annotations_path']
        class_labels_path = config['data'][f'openimages_class_labels_path']
        self.split = split

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Load the annotations
        annotations = pd.read_csv(annotations_path)
        class_labels = pd.read_csv(class_labels_path)
        id_to_label = dict(zip(class_labels['LabelName'], class_labels['DisplayName']))
        self.image_ids = list(annotations['ImageID'].unique())
        self.mask_ids = annotations.groupby('ImageID')['MaskPath'].apply(list).to_dict()
        self.label_ids = annotations.groupby('ImageID')['LabelName'].apply(list).to_dict()
        for image_id in self.label_ids:
            self.label_ids[image_id] = [id_to_label[label_id].lower() for label_id in self.label_ids[image_id]]
        CATEGORIES.update([label.lower() for label in class_labels['DisplayName'].unique().tolist()])

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get regions
        regions = []
        for mask_id in self.mask_ids[image_id]:
            mask = Image.open(os.path.join(self.masks_dir, mask_id))
            mask = np.array(mask).astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[2]))
            regions.append(mask)
        
        # Get categories and descriptions
        categories = self.label_ids[image_id]
        descriptions = ['none'] * len(categories)
        return image, regions, categories, descriptions


class MapillaryDataset(Dataset):
    def __init__(self, config, split):
        self.image_dir = config['data'][f'mapillary_{split}_images_dir']
        self.image_ids = [f.split('.')[0] for f in sorted(os.listdir(self.image_dir))]
        annotations_dir = config['data'][f'mapillary_{split}_annotations_dir']
        self.instance_masks_dir = os.path.join(annotations_dir, 'instances')
        self.labels_dir = os.path.join(annotations_dir, 'labels')
        object_categories_path = config['data'][f'mapillary_object_categories_path']

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Load the object categories
        self.categories = {}
        with open(object_categories_path, 'r') as f:
            data = json.load(f)
            labels = data['labels']
            for category_id in range(len(labels)):
                self.categories[category_id] = labels[category_id]['readable'].lower()
                CATEGORIES.add(labels[category_id]['readable'].lower())

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Load instance mask and label mask
        mask_path = os.path.join(self.instance_masks_dir, f'{image_id}.png')
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.uint32)
        label_path = os.path.join(self.labels_dir, f'{image_id}.png')
        label = Image.open(label_path)
        label = np.array(label).astype(np.uint32)

        # Get regions, categories and descriptions
        regions, categories, descriptions = [], [], []
        for instance_id in np.unique(mask):
            region = (mask == instance_id).astype(np.uint8)
            label_id = np.unique(label[region == 1])[0]
            assert len(np.unique(label[region == 1])) == 1, 'Multiple label ids found in the same region'
            region = cv2.resize(region, (image.shape[1], image.shape[2]))
            if np.sum(region) < 400:
                continue
            category = self.categories[label_id]
            if category  == 'unlabeled':
                continue
            regions.append(region)
            categories.append(category)
            descriptions.append('none')
        return image, regions, categories, descriptions


class PhraseCutDataset(Dataset):
    def __init__(self, config, split):
        image_dir = config['data'][f'phrasecut_{split}_images_dir']
        annotations_path = config['data'][f'phrasecut_{split}_annotations_path']
        split_info_path = config['data'][f'phrasecut_split_info_path']
        self.split = split

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Load the split info
        self.image_ids, self.data_info = [], {}
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        for row in split_info:
            if row['split'] == self.split:
                self.image_ids.append(row['image_id'])
                self.data_info[row['image_id']] = {
                    'image_path': os.path.join(image_dir, f"{row['image_id']}.jpg"),
                    'height': row['height'],
                    'width': row['width'],
                    'region_polygons': [],
                    'region_categories': [],
                    'region_descriptions': [],
                }

        # Load the annotations
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        for annotation in annotations:
            self.data_info[annotation['image_id']]['region_polygons'].append(annotation['Polygons'])
            self.data_info[annotation['image_id']]['region_categories'].append(annotation['phrase_structure']['name'].lower())
            self.data_info[annotation['image_id']]['region_descriptions'].append(annotation['phrase'].lower())
            CATEGORIES.add(annotation['phrase_structure']['name'].lower())

    def polygons_to_mask(self, polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        rings = [np.array(ring, dtype=np.float32) for ring in polygons[0]]
        rings = [np.round(r).astype(np.int32) for r in rings]
        if len(rings) >= 1:
            cv2.fillPoly(mask, [rings[0]], color=1)
        for hole in rings[1:]:
            cv2.fillPoly(mask, [hole], color=0)
        return mask
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_data = self.data_info[image_id]
        image_path = image_data['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get regions, categories and descriptions
        regions = []
        for region_polygons in image_data['region_polygons']:
            region_mask = self.polygons_to_mask(region_polygons, image_data['height'], image_data['width'])
            region_mask = cv2.resize(region_mask, (image.shape[1], image.shape[2]))
            regions.append(region_mask)
        categories = image_data['region_categories']
        descriptions = image_data['region_descriptions']
        return image, regions, categories, descriptions


class MUSEDataset(Dataset):
    def __init__(self, config, split):
        self.image_dir = config['data'][f'muse_{split}_images_dir']
        self.annotations_path = config['data'][f'muse_{split}_annotations_path']
        self.split = split

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Cache images if they are not already cached
        if not os.path.exists(self.image_dir):
            self._cache_images()
        self.image_paths = [os.path.join(self.image_dir, f) for f in sorted(os.listdir(self.image_dir))]

        # Create a look up table for region texts
        self.region_category, self.region_description = {}, {}
        with open(self.annotations_path, 'r') as f:
            annotations = json.load(f)

        for annotation in annotations:
            image_id = annotation['id']
            region_categories, region_descriptions = [], []
            for ann in annotation['ann_list']:
                category_name = ann['category_name'].replace('_', ' ').lower()
                CATEGORIES.add(category_name)
                region_categories.append(category_name)
                if 'rephrased_name' in ann:
                    region_descriptions.append(ann['rephrased_name'])
                else:
                    region_descriptions.append('none')
            self.region_category[image_id] = region_categories
            self.region_description[image_id] = region_descriptions

        # Cache regions masks as binaries for faster processing
        self.binary_cache_dir = Path(config['data'][f'muse_{split}_regions_binary_cache_dir'])
        os.makedirs(self.binary_cache_dir, exist_ok=True)
        self.cache_metadata_path = os.path.join(self.binary_cache_dir, f'{split}_cache_metadata.pkl')
        self.binary_cache_path = os.path.join(self.binary_cache_dir, f'{split}_binary_cache.bin')
        if not (os.path.exists(self.cache_metadata_path) and os.path.exists(self.binary_cache_path)):
            self._cache_mask_binaries()

        # Initialize the binary cache by loading the meta-data and performing memory mapping of the cache file
        with open(self.cache_metadata_path, 'rb') as f:
            self.cache_metadata = pickle.load(f)
        self.binary_file = open(self.binary_cache_path, 'rb')
        self.mm = mmap.mmap(self.binary_file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def _cache_images(self):
        os.makedirs(self.image_dir, exist_ok=True)
        with open(self.annotations_path, 'r') as f:
            annotations = json.load(f)

        for annotation in tqdm(annotations, desc=f'Caching images for muse {self.split} split'):
            image_id = annotation['id']
            image_url = annotation['coco_url']
            image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
            if os.path.exists(image_path):
                continue
            try:
                image = requests.get(image_url)
            except Exception as e:
                print(f'Error caching image {image_url}: {e}')
                continue
            with open(image_path, 'wb') as f: 
                f.write(image.content)

    def _cache_mask_binaries(self):
        metadata = {}
        offset = 0
        with open(self.annotations_path, 'r') as f:
            annotations = json.load(f)

        with open(self.binary_cache_path, 'wb') as binary_file:
            for annotation in tqdm(annotations, desc=f'Caching masks for muse {self.split} split'):
                image_id = annotation['id']
                height, width = annotation['height'], annotation['width']
                masks = []
                total_size = 0
                for ann in annotation['ann_list']:
                    # Decode the mask RLE
                    segmentation_rle = mask.frPyObjects(ann['segmentation'], height, width)
                    segmentation_rle = mask.merge(segmentation_rle)
                    segmentation_mask = mask.decode(segmentation_rle)
                    
                    # Convert the mask to binary format
                    packed_mask = np.packbits(segmentation_mask.astype(np.uint8))
                    mask_bytes = packed_mask.tobytes()
                    masks.append(mask_bytes)
                    total_size += len(mask_bytes)
                
                # Save metadata for the masks
                metadata[image_id] = {
                    'offset': offset,
                    'sizes': [len(m) for m in masks],
                    'shape': segmentation_mask.shape
                }

                # Write masks to binary file
                for mask_bytes in masks:
                    binary_file.write(mask_bytes)
                    offset += len(mask_bytes)
        
        with open(self.cache_metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image_id = int(image_path.split('/')[-1].split('.')[0])

        # Get region categories and descriptions
        categories = self.region_category[image_id]
        descriptions = self.region_description[image_id]

        # Read mask from the memory-mapped file
        regions = []
        metadata = self.cache_metadata[image_id]
        offset = metadata['offset']
        shape = metadata['shape']
        for size in metadata['sizes']:
            self.mm.seek(offset)
            mask_bytes = self.mm.read(size)
            packed_mask = np.frombuffer(mask_bytes, dtype=np.uint8)
            unpacked_mask = np.unpackbits(packed_mask)[:np.prod(shape)].reshape(shape)
            resized_mask = cv2.resize(unpacked_mask, (image.shape[1], image.shape[2]))
            regions.append(resized_mask)
            offset += size

        return image, regions, categories, descriptions

    def __del__(self):
        self.mm.close()
        self.binary_file.close()


class RENDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        
        # Create a dictionary of all datasets
        image_datasets = {}
        if f'sa1b_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'sa1b_{split}'] = SA1BDataset(config, split)
        if f'cocostuff_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'cocostuff_{split}'] = COCOStuffDataset(config, split)
        if f'openimages_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'openimages_{split}'] = OpenImagesDataset(config, split)
        if f'phrasecut_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'phrasecut_{split}'] = PhraseCutDataset(config, split)
        if f'mapillary_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'mapillary_{split}'] = MapillaryDataset(config, split)

        # Define datasets for loss conditioning
        self.region_region_datasets = ['sa1b_train', 'sa1b_val']
        self.region_text_datasets = ['cocostuff_train', 'cocostuff_val', 'openimages_train', 'openimages_val', 
                                     'phrasecut_train', 'phrasecut_val', 'mapillary_train', 'mapillary_val']
        
        # Create a list of all datasets and a look-up table to index images in the cumulative data pool
        self.datasets = []
        self.dataset_idxs = {}
        idx = 0
        for dataset_idx, dataset_name in enumerate(config['data'][f'{split}_datasets']):
            dataset = image_datasets[dataset_name]
            self.datasets.append(dataset)
            for image_idx in range(len(dataset)):
                self.dataset_idxs[idx] = (dataset_idx, image_idx)
                idx += 1
        assert len(self.datasets) > 0, 'No dataset is specified'

        # Define weights for our data sampler
        dataset_weights = config['data']['weights']
        dataset_weights = [w / sum(dataset_weights) for w in dataset_weights]
        sample_weights = []
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_weight = dataset_weights[dataset_idx]
            dataset_samples = len(dataset)
            sample_weights.extend([dataset_weight / dataset_samples] * dataset_samples)
        self.sample_weights = torch.tensor(sample_weights)

        # Additional parameters
        self.image_resolution = config['parameters']['image_resolution']
        self.grid_size = config['architecture']['grid_size']
        x_coords = np.linspace(0, self.image_resolution - 1, self.grid_size, dtype=int)
        y_coords = np.linspace(0, self.image_resolution - 1, self.grid_size, dtype=int)
        self.grid_points = np.array([(y, x) for y in y_coords for x in x_coords])
        self.region_region_max_prompts = config['parameters']['region_region_max_prompts']
        self.region_text_max_prompts = config['parameters']['region_text_max_prompts']
        self.num_multiscale_regions = config['parameters']['num_multiscale_regions']
        self.downsample_masks = config['parameters']['downsample_masks']
        self.mask_resolution = config['parameters']['mask_resolution']
        self.patch_size = config['architecture']['patch_size']
        self.crop_ratio = config['parameters']['crop_ratio']

    def __len__(self):
        return len(self.dataset_idxs)
    
    def __getitem__(self, idx):
        dataset_idx, image_idx = self.dataset_idxs[idx]
        dataset = self.datasets[dataset_idx]
        image, regions, categories, descriptions = dataset[image_idx]

        # Get dataset name for loss conditioning
        dataset_name = self.config['data'][f'{self.split}_datasets'][dataset_idx]
        dataset_type = 'region_region' if dataset_name in self.region_region_datasets else 'region_text'
        if dataset_name in self.region_region_datasets:
            max_prompts = self.region_region_max_prompts
        elif dataset_name in self.region_text_datasets:
            max_prompts = self.region_text_max_prompts
        else:
            raise ValueError(f'Dataset {dataset_name} is not categorized for loss conditioning')

        # Get two views of the image and regions
        image_v1, regions_v1 = self.apply_transforms(image, regions, crop_ratio=self.crop_ratio)
        image_v2, regions_v2 = self.apply_transforms(image, regions, crop_ratio=self.crop_ratio)
        
        # Sample grid points for the two views
        grid_points_v1 = self.sample_grid_points(regions_v1, max_prompts)
        grid_points_v2 = self.sample_grid_points(regions_v2, max_prompts)

        # Arrange the regions in prompt order
        arranged_v1 = self.arrange_regions(regions_v1, grid_points_v1, categories, descriptions, null_region_id=-1)
        arranged_v2 = self.arrange_regions(regions_v2, grid_points_v2, categories, descriptions, null_region_id=-2)

        # Downsample the masks for attention supervision
        masks_v1 = [torch.tensor(np.array(mask)).unsqueeze(1).float() for mask in arranged_v1['region_masks']]
        masks_v2 = [torch.tensor(np.array(mask)).unsqueeze(1).float() for mask in arranged_v2['region_masks']]
        mask_size = (image.shape[1] // self.patch_size, image.shape[2] // self.patch_size)
        masks_v1_resized = [F.interpolate(mask, size=mask_size, mode='area').squeeze(1).numpy() for mask in masks_v1]
        masks_v2_resized = [F.interpolate(mask, size=mask_size, mode='area').squeeze(1).numpy() for mask in masks_v2] 
        arranged_v1['attn_masks'] = masks_v1_resized
        arranged_v2['attn_masks'] = masks_v2_resized

        # Downsample the masks for region mask prediction
        if self.downsample_masks:
            masks_v1 = [torch.tensor(np.array(mask)).unsqueeze(1).float() for mask in arranged_v1['region_masks']]
            masks_v2 = [torch.tensor(np.array(mask)).unsqueeze(1).float() for mask in arranged_v2['region_masks']]
            mask_size = (self.mask_resolution, self.mask_resolution)
            masks_v1_resized = [F.interpolate(mask, size=mask_size, mode='area').squeeze(1).numpy() for mask in masks_v1]
            masks_v2_resized = [F.interpolate(mask, size=mask_size, mode='area').squeeze(1).numpy() for mask in masks_v2] 
            arranged_v1['region_masks'] = masks_v1_resized
            arranged_v2['region_masks'] = masks_v2_resized

        # Collate info for the two views
        v1 = {
            'image': image_v1,
            'region_masks': arranged_v1['region_masks'],
            'region_ids': arranged_v1['region_ids'],
            'attn_masks': arranged_v1['attn_masks'],
            'categories': arranged_v1['categories'],
            'descriptions': arranged_v1['descriptions'],
            'loss_mask': arranged_v1['loss_mask'],
            'grid_points': arranged_v1['grid_points'],
            'dataset_type': dataset_type,
        }
        v2 = {
            'image': image_v2,
            'region_masks': arranged_v2['region_masks'],
            'region_ids': arranged_v2['region_ids'],
            'attn_masks': arranged_v2['attn_masks'],
            'categories': arranged_v2['categories'],
            'descriptions': arranged_v2['descriptions'],
            'loss_mask': arranged_v2['loss_mask'],
            'grid_points': arranged_v2['grid_points'],
            'dataset_type': dataset_type,
        }
        return v1, v2
    
    def apply_transforms(self, image, regions, flip=True, brightness_param=0.4, contrast_param=0.4, crop_ratio=0.3, 
                         saturation_param=0.4, sharpness_param=0.4, rotation_angle=15, shear_x=15, shear_y=15):
        image_pil = T.ToPILImage()(image)
        regions_pil = [Image.fromarray(r.astype(np.uint8) * 255) for r in regions]

        if flip:
            do_flip = np.random.rand() > 0.5
        else:
            do_flip = False
        crop_size = np.random.randint(int(crop_ratio * image.shape[2]), image.shape[2])
        crop_params = T.RandomCrop.get_params(image_pil, output_size=(crop_size, crop_size))

        def synchronized_transform(img, blur_and_jitter=True, skip_transforms=True):
            if skip_transforms:
                return img
            if do_flip:
                img = T.functional.hflip(img)
            if blur_and_jitter:
                img = T.ColorJitter(brightness=brightness_param, contrast=contrast_param, saturation=saturation_param)(img)
                img = T.RandomAdjustSharpness(sharpness_factor=sharpness_param, p=1.0)(img)
            img = T.functional.affine(img, translate=(0.0, 0.0), scale=1.0, angle=rotation_angle, shear=(shear_x, shear_y))
            img = T.functional.crop(img, *crop_params)
            img = T.functional.resize(img, (image.shape[1], image.shape[2]), interpolation=Image.BICUBIC)
            return img

        transformed_image = T.ToTensor()(synchronized_transform(image_pil))
        transformed_regions = [np.array(synchronized_transform(mask_pil)) // 255 for mask_pil in regions_pil]
        return transformed_image, transformed_regions
    
    def arrange_regions(self, regions, grid_points, categories, descriptions, null_region_id=-1):
        arranged = {}
        arranged['region_masks'] = []
        arranged['region_ids'] = []
        arranged['grid_points'] = []
        arranged['categories'] = []
        arranged['descriptions'] = []
        arranged['loss_mask'] = []

        for point in grid_points:
            y, x = point
            regions_on_point, region_ids_on_point = [], []
            categories_on_point, descriptions_on_point, loss_mask_on_point = [], [], []
            for region_id, (region, category, description) in enumerate(zip(regions, categories, descriptions)):
                if region[y, x]:
                    regions_on_point.append(region)
                    region_ids_on_point.append(region_id)
                    categories_on_point.append(category)
                    descriptions_on_point.append(description)
                    loss_mask_on_point.append(1)
                    if len(regions_on_point) == self.num_multiscale_regions:
                        break
            
            pad_length = self.num_multiscale_regions - len(regions_on_point)
            regions_on_point.extend([np.zeros((self.image_resolution, self.image_resolution))] * pad_length)
            region_ids_on_point.extend([null_region_id] * pad_length)
            categories_on_point.extend(['none'] * pad_length)
            descriptions_on_point.extend(['none'] * pad_length)
            loss_mask_on_point.extend([0] * pad_length)

            arranged['region_masks'].append(regions_on_point)
            arranged['region_ids'].append(region_ids_on_point)
            arranged['categories'].append(categories_on_point)
            arranged['descriptions'].append(descriptions_on_point)
            arranged['loss_mask'].append(loss_mask_on_point)
            arranged['grid_points'].append(point)
        return arranged

    def sample_grid_points(self, regions, max_prompts):
        if len(regions) == 0:
            sampled_grid_points = random.choices(range(len(self.grid_points)), k=max_prompts)
            return self.grid_points[sampled_grid_points]

        # Compute the number of regions on each grid point
        union_mask = np.zeros_like(regions[0], dtype=int)
        for region in regions:
            if region is not None:
                union_mask += region.astype(int)
                
        # Sample grid points using the union mask as a probability distribution
        point_weights = [union_mask[y, x] * union_mask[y, x] for y, x in self.grid_points]
        
        # Check if all weights are zero 
        if sum(point_weights) == 0:
            sampled_grid_points = random.choices(range(len(self.grid_points)), k=max_prompts)
        else:
            sampled_grid_points = random.choices(range(len(self.grid_points)), weights=point_weights, k=max_prompts)
        
        grid_points = self.grid_points[sampled_grid_points]
        return grid_points

    def get_weighted_sampler(self):
        return WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self), replacement=True)


if __name__ == '__main__':
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    import yaml

    with open(f'configs/train_dinov3_vitl16.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = RENDataset(config, 'train')
    for data_idx, item in enumerate(dataset):
        v1, v2 = item
        image = v1['image']
        regions = v1['region_masks']
        region_ids = v1['region_ids']
        categories = v1['categories']
        descriptions = v1['descriptions']
        grid_points = v1['grid_points']

        for point_idx in range(len(regions)):
            plt.subplot(1, 4, 1)
            plt.imshow(image.permute(1, 2, 0).numpy())
            plt.scatter([grid_points[point_idx][1]], [grid_points[point_idx][0]], marker='x', s=18, c='red')
            plt.axis('off')

            for region_idx in range(len(regions[point_idx])):
                plt.subplot(1, 4, region_idx + 2)
                region = cv2.resize(regions[point_idx][region_idx], (image.shape[2], image.shape[1]))
                plt.imshow(region)
                plt.scatter([grid_points[point_idx][1]], [grid_points[point_idx][0]], marker='x', s=18, c='red')
                plt.axis('off')
            
            os.makedirs(f'test-data-vis-{data_idx}', exist_ok=True)
            plt.tight_layout(pad=0.5)
            plt.savefig(f'test-data-vis-{data_idx}/x{grid_points[point_idx][0]}-y{grid_points[point_idx][1]}.jpg', bbox_inches='tight', pad_inches=0.2)
            plt.clf()
            plt.close()