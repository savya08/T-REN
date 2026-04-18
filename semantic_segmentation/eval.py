import os
import sys
import yaml
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
from fast_slic import Slic
from scipy import ndimage as ndi
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from dataloader import ADESegmentation, CityscapesSegmentation

sys.path.append('..')
from model import FeatureExtractor, RegionEncoder, TextEncoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
logging.getLogger().setLevel(logging.WARNING)


PROMPT_TEMPLATES = (
    "a bad photo of a {0}.",
    "a photo of many {0}.",
    "a sculpture of a {0}.",
    "a photo of the hard to see {0}.",
    "a low resolution photo of the {0}.",
    "a rendering of a {0}.",
    "graffiti of a {0}.",
    "a bad photo of the {0}.",
    "a cropped photo of the {0}.",
    "a tattoo of a {0}.",
    "the embroidered {0}.",
    "a photo of a hard to see {0}.",
    "a bright photo of a {0}.",
    "a photo of a clean {0}.",
    "a photo of a dirty {0}.",
    "a dark photo of the {0}.",
    "a drawing of a {0}.",
    "a photo of my {0}.",
    "the plastic {0}.",
    "a photo of the cool {0}.",
    "a close-up photo of a {0}.",
    "a black and white photo of the {0}.",
    "a painting of the {0}.",
    "a painting of a {0}.",
    "a pixelated photo of the {0}.",
    "a sculpture of the {0}.",
    "a bright photo of the {0}.",
    "a cropped photo of a {0}.",
    "a plastic {0}.",
    "a photo of the dirty {0}.",
    "a jpeg corrupted photo of a {0}.",
    "a blurry photo of the {0}.",
    "a photo of the {0}.",
    "a good photo of the {0}.",
    "a rendering of the {0}.",
    "a {0} in a video game.",
    "a photo of one {0}.",
    "a doodle of a {0}.",
    "a close-up photo of the {0}.",
    "a photo of a {0}.",
    "the origami {0}.",
    "the {0} in a video game.",
    "a sketch of a {0}.",
    "a doodle of the {0}.",
    "a origami {0}.",
    "a low resolution photo of a {0}.",
    "the toy {0}.",
    "a rendition of the {0}.",
    "a photo of the clean {0}.",
    "a photo of a large {0}.",
    "a rendition of a {0}.",
    "a photo of a nice {0}.",
    "a photo of a weird {0}.",
    "a blurry photo of a {0}.",
    "a cartoon {0}.",
    "art of a {0}.",
    "a sketch of the {0}.",
    "a embroidered {0}.",
    "a pixelated photo of a {0}.",
    "itap of the {0}.",
    "a jpeg corrupted photo of the {0}.",
    "a good photo of a {0}.",
    "a plushie {0}.",
    "a photo of the nice {0}.",
    "a photo of the small {0}.",
    "a photo of the weird {0}.",
    "the cartoon {0}.",
    "art of the {0}.",
    "a drawing of the {0}.",
    "a photo of the large {0}.",
    "a black and white photo of a {0}.",
    "the plushie {0}.",
    "a dark photo of a {0}.",
    "itap of a {0}.",
    "graffiti of the {0}.",
    "a toy {0}.",
    "itap of my {0}.",
    "a photo of a cool {0}.",
    "a photo of a small {0}.",
    "a tattoo of the {0}.",
)


def intersect_and_union(prediction, label, num_labels, ignore_index, label_map=None, reduce_labels=False,
                        reduce_predictions=False):
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    prediction = np.array(prediction)
    label = np.array(label)

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    if reduce_predictions:
        prediction[prediction == 0] = 255
        prediction = prediction - 1
        prediction[prediction == 254] = 255

    prediction = prediction[label != ignore_index]

    label = label[label!= ignore_index]
    intersect = prediction[prediction == label]
    area_intersect = np.histogram(intersect, bins=num_labels, range=(0, num_labels))[0]
    area_pred_label = np.histogram(prediction, bins=num_labels, range=(0, num_labels))[0]
    area_label = np.histogram(label, bins=num_labels, range=(0, num_labels))[0]
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(predictions, targets, num_labels, ignore_index, label_map=None, reduce_labels=False,
                              reduce_pred_labels=False):
    total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
    total_area_union = np.zeros((num_labels,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
    total_area_label = np.zeros((num_labels,), dtype=np.float64)
    for prediction, target in zip(predictions, targets):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(prediction, target, num_labels, 
                                                                                      ignore_index, label_map, reduce_labels, 
                                                                                      reduce_pred_labels)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def mean_iou(predictions, targets, num_labels, ignore_index, nan_to_num=None, label_map=None, reduce_labels=False,
             reduce_pred_labels=False):
    total_area_intersect, total_area_union, _, total_area_label = total_intersect_and_union(predictions, targets, num_labels,
                                                                                            ignore_index, label_map, reduce_labels,
                                                                                            reduce_pred_labels)
    return mean_iou_from_totals(total_area_intersect, total_area_union, total_area_label, nan_to_num=nan_to_num)


def mean_iou_from_totals(total_area_intersect, total_area_union, total_area_label, nan_to_num=None):
    metrics = {}
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label

    metrics['mean_iou'] = np.nanmean(iou)
    metrics['mean_accuracy'] = np.nanmean(acc)
    metrics['overall_accuracy'] = all_acc
    metrics['per_category_iou'] = iou
    metrics['per_category_accuracy'] = acc
    if nan_to_num is not None:
        metrics = dict({metric: np.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in metrics.items()})
    return metrics


def get_slic_points(images, num_segments):
    prompts, superpixels = [], []
    for image in images:
        image = (image.permute(1, 2, 0).cpu().numpy().copy() * 255).astype(np.uint8)

        # Get SLIC superpixels
        slic = Slic(num_components=num_segments, compactness=256)
        segments = slic.iterate(image)
        slic_segments = segments.max() + 1

        # Get center of mass for all segments
        centers = np.array(ndi.center_of_mass(np.ones_like(segments), labels=segments, index=np.arange(slic_segments)))
        centers = np.round(centers).astype(int)

        centers[:, 0] = np.clip(centers[:, 0], 0, segments.shape[0] - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, segments.shape[1] - 1)

        # Check which centers are outside their own superpixel
        valid = segments[centers[:, 0], centers[:, 1]] == np.arange(slic_segments)

        # For invalid centers, pick a pixel inside the superpixel
        if not np.all(valid):
            for seg_id in np.where(~valid)[0]:
                mask = (segments == seg_id)
                yx = np.argwhere(mask)
                if len(yx) > 0:
                    centers[seg_id] = yx[len(yx) // 2]
        centers = torch.tensor(centers, dtype=torch.int64)

        # Pad if needed
        pad_len = num_segments - len(centers)
        if pad_len > 0:
            center_padding = torch.stack([centers[-1]] * pad_len)
            centers = torch.cat([centers, center_padding], dim=0)

        prompts.append(centers)
        superpixels.append(segments)
    return prompts, superpixels


class Evaluator():
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f'Configs: {config}')

        # Instantiate the dataloaders
        self.target_data = config['data']['target_data']
        if self.target_data == 'ade20k':
            val_dataset = ADESegmentation(config, split='validation')
            self.val_loader = DataLoader(val_dataset, batch_size=1)
            self.num_classes = config['data']['ade_num_classes']
        elif self.target_data == 'cityscapes':
            val_dataset = CityscapesSegmentation(config, split='val')
            self.val_loader = DataLoader(val_dataset, batch_size=1)
            self.num_classes = config['data']['cityscapes_num_classes']

        # Create the models
        self.tren_image_encoder = FeatureExtractor(config['tren'], device=device).eval()
        self.tren_region_encoder = RegionEncoder(config['tren']).to(device).eval()
        self.tren_text_encoder = TextEncoder(config['tren'], device=device, prompt='').eval()
        self.patch_size = config['tren']['architecture']['patch_size']
        if self.target_data == 'ade20k':
            self.text_embeddings = self.get_ade_text_embeddings()
        elif self.target_data == 'cityscapes':
            self.text_embeddings = self.get_cityscapes_text_embeddings()

        # Create prompts for region encoder
        self.image_resolution = config['tren']['parameters']['image_resolution']
        self.grid_size = self.image_resolution // self.patch_size
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Load checkpoints
        self.tren_checkpoint = os.path.join(config['tren']['logging']['save_dir'], config['tren']['logging']['exp_name'], 'tren_region_encoder.pth')
        self.load_tren()

        # Colormap for visualizing results
        self.colormap = np.random.randint(0, 256, size=(self.num_classes + 1, 3), dtype=np.uint8)

    def visualize(self, image, prediction, target, save_path, ignore_index=255):
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if torch.is_tensor(prediction):
            prediction = prediction.numpy()
        if torch.is_tensor(target):
            target = target.numpy()
        prediction = prediction.copy()
        target = target.copy()
        prediction[prediction == ignore_index] = self.num_classes
        target[target == ignore_index] = self.num_classes
        prediction = self.colormap[prediction]
        target = self.colormap[target]

        _, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].imshow(prediction)
        axes[1].axis('off')
        axes[2].imshow(target)
        axes[2].axis('off')
        plt.savefig(save_path)
        plt.clf()
        plt.close()

    def load_tren(self):
        if os.path.exists(self.tren_checkpoint):
            checkpoint = torch.load(self.tren_checkpoint)
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

    def get_ade_text_embeddings(self):
        categories = [
            "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
            "cabinet", "sidewalk", "person", "ground", "door", "table", "mountain", "plant", "curtain", "chair",
            "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
            "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
            "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
            "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
            "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
            "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
            "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
            "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
            "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
            "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
            "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
            "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
            "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
        ]
        text_feats = []
        for class_name in tqdm(categories, desc="Class names"):
            text = [template.format(class_name) for template in PROMPT_TEMPLATES]
            feats = self.tren_text_encoder(text)
            feats = F.normalize(feats, p=2, dim=-1)
            feats = feats.mean(dim=0)
            feats = F.normalize(feats, p=2, dim=-1)
            text_feats.append(feats)
        text_feats = torch.stack(text_feats)
        return text_feats
    
    def get_cityscapes_text_embeddings(self):
        categories = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
        ]
        text_feats = []
        for class_name in tqdm(categories, desc="Class names"):
            text = [template.format(class_name) for template in PROMPT_TEMPLATES]
            feats = self.tren_text_encoder(text)
            feats = F.normalize(feats, p=2, dim=-1)
            feats = feats.mean(dim=0)
            feats = F.normalize(feats, p=2, dim=-1)
            text_feats.append(feats)
        text_feats = torch.stack(text_feats)
        return text_feats
    
    def step(self, batch, use_slic, aggregate_tokens):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        batch_size = images.shape[0]
        token_count = 0

        with torch.inference_mode():
            with autocast('cuda', dtype=torch.bfloat16):
                # Compute the outputs
                tren_image_encoder_outputs = self.tren_image_encoder(images, resize=False)
                feature_maps = tren_image_encoder_outputs['feature_maps']
                if use_slic:
                    prompts, superpixels = get_slic_points(images, self.grid_size * self.grid_size)
                else:
                    prompts = [self.grid_points for _ in range(batch_size)]
                tren_outputs = self.tren_region_encoder(feature_maps, prompts, aggregate_tokens=aggregate_tokens)
                if aggregate_tokens:
                    outputs = torch.zeros(batch_size, self.num_classes, self.grid_size, self.grid_size).to(device)
                    for batch_idx in range(batch_size):
                        region_tokens_b = tren_outputs['text_aligned_tokens'][batch_idx]
                        region_masks_b = tren_outputs['region_masks'][batch_idx]
                        token_count += region_tokens_b.shape[0]
                        similarity = F.cosine_similarity(region_tokens_b.unsqueeze(1), self.text_embeddings.unsqueeze(0), dim=-1)
                        mask_sum = region_masks_b.sum(dim=0, keepdim=True)
                        eps = 1e-6
                        for region_idx in range(region_masks_b.shape[0]):
                            sim = similarity[region_idx].unsqueeze(-1).unsqueeze(-1)
                            mask = region_masks_b[region_idx].unsqueeze(0)
                            outputs[batch_idx] += sim * mask
                        outputs[batch_idx] = outputs[batch_idx] / (mask_sum + eps)
                        no_coverage = (mask_sum.squeeze(0) < eps)
                        if no_coverage.any():
                            outputs[batch_idx][:, no_coverage] = similarity.mean(dim=0).unsqueeze(1)
                else:
                    region_tokens = tren_outputs['text_aligned_tokens'].mean(dim=-2)
                    token_count += region_tokens.shape[1]
                    text_embeddings = self.text_embeddings.repeat(batch_size, 1, 1)
                    similarity = F.cosine_similarity(region_tokens.unsqueeze(1), text_embeddings.unsqueeze(2), dim=-1)
                    outputs = similarity.view(batch_size, -1, self.grid_size, self.grid_size)
                
                # Resize the outputs to the desired dimensions
                if use_slic:
                    outputs = torch.argmax(outputs, dim=1).flatten(-2)[0].cpu()
                    predictions = superpixels[0].copy()
                    for segment_idx, segment_id in enumerate(np.unique(superpixels)):
                        predictions[superpixels[0] == segment_id] = outputs[segment_idx]
                    predictions = torch.tensor(predictions)[None]
                else:
                    resized_outputs = torch.nn.functional.interpolate(outputs, size=[self.image_resolution, self.image_resolution],
                                                                      mode='bilinear')
                    resized_outputs = resized_outputs.flatten(-2).permute(0, 2, 1).reshape(-1, outputs.shape[1])
                    predictions = torch.argmax(resized_outputs, dim=-1).view(-1, self.image_resolution, self.image_resolution)

        return {
            'images': images,
            'predictions': predictions,
            'targets': masks.to(predictions.dtype),
            'token_count': token_count,
        }

    def run(self, split='val', use_slic=False, aggregate_tokens=False, visualize_predictions=True):
        if split == 'val':
            dataloader = self.val_loader

        # Accumulate per-class histogram totals
        num_labels = self.num_classes
        ignore_index = 255
        total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
        total_area_union = np.zeros((num_labels,), dtype=np.float64)
        total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
        total_area_label = np.zeros((num_labels,), dtype=np.float64)

        num_vis = 100 if visualize_predictions else 0
        vis_count = 0

        token_count = 0
        for batch in tqdm(dataloader, desc=f'Running eval'):
            outputs = self.step(batch, use_slic, aggregate_tokens)
            predictions = outputs['predictions'].cpu().numpy()
            targets = outputs['targets'].cpu().numpy()
            token_count += outputs['token_count']
            del outputs['predictions'], outputs['targets']

            for i in range(predictions.shape[0]):
                area_intersect, area_union, area_p, area_l = intersect_and_union(predictions[i], targets[i], num_labels, ignore_index)
                total_area_intersect += area_intersect
                total_area_union += area_union
                total_area_pred_label += area_p
                total_area_label += area_l

            if visualize_predictions and vis_count < num_vis:
                images = outputs['images'].cpu()
                for i in range(images.shape[0]):
                    if vis_count >= num_vis:
                        break
                    self.visualize(images[i].permute(1, 2, 0), torch.from_numpy(predictions[i]), torch.from_numpy(targets[i]),
                                   os.path.join(self.exp_dir, f'vis/{vis_count}.jpg'))
                    vis_count += 1
            del outputs['images']

        metrics = mean_iou_from_totals(total_area_intersect, total_area_union, total_area_label)
        print(f'Mean IoU: {metrics["mean_iou"] * 100:.2f}%')
        print(f'Average token count: {token_count / len(dataloader)}')


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

    evaluator = Evaluator(config)
    evaluator.run()