import os
import yaml
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from dataloader import VSPWSegmentation
from models import VideoREN


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


class Evaluator():
    def __init__(self, config):
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f'Configs: {config}')

        # Instantiate the dataloaders
        self.target_data = config['data']['target_data']
        if self.target_data == 'vspw':
            val_dataset = VSPWSegmentation(config, split='val')
            self.val_loader = DataLoader(val_dataset, batch_size=1)
            self.num_classes = config['data']['vspw_num_classes']

        # Create the models
        self.video_ren = VideoREN(config['tren']).to(device).eval()
        self.tren_text_encoder = self.video_ren.tren_text_encoder
        if self.target_data == 'vspw':
            self.text_embeddings = self.get_vspw_text_embeddings()

        # Create prompts for region encoder
        self.image_resolution = config['tren']['parameters']['image_resolution']
        self.patch_size = config['tren']['architecture']['patch_size']
        self.grid_size = self.image_resolution // self.patch_size
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Colormap for visualizing results
        self.colormap = np.random.randint(0, 256, size=(self.num_classes + 1, 3), dtype=np.uint8)

    def visualize(self, image, prediction, target, save_path, ignore_index=255):
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
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

    def get_vspw_text_embeddings(self):
        categories = [
            'wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground slide', 'handrail or fence', 
            'window', 'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic field', 
            'road', 'path', 'crosswalk', 'building', 'house', 'bridge', 'tower', 'windmill', 'well or well lid', 
            'other construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 
            'river', 'lake', 'waterfall', 'water', 'billboard or Bulletin Board', 'sculpture', 'pipeline', 'flag', 
            'parasol or umbrella', 'cushion or carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle', 
            'motorcycle', 'wheeled machine', 'ship or boat', 'raft', 'airplane', 'tyre', 'traffic light', 'lamp', 
            'person', 'cat', 'dog', 'horse', 'cattle', 'other animal', 'tree', 'flower', 'other plant', 'toy', 
            'ball net', 'backboard', 'skateboard', 'bat', 'ball', 'cupboard or showcase or storage rack', 'box', 
            'traveling case or trolley case', 'basket', 'bag or package', 'trash can', 'cage', 'plate', 
            'tub or bowl or pot', 'bottle or cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table or desk', 
            'chair or seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other machine', 
            'refrigerator', 'washing machine', 'Microwave oven', 'fan', 'curtain', 'textiles', 'clothes', 
            'painting or poster', 'mirror', 'flower pot or vase', 'clock', 'book', 'tool', 'blackboard', 'tissue', 
            'screen or television', 'computer', 'printer', 'Mobile phone', 'keyboard', 'other electronic product', 
            'fruit', 'food', 'instrument', 'train'
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
    
    def step(self, batch):
        images = batch['images'].numpy()
        masks = batch['masks'].numpy()
        batch_size = images.shape[0]
        assert batch_size == 1, 'Batch size must be 1 for evaluation.'
        images = images[0]
        masks = masks[0]
        num_frames = images.shape[0]

        with torch.inference_mode():
            with autocast('cuda', dtype=torch.bfloat16):
                # Compute the track-level region tokens and masks
                tren_outputs, compression = self.video_ren(images)
                region_tokens = tren_outputs['track_text_aligned_tokens']
                region_masks = tren_outputs['track_region_masks']
                track_members = tren_outputs['track_members']

                # Compute the token-level similarity scores
                similarity = F.cosine_similarity(region_tokens.unsqueeze(1), self.text_embeddings.unsqueeze(0), dim=-1)

                # Accumulate weighted class scores per frame using track masks
                outputs = torch.zeros(num_frames, self.num_classes, self.grid_size, self.grid_size, device=device)
                mask_sum = torch.zeros(num_frames, 1, self.grid_size, self.grid_size, device=device)
                eps = 1e-6
                for track_idx in range(len(track_members)):
                    sim = similarity[track_idx]
                    for member_idx, (frame_id, _) in enumerate(track_members[track_idx]):
                        mask = region_masks[track_idx][member_idx].to(device).float()
                        outputs[frame_id] += sim[:, None, None] * mask[None]
                        mask_sum[frame_id] += mask[None]
                outputs = outputs / (mask_sum + eps)
                no_coverage = (mask_sum.squeeze(1) < eps)
                mean_sim = similarity.mean(dim=0)
                for f in range(num_frames):
                    if no_coverage[f].any():
                        outputs[f, :, no_coverage[f]] = mean_sim[:, None].expand(-1, no_coverage[f].sum())

                # Resize to image resolution and take argmax
                resized_outputs = F.interpolate(outputs, size=[self.image_resolution, self.image_resolution], mode='bilinear')
                predictions = torch.argmax(resized_outputs, dim=1)
                predictions = predictions.cpu().numpy()
                targets = torch.from_numpy(masks).float().unsqueeze(1)
                targets = F.interpolate(targets, size=[self.image_resolution, self.image_resolution], mode='nearest')
                targets = targets.squeeze(1).numpy().astype(predictions.dtype)

        return {
            'images': images,
            'predictions': predictions,
            'targets': targets,
            'compression': compression,
        }

    def run(self, split='val', visualize_predictions=False):
        if split == 'val':
            dataloader = self.val_loader

        # Accumulate per-class histogram totals
        num_labels = self.num_classes
        ignore_index = 255
        total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
        total_area_union = np.zeros((num_labels,), dtype=np.float64)
        total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
        total_area_label = np.zeros((num_labels,), dtype=np.float64)

        num_vis = 10 if visualize_predictions else 0
        vis_count = 0
        compressions = []
        for batch in tqdm(dataloader, desc=f'Running eval'):
            outputs = self.step(batch)
            predictions = outputs['predictions']
            targets = outputs['targets']
            del outputs['predictions'], outputs['targets']
            compressions.append(outputs['compression']['from_patches'])

            for i in range(predictions.shape[0]):
                area_intersect, area_union, area_p, area_l = intersect_and_union(predictions[i], targets[i], num_labels, ignore_index)
                total_area_intersect += area_intersect
                total_area_union += area_union
                total_area_pred_label += area_p
                total_area_label += area_l

            if visualize_predictions and vis_count < num_vis:
                images = outputs['images']
                for i in range(images.shape[0]):
                    self.visualize(images[i], predictions[i], targets[i], os.path.join(self.exp_dir, f'vis/{vis_count}/{i}.jpg'))
                vis_count += 1
            del outputs['images']

        metrics = mean_iou_from_totals(total_area_intersect, total_area_union, total_area_label)
        print(f'Mean IoU: {metrics["mean_iou"] * 100:.2f}%')
        print(f'Average compression: {np.mean(compressions)}')


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])

    evaluator = Evaluator(config)
    evaluator.run()