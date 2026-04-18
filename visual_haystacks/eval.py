import os
import sys
import yaml
import json
import logging
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from torch.amp import autocast
from sklearn.metrics import accuracy_score

sys.path.append('..')
sys.path.append('../segment_anything/')
from model import FeatureExtractor, RegionEncoder, TextEncoder, TokenAggregator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
logging.getLogger().setLevel(logging.WARNING)


class Evaluator:
    def __init__(self, config):
        self.coco_root_dir = config['data']['coco_root_dir']
        self.vhqa_root_dir = config['data']['vhqa_root_dir']
        self.grid_size = config['parameters']['grid_size']
        self.similarity_threshold = config['parameters']['similarity_threshold']
        self.image_resolution = config['tren']['parameters']['image_resolution']
        self.batch_size = config['parameters'].get('batch_size', 8)

        # Create the model
        self.tren_image_encoder = FeatureExtractor(config['tren'], device=device).eval()
        self.tren_region_encoder = RegionEncoder(config['tren']).to(device).eval()
        self.tren_text_encoder = TextEncoder(config['tren'], device=device).eval()
        self.token_aggregator = TokenAggregator(config['tren'])
        self.patch_encoder = torch.hub.load('facebookresearch/dinov3', 
                                            'dinov3_vitl16_dinotxt_tet1280d20h24l')[0].to(device).eval()
        self.patch_size = config['tren']['architecture']['patch_size']

        # Define the image transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_resolution, self.image_resolution), antialias=True),
        ])

        # Create prompts for region encoder
        x_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        y_coords = np.linspace(self.patch_size // 2, self.image_resolution - self.patch_size // 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

        # Load checkpoints
        self.tren_checkpoint = os.path.join(config['tren']['logging']['save_dir'], config['tren']['logging']['exp_name'], 'tren_region_encoder.pth')
        self.load_tren()

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

    def load_images(self, image_paths):
        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(self.coco_root_dir, image_path)).convert('RGB')
            images.append(image)
        return images
    
    def get_region_tokens(self, images):
        all_region_tokens = []
        for start in range(0, len(images), self.batch_size):
            end = min(start + self.batch_size, len(images))
            batch_images = images[start:end]
            batch_tensors = torch.stack([self.transform(image) for image in batch_images]).to(device)
            batch_size = batch_tensors.shape[0]

            with torch.no_grad():
                with autocast('cuda', dtype=torch.bfloat16):
                    tren_image_encoder_outputs = self.tren_image_encoder(batch_tensors)
                    feature_maps = tren_image_encoder_outputs['feature_maps']
                    prompts = [self.grid_points for _ in range(batch_size)]
                    tren_outputs = self.tren_region_encoder(feature_maps, prompts, aggregate_tokens=True)
                    region_tokens = tren_outputs['text_aligned_tokens']

            if isinstance(region_tokens, list):
                all_region_tokens.extend(region_tokens)
                return_list = True
            else:
                all_region_tokens.append(region_tokens)
                return_list = False
            del batch_tensors, feature_maps, tren_outputs
            if device == 'cuda':
                torch.cuda.empty_cache()

        if return_list:
            return all_region_tokens
        return torch.cat(all_region_tokens, dim=0)
    
    def get_patch_tokens(self, images):
        images = [self.transform(image) for image in images]
        images = torch.stack(images).to(device)
        with torch.no_grad():
            with autocast('cuda', dtype=torch.bfloat16):
                patch_tokens = self.patch_encoder.visual_model(images)[1]
        return patch_tokens.to(torch.bfloat16)
    
    def get_text_embeddings(self, needle, target):
        with torch.no_grad():
            with autocast('cuda', dtype=torch.bfloat16):
                text_embeddings = self.tren_text_encoder([needle, target])
        needle_embedding = text_embeddings[0]
        target_embedding = text_embeddings[1]
        return needle_embedding, target_embedding
    
    def find_needle_image(self, region_tokens, needle_embedding):
        needle_embedding = needle_embedding / needle_embedding.norm(p=2, dim=-1, keepdim=True)
        if isinstance(region_tokens, list):
            image_scores = []
            for t in region_tokens:
                t_norm = t / t.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                sim = torch.matmul(t_norm, needle_embedding)
                image_scores.append(torch.max(sim).item())
            image_scores = torch.tensor(image_scores, device=needle_embedding.device)
        else:
            region_tokens = region_tokens / region_tokens.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            similarity = torch.matmul(region_tokens, needle_embedding)
            image_scores = torch.max(similarity, dim=1).values
        image_idx = torch.argmax(image_scores)
        image_score = image_scores[image_idx]
        return image_idx.item(), image_score
    
    def check_target(self, region_tokens, target_embedding):
        region_tokens = region_tokens / region_tokens.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        target_embedding = target_embedding / target_embedding.norm(p=2, dim=-1, keepdim=True)
        similarity = torch.matmul(region_tokens, target_embedding)
        return 'yes' if torch.max(similarity) > self.similarity_threshold else 'no'
    
    def compute_metrics(self, predictions, targets):
        # Overall accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Accuracy for "yes" cases
        yes_indices = [i for i, t in enumerate(targets) if t == "yes"]
        if yes_indices:
            yes_targets = [targets[i] for i in yes_indices]
            yes_predictions = [predictions[i] for i in yes_indices]
            yes_accuracy = accuracy_score(yes_targets, yes_predictions)
        else:
            yes_accuracy = 0.0
        
        # Accuracy for "no" cases
        no_indices = [i for i, t in enumerate(targets) if t == "no"]
        if no_indices:
            no_targets = [targets[i] for i in no_indices]
            no_predictions = [predictions[i] for i in no_indices]
            no_accuracy = accuracy_score(no_targets, no_predictions)
        else:
            no_accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'yes_accuracy': yes_accuracy,
            'no_accuracy': no_accuracy,
            'yes_indices': yes_indices,
            'no_indices': no_indices
        }
    
    def run(self, n):
        json_file = os.path.join(self.vhqa_root_dir, f'visual_haystack_{n}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Process each record
        predictions, answers = [], []
        recall, token_count, num_records = 0, 0, 0
        for idx, record in enumerate(tqdm(data, desc="Processing records")):
            needle = record.get("needle")
            target = record.get("target")
            pos_images_paths = record.get("pos_image", [])
            neg_images_paths = record.get("neg_image", [])
            conversations = record.get("conversations", [])
            answer = None
            for conv in conversations:
                if conv.get("from") == "gpt" and conv.get("value") in ["yes", "no"]:
                    answer = conv.get("value")
                    break
            if answer is None:
                print(f'Could not extract valid ground truth for record {idx}. Skipping this record.')
                continue
            
            # Load all images
            images = self.load_images(pos_images_paths + neg_images_paths)

            # Get region token representations for all images
            region_tokens = self.get_region_tokens(images)
            token_count_record = sum(t.shape[0] for t in region_tokens)
            token_count += token_count_record
            num_records += 1
            if not isinstance(region_tokens, list):
                region_tokens = region_tokens.flatten(1, 2)

            # Get text embeddings for needle and target
            needle_embedding, target_embedding = self.get_text_embeddings(needle, target)

            # Find the image containing the needle
            needle_image_idx, _ = self.find_needle_image(region_tokens, needle_embedding)
            recall += 1 if needle_image_idx == 0 else 0

            # Check if the target is present in the needle image
            prediction = self.check_target(region_tokens[needle_image_idx], target_embedding)
            predictions.append(prediction)
            answers.append(answer)
        
        # Compute and report the metrics
        metrics = self.compute_metrics(predictions, answers)
        print(f"Recall: {recall} / {len(answers)}")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Accuracy for 'yes': {metrics['yes_accuracy']:.4f} ({len(metrics['yes_indices'])} samples)")
        print(f"Accuracy for 'no': {metrics['no_accuracy']:.4f} ({len(metrics['no_indices'])} samples)")
        print(f"Average number of tokens needed to represent a datbase of {n} images: {token_count / num_records}")


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    evaluator = Evaluator(config)
    evaluator.run(n=10)