import os
import pickle
import numpy as np
import pandas as pd
import torch
import random
import logging
from models import QuerySearch


seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
logging.getLogger().setLevel(logging.WARNING)


class Evaluator():
    def __init__(self, config, split='val'):
        self.exp_dir = os.path.join(config['parameters']['save_dir'], config['parameters']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)

        self.query_search = QuerySearch(config=config)
        self.data_dir = config['data'][f'{split}_data_dir']
        self.tiou_thresholds = np.array([0.25, 0.5, 0.75, 0.95])

    def load_data(self, run_id):
        data = []
        for fname in sorted(os.listdir(self.data_dir))[run_id:(run_id + 1)]:
            record_file = os.path.join(self.data_dir, fname)
            with open(record_file, 'rb') as fp:
                record = pickle.load(fp)
            clip_uid = record['clip_uid'].decode('utf-8')
            for annotation in record['annotations']:
                data.append((clip_uid, record_file, annotation))
        return data

    def evaluate(self, run_ids):
        for run_id in run_ids:
            print(f'Run {run_id}.')
            data = self.load_data(run_id)

            video_ids_gt, t_start_gt, t_end_gt = [], [], []
            video_ids_pred, t_start_pred, t_end_pred, scores_pred = [], [], [], []
            video_cache = []
            video_compressions = {
                'from_patches': [],
                'from_regions': [],
            }

            # Evaluate for each record in the data
            already_cached = False
            for clip_uid, record_file, annotation in data:
                pred_save_path = os.path.join(self.exp_dir, f'{run_id}-{clip_uid}-pred.csv')
                target_save_path = os.path.join(self.exp_dir, f'{run_id}-{clip_uid}-target.csv')
                cache_save_path = os.path.join(self.exp_dir, f'{run_id}-{clip_uid}-cache.pkl')
                if os.path.exists(pred_save_path) and os.path.exists(target_save_path):
                    print(f'Skipping clip {clip_uid} because it already exists.')
                    already_cached = True
                    break

                # Search for the query in the video and get the selected frame ids and compression stats
                selected_frame_ids, compression = self.query_search(record_file, annotation)
                video_cache.append({'frame_ids': selected_frame_ids, 'compression': compression})

                # Get the last contiguous run of frames ending at the last selected frame
                selected_frame_ids = sorted(set(selected_frame_ids))
                if not selected_frame_ids:
                    selected_track_start = selected_track_end = 0
                    score = 0.0
                else:
                    selected_track_end = selected_frame_ids[-1]
                    frame_set = set(selected_frame_ids)
                    selected_track_start = selected_track_end
                    while selected_track_start - 1 in frame_set:
                        selected_track_start -= 1
                    score = 1.0

                # Get the target
                target = [track['frame_number'] for track in annotation['response_track']]
                video_ids_gt.append(clip_uid)
                t_start_gt.append(min(target) if target else 0)
                t_end_gt.append(max(target) if target else 0)

                # Record the prediction
                video_ids_pred.append(clip_uid)
                t_start_pred.append(selected_track_start)
                t_end_pred.append(selected_track_end)
                scores_pred.append(score)
                video_compressions['from_patches'].append(compression['from_patches'])
                video_compressions['from_regions'].append(compression['from_regions'])
            
            # Cache the prediction and target for the run
            if not already_cached:
                pd.DataFrame({
                    'video_id': video_ids_pred,
                    't_start': t_start_pred,
                    't_end': t_end_pred,
                    'score': scores_pred,
                }).to_csv(pred_save_path, index=False)
                pd.DataFrame({
                    'video_id': video_ids_gt,
                    't_start': t_start_gt,
                    't_end': t_end_gt,
                }).to_csv(target_save_path, index=False)
                with open(cache_save_path, 'wb') as f:
                    pickle.dump(video_cache, f)


if __name__ == '__main__':
    import yaml
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    evaluator = Evaluator(config)
    start_idx = 1
    end_idx = 101
    evaluator.evaluate(list(range(start_idx, end_idx)))