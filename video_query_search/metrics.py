import os
import yaml
import pickle
import numpy as np
import pandas as pd


def segment_iou(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])

    # Compute intersection including non-negative overlap score
    segments_intersection = (tt2 - tt1 + 1).clip(0)

    # Compute segment union
    segments_union = ((candidate_segments[:, 1] - candidate_segments[:, 0] + 1) +
                      (target_segment[1] - target_segment[0] + 1) - segments_intersection)

    # Compute overlap as the ratio of the intersection over union of two segments
    iou = segments_intersection.astype(float) / segments_union
    return iou


def interpolated_prec_rec(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_tap_detection(target, prediction, tiou_thresholds):
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(target))
    lock_gt = np.ones((len(tiou_thresholds), len(target))) * -1

    # Sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    target_gbvn = target.groupby('video_id')

    # Assign true positive to truly ground truth instances
    for idx, this_pred in prediction.iterrows():
        try:
            target_videoid = target_gbvn.get_group(this_pred['video_id'])
        except Exception:
            fp[:, idx] = 1
            continue

        this_gt = target_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t_start', 't_end']].values, this_gt[['t_start', 't_end']].values)

        # Retrieve the predictions with highest tiou score
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue

                # Assign as true positive after the filters above
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])
    return ap


def compute_match_rate(target, prediction, tiou_thresholds):
    match_rates = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return match_rates

    target_gbvn = target.groupby('video_id')
    num_pred = float(len(prediction))

    for _, this_pred in prediction.iterrows():
        try:
            target_videoid = target_gbvn.get_group(this_pred['video_id'])
        except Exception:
            continue
        this_gt = target_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t_start', 't_end']].values, this_gt[['t_start', 't_end']].values)
        max_tiou = float(tiou_arr.max())
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            if max_tiou >= tiou_thr:
                match_rates[tidx] += 1

    match_rates /= num_pred
    return match_rates


def compute_frame_hit_rate(target_files, cache_files, tiou_thresholds):
    hit_counts = np.zeros(len(tiou_thresholds), dtype=float)
    total_gt = 0.0

    # Ensure consistent pairing of target and cache files
    target_files = sorted(target_files)
    cache_files = sorted(cache_files)

    for target_path, cache_path in zip(target_files, cache_files):
        target_df = pd.read_csv(target_path)
        cache_list = pickle.load(open(cache_path, 'rb'))

        # Each row in target_df corresponds to one annotation; assume same order as cache_list
        num_ann = min(len(target_df), len(cache_list))
        for ann_idx in range(num_ann):
            gt_row = target_df.iloc[ann_idx]
            target_segment = np.array([gt_row['t_start'], gt_row['t_end']], dtype=float)
            frame_ids = cache_list[ann_idx].get('frame_ids', [])

            total_gt += 1.0
            if len(frame_ids) == 0:
                continue

            # Build contiguous tracks from the sorted unique frame ids
            unique_frames = sorted(set(frame_ids))
            tracks = []
            start = unique_frames[0]
            prev = unique_frames[0]
            for f in unique_frames[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    tracks.append([start, prev])
                    start = f
                    prev = f
            tracks.append([start, prev])

            candidate_segments = np.array(tracks, dtype=float)
            tiou_arr = segment_iou(target_segment, candidate_segments)
            max_tiou = float(tiou_arr.max()) if tiou_arr.size > 0 else 0.0

            for tidx, thr in enumerate(tiou_thresholds):
                if max_tiou >= thr:
                    hit_counts[tidx] += 1.0

    if total_gt == 0:
        return np.zeros(len(tiou_thresholds), dtype=float)
    return hit_counts / total_gt


if __name__ == "__main__":
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['parameters']['save_dir'], config['parameters']['exp_name'])
    
    pred_files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('-pred.csv')]
    target_files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('-target.csv')]
    cache_files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('-cache.pkl')]
    print(f'Evaluating over {len(pred_files)} videos from {exp_dir}.')
    
    predictions = pd.concat([pd.read_csv(file) for file in pred_files], ignore_index=True)
    targets = pd.concat([pd.read_csv(file) for file in target_files], ignore_index=True)
    tiou_thresholds = [0.25, 0.50, 0.75, 0.95]

    # Compute temporal AP
    ap = compute_tap_detection(targets, predictions, tiou_thresholds=tiou_thresholds)
    print('Temporal AP @ IoU 0.25:              {:.4f}'.format(ap[0]))
    print('Temporal AP @ IoU 0.50:              {:.4f}'.format(ap[1]))
    print('Temporal AP (avg):                   {:.4f}'.format(ap.mean()))
    
    # Compute temporal match rate
    mr = compute_match_rate(targets, predictions, tiou_thresholds=tiou_thresholds)
    print('Temporal MR @ IoU 0.25:              {:.4f}'.format(mr[0]))
    print('Temporal MR @ IoU 0.50:              {:.4f}'.format(mr[1]))
    print('Temporal MR (avg):                   {:.4f}'.format(mr.mean()))

    # Compute frame-level hit rate based on cached frame_ids
    frame_hit = compute_frame_hit_rate(target_files, cache_files, tiou_thresholds=tiou_thresholds)
    print('Frame hit rate @ IoU 0.25:           {:.4f}'.format(frame_hit[0]))
    print('Frame hit rate @ IoU 0.50:           {:.4f}'.format(frame_hit[1]))
    print('Frame hit rate (avg):                {:.4f}'.format(frame_hit.mean()))

    # Print the compression ratio which is in cache_files
    compression_ratios = []
    for cache_file in cache_files:
        cache_list = pickle.load(open(cache_file, 'rb'))
        for clip_cache in cache_list:
            compression_ratios.append(clip_cache['compression']['from_patches'])
    print('Compression ratio:                   {:.4f}'.format(np.mean(compression_ratios)))