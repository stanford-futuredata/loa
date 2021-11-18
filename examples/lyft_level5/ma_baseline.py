import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json as json
from collections import defaultdict
from pyquaternion import Quaternion
from lyft_dataset_sdk.lyftdataset import LyftDataset
from loa.datum.datum_3d import make_datum_from_gt, make_datum_from_pred
from loa.datum.track import Track, TrackDatum
from prior_lyft import serialize_track, get_idx_to_rot2
from constants import LYFT_DATA_DIR, PRIOR_DIR, LOA_DATA_DIR


def process_scene(level5data, preds, scene_record, scene_idx, rot2=None, score_cutoff=0.2, seed=1):
    assert seed >= 1
    pred_cls_names = ['car', 'pedestrian', 'motorcycle', 'bicycle', 'other_vehicle', 'bus', 'truck']
    print('Processing scene', scene_record['token'])
    last_token = scene_record['last_sample_token']
    sample_token = scene_record['first_sample_token']
    sample_record = level5data.get('sample', sample_token)
    next_token = sample_record['next']

    all_tracks = []

    while next_token != '':
        timestamp = sample_record['timestamp']
        sd_record = level5data.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = level5data.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = level5data.get('ego_pose', sd_record['ego_pose_token'])

        gt_data = []
        for ann_token in sample_record['anns']:
            ann_record = level5data.get('sample_annotation', ann_token)
            ann_record['name'] = ann_record['category_name']
            gt_datum = make_datum_from_gt(ann_record, pose_record, cs_record, ts=timestamp, identifier=ann_token)
            gt_data.append(gt_datum)

        pred_data = []
        dets = preds[sample_token]
        box3d = dets['box3d_lidar'].detach().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        scores = dets['scores'].detach().cpu().numpy()
        labels = dets['label_preds'].detach().cpu().numpy()
        pred_idens = dets['idens']
        pred_trans = - np.array(cs_record['translation'])
        pred_rot = Quaternion(cs_record['rotation']).inverse
        for pred_idx in range(box3d.shape[0]):
            pred_datum = make_datum_from_pred(
                sample_token,
                box3d[pred_idx],
                scores[pred_idx],
                pred_cls_names[labels[pred_idx]],
                pred_trans,
                pred_rot,
                rot2=rot2,
                ts=timestamp,
                identifier=pred_idens[pred_idx]
            )
            pred_data.append(pred_datum)


        for pred_datum in pred_data:
            if pred_datum.score < score_cutoff:
                continue
            has_overlap = False
            for gt_datum in gt_data:
                if pred_datum.intersects(gt_datum):
                    has_overlap = True
                    break
            if has_overlap:
                continue

            add_datum = TrackDatum({'pred': pred_datum}, [])
            new_track = Track()
            new_track.add_datum(timestamp, add_datum)
            all_tracks.append(new_track)

        sample_token = next_token
        sample_record = level5data.get('sample', sample_token)
        next_token = sample_record['next']

    rand = np.random.RandomState(seed=seed)
    rand.shuffle(all_tracks)

    scored = []
    for track_idx, track in enumerate(all_tracks):
        cls = track.data[0].observations['pred'].cls
        score = track.data[0].observations['pred'].score
        elem = (track_idx, cls, score)
        scored.append(elem)

    df_scored = pd.DataFrame(
        scored,
        columns=['track_idx', 'cls', 'score']
    )
    df_scored = df_scored.sort_values(by=['score'], ascending=False).reset_index(drop=True)

    track_dir = f'{LOA_DATA_DIR}/ma-conf/tracks/{scene_idx}'
    os.makedirs(track_dir, exist_ok=True)
    df_scored.to_csv(f'{track_dir}/tracks.csv')
    print(df_scored)

    for idx, row in df_scored.iterrows():
        track_idx = int(row['track_idx'])
        track = all_tracks[track_idx]
        serialize_track(track, f'{track_dir}/{idx}-{track_idx}.json')


def main():
    level5data = LyftDataset(
        data_path=f'{LYFT_DATA_DIR}',
        json_path=f'{LYFT_DATA_DIR}/data',
        verbose=True
    )

    with open('./preds_id.p', 'rb') as f:
        preds = pickle.load(f)

    idx_to_rot2 = get_idx_to_rot2()

    for scene_idx in range(150):
        try:
            process_scene(
                level5data,
                preds,
                level5data.scene[scene_idx],
                scene_idx,
                rot2=idx_to_rot2[scene_idx],
                score_cutoff=0.2
            )
        except:
            pass


if __name__ == '__main__':
    main()
