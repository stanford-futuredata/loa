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
from loa.prior.av_priors import VolumeObsPrior, SpeedTransitionPrior, TrackLengthPrior
from prior_lyft import get_idx_to_rot2, serialize_track, get_overlapping
from constants import LYFT_DATA_DIR, PRIOR_DIR, LOA_DATA_DIR


def serialize_by_conf(level5data, preds, scene_record, scene_idx, rot2=None):
    pred_cls_names = ['car', 'pedestrian', 'motorcycle', 'bicycle', 'other_vehicle', 'bus', 'truck']
    print('Processing scene', scene_record['token'])
    sample_token = scene_record['first_sample_token']
    sample_record = level5data.get('sample', sample_token)
    next_token = sample_record['next']

    all_tracks = []
    cur_tracks = []

    while next_token != '':
        timestamp = sample_record['timestamp']
        sd_record = level5data.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = level5data.get('calibrated_sensor', sd_record['calibrated_sensor_token'])

        pred_data = []
        if sample_token not in preds:
            print(sample_token)
            sys.exit(0)
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

        def track_argmin(overlapping):
            add_track, max_val = overlapping[0]
            for track, area in overlapping[1:]:
                if area > max_val:
                    max_val = area
                    add_track = track
            return add_track

        for pred_datum in pred_data:
            add_datum = TrackDatum({'pred': pred_datum}, [])
            new_track = Track()
            new_track.add_datum(timestamp, add_datum)
            cur_tracks.append(new_track)

        for track in cur_tracks:
            # FIXME
            if timestamp - track.data[-1].ts >= 450000.:
                track.finish()
        all_tracks += [track for track in cur_tracks if track.is_finished]
        cur_tracks = [track for track in cur_tracks if not track.is_finished]

        sample_token = next_token
        sample_record = level5data.get('sample', sample_token)
        next_token = sample_record['next']

    all_tracks += cur_tracks
    print(len(all_tracks))
    
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
    df_scored['uncertainty'] = (df_scored['score'] - 0.2).abs()
    df_scored = df_scored.sort_values(by=['uncertainty'], ascending=True, ignore_index=True)
    print(df_scored)

    track_dir = f'{LOA_DATA_DIR}/conf/tracks/{scene_idx}'
    os.makedirs(track_dir, exist_ok=True)
    df_scored.to_csv(f'{track_dir}/tracks.csv')
    for idx, row in df_scored.iterrows():
        track_idx = int(row['track_idx'])
        track = all_tracks[track_idx]
        serialize_track(track, f'{track_dir}/{idx}-{track_idx}.json')



def process_scene(level5data, preds, scene_record, scene_idx, rot2=None):
    pred_cls_names = ['car', 'pedestrian', 'motorcycle', 'bicycle', 'other_vehicle', 'bus', 'truck']
    print('Processing scene', scene_record['token'])
    sample_token = scene_record['first_sample_token']
    sample_record = level5data.get('sample', sample_token)
    next_token = sample_record['next']

    all_tracks = []
    cur_tracks = []

    while next_token != '':
        timestamp = sample_record['timestamp']
        sd_record = level5data.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = level5data.get('calibrated_sensor', sd_record['calibrated_sensor_token'])

        pred_data = []
        if sample_token not in preds:
            print(sample_token)
            sys.exit(0)
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

        def track_argmin(overlapping):
            add_track, max_val = overlapping[0]
            for track, area in overlapping[1:]:
                if area > max_val:
                    max_val = area
                    add_track = track
            return add_track

        nb_combined = 0
        for pred_datum in pred_data:
            overlapping_pred_tracks = get_overlapping(cur_tracks, pred_datum, 'pred')
            overlapping_gt_tracks = get_overlapping(cur_tracks, pred_datum, 'gt')
            add_datum = TrackDatum({'pred': pred_datum}, [])
            if len(overlapping_pred_tracks) == 0 and len(overlapping_gt_tracks) == 0:
                new_track = Track()
                new_track.add_datum(timestamp, add_datum)
                cur_tracks.append(new_track)
            else:
                nb_combined += 1
                if len(overlapping_gt_tracks) != 0:
                    add_track = track_argmin(overlapping_gt_tracks)
                else:
                    add_track = track_argmin(overlapping_pred_tracks)
                if timestamp == add_track.data[-1].ts:
                    add_track.data[-1].merge(add_datum)
                else:
                    add_track.add_datum(timestamp, add_datum)

        for track in cur_tracks:
            # FIXME
            if timestamp - track.data[-1].ts >= 450000.:
                track.finish()
        all_tracks += [track for track in cur_tracks if track.is_finished]
        cur_tracks = [track for track in cur_tracks if not track.is_finished]

        sample_token = next_token
        sample_record = level5data.get('sample', sample_token)
        next_token = sample_record['next']

    all_tracks += cur_tracks
    print(len(all_tracks))

    vol_obs_prior = VolumeObsPrior(serialized_path=f'{PRIOR_DIR}/volume_obs_prior.p')
    speed_trans_prior = SpeedTransitionPrior(
        serialized_path=f'{PRIOR_DIR}/speed_trans_prior.p')
    track_length_prior = TrackLengthPrior(serialized_path=f'{PRIOR_DIR}/track_length_prior.p')

    scored = []
    both_contains = 0
    for track_idx, track in enumerate(all_tracks):
        if len(track.data) == 1:
            continue
        max_score = 0.0
        min_x = 100.
        min_y = 100.
        for datum_idx, cur_datum in enumerate(track.data):
            if 'pred' in cur_datum.observations:
                max_score = max(max_score, cur_datum.observations['pred'].score)
                box = cur_datum.observations['pred'].data_box
                xyz = np.abs(box.center)
                min_x = min(abs(xyz[0]), min_x)
                min_y = min(abs(xyz[1]), min_y)
        if max_score < 0.2:
            continue
        if min_x > 40 or min_y > 40:
            continue

        # print(track_idx)
        # print(track.data[0].observations['pred'])

        score = 0.
        nb_datum = 0.
        nb_priors = 0.
        nb_priors += 1
        len_score, _ = track_length_prior.score(track)
        score += len_score
        for datum_idx, cur_datum in enumerate(track.data):
            datum = cur_datum.observations['pred']
            nb_datum += 1
            nb_priors += 1
            vol_score, vol_valid = vol_obs_prior.score(datum)
            score += vol_score

            if datum_idx != 0:
                try:
                    prev_datum = track.data[datum_idx - 1].observations['pred']
                except:
                    prev_datum = track.data[datum_idx - 1].observations['gt']

                nb_priors += 1
                speed_score, speed_valid = speed_trans_prior.score(
                    [datum],
                    [prev_datum]
                )
                score += speed_score

        cls = track.data[0].observations['pred'].cls
        scored.append((track_idx, cls, nb_datum, nb_priors, score, score / nb_datum, score / nb_priors))

    print(len(all_tracks))
    print(len(scored))
    print(both_contains)

    df_scored = pd.DataFrame(
        scored,
        columns=['track_idx', 'cls', 'nb_datum', 'nb_priors', 'score', 'score_d', 'score_p']
    )

    df_sort = df_scored.sort_values(by=['score_p'], ascending=False)
    df_sort = df_sort[df_sort['nb_datum'] > 2].reset_index(drop=True)
    track_dir = f'{LOA_DATA_DIR}/model_only/tracks/{scene_idx}'
    os.makedirs(track_dir, exist_ok=True)
    df_sort.to_csv(f'{track_dir}/tracks.csv')

    print(df_sort)

    for idx, row in df_sort.iterrows():
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

    # For whatever reason, some of the predictions are rotated
    # Not sure why, but for now manual fix
    idx_to_rot2 = get_idx_to_rot2()

    for scene_idx in range(150):
        try:
            serialize_by_conf(
                level5data,
                preds,
                level5data.scene[scene_idx],
                scene_idx,
                idx_to_rot2[scene_idx]
            )
        except:
            pass


if __name__ == '__main__':
    main()
