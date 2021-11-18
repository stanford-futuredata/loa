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
from loa.prior.av_priors import VolumeObsPrior, SpeedTransitionPrior, MinDistanceTrackPrior
from constants import LYFT_DATA_DIR, PRIOR_DIR, LOA_DATA_DIR


def get_idx_to_rot2():
    idx_to_rot2 = defaultdict(lambda: None)
    idx_to_rot2[1] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[2] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[3] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[13] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[14] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[15] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[21] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[22] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[27] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[35] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[36] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[38] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[39] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[40] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[47] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[49] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[52] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[56] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[59] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[60] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[65] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[68] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[69] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[77] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[78] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[88] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[89] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[90] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[91] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[97] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[98] = Quaternion(axis=[0, 0, 1], radians=-0.1)
    idx_to_rot2[99] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[101] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[103] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[106] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[107] = Quaternion(axis=[0, 0, 1], radians=-0.125)
    idx_to_rot2[109] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[111] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[115] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[116] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[117] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[119] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[122] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[124] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[126] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[128] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[133] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[136] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[138] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[140] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    idx_to_rot2[144] = Quaternion(axis=[0, 0, 1], radians=-0.125)
    idx_to_rot2[145] = Quaternion(axis=[0, 0, 1], radians=-0.11)
    idx_to_rot2[148] = Quaternion(axis=[0, 0, 1], radians=-0.09)
    return idx_to_rot2

def get_gt_track(level5data, cur_tracks, gt_datum):
    def get_instance_token(ann_token):
        ann_record = level5data.get('sample_annotation', ann_token)
        return ann_record['instance_token']

    cur_instance_token = get_instance_token(gt_datum.identifier)
    for track in cur_tracks:
        for datum in track.data:
            if 'gt' not in datum.observations:
                continue
            datum = datum.observations['gt']
            if get_instance_token(datum.identifier) == cur_instance_token:
                return track
    return None


def get_overlapping(tracks, datum, obs_type):
    overlapping = []
    for track in tracks:
        max_area = -1
        for cur_obs_type, observation in track.data[-1].observations.items():
            if obs_type != cur_obs_type:
                continue
            if True or observation.cls == datum.cls:
                area = observation.intersects_iou(datum)
            else:
                area = -1
            max_area = max(max_area, area)
        if max_area > 0:
            overlapping.append((track, max_area))
    return overlapping


def serialize_track(track, out_fname):
    data = []
    for datum in track.data:
        for obs_type, obs in datum.observations.items():
            d = {
                'ts': float(obs.ts),
                'iden': obs.identifier,
                'obs_type': obs_type,
                'sample_token': obs.data_box.token,
                'translation': list(map(float, obs.data_box.center)),
                'size': list(map(float, obs.data_box.wlh)),
                'orientation': list(map(float, obs.data_box.orientation)),
                'label': obs.cls,
                'score': float(obs.data_box.score),
            }
            data.append(d)
    with open(out_fname, 'w') as f:
        json.dump(data, f, indent=2)

def process_scene(level5data, preds, scene_record, scene_idx, rot2=None):
    pred_cls_names = ['car', 'pedestrian', 'motorcycle', 'bicycle', 'other_vehicle', 'bus', 'truck']
    print('Processing scene', scene_record['token'])
    last_token = scene_record['last_sample_token']
    sample_token = scene_record['first_sample_token']
    sample_record = level5data.get('sample', sample_token)
    next_token = sample_record['next']

    all_tracks = []
    cur_tracks = []

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
        if sample_token not in preds:
            print(sample_token)
            return
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

        # _, ax = plt.subplots(1, 1, figsize=(9, 9))
        # ax.plot(0, 0, "x", color="red")
        # for datum in gt_data:
        #     box = datum.data_box
        #     c = np.array(level5data.explorer.get_color(box.name)) / 255.0
        #     box.render(ax, view=np.eye(4), colors=(c, c, c))
        #     # xs, ys = zip(*datum.poly)
        #     # print(xs, ys)
        #     # ax.plot(xs, ys)
        # for datum in pred_data:
        #     box = datum.data_box
        #     c = (0, 0, 0)
        #     box.render(ax, view=np.eye(4), colors=(c, c, c))
        # axes_limit = 40
        # ax.set_xlim(-axes_limit, axes_limit)
        # ax.set_ylim(-axes_limit, axes_limit)
        # plt.show()
        # sys.exit(0)

        for gt_datum in gt_data:
            add_datum = TrackDatum({'gt': gt_datum}, [])
            overlapping_track = get_gt_track(level5data, cur_tracks, gt_datum)
            if overlapping_track is None:
                new_track = Track()
                new_track.add_datum(timestamp, add_datum)
                cur_tracks.append(new_track)
            else:
                overlapping_track.add_datum(timestamp, add_datum)

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
    min_dist_prior = MinDistanceTrackPrior()

    scored = []
    both_contains = 0
    for track_idx, track in enumerate(all_tracks):
        if len(track.data) == 1:
            continue
        valid = True
        max_score = 0.0
        min_x = 100.
        min_y = 100.
        for datum_idx, cur_datum in enumerate(track.data):
            if 'gt' in cur_datum.observations:
                valid = False
                break
            if 'pred' in cur_datum.observations:
                max_score = max(max_score, cur_datum.observations['pred'].score)
                box = cur_datum.observations['pred'].data_box
                xyz = np.abs(box.center)
                min_x = min(abs(xyz[0]), min_x)
                min_y = min(abs(xyz[1]), min_y)
        if not valid:
            continue
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
        dist_score, _ = min_dist_prior.score(track)
        score += dist_score
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

    print(df_sort)

    track_output_dir = f'{LOA_DATA_DIR}/tracks/{scene_idx}'
    os.makedirs(track_output_dir, exist_ok=True)
    df_sort.to_csv(f'{track_output_dir}/tracks.csv')
    for idx, row in df_sort.iterrows():
        track_idx = int(row['track_idx'])
        track = all_tracks[track_idx]
        serialize_track(
            track,
            f'{track_output_dir}/{idx}-{track_idx}.json'
        )


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

    leftovers = [1, 2, 3, 13, 14, 15, 22, 27,
                 35, 36, 38, 39, 40, 47, 49, 52, 56, 59,
                 68, 69, 77, 78, 88, 89,
                 90, 91, 97, 99, 101, 103, 106, 109, 111, 115, 116, 117, 119,
                 122, 124, 126, 128, 133, 136, 138, 140, 145, 148]

    for scene_idx in leftovers: # range(150):
        process_scene(
            level5data,
            preds,
            level5data.scene[scene_idx],
            scene_idx,
            idx_to_rot2[scene_idx]
        )


if __name__ == '__main__':
    main()
