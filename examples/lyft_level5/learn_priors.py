import argparse
import pandas as pd
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
from loa.datum.track import Track, TrackDatum
from loa.datum.datum_3d import make_datum_from_gt
from loa.prior.av_priors import VolumeObsPrior, SpeedTransitionPrior, TrackLengthPrior
from constants import LYFT_DATA_DIR, PRIOR_DIR


def get_instance_vals(instance, level5data, prior, prior_type):
    first_ann_token = instance['first_annotation_token']
    last_ann_token = instance['last_annotation_token']
    instance_anns = []
    data = []
    current_token = first_ann_token
    while current_token != last_ann_token:
        ann_record = level5data.get('sample_annotation', current_token)
        ann_record['name'] = ann_record['category_name']
        sample_record = level5data.get('sample', ann_record['sample_token'])
        sd_record = level5data.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = level5data.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = level5data.get('ego_pose', sd_record['ego_pose_token'])
        datum = make_datum_from_gt(ann_record, pose_record, cs_record, ts=sample_record['timestamp'])
        data.append(datum)
        # sample_record = level5data.get('sample', ann_record['sample_token'])
        # _, boxes, _ = level5data.get_sample_data(sample_record['data']['LIDAR_TOP'], selected_anntokens=[current_token])
        # for box in boxes:
        #     print(box)
        #     corners = view_points(box.corners(), np.eye(4), normalize=False)[:2, :]
        #     print(corners)
        #     print(corners.T)
        # break
        instance_anns.append(ann_record)
        current_token = ann_record['next']

    vals = []
    if prior_type == 'obs':
        for datum in data:
            val = prior.get_val(datum)
            vals.append(val)
    elif prior_type == 'transition':
        for d1, d2 in zip(data[:-1], data[1:]):
            val = prior.get_val((d1, None), (d2, None))
            vals.append(val)
    elif prior_type == 'track':
        if len(data) == 0:
            print('wtf')
            return []
        track = Track()
        for idx, datum in enumerate(data):
            d = TrackDatum({'gt': datum}, [])
            track.add_datum(idx, d)
        val = prior.get_val(track)
        vals.append(val)
    else:
        raise NotImplementedError
    return vals


def main():
    level5data = LyftDataset(
        data_path=f'{LYFT_DATA_DIR}',
        json_path=f'{LYFT_DATA_DIR}/data',
        verbose=True
    )

    for scene in level5data.scene:
        scene_token = scene['token']
        break

    priors = [
        (VolumeObsPrior(), 'obs', f'{PRIOR_DIR}/volume_obs_prior.p'),
        (SpeedTransitionPrior(), 'transition', f'{PRIOR_DIR}/speed_trans_prior.p'),
        (TrackLengthPrior(), 'track', f'{PRIOR_DIR}/track_length_prior.p')
    ]

    for (prior, prior_type, out_fname) in priors:
        all_vals = []
        for idx, instance in enumerate(level5data.instance):
            all_vals += get_instance_vals(instance, level5data, prior, prior_type)

        prior._init_empty()
        prior._fit(all_vals)
        prior.serialize(out_fname)


if __name__ == '__main__':
    main()
