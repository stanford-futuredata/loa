import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ujson as json
import time
import os
from pyquaternion import Quaternion
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import view_points
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from loa.datum.datum_3d import PredBox3DDatum, GTBox3DDatum
from constants import LYFT_DATA_DIR, PRIOR_DIR, LOA_DATA_DIR


def dict_to_datum(d):
    trans = np.array(d['translation'])
    rotation = np.array(d['orientation'])
    size = np.array(d['size'])
    sample_token = d['sample_token']
    label = d['label']
    score = d['score']
    ts = d['ts']
    
    data_box = Box(
        trans,
        size,
        Quaternion(rotation),
        name=label,
        score=score,
        token=sample_token
    )
    eval_box = Box3D(
        sample_token=sample_token,
        translation=data_box.center,
        size=data_box.wlh,
        rotation=list(data_box.orientation),
        name=label
    )

    if d['obs_type'] == 'pred':
        return PredBox3DDatum(data_box, eval_box, ts, score=score, cls=label, identifier=d['iden'])
    elif d['obs_type'] == 'gt':
        return GTBox3DDatum(data_box, eval_box, ts, score=score, cls=label, identifier=d['iden'])
    else:
        raise NotImplementedError


def render_lidar_pc(level5data, sample_token, ax, axes_limit):
    sample_record = level5data.get('sample', sample_token)
    sd_record = level5data.get('sample_data', sample_record['data']['LIDAR_TOP'])
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])

    lidar_fname = level5data.data_path / sd_record['filename']
    pc = LidarPointCloud.from_file(lidar_fname)

    vehicle_from_sensor = np.eye(4)
    vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
    vehicle_from_sensor[:3, 3] = cs_record["translation"]

    ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
    rot_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
    )

    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle

    points = view_points(
        pc.points[:3, :], np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor), normalize=False
    )
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)


def render_gt_boxes(level5data, sample_token, ax):
    sample_record = level5data.get('sample', sample_token)
    sd_record = level5data.get('sample_data', sample_record['data']['LIDAR_TOP'])
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])

    global_boxes = [level5data.get_box(record) for record in sample_record['anns']]
    gt_boxes = []
    for box in global_boxes:
        ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
        yaw = ypr[0]

        box.translate(-np.array(pose_record["translation"]))
        box.rotate_around_origin(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        gt_boxes.append(box)

    for box in gt_boxes:
        c = np.array(level5data.explorer.get_color(box.name)) / 255.0
        box.render(ax, view=np.eye(4), colors=(c, c, c))


def render_single(
        level5data, data, frame_idx,
        ax=None,
        axes_limit=100,
        viz_lidar_pc=False,
        viz_gt_boxes=False,
        out_dir=None
):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    sample_token = data[0].data_box.token
    if viz_lidar_pc:
        render_lidar_pc(level5data, sample_token, ax, axes_limit)
    if viz_gt_boxes:
        render_gt_boxes(level5data, sample_token, ax)

    ax.plot(0, 0, "x", color="red")
    max_conf = 0.
    for datum in data:
        box = datum.data_box
        if datum.is_prediction:
            c = (0, 0, 0)
            max_conf = max(max_conf, box.score)
        else:
            c = np.array(level5data.explorer.get_color(box.name)) / 255.0
        box.render(ax, view=np.eye(4), colors=(c, c, c))

    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_title(f'{frame_idx}, {max_conf * 100:2.2f}')
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, f'{frame_idx}.png'))
    else:
        plt.show()
    fig.clear()
    plt.close(fig)


def viz_one(level5data, track_fname, out_dir=None):
    with open(track_fname, 'r') as f:
        track_raw_data = json.load(f)
    track_data = list(map(dict_to_datum, track_raw_data))
    print(track_data[0].data_box)

    def get_st_to_frame_idx(curr_token):
        d = {}
        frame_idx = 0
        next_token = level5data.get('sample', curr_token)['next']
        while next_token != '':
            d[curr_token] = frame_idx
            frame_idx += 1
            curr_token = next_token
            next_token = level5data.get('sample', curr_token)['next']
        return d

    sample_token = track_data[0].data_box.token
    sample_rec = level5data.get('sample', sample_token)
    scene_token = sample_rec['scene_token']
    scene_rec = level5data.get('scene', scene_token)
    st_to_frame_idx = get_st_to_frame_idx(scene_rec['first_sample_token'])

    for idx, track_datum in enumerate(track_data):
        frame_idx = st_to_frame_idx[track_datum.data_box.token]
        print(idx, frame_idx)
        render_single(
            level5data, [track_datum], frame_idx,
            viz_lidar_pc=True,
            viz_gt_boxes=True,
            out_dir=out_dir
        )
        # time.sleep(0.5)
        plt.cla()
        plt.clf()


def main():
    level5data = LyftDataset(
        data_path=f'{LYFT_DATA_DIR}',
        json_path=f'{LYFT_DATA_DIR}/data',
        verbose=True
    )

    leftovers = [117]
    for scene_idx in leftovers:
        base_path = f'{LOA_DATA_DIR}/ma-conf/'
        track_path = f'{base_path}/tracks/{scene_idx}'
        try:
            df = pd.read_csv(os.path.join(track_path, 'tracks.csv'))
        except:
            continue
        for idx, row in df.iterrows():
            rank_idx = row['Unnamed: 0']
            if rank_idx < 150 or rank_idx > 450:
                continue
            track_idx = row['track_idx']
            json_fname = os.path.join(track_path, f'{rank_idx}-{track_idx}.json')
            out_dir = os.path.join(base_path, 'viz', str(scene_idx), f'{rank_idx}-{track_idx}')
            os.makedirs(out_dir, exist_ok=True)

            viz_one(
                level5data,
                json_fname,
                out_dir=out_dir
            )

if __name__ == '__main__':
    main()
