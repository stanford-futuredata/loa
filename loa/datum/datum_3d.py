import numpy as np
import pyclipper
from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.utils.geometry_utils import view_points
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D

def make_datum_from_gt(ann, pose_record, cs_record, ts=None, identifier=None):
    data_box = Box(
            ann['translation'],
            ann['size'],
            Quaternion(ann['rotation']),
            name=ann['category_name'],
            token=ann['token'],
    )
    # Translate by pose
    data_box = data_box.translate(-np.array(pose_record['translation']))
    ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
    yaw = ypr[0]
    data_box.rotate_around_origin(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
    # data_box = data_box.rotate_around_origin(Quaternion(pose_record['rotation']).inverse)
    # Translate by sensor
    # data_box = data_box.translate(-np.array(cs_record['translation']))
    # data_box = data_box.rotate_around_origin(Quaternion(cs_record['rotation']).inverse)

    # Because the Box is in absolute coordinates, the 3D box needs to be made from the Box post translation
    eval_box = Box3D(
        sample_token=ann['token'],
        translation=data_box.center,
        size=data_box.wlh,
        rotation=list(data_box.orientation),
        name=ann['category_name']
    )

    return GTBox3DDatum(data_box, eval_box, ts, cls=ann['category_name'], identifier=identifier)


def make_datum_from_pred(
        sample_token,
        box3d,
        score,
        label,
        trans,
        rot,
        rot2=None,
        ts=None,
        identifier=None
):
    quat = Quaternion(axis=[0, 0, 1], radians=box3d[-1])
    velocity = (*box3d[6:8], 0.0, 0.0)  # TODO: ??
    data_box = Box(
        box3d[:3],
        box3d[3:6],
        quat,
        name=label,
        score=score,
        velocity=velocity,
        token=sample_token
    )
    data_box.translate(trans)
    data_box.rotate_around_origin(rot)
    if rot2 is not None:
        data_box.rotate_around_origin(rot2)
    # The eval box should be made post translation
    eval_box = Box3D(
        sample_token=sample_token,
        translation=data_box.center,
        size=data_box.wlh,
        rotation=list(data_box.orientation),
        name=label
    )

    return PredBox3DDatum(data_box, eval_box, ts, score=score, cls=label, identifier=identifier)

class Box3DDatum(object):
    def __init__(
            self,
            data_box,
            eval_box,
            ts,
            score=0.0,
            cls=None,
            identifier=None
    ):
        self.data_box = data_box
        self.corners = view_points(data_box.corners(), view=np.eye(4), normalize=False)[:2, :].T
        self.poly = [
            (self.corners[0, 0], self.corners[0, 1]),
            (self.corners[4, 0], self.corners[4, 1]),
            (self.corners[5, 0], self.corners[5, 1]),
            (self.corners[1, 0], self.corners[1, 1]),
            (self.corners[0, 0], self.corners[0, 1]),
        ]
        self.poly = np.array(self.poly)
        self.eval_box = eval_box
        self.ts = ts
        self.score = score
        self.cls = cls
        self.identifier = identifier

    def _poly_intersect(self, poly1, poly2):
        pc = pyclipper.Pyclipper()
        poly1 = (poly1 * 100).astype(np.int32)
        poly2 = (poly2 * 100).astype(np.int32)
        pc.AddPath(poly1, pyclipper.PT_CLIP, True)
        pc.AddPath(poly2, pyclipper.PT_SUBJECT, True)
        solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        return solution

    # def intersect_iou(self, other):
    #     return self.eval_box.get_iou(other.eval_box)
    def intersects_iou(self, other):
        intersect = self._poly_intersect(self.poly, other.poly)
        if len(intersect) == 0:
            return 0
        return sum([pyclipper.Area(inter_poly) for inter_poly in intersect])

    def intersects(self, other):
        iou = self.intersects_iou(other)
        return iou > 0.0

    def is_valid(self):
        return True

    def __repr__(self):
        return str(self.data_box)


class PredBox3DDatum(Box3DDatum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_prediction = True


class GTBox3DDatum(Box3DDatum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_prediction = False