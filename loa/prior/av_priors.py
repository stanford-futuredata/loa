import numpy as np
from .priors import KDEScalarPrior
from .priors import Prior

class VolumeObsPrior(KDEScalarPrior):
    def __init__(self, is_prediction=True, serialized_path=None):
        self.is_prediction = is_prediction
        super().__init__(serialized_path=serialized_path)

    def get_val(self, observation):
        wlh = observation.data_box.wlh
        return observation.cls, wlh[0] * wlh[1] * wlh[2]

    def score(self, observation):
        if self.is_prediction != observation.is_prediction:
            return 1., False
        return super().score(observation)


class MinDistanceTrackPrior(KDEScalarPrior):
    def score(self, track):
        def get_dist(track_datum):
            for d in track_datum.observations.values():
                box = d.data_box
                break
            return box.center[0] ** 2 + box.center[1] ** 2
        dist = min([get_dist(datum) for datum in track.data])
        return -dist, True


class SpeedTransitionPrior(KDEScalarPrior):
    def get_val(self, bundle1, bundle2):
        xyz1 = bundle1[0].data_box.center
        xyz2 = bundle2[0].data_box.center
        dist = np.linalg.norm(xyz1 - xyz2)
        tdiff = bundle1[0].ts - bundle2[0].ts
        return bundle1[0].cls, dist / tdiff


class TrackLengthPrior(KDEScalarPrior):
    def get_val(self, track):
        datum = track.data[0]
        for d in datum.observations.values():
            cls = d.cls
            break
        return cls, len(track.data)


class NoShortTracksPrior(Prior):
    def score(self, track):
        if len(self.track) <= 2:
            return -np.inf, True
        else:
            return 0, True
