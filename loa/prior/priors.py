import pickle
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KernelDensity


class Prior(object):
    def __init__(self):
        pass

    def score(self, *args, **kwargs):
        raise NotImplementedError

    def serialize(self, out_path):
        raise NotImplementedError

    def deserialize(self, out_path):
        raise NotImplementedError


class KDEScalarPrior(Prior):
    def __init__(self, serialized_path=None, class_conditional=True):
        if serialized_path is not None:
            self.deserialize(serialized_path)
        else:
            self.model = None
        self.class_conditional = class_conditional

    def _init_empty(self):
        self.model = defaultdict(KernelDensity)
        # self.model = {None: KernelDensity()}

    def _score(self, val):
        cls, val = val
        # print(cls, cls in self.model, self.model.keys())
        log_prob = self.model[cls].score([[val]])
        return log_prob
    def get_val(self, *args, **kwargs):
        raise NotImplementedError
    def score(self, *args, **kwargs):
        if self.model is None:
            return 1., False # TODO: check if it's 0 or 1
        else:
            val = self.get_val(*args, **kwargs)
            return self._score(val), True

    def _fit(self, vals):
        cls_to_val = defaultdict(list)
        for cls, val in vals:
            cls_to_val[cls].append(val)
        for cls, vals in cls_to_val.items():
            self.model[cls].fit(np.array(vals).reshape(-1, 1))

    def serialize(self, out_path):
        pickle.dump(self.model, open(out_path, 'wb'))

    def deserialize(self, serialized_path):
        self.model = pickle.load(open(serialized_path, 'rb'))

