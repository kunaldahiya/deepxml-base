from typing import Union
from numpy import ndarray

import numpy as np
from .utils import filter_predictions
from dataclasses import dataclass, field
from scipy.sparse import issparse, spmatrix
import xclib.evaluation.xc_metrics as xc_metrics


@dataclass
class MetricRecord:
    precision: ndarray
    ndcg: ndarray
    ps_precision: ndarray=field(default_factory=lambda: np.array(-1))
    ps_ndcg: ndarray=field(default_factory=lambda: np.array(-1))

    def __format(self, x):
        return ",".join(map(lambda z: f"{z:.2f}", x))

    def __str__(self):
        s = []
        s.append(self.__format(self.precision))
        s.append(self.__format(self.ndcg))
        if self.ps_precision.sum().item() > 0:
            s.append(self.__format(self.ps_precision))
        if self.ps_ndcg.sum().item() > 0:
            s.append(self.__format(self.ps_ndcg))
        return "\n".join(s)

    def summary(self):
        return f"P@1 (%): {self.precision[0]*100:.2f}"


class Evaluater:
    def __init__(self, k=5, A=0.55, B=1.5, labels=None, filter_map=None):
        self.filter_map = filter_map
        self.k = k
        self.inv_prop = self.compute_inv_psp(labels, A, B)

    def setup(self):
        pass

    def compute_inv_psp(self, labels, A, B):
        if labels is not None:
            return xc_metrics.compute_inv_propensity(labels, A, B)
        else:
            return None

    def _evaluate(self, _true: spmatrix, _pred: spmatrix, k: int):
        acc = xc_metrics.Metrics(_true, self.inv_prop)
        acc = acc.eval(_pred.tocsr(), k)
        return MetricRecord(*acc)

    def _load_filter_map(self, filter_map):
        filter_map = self.filter_map if filter_map is None else filter_map
        if isinstance(filter_map, str):
            filter_map = np.loadtxt(filter_map, dtype='int')
        return filter_map

    def __call__(
            self, 
            _true: spmatrix, 
            _pred: Union[spmatrix, dict],
            k: int=5,
            filter_map=None):
        k = k if k > 0 else self.k
        filter_map = self._load_filter_map(filter_map)
        if issparse(_pred):
            return self._evaluate(
                _true, filter_predictions(_pred, filter_map), k)
        else:  # Multiple set of predictions
            acc = {}
            for key, val in _pred.items():
                acc[key] = self._evaluate(
                    _true, filter_predictions(val, filter_map), k)
            return acc