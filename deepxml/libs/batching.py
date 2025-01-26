import torch
import numpy as np
from typing import Iterable


class MySampler(torch.utils.data.Sampler[int]):
    def __init__(self, order: np.ndarray):
        self.order = order.copy()

    def update_order(self, x: np.ndarray) -> None:
        self.order[:] = x[:]

    def __iter__(self) -> Iterable[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


class DummyLoader():
    def __init__(self, dataset, *args, **kwargs) -> None:
        pass