import numpy as np
from torch import Tensor
from typing import Union


class MeanMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, value: Union[float, Tensor]):
        if isinstance(value, Tensor):
            value = value.item()
        self.value = value
        self.sum += value
        self.count += 1
        self.mean = self.sum / self.count

    def reset_and_compute(self) -> float:
        mean = self.mean
        self.reset()
        return mean
