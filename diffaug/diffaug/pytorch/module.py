from torch import Tensor, nn
from abc import ABCMeta, abstractmethod
from . import functional as DF


class DiffAugmentation(nn.Module):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class RandomBrightness(DiffAugmentation):
    def forward(self, x: Tensor) -> Tensor:
        return DF.random_brightness(x=x)


class RandomSaturation(DiffAugmentation):
    def forward(self, x: Tensor) -> Tensor:
        return DF.random_saturation(x=x)


class RandomContrast(DiffAugmentation):
    def forward(self, x: Tensor) -> Tensor:
        return DF.random_contrast(x=x)


class RandomTranslation(DiffAugmentation):
    def __init__(self, ratio: float = 0.125) -> None:
        super().__init__()
        self._ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        return DF.random_translation(x=x, ratio=self._ratio)


class RandomCutout(DiffAugmentation):
    def __init__(self, ratio: float = 0.5) -> None:
        super().__init__()
        self._ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        return DF.random_cutout(x=x, ratio=self._ratio)
