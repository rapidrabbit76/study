import os
import random
import shutil

import numpy as np
import torch
from copy import deepcopy

import torch
from torch import Tensor
import torch.nn as nn


def seed_everything(seed: int = 42) -> int:
    if not isinstance(seed, int):
        seed = int(seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def forward(self, input):
        return self.module(input)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(
            model,
            update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m,
        )

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


def interleave(x: Tensor, size) -> Tensor:
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x: Tensor, size) -> Tensor:
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class BestMobelCheckPoint:
    def __init__(self, ckpt_dir="./ckpt") -> None:
        self.best_metric = None
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def is_best(self, metric) -> bool:
        if not self.best_metric:
            return True
        return metric >= self.best_metric

    def save_checkpoint(self, ckpt, metric, filename="cpkt.pth.tar"):
        filepath = os.path.join(self.ckpt_dir, filename)
        torch.save(ckpt, filepath)
        if self.is_best(metric):
            shutil.copyfile(filepath, os.path.join(self.ckpt_dir, "best_model.pth.tar"))


def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
