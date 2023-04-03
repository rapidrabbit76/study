from dataclasses import dataclass


@dataclass(frozen=True)
class Hparams:
    gpu_id = 0
    num_workers = 1
    dataset = "cifar10"
    epochs: int = 100
    num_classes: int = 10
    num_labeled = 4000
    expand_labels = False
    backbone: str = "resnet50"
    total_steps: int = 1048576
    device: str = "cuda:0"
    batch_size: int = 64
    lr: float = 0.03
    warmup: int = 0
    weight_decay: float = 0.0005
    nesterov: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    mu: int = 7
    lambda_u: float = 1
    T: float = 0.4
    threshold: float = 0.8
    seed: int = 42
    amp = False
