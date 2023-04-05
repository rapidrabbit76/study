from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Hparams:
    name: str = "cifar10-4K.2"
    data_path: str = "./data"
    save_path: str = "./ckpt"
    dataset: str = "cifar10"
    num_labeled: int = 4000
    expand_labels: bool = True
    total_steps: int = 300000
    eval_step: int = 1000
    start_step: int = 0
    num_workers: int = 4
    num_classes: int = 10
    resize: int = 32
    batch_size: int = 128
    teacher_dropout: float = 0.2
    student_dropout: float = 0.2
    teacher_lr: float = 0.05
    student_lr: float = 0.05
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0005
    ema: float = 0.995
    warmup_steps: int = 5000
    student_wait_steps: int = 3000
    grad_clip: float = 1000000000.0
    evaluate: bool = False
    finetune: bool = False
    finetune_epochs: int = 625
    finetune_batch_size: int = 512
    finetune_lr: float = 3e-05
    finetune_weight_decay: float = 0.0
    finetune_momentum: float = 0.9
    seed: int = 2
    label_smoothing: float = 0.15
    mu: int = 7
    threshold: float = 0.6
    temperature: float = 0.7
    lambda_u: float = 8.0
    uda_steps: float = 5000.0
    randaug: List[int] = [2, 16]
    amp: bool = True
