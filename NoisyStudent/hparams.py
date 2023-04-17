from dataclasses import dataclass


@dataclass(frozen=True)
class Hparams:
    num_workers = 8
    dataset = "cifar10"
    resize = 32
    data_path: str = "./data"
    num_classes: int = 10
    num_labeled = 4000
    expand_labels = True
    randaug = [2, 16]
    eval_step: int = 1000
    batch_size: int = 128

    backbone: str = "WideResNet"

    teacher_backbone: str = "WideResNet"
    teacher_ckpt_path: str = "./WideResNet.teacher.pth"

    student_backbone: str = "WideResNet"
    student_dropout: float = 0.2
    student_lr: float = 0.05
    student_momentum = 0.9
    student_scheduler_wait_steps: int = 3000

    log_dir: str = "./logs"
    total_steps: int = 300000
    uda_steps = 5000.0
    lr: float = 0.03
    warmup_steps: int = 5000
    weight_decay: float = 0.0005
    nesterov: bool = True
    use_ema: bool = False
    ema_decay: float = 0.995
    mu: int = 7
    lambda_u: float = 8.0
    label_smoothing: float = 0.15
    T: float = 0.7
    threshold: float = 0.6
    seed: int = 2
    grad_clip: float = 1000000000.0
