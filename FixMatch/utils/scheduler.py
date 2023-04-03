import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def CosineScheduleWithWarmup(
    optimizer: Optimizer,
    num_warmup_steps: float,
    num_training_steps: int,
    num_cycles: float = 7.0 / 16.0,
    last_epoch: int = -1,
) -> LambdaLR:
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
