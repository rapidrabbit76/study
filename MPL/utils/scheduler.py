import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def CosineScheduleWithWarmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_wait_steps: int = 0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(
                max(1, num_warmup_steps + num_wait_steps)
            )

        progress = float(current_step - num_warmup_steps - num_wait_steps) / float(
            max(1, num_training_steps - num_warmup_steps - num_wait_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
