import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, SequentialSampler
from torch import optim
from tqdm.auto import tqdm
from utils.dataset import InfiniteSampler
from utils.scheduler import CosineScheduleWithWarmup
import utils
from hparams import Hparams
from dataset.cifar import DATASET_GETTERS
import torchmetrics.functional as TF
import transformers
import numpy as np
from accelerate import Accelerator
from datetime import datetime
import os
import models
from accelerate.utils import DistributedDataParallelKwargs

torch.backends.cudnn.benchmark = True


def hparams_check(hp: Hparams):
    assert os.path.exists(hp.teacher_ckpt_path)


def main():
    parser = transformers.HfArgumentParser((Hparams))
    hp: Hparams = parser.parse_args_into_dataclasses()[0]
    # hparams_check(hp)
    utils.seed_everything(hp.seed)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="tensorboard", logging_dir=hp.log_dir, kwargs_handlers=[kwargs]
    )

    with accelerator.main_process_first():
        labeled_ds, unlabeled_ds, test_ds, finetune_ds = DATASET_GETTERS[hp.dataset](hp)

    accelerator.print(
        f"""
        labeled_data: {len(labeled_ds)}
        unlabeled_data: {len(unlabeled_ds)}
        test_data: {len(test_ds)}
        finetune_data: {len(finetune_ds)}
        {hp}
        """,
    )

    labeled_dl = DataLoader(
        labeled_ds,
        sampler=InfiniteSampler(labeled_ds),
        num_workers=hp.num_workers,
        batch_size=hp.batch_size,
        drop_last=True,
        pin_memory=True,
    )
    unlabeled_dl = DataLoader(
        unlabeled_ds,
        sampler=InfiniteSampler(unlabeled_ds),
        num_workers=hp.num_workers,
        batch_size=hp.batch_size * hp.mu,
        drop_last=True,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        sampler=SequentialSampler(test_ds),
        num_workers=hp.num_workers,
        batch_size=hp.batch_size,
    )

    teacher = models.create_model(
        hp.teacher_backbone, num_classes=hp.num_classes, pretrained=False
    )
    # teacher.load_state_dict(torch.load(hp.teacher_ckpt_path, map_location="cpu"))
    student = models.create_model(
        hp.student_backbone,
        num_classes=hp.num_classes,
        pretrained=False,
        dropout=hp.student_dropout,
    )
    params = utils.etc.get_training_params(student, hp.weight_decay)
    optim = optim.SGD(
        params=params,
        lr=hp.lr,
        momentum=hp.student_momentum,
        nesterov=hp.nesterov,
    )
    optim_scheduler = CosineScheduleWithWarmup(
        optimizer=optim,
        num_warmup_steps=hp.warmup_steps,
        num_training_steps=hp.total_steps,
        num_wait_steps=hp.student_scheduler_wait_steps,
    )
    loss_met = utils.meters.MeanMetric()

    (
        teacher,
        student,
        optim,
        optim_scheduler,
        labeled_dl,
        unlabeled_dl,
        test_dl,
    ) = accelerator.prepare(
        teacher, student, optim, optim_scheduler, labeled_dl, unlabeled_dl, test_dl
    )
    if hp.use_ema:
        ema_student_model = utils.ModelEMA(student, decay=hp.ema_decay)
        
    teacher.eval()
    student.train()
    
    labeled_iter = iter(labeled_dl)
    unlabeled_iter = iter(unlabeled_dl)
    
    
    pbar = tqdm(
        range(hp.total_steps),
        bar_format="{desc} | {r_bar}",
        disable=not accelerator.is_local_main_process,
    )
    checkpoint_manager = utils.etc.BestMobelCheckPoint()
    
    if accelerator.is_main_process:
        accelerator.init_trackers(f"{datetime.now()}", config=vars(hp))
        
    for gs in pbar:
        images_l, targets_l = next(labeled_iter)
        (images_uw, images_us), _ = next(unlabeled_iter)

    
    
    


if __name__ == "__main__":
    main()
