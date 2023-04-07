from random import Random
import shutil
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, SequentialSampler
from torch import optim
from tqdm.auto import tqdm
from utils.dataset import InfiniteSampler
from utils.scheduler import CosineScheduleWithWarmup
import utils
import timm
from torch.utils.tensorboard import SummaryWriter
from argparse_dataclass import ArgumentParser
from hparams import Hparams
from dataset.cifar import DATASET_GETTERS
import torchmetrics.functional as TF
from utils.meters import MeanMetric
import transformers
import numpy as np
from accelerate import Accelerator
from datetime import datetime
import os

torch.backends.cudnn.benchmark = True


def main():
    parser = transformers.HfArgumentParser((Hparams))
    hp: Hparams = parser.parse_args_into_dataclasses()[0]
    utils.seed_everything(hp.seed)
    logging_dir = os.path.join(
        hp.log_dir, f"{hp.dataset}.{hp.num_labeled}.{hp.backbone}"
    )
    os.makedirs(logging_dir, exist_ok=True)
    accelerator = Accelerator(log_with="tensorboard", logging_dir=logging_dir)

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
    finetune_dl = DataLoader(
        finetune_ds,
        sampler=InfiniteSampler(finetune_ds),
        num_workers=hp.num_workers,
        batch_size=hp.finetune_batch_size,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        sampler=SequentialSampler(test_ds),
        num_workers=hp.num_workers,
        batch_size=hp.batch_size,
    )

    teacher_model = timm.create_model(
        hp.backbone, num_classes=hp.num_classes, pretrained=False
    )

    student_model = timm.create_model(
        hp.backbone, num_classes=hp.num_classes, pretrained=False
    )
    if hp.use_ema:
        ema_student_model = utils.ModelEMA(student_model, decay=hp.ema_decay)

    no_decay = ["bias", "bn"]
    teacher_params = [
        {
            "params": [
                p
                for n, p in teacher_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": hp.weight_decay,
        },
        {
            "params": [
                p
                for n, p in teacher_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    student_params = [
        {
            "params": [
                p
                for n, p in student_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": hp.weight_decay,
        },
        {
            "params": [
                p
                for n, p in student_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    t_optim = optim.SGD(
        params=teacher_params,
        lr=hp.lr,
        momentum=hp.teacher_momentum,
        nesterov=hp.nesterov,
    )
    s_optim = optim.SGD(
        params=student_params,
        lr=hp.student_lr,
        momentum=hp.student_momentum,
        nesterov=hp.nesterov,
    )
    t_scheduler = CosineScheduleWithWarmup(
        optimizer=t_optim,
        num_warmup_steps=hp.warmup_steps,
        num_training_steps=hp.total_steps,
    )
    s_scheduler = CosineScheduleWithWarmup(
        optimizer=s_optim,
        num_warmup_steps=hp.warmup_steps,
        num_training_steps=hp.total_steps,
        num_wait_steps=hp.student_scheduler_wait_steps,
    )

    metric_t_loss = MeanMetric()
    metric_t_loss_mpl = MeanMetric()
    metric_t_loss_uda = MeanMetric()
    metric_t_loss_l = MeanMetric()
    metric_t_loss_u = MeanMetric()
    metric_dot_product = MeanMetric()
    metric_mask = MeanMetric()
    metric_s_loss = MeanMetric()
    metric_s_loss_l_new = MeanMetric()
    metric_s_loss_l_old = MeanMetric()

    (
        teacher_model,
        t_optim,
        t_scheduler,
        student_model,
        s_optim,
        s_scheduler,
        labeled_dl,
        unlabeled_dl,
        test_dl,
        finetune_dl,
    ) = accelerator.prepare(
        teacher_model,
        t_optim,
        t_scheduler,
        student_model,
        s_optim,
        s_scheduler,
        labeled_dl,
        unlabeled_dl,
        test_dl,
        finetune_dl,
    )

    teacher_model.train()
    student_model.train()

    labeled_iter = iter(labeled_dl)
    unlabeled_iter = iter(unlabeled_dl)
    finetune_iter = iter(finetune_dl)

    pbar = tqdm(
        range(hp.total_steps),
        bar_format="{desc} | {r_bar}",
        disable=not accelerator.is_local_main_process,
    )
    checkpoint_manager = utils.etc.BestMobelCheckPoint()
    if accelerator.is_main_process:
        accelerator.init_trackers(f"{datetime.now()}", config=vars(hp))

    # training loop
    for gs in pbar:
        images_l, targets_l = next(labeled_iter)
        (images_uw, images_us), _ = next(unlabeled_iter)

        batch_size = images_l.shape[0]
        images_l = images_l.to(accelerator.device)
        images_uw = images_uw.to(accelerator.device)
        images_us = images_us.to(accelerator.device)
        targets_l = targets_l.to(accelerator.device)

        # Teacher step
        t_images = torch.cat((images_l, images_uw, images_us))
        t_logits = teacher_model(t_images)
        t_logits_l = t_logits[:batch_size]
        t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)

        t_loss_l = F.cross_entropy(
            t_logits_l, targets_l, label_smoothing=hp.label_smoothing
        )

        soft_pseudo_label = torch.softmax(t_logits_uw.detach() / hp.T, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(hp.threshold).float()
        # UDA loss (KLD)
        t_loss_u = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1)
            * mask
        )
        weight_u = hp.lambda_u * min(1.0, (gs + 1) / hp.uda_steps)
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        # student step
        s_images = torch.cat((images_l, images_us))
        s_logits = student_model(s_images)
        s_logits_l = s_logits[:batch_size]
        s_logits_us = s_logits[batch_size:]
        del s_logits

        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets_l)
        s_loss = F.cross_entropy(
            s_logits_us, hard_pseudo_label, label_smoothing=hp.label_smoothing
        )

        accelerator.backward(s_loss)
        if hp.grad_clip > 0 and accelerator.sync_gradients:
            nn.utils.clip_grad_norm_(student_model.parameters(), hp.grad_clip)
        s_optim.step()
        s_scheduler.step()

        if hp.ema_decay > 0 and hp.use_ema:
            eam_model_ = accelerator.unwrap_model(student_model)
            ema_student_model.update_parameters(eam_model_)

        # MPL step
        with torch.no_grad():
            s_logits_l = student_model(images_l)
        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets_l)
        dot_product = s_loss_l_new - s_loss_l_old
        _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
        t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
        t_loss = t_loss_uda + t_loss_mpl
        accelerator.backward(t_loss)
        if hp.grad_clip > 0 and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(teacher_model.parameters(), hp.grad_clip)
        t_optim.step()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        t_loss_mean = accelerator.gather(t_loss).mean()
        t_loss_mpl_mean = accelerator.gather(t_loss_mpl).mean()
        t_loss_uda_mean = accelerator.gather(t_loss_uda).mean()
        t_loss_l_mean = accelerator.gather(t_loss_l).mean()
        t_loss_u_mean = accelerator.gather(t_loss_u).mean()
        dot_product_mean = accelerator.gather(dot_product).mean()
        mask_mean = accelerator.gather(mask).mean()

        s_loss_mean = accelerator.gather(s_loss).mean()
        s_loss_l_new_mean = accelerator.gather(s_loss_l_new).mean()
        s_loss_l_old_mean = accelerator.gather(s_loss_l_old).mean()

        metric_t_loss.update(t_loss_mean)
        metric_t_loss_mpl.update(t_loss_mpl_mean)
        metric_t_loss_uda.update(t_loss_uda_mean)
        metric_t_loss_l.update(t_loss_l_mean)
        metric_t_loss_u.update(t_loss_u_mean)
        metric_dot_product.update(dot_product_mean)
        metric_mask.update(mask_mean)
        metric_s_loss.update(s_loss_mean)
        metric_s_loss_l_new.update(s_loss_l_new_mean)
        metric_s_loss_l_old.update(s_loss_l_old_mean)

        msg = {"t_loss": metric_s_loss.mean, "s_loss": metric_s_loss.mean}

        if gs % 50 == 0 or (gs + 1 == hp.total_steps):
            log_dict = {
                "train/t_loss": metric_s_loss.reset_and_compute(),
                "train/t_loss_mpl": metric_t_loss_mpl.reset_and_compute(),
                "train/t_loss_uda": metric_t_loss_uda.reset_and_compute(),
                "train/t_loss_l": metric_t_loss_l.reset_and_compute(),
                "train/t_loss_u": metric_t_loss_u.reset_and_compute(),
                "train/dot_product": metric_dot_product.reset_and_compute(),
                "train/mask": metric_mask.reset_and_compute(),
                "train/s_loss": metric_s_loss.reset_and_compute(),
                "train/s_loss_l_new": metric_s_loss_l_new.reset_and_compute(),
                "train/s_loss_l_old": metric_s_loss_l_old.reset_and_compute(),
            }
            accelerator.log(log_dict, step=gs)

        if gs % 1000 == 0 or (gs + 1 == hp.total_steps):
            test_model = ema_student_model if hp.use_ema else student_model
            test_model.eval()
            info_dict = eval_loop(hp, test_dl, test_model, accelerator=accelerator)
            test_model.train()
            accelerator.log(log_dict, step=gs)

            ckpt = {
                "teacher_state_dict": teacher_model.state_dict(),
                "student_state_dict": student_model.state_dict(),
                "acc": info_dict["acc"],
                "gs": gs,
            }
            file_path = "./ckpt/ckpt.pth"
            accelerator.save(ckpt, file_path)
            if accelerator.is_local_main_process:
                if checkpoint_manager.is_best(info_dict["acc"]):
                    shutil.copyfile(file_path, "./ckpt/best_model.pth.tar")
            accelerator.wait_for_everyone()

        msg = dict(msg, **info_dict)
        pbar.set_postfix(**msg)

    accelerator.wait_for_everyone()
    # finetune loop
    ft_total_step = hp.finetune_epochs * len(finetune_ds) // hp.finetune_batch_size
    pbar = tqdm(range(ft_total_step), disable=not accelerator.is_local_main_process)
    model = student_model
    metric_ft_loss = MeanMetric()
    f_optim = optim.SGD(
        params=model.parameters(),
        lr=hp.finetune_lr,
        momentum=hp.finetune_momentum,
        nesterov=hp.nesterov,
    )
    f_optim = accelerator.prepare(f_optim)

    model.train()
    for ft_gs in pbar:
        images, targets = next(finetune_iter)
        batch_size = images.shape[0]
        images = images.to(accelerator.device)
        targets = targets.to(accelerator.device)

        model.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss_mean = accelerator.gather(loss).mean()
        accelerator.backward(loss)
        f_optim.step()
        metric_ft_loss.update(loss_mean.item())
        logs = {"loss": metric_ft_loss.mean}
        pbar.set_postfix(**logs)

        if ft_gs % 50 == 0 and (ft_gs + 1 == ft_total_step):
            accelerator.log({"ft/loss": metric_ft_loss.reset_and_compute()}, ft_gs)

        if ft_gs % (len(finetune_ds) // hp.finetune_batch_size) == 0 and (
            ft_gs + 1 == ft_total_step
        ):
            test_model = ema_student_model if hp.use_ema else student_model
            test_model.eval()
            info_dict = eval_loop(hp, test_dl, test_model, accelerator=accelerator)
            test_model.train()
            accelerator.log(log_dict, step=ft_gs)
    ckpt = {
        "teacher_state_dict": teacher_model.state_dict(),
        "student_state_dict": student_model.state_dict(),
        "acc": info_dict["acc"],
        "gs": gs,
    }
    file_path = "./ckpt/end.pth"
    accelerator.save(ckpt, file_path)
    accelerator.save(student_model.state_dict(), "./ckpt/student_model.pth")
    accelerator.end_training()


@torch.inference_mode()
def eval_loop(hp: Hparams, test_dl, model, accelerator: Accelerator):
    acc_meter = []
    acc_t3_meter = []
    loss_meter = []

    for bidx, (x, y) in enumerate(test_dl):
        x, y = x.to(accelerator.device), y.to(accelerator.device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_meter.append(loss.item())

        logits = accelerator.gather_for_metrics(logits)
        y = accelerator.gather_for_metrics(y)

        acc = TF.accuracy(
            logits, y, num_classes=hp.num_classes, task="multiclass"
        ).item()
        acc_t3 = TF.accuracy(
            logits, y, num_classes=hp.num_classes, top_k=3, task="multiclass"
        ).item()
        acc_meter.append(acc)
        acc_t3_meter.append(acc_t3)
    return {
        "loss": np.mean(loss_meter),
        "acc": np.mean(acc_meter),
        "acc_t3": np.mean(acc_t3_meter),
    }


if __name__ == "__main__":
    main()
