from random import Random
import torch
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
from torchmetrics import MeanMetric
import numpy as np


torch.backends.cudnn.benchmark = True


tb = SummaryWriter("./logs")


def main():
    parser = ArgumentParser(Hparams)
    hp = parser.parse_args([])
    utils.seed_everything(hp.seed)

    metric_loss = MeanMetric()
    metric_loss_sv = MeanMetric()
    metric_loss_usv = MeanMetric()
    metric_mask = MeanMetric()

    labeled_ds, unlabeled_ds, test_ds = DATASET_GETTERS[hp.dataset](hp, "./data")
    print(
        f"""
        labeled_data: {len(labeled_ds)}
        unlabeled_data: {len(unlabeled_ds)}
        {hp}""",
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

    model = timm.create_model(
        hp.backbone, num_classes=hp.num_classes, pretrained=False
    ).to(hp.device)

    if hp.use_ema:
        ema_model = utils.ModelEMA(model, decay=hp.ema_decay)

    no_decay = ["bias", "bn"]
    params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": hp.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.SGD(params=params, lr=hp.lr, momentum=0.9, nesterov=hp.nesterov)
    scheduler = CosineScheduleWithWarmup(
        optimizer=optimizer,
        num_warmup_steps=hp.warmup,
        num_training_steps=hp.total_steps,
    )
    model.train()

    labeled_iter = iter(labeled_dl)
    unlabeled_iter = iter(unlabeled_dl)

    pbar = tqdm(range(hp.total_steps), bar_format="{desc} | {r_bar}")
    checkpoint_manager = utils.etc.BestMobelCheckPoint()
    for gs in pbar:
        inputs_x, targets_x = next(labeled_iter)
        (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
        inputs = utils.etc.interleave(inputs, 2 * hp.mu + 1).to(hp.device)
        targets_x = targets_x.to(hp.device)
        logits = model(inputs)
        logits = utils.etc.de_interleave(logits, 2 * hp.mu + 1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        sv_loss = F.cross_entropy(logits_x, targets_x, reduction="mean")

        pseudo_label = torch.softmax(logits_u_w.detach() / hp.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(hp.threshold).float()

        usv_loss = (
            F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
        ).mean()
        loss = sv_loss + usv_loss * hp.lambda_u

        loss.backward()

        optimizer.step()
        scheduler.step()

        if hp.use_ema:
            ema_model.update_parameters(model)

        model.zero_grad()
        metric_loss.update(loss.item())
        metric_loss_sv.update(sv_loss.item())
        metric_loss_usv.update(usv_loss.item())
        metric_mask.update(mask.mean().item())

        if gs % 50 == 0 or (gs + 1 == hp.total_steps):
            tb.add_scalar("train/1.loss", metric_loss.compute(), global_step=gs)
            tb.add_scalar("train/2.loss.sv", metric_loss_sv.compute(), global_step=gs)
            tb.add_scalar("train/3.loss.usv", metric_loss_usv.compute(), global_step=gs)
            tb.add_scalar("train/4.mask", metric_mask.compute(), global_step=gs)

        if gs % 1000 == 0 or (gs + 1 == hp.total_steps):
            # test
            test_model = ema_model if hp.use_ema else model
            test_model.eval()
            info_dict = eval_loop(hp, test_dl, test_model)
            test_model.train()

            tb.add_scalar("validation/1.loss", info_dict["loss"], global_step=gs)
            tb.add_scalar("validation/2.acc", info_dict["acc"], global_step=gs)
            tb.add_scalar("validation/3.acc_t3", info_dict["acc_t3"], global_step=gs)
            metric_loss.reset()
            metric_loss_sv.reset()
            metric_loss_usv.reset()
            metric_mask.reset()

            ckpt = {
                "gs": gs,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if hp.use_ema else None,
                "acc": info_dict["acc"],
            }
            checkpoint_manager.save_checkpoint(ckpt, info_dict["acc"])

        msg = (
            f"lr: {scheduler.get_last_lr()[0] :.4f} "
            + f"loss: {metric_loss.compute(): .4f} "
            + f"loss_sv: {metric_loss_sv.compute(): .4f} "
            + f"loss_usv: {metric_loss_usv.compute(): .4f} "
            + f"mask: {metric_mask.compute():.2f} "
            + f"test_loss: {info_dict['loss']: .4f} "
            + f"test_acc: {info_dict['acc']: .4f} "
            + f"test_acc_t3: {info_dict['acc_t3']: .4f}"
        )

        pbar.set_description_str(msg)

    return -1


@torch.inference_mode()
def eval_loop(hp, test_dl, model):
    acc_meter = []
    acc_t3_meter = []
    loss_meter = []

    for bidx, (x, y) in enumerate(test_dl):
        x, y = x.to(hp.device), y.to(hp.device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_meter.append(loss.item())

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
