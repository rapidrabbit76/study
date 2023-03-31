from cProfile import label
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
from tqdm.auto import tqdm
from utils.dataset import build_dataset, InfiniteSampler
from utils.scheduler import CosineScheduleWithWarmup
import utils
import timm
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from argparse_dataclass import ArgumentParser
from hparams import Hparams
from dataset.cifar import DATASET_GETTERS
import torchmetrics.functional as TF
import numpy as np

torch.backends.cudnn.benchmark = True


def main():
    accelerator = Accelerator()
    parser = ArgumentParser(Hparams)
    hp = parser.parse_args([])
    utils.seed_everything(hp.seed)

    if accelerator.is_local_main_process():
        tb = SummaryWriter("./logs")

    with accelerator.main_process_first():
        labeled_ds, unlabeled_ds, test_ds = DATASET_GETTERS[hp.dataset](hp, "./data")
        labeled_dl = DataLoader(
            labeled_ds,
            sampler=InfiniteSampler(labeled_ds, shuffle=True),
            num_workers=hp.num_workers,
            batch_size=hp.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        unlabeled_dl = DataLoader(
            unlabeled_ds,
            sampler=InfiniteSampler(unlabeled_ds, shuffle=True),
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
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process():
        print(
            f"""
            labeled_data: {len(labeled_ds)}
            unlabeled_data: {len(unlabeled_ds)}""",
        )

    model = timm.create_model(
        "resnet50", num_classes=hp.num_classes, pretrained=False
    ).to(accelerator.device)

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

    (
        model,
        optimizer,
        labeled_dl,
        unlabeled_dl,
        test_dl,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        labeled_dl,
        unlabeled_dl,
        test_dl,
        scheduler,
    )

    labeled_iter = iter(labeled_dl)
    unlabeled_iter = iter(unlabeled_dl)

    pbar = tqdm(
        range(hp.total_steps),
        bar_format="{desc} | {r_bar}",
        disable=not accelerator.is_local_main_process,
    )
    for gs in pbar:
        inputs_x, targets_x = next(labeled_iter)
        (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(hp.device)
        targets_x = targets_x.to(hp.device)
        logits = model(inputs)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        sv_loss = F.cross_entropy(logits_x, targets_x)

        targets_u = torch.softmax(logits_u_w.detach() / hp.T, dim=-1)
        max_probs, _ = torch.max(targets_u, dim=-1)
        mask = max_probs.ge(hp.threshold).float()

        uda_loss = torch.mean(
            -(targets_u * torch.log_softmax(logits_u_s, dim=-1)).sum(dim=-1) * mask
        )
        loss = sv_loss + uda_loss * hp.lambda_u

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        if hp.use_ema:
            ema_model.update_parameters(model)
        model.zero_grad()

        if gs % 5 == 0:
            if accelerator.is_local_main_process:
                tb.add_scalar("train/1.loss", loss.item(), global_step=gs)
                tb.add_scalar("train/2.loss.sv", sv_loss.item(), global_step=gs)
                tb.add_scalar("train/3.loss.uda", uda_loss.item(), global_step=gs)

        # test
        if gs % 1000 == 0:
            test_model = ema_model if hp.use_ema else model
            test_model.eval()
            info_dict = eval_loop(hp, test_dl, test_model, accelerator)
            test_model.train()

            if accelerator.is_local_main_process:
                tb.add_scalar("validation/1.loss", info_dict["loss"], global_step=gs)
                tb.add_scalar("validation/2.acc", info_dict["acc"], global_step=gs)
                tb.add_scalar(
                    "validation/3.acc_t3", info_dict["acc_t3"], global_step=gs
                )

        if accelerator.is_local_main_process:
            msg = (
                f"lr: {scheduler.get_last_lr()[0] :.4f} "
                + f"loss: {loss.item(): .4f} "
                + f"loss_sv: {sv_loss.item(): .4f} "
                + f"uda_loss: {uda_loss.item(): .4f} "
                + f"mask: {mask.mean().item():.2f} "
                + f"test_loss: {info_dict['loss']: .4f} "
                + f"test_acc: {info_dict['acc']: .4f} "
                + f"test_acc_t3: {info_dict['acc_t3']: .4f}"
            )
            pbar.set_description_str(msg)
        accelerator.wait_for_everyone()

    return -1


@torch.inference_mode()
def eval_loop(hp, test_dl, model, accelerator: Accelerator):
    accelerator.wait_for_everyone()
    acc_meter = []
    acc_t3_meter = []
    loss_meter = []

    for bidx, (x, y) in enumerate(test_dl):
        x, y = x.to(hp.device), y.to(hp.device)
        logits = model(x)

        logits = accelerator.gather_for_metrics(logits)
        y = accelerator.gather_for_metrics(y)

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
