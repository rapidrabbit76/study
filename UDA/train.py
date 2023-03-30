import torch
from torch.utils.data import DataLoader
from utils.dataset import build_dataset


def train_collate_fn(batch):
    images = [row["x"] for row in batch]
    images = torch.vstack(images)
    labels = torch.tensor([row["label"] for row in batch]).long()
    return {"x": images, "labels": labels}


def uda_collate_fn(batch):
    x_w = [row["x_w"] for row in batch]
    x_w = torch.vstack(x_w)

    x_s = [row["x_s"] for row in batch]
    x_s = torch.vstack(x_s)
    return {"x_s": x_s, "x_w": x_w}


def main():
    train_ds, val_ds, test_ds, uda_ds = build_dataset()
    train_dl_args = dict(
        batch_size=4, num_workers=4, drop_last=True, pin_memory=True
    )
    DataLoader(train_ds, collate_fn=train_collate_fn, **train_dl_args)
    DataLoader(uda_ds, collate_fn=uda_collate_fn, **train_dl_args)
    DataLoader(val_ds, batch_size=4, num_workers=4, collate_fn=train_collate_fn)
    DataLoader(
        test_ds, batch_size=4, num_workers=4, collate_fn=train_collate_fn
    )

    return -1


if __name__ == "__main__":
    main()
