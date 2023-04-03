from datasets import load_dataset
from torchvision import transforms
import copy
from utils.randaugment import RandAugment
import random


class LabeledTRansform:
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, row):
        images = row["img"]
        row["x"] = [self.transforms(image) for image in images]
        return row


class UDATransform:
    def __init__(self, weak, strong) -> None:
        self.weak = weak
        self.strong = strong

    def __call__(self, row):
        images = row["img"]
        row["x_w"] = [self.weak(image) for image in images]
        row["x_s"] = [self.strong(image) for image in images]
        return row


class InfiniteSampler:
    def __init__(self, dataset, shuffle=False):
        assert len(dataset) > 0
        self.dataset_len = len(dataset)
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range((self.dataset_len)))
        idx = 0
        while True:
            yield order[idx]
            idx += 1
            if idx == len(order):
                if self.shuffle:
                    random.shuffle(order)
                idx = 0


def build_dataset(datatype="cifar10"):
    dataset = load_dataset(datatype)
    ds = dataset["train"].train_test_split(0.1)
    train_ds = ds["train"]
    val_ds = ds["test"]
    test_ds = dataset["test"]
    uda_ds = copy.deepcopy(train_ds)

    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2471, 0.2435, 0.2616],
            ),
        ]
    )
    transform_train = LabeledTRansform(transforms=transform_labeled)

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2471, 0.2435, 0.2616],
            ),
        ]
    )
    transform_val = LabeledTRansform(transforms=transform_labeled)

    weak = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2471, 0.2435, 0.2616],
            ),
        ]
    )
    strong = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            RandAugment(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2471, 0.2435, 0.2616],
            ),
        ]
    )
    transform_uda = UDATransform(weak=weak, strong=strong)

    train_ds.set_transform(transform_train)
    uda_ds.set_transform(transform_uda)
    val_ds.set_transform(transform_val)
    test_ds.set_transform(transform_val)
    return train_ds, val_ds, test_ds, uda_ds
