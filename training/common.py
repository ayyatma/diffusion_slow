from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mobilevit_s.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_cifar10_loaders(config: dict, train_transform=None, eval_transform=None):
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    input_size = config["input_size"]
    data_root = resolve_project_path(dataset_cfg["root"])
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)
    val_size = dataset_cfg.get("val_size", 5000)
    seed = dataset_cfg.get("split_seed", 0)

    train_transform = train_transform or build_train_transform(input_size)
    eval_transform = eval_transform or build_eval_transform(input_size)

    full_train = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=dataset_cfg.get("download", True),
        transform=train_transform,
    )
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, _ = random_split(full_train, [train_size, val_size], generator=generator)

    full_val = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=dataset_cfg.get("download", True),
        transform=eval_transform,
    )
    _, val_dataset = random_split(full_val, [train_size, val_size], generator=generator)

    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=dataset_cfg.get("download", True),
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def limited_batches(loader: Iterable, max_batches: int | None):
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser
