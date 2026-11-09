"""Data transforms and DataLoaders based on torchvision ImageFolder.

Expected layout:
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg
    data/test/<class_name>/*.jpg   (optional)
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import MEAN, STD


def train_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def make_loader(split_dir: Path, tf, batch_size: int, num_workers: int,
                shuffle: bool) -> tuple[DataLoader, list[str]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split folder not found: {split_dir}. "
            "Expected data/<split>/<class_name>/<images>."
        )
    ds = datasets.ImageFolder(split_dir, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=False, persistent_workers=num_workers > 0)
    return loader, ds.classes


def get_train_val_loaders(cfg):
    train_loader, classes = make_loader(
        cfg.data_dir / "train", train_transforms(cfg.img_size),
        cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader, val_classes = make_loader(
        cfg.data_dir / "val", eval_transforms(cfg.img_size),
        cfg.batch_size, cfg.num_workers, shuffle=False)
    if classes != val_classes:
        raise ValueError(f"Class mismatch between train {classes} and val {val_classes}")
    return train_loader, val_loader, classes
