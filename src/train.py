"""Two-stage transfer-learning training.

Stage 1: freeze backbone, train classifier head.
Stage 2: unfreeze last backbone block, fine-tune with a lower LR.

Usage:
    python -m src.train
    python -m src.train --arch mobilenet_v3_small --head-epochs 3 --finetune-epochs 3
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .dataset import get_train_val_loaders
from .model import build_model, count_trainable, freeze_backbone, unfreeze_last_block
from .utils import run_epoch, save_checkpoint, set_seed


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Train an image classifier")
    p.add_argument("--data-dir", default=str(cfg.data_dir))
    p.add_argument("--arch", default=cfg.arch,
                   choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--img-size", type=int, default=cfg.img_size)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--head-epochs", type=int, default=cfg.head_epochs)
    p.add_argument("--finetune-epochs", type=int, default=cfg.finetune_epochs)
    p.add_argument("--head-lr", type=float, default=cfg.head_lr)
    p.add_argument("--finetune-lr", type=float, default=cfg.finetune_lr)
    p.add_argument("--patience", type=int, default=cfg.early_stop_patience)
    a = p.parse_args()

    cfg.data_dir = Path(a.data_dir)
    cfg.arch, cfg.img_size, cfg.batch_size = a.arch, a.img_size, a.batch_size
    cfg.num_workers = a.num_workers
    cfg.head_epochs, cfg.finetune_epochs = a.head_epochs, a.finetune_epochs
    cfg.head_lr, cfg.finetune_lr = a.head_lr, a.finetune_lr
    cfg.early_stop_patience = a.patience
    return cfg


def train_stage(name, model, train_loader, val_loader, criterion, optimizer, epochs,
                cfg, classes, state, history):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    stale = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, cfg.device,
                                    optimizer, desc=f"{name} {epoch} train")
        va_loss, va_acc = run_epoch(model, val_loader, criterion, cfg.device,
                                    desc=f"{name} {epoch} val")
        scheduler.step()
        state["global_epoch"] += 1
        history.append([state["global_epoch"], name, tr_loss, tr_acc, va_loss, va_acc])
        print(f"[{name}] epoch {epoch}/{epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}  ({time.time() - t0:.0f}s)")

        if va_acc > state["best_acc"]:
            state["best_acc"] = va_acc
            stale = 0
            save_checkpoint(cfg.checkpoint_dir / "best.pt", model, classes, cfg.arch,
                            cfg.img_size, state["global_epoch"], va_acc)
            print(f"  -> new best val acc {va_acc:.4f}, saved checkpoints/best.pt")
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"  early stopping {name} (no improvement for {stale} epochs)")
                break


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device} | arch: {cfg.arch} | data: {cfg.data_dir}")

    train_loader, val_loader, classes = get_train_val_loaders(cfg)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train images: {len(train_loader.dataset)} | Val images: {len(val_loader.dataset)}")

    model = build_model(cfg.arch, len(classes)).to(cfg.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    state = {"best_acc": 0.0, "global_epoch": 0}
    history = []

    # Stage 1: head only
    freeze_backbone(model, cfg.arch)
    print(f"\nStage 1 (head) trainable params: {count_trainable(model):,}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    train_stage("head", model, train_loader, val_loader, criterion, opt,
                cfg.head_epochs, cfg, classes, state, history)

    # Stage 2: fine-tune last block + head
    if cfg.finetune_epochs > 0:
        unfreeze_last_block(model, cfg.arch)
        print(f"\nStage 2 (finetune) trainable params: {count_trainable(model):,}")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
        train_stage("finetune", model, train_loader, val_loader, criterion, opt,
                    cfg.finetune_epochs, cfg, classes, state, history)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = cfg.output_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)
    print(f"\nDone. Best val acc: {state['best_acc']:.4f}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
