"""Evaluate the best checkpoint on a data split and save plots.

Usage:
    python -m src.evaluate
    python -m src.evaluate --split val
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from .config import Config
from .dataset import eval_transforms, make_loader
from .utils import load_checkpoint, predict_loader


def plot_history(history_csv: Path, out: Path) -> None:
    if not history_csv.exists():
        return
    with open(history_csv) as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    a1.plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    a1.set_title("Loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(ep, [float(r["train_acc"]) for r in rows], label="train")
    a2.plot(ep, [float(r["val_acc"]) for r in rows], label="val")
    a2.set_title("Accuracy")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to {out}")


def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(cfg.data_dir))
    p.add_argument("--checkpoint", default=str(cfg.checkpoint_dir / "best.pt"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    a = p.parse_args()

    model, ckpt = load_checkpoint(Path(a.checkpoint), cfg.device)
    classes = ckpt["classes"]
    loader, split_classes = make_loader(Path(a.data_dir) / a.split,
                                        eval_transforms(ckpt["img_size"]),
                                        cfg.batch_size, a.num_workers, shuffle=False)
    if split_classes != classes:
        raise ValueError(f"Class mismatch: checkpoint {classes} vs {a.split} {split_classes}")

    y_true, y_pred = predict_loader(model, loader, cfg.device)
    labels = list(range(len(classes)))
    print(f"\n{a.split} accuracy: {(y_true == y_pred).mean():.4f}\n")
    print(classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                digits=4, zero_division=0))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(5, len(classes) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    cm_path = cfg.output_dir / f"confusion_matrix_{a.split}.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")
    plot_history(cfg.output_dir / "history.csv", cfg.output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
