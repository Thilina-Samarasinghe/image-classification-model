"""Shared helpers: seeding, checkpoint IO, epoch loops."""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: nn.Module, classes, arch: str, img_size: int,
                    epoch: int, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "classes": classes, "arch": arch,
        "img_size": img_size, "epoch": epoch, "val_acc": val_acc,
    }, path)


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over loader. Trains if optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        preds.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()
