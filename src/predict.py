"""Classify one or more images with the trained model.

Usage:
    python -m src.predict path/to/image.jpg [more.jpg ...] --topk 3
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import Config
from .dataset import eval_transforms
from .utils import load_checkpoint


@torch.no_grad()
def predict(model, image_path: Path, tf, classes, device, topk: int):
    img = Image.open(image_path).convert("RGB")
    probs = torch.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)[0]
    values, idxs = probs.topk(min(topk, len(classes)))
    return [(classes[i], v.item()) for v, i in zip(values, idxs)]


def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default=str(cfg.checkpoint_dir / "best.pt"))
    p.add_argument("--topk", type=int, default=3)
    a = p.parse_args()

    model, ckpt = load_checkpoint(Path(a.checkpoint), cfg.device)
    tf = eval_transforms(ckpt["img_size"])
    for path in a.images:
        print(f"\n{path}")
        for label, prob in predict(model, Path(path), tf, ckpt["classes"], cfg.device, a.topk):
            print(f"  {label:<25} {prob:.4f}")


if __name__ == "__main__":
    main()
