"""Split a flat class-folder dataset into train/val/test.

Input:   raw/<class_name>/*.jpg
Output:  data/train|val|test/<class_name>/*.jpg   (files are copied)

Usage:
    python -m src.split_data --src raw --dst data --val 0.15 --test 0.15
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", default="data")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    rng = random.Random(a.seed)
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {src}")

    for cdir in class_dirs:
        files = sorted(f for f in cdir.iterdir() if f.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        n_val, n_test = int(len(files) * a.val), int(len(files) * a.test)
        splits = {
            "test": files[:n_test],
            "val": files[n_test:n_test + n_val],
            "train": files[n_test + n_val:],
        }
        for split, items in splits.items():
            out = dst / split / cdir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
        print(f"{cdir.name:<25} train {len(splits['train']):>5} | "
              f"val {len(splits['val']):>5} | test {len(splits['test']):>5}")


if __name__ == "__main__":
    main()
