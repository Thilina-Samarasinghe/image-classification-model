"""Central configuration for training, evaluation and inference."""
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Paths
    data_dir: Path = ROOT / "data"
    checkpoint_dir: Path = ROOT / "checkpoints"
    output_dir: Path = ROOT / "outputs"

    # Data
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    # Model: "resnet18" | "mobilenet_v3_small" | "efficientnet_b0"
    arch: str = "resnet18"

    # Stage 1: train classifier head only
    head_epochs: int = 5
    head_lr: float = 1e-3

    # Stage 2: fine-tune last block(s) + head
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4

    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    seed: int = 42

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


# ImageNet normalisation (pretrained weights expect these)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
