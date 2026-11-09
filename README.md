# Image Classification Model (PyTorch)

Transfer-learning image classifier built on pretrained torchvision backbones
(ResNet18 by default; MobileNetV3-Small and EfficientNet-B0 are also supported).
It runs on CPU and uses a GPU automatically when CUDA is available.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare data

Arrange images one folder per class:

```
data/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

If you only have `raw/<class_name>/*.jpg`, split it automatically:

```bash
python -m src.split_data --src raw --dst data --val 0.15 --test 0.15
```

## 2. Train

```bash
python -m src.train
# faster on CPU:
python -m src.train --arch mobilenet_v3_small --img-size 160 --head-epochs 3 --finetune-epochs 3
```

Training happens in two stages:
1. **Head:** the backbone is frozen and only the new classifier head is trained.
2. **Fine-tune:** the last backbone block is unfrozen and trained at a lower learning rate.

The best model by validation accuracy is saved to `checkpoints/best.pt`, and the
per-epoch metrics go to `outputs/history.csv`.

## 3. Evaluate

```bash
python -m src.evaluate            # test split
python -m src.evaluate --split val
```

This prints accuracy and per-class precision/recall/F1, and saves
`outputs/confusion_matrix_<split>.png` and `outputs/training_curves.png`.

## 4. Predict

```bash
python -m src.predict path/to/image.jpg --topk 3
```

## Project layout

| File | Purpose |
|---|---|
| `src/config.py` | Default hyperparameters and paths |
| `src/dataset.py` | Transforms / augmentation and DataLoaders |
| `src/model.py` | Backbone construction, freeze/unfreeze helpers |
| `src/utils.py` | Seeding, checkpoint IO, train/eval loops |
| `src/train.py` | Two-stage training script |
| `src/evaluate.py` | Test metrics, confusion matrix, curves |
| `src/predict.py` | Single-image inference |
| `src/split_data.py` | Split raw class folders into train/val/test |
