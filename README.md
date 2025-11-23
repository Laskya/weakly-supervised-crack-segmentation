# Weakly-supervised crack segmentation

DL project 2 (pro version). Detecting cracks pixel by pixel while training on one bit per image,
crack or no crack. The masks are only used at the end to score the result.

Jakub Laskowski (160287), Jakub Górniak (160326)

Dataset: [Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset), only the `train/` directory.

## Pipeline

MIL classifier (ResNet-34 with top-k pooling) -> CAM with TTA and a guided filter -> pseudo-masks
with an ignore band -> U-Net trained on those, plus one round of self-training.

## How to run

The notebook is made for Kaggle, so add the dataset as an input, turn the GPU on and run it top to
bottom. Locally:

```bash
pip install -r requirements.txt
DATA_ROOT=path/to/crack_segmentation_dataset WORK_DIR=./work jupyter notebook
```
