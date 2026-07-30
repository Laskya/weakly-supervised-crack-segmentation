# Weakly-supervised crack segmentation

DL project 2 (pro version). Segmenting cracks pixel by pixel while training on one bit per image:
does this image contain a crack or not. No mask is ever used for training, only for the evaluation
at the end.

Jakub Laskowski (160287), Jakub Górniak (160326)

Competition: [DL_2025-Project2-Pro](https://www.kaggle.com/competitions/dl-2025-project-2-pro)
Dataset: [Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset), only the `train/` directory.

## How it works

Three stages:

1. A fully convolutional ResNet-34 classifier with top-k MIL pooling. It is trained on the image
   level bit, but the map before the pooling is already a class activation map.
2. That map is turned into a pseudo-mask: multi-scale and flip TTA, a guided filter to snap it to
   the edges of the image, then two thresholds with an ignore band in between.
3. A U-Net is trained on the pseudo-masks and then relabels the data itself for a second round.

Nothing in the pipeline knows what a crack looks like, so pointing `DATA_ROOT` somewhere else is
enough to run it on a different problem of the same kind.

## Results

The last part of the notebook opens the real masks on a held out split and compares four variants
(plain CAM, plus TTA, plus the guided filter, plus the U-Net) on IoU, Dice and a boundary F1 that is
less brutal on structures a few pixels wide. The tables and the plots are in the notebook.

## How to run

The notebook is written for Kaggle. Add the crack segmentation dataset and the competition data as
inputs, turn the GPU on and run everything from top to bottom.

Locally:

```bash
pip install -r requirements.txt
DATA_ROOT=path/to/crack_segmentation_dataset WORK_DIR=./work jupyter notebook
```

The last section runs the model on the competition test images and writes `submission.csv` with the
masks in RLE.

## Files

```
crack_segmentation_weakly_supervised.ipynb   the whole thing
requirements.txt
pyproject.toml                               ruff config
```
