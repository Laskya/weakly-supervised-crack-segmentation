# Weakly-supervised crack segmentation

DL project 2 (pro version). Segmenting cracks pixel by pixel while training on one bit per image:
does this image contain a crack or not. No mask is ever used for training, only for the evaluation
at the end.

Jakub Laskowski (160287), Jakub Górniak (160326)

Competition: [DL_2025-Project2-Pro](https://www.kaggle.com/competitions/dl-2025-project-2-pro)
Dataset: [Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset), only the `train/` directory.

## The rule we are working under

| Stage | What it is allowed to see |
|---|---|
| Label creation | Each training mask is opened once and becomes one bit, is there any crack pixel. The pixels are then thrown away and the bit is cached in a CSV. |
| Training, all three stages | Images and that bit. |
| Every threshold and every model choice | Images, the bit, and the pseudo-masks we generated ourselves. |
| Evaluation | The real masks, on a held out split, once, after everything is frozen. |

`load_gt_mask` counts every mask it opens and the results section asserts that the counter for the
training split is still zero. Everything diagnostic, the failure ranking and the threshold sweep,
runs on validation, so that the test split really is scored only once.

## How it works

Three stages:

1. A fully convolutional ResNet-34 classifier with top-k MIL pooling, trained on the image level bit.
   The map before the pooling is already a class activation map. A Puzzle-CAM style tile consistency
   term and a small area prior stop it from lighting up one blob and calling it a day.
2. That map is turned into a pseudo-mask: multi-scale and flip TTA, a guided filter to snap it to the
   edges of the image, then two thresholds with an ignore band in between. The foreground threshold
   is a per image quantile that keeps about 2% of the pixels, the same object size prior the MIL
   pooling uses.
3. A U-Net is trained on the confident parts of the pseudo-masks, with partial BCE, an asymmetric
   Tversky term for the width and a clDice term for the topology. It then relabels the data itself
   for a second round.

Nothing in the pipeline knows what a crack looks like, so pointing `DATA_ROOT` somewhere else is
enough to run it on a different problem of the same kind.

## Data

9603 images in `train/`, 8370 with a crack and 1233 without, split 75 / 10 / 15 into train, val and
test, stratified on the label and on the source sub-dataset. The dataset is a merge of eleven sources
(CFD, DeepCrack, Rissbilder, GAPs, CrackTree, Volker, Eugen Muller, Sylvie Chambon, forest, a plain
`crack` folder and a crack free one) and they look very different, so a plain random split gives a
validation set that is too easy.

Worth knowing before reading the image level numbers: nine of the eleven sources contain only cracked
images and about 96% of the crack free ones come from a single source. "Which folder is this"
therefore answers "is there a crack" almost perfectly, and the notebook says so in as many words.

## Results

From the first full run on Kaggle, 3.6 h on a T4. The notebook regenerates this table every run.

Image level, on the test split: accuracy 0.997, F1 0.998, AUROC 0.999, with the caveat above.

Pixel level, one row per idea, each row adding to the one above it:

| Variant | IoU | Dice | Boundary F1 |
|---|---|---|---|
| CAM, single scale | 0.132 | 0.234 | 0.259 |
| + multi-scale and flip TTA | 0.178 | 0.302 | 0.277 |
| + guided filter | 0.171 | 0.292 | 0.301 |
| + U-Net distillation, 2 rounds | 0.246 | 0.395 | 0.205 |

A fully supervised U-Net on this dataset gets somewhere around 0.65-0.75 IoU, so this is about a
third of the way there for one click per image instead of a traced mask.

That run is also what the current version is a reaction to. It showed that the classifier had peaked
by epoch 3 and kept training for five more, that the second self-training round was being kept even
when it was worse, and that the U-Net was gaining Dice while losing boundary F1, drawing better
regions and worse outlines. Early stopping, model selection against a fixed reference, a calibrated
binarisation threshold and the clDice term are the answers to those three.

## How to run

The notebook is written for Kaggle. Add the crack segmentation dataset and the competition data as
inputs, turn the GPU on and run everything from top to bottom.

Locally:

```bash
pip install -r requirements.txt
DATA_ROOT=path/to/crack_segmentation_dataset WORK_DIR=./work jupyter notebook
```

`IMG_SIZE`, `BATCH_SIZE`, `CLS_EPOCHS`, `SEG_EPOCHS` and `SELF_TRAINING_ROUNDS` are environment
overrides, so the whole notebook can be run small as a smoke test.

The last section runs the model on the competition test images and writes `submission.csv` with the
masks in RLE.

## References

The papers behind each component are listed in the last cell of the notebook: Puzzle-CAM, CAM,
multiple instance learning, guided filtering, Tversky, clDice, U-Net, ResNet and atrous convolution.

## Files

```
crack_segmentation_weakly_supervised.ipynb   the whole thing
kernel-metadata.json                         kaggle kernel config
requirements.txt
pyproject.toml                               ruff config
```

## Licence

Our code is MIT. The dataset and the competition data are covered by their own terms on Kaggle.
