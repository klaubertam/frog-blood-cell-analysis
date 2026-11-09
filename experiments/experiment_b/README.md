# Experiment B — U-Net Segmentation with 60 Masks

## Objective

In Experiment B, I extended the initial segmentation
experiment by using 60 annotated microscope images.

The goal was to evaluate whether the U-Net model could
identify frog blood-cell regions more reliably.

## Dataset

- 60 annotated frog blood-cell photographs
- Whole-cell segmentation masks
- Masks prepared using Fiji/ImageJ

## Method

I trained a U-Net model to separate blood-cell regions
from the image background.

I evaluated the model using five outer folds,
with validation used to select the best checkpoint
within each fold.

Dice and IoU measure the overlap between predicted
cell regions and reference masks.

## Results

Mean test Dice: 0.8720
Mean test IoU: 0.7866

These are test results from the five-fold evaluation.

## Interpretation

Experiment B provided a broader segmentation evaluation
using 60 annotated images.

Its scores were lower than the selected-validation
scores reported for Experiment A.

However, the evaluation procedures were different,
so this does not prove that adding more masks reduced
model performance.

## Files

- train_unet.py: Segmentation training script
- fold_metrics.csv: Results for each evaluation fold
- per_image_test_metrics.csv: Results for individual images
- summary.json: Overall experiment results

## Next Experiment

Experiment C introduces more detailed segmentation,
including cytoplasm, nucleus and pink regions.
