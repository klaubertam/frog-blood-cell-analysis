# Reproducibility Guide

## Software

The experiments use Python and packages such as:

- PyTorch
- NumPy
- SciPy
- scikit-learn
- pandas
- Pillow
- OpenCV

See the requirements.txt supplied with
Experiment C segmentation.

The scripts may require additional dependencies
and paths matching the original research environment.

## Data requirements

The complete experiments require:

1. Original frog blood-cell photographs
2. Corresponding annotated segmentation masks
3. Image-level labels from the Excel workbook

These data are not published by default.

## Experiments

Experiment A:
experiments/experiment_a/

Experiment B:
experiments/experiment_b/

Experiment C segmentation:
experiments/experiment_c/segmentation/

Watershed:
experiments/experiment_c/watershed/

Classification:
experiments/experiment_c/classification/

## Evaluation

Experiment A reports selected-validation results.

Experiment B reports results across five outer folds.

Experiment C segmentation uses 21 training,
4 validation and 5 held-out test photographs.

Later Logistic Regression and Random Forest
evaluations use five folds across 211 photographs
outside the original 30-image annotated set.

## Important notes

This repository documents the original code
and recorded experiment outputs.

Trained model checkpoint files are not included.

The code may require updates to input paths,
dependencies and configuration before execution.

Full reproduction requires permission to access
the original research dataset.

The current results should be considered
exploratory rather than independently validated.
