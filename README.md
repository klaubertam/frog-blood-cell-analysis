# Frog Blood-Cell Image Analysis

## Overview

This research investigates automatic segmentation
and normal/anomaly classification of frog blood-cell
microscope photographs.

The project combines deep learning, image processing
and conventional machine-learning methods.

## Dataset

- 241 original microscope photographs
- Image-level labels: 0 = normal, 1 = anomaly
- Manual annotations prepared using Fiji/ImageJ

Original research photographs and labels are not
included in this public repository by default.

## Research pipeline

Microscope photographs
        |
        v
U-Net segmentation
        |
        v
Nucleus-guided watershed
        |
        v
Cell measurements and feature extraction
        |
        v
Photograph-level classification
        |
        v
Normal (0) or Anomaly (1)

## Experiments

### Experiment A
Initial whole-cell U-Net segmentation using 30 masks.

Reported selected-validation Dice: 0.9282.

### Experiment B
Whole-cell U-Net segmentation using 60 masks.

Mean test Dice: 0.8720.

### Experiment C
Multiclass U-Net segmentation identifying cytoplasm,
nuclei and pink regions.

Test foreground Dice: 0.9043 on five photographs.

The segmentation outputs were used for watershed
and photograph-level classification.

## Classification results

| Model | Accuracy | Balanced accuracy | ROC-AUC |
| --- | ---: | ---: | ---: |
| MIL | 47.3% | 48.0% | 0.497 |
| Logistic Regression | 63.5% | 67.7% | 0.724 |
| Random Forest | 70.6% | 68.7% | 0.792 |

MIL was evaluated on 205 usable photographs.
The two other rows use keep-pink extraction and
five-fold evaluation on 211 photographs.

These are exploratory results.

## Repository structure

- experiments/experiment_a/ - Initial U-Net experiment
- experiments/experiment_b/ - 60-mask segmentation
- experiments/experiment_c/segmentation/ - Multiclass U-Net
- experiments/experiment_c/watershed/ - Cell extraction
- experiments/experiment_c/classification/ - Classifiers
- docs/ - Research report and documentation

## Limitations

Reference masks and spreadsheet labels require
further biological quality verification.

The five-image segmentation test is small.

Classification performance does not establish
pollutant exposure or the cause of abnormalities.

## Reproducibility

See docs/REPRODUCIBILITY.md for the requirements,
data limitations and information needed to rerun
the experiments.

## Publication note

The experiments were conducted before publication.
Git commits document the staged publication of
completed research work.
