# Experiment A — Initial U-Net Segmentation

## Objective

The aim of Experiment A was to train a U-Net model
to identify frog blood cells in microscope photographs.

## Dataset

I used 30 manually annotated blood-cell photographs.

I prepared whole-cell masks using Fiji/ImageJ.
These masks showed the model where the cells were located.

## Method

I used a U-Net neural network for image segmentation.

The model learned to distinguish cell regions from
the image background.

I evaluated the segmentation using Dice and IoU,
which measure how closely predicted cell regions
overlap with the reference masks.

## Results

Mean reported Dice: 0.9282
Mean reported IoU: 0.8686

These are selected-validation results from five folds,
not independent held-out test results.

## Conclusion

Experiment A established the first whole-cell
segmentation approach.

It provided the foundation for later experiments
with additional masks and more detailed annotations.

## Files

- train_unet1.py: U-Net training script.
- segmentation_cv.csv: Recorded segmentation results.
