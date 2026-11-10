# Experiment C — Multiclass U-Net Segmentation

## Objective

The goal of Experiment C was to improve frog blood-cell
segmentation by identifying different cell structures.

Unlike Experiments A and B, which focused on whole-cell
segmentation, Experiment C distinguishes cytoplasm,
nucleus and pink-stained regions.

## Dataset

I used 30 annotated microscope photographs.

The data were divided into:

- 21 training photographs
- 4 validation photographs
- 5 test photographs

Training images were rotated by 0, 90, 180 and 270 degrees.
This produced 84 training views per epoch, not 84
independent biological samples.

## Segmentation classes

The masks use the following grayscale values:

| Value | Meaning |
| --- | --- |
| 0 | Background |
| 85 | Cytoplasm |
| 170 | Nucleus |
| 255 | Pink region |
| 200 | Ignored or uncertain pixels |

Some nucleus regions were proposed during mask preparation.
These reference masks require biological quality verification.

## Method

A four-class U-Net was trained to identify the different
regions in each microscope photograph.

The model was evaluated using Dice and IoU, which measure
overlap between predicted and reference masks.

## Results

The five held-out test photographs produced:

| Metric | Result |
| --- | --- |
| Macro foreground Dice | 0.9043 |
| Whole-cell Dice | 0.9476 |
| Whole-cell IoU | 0.9004 |
| Nucleus Dice | 0.9102 |
| Pink-region Dice | 0.8641 |

The results show encouraging agreement with the prepared
reference masks on a small test set.

These are segmentation metrics, not normal/anomaly
classification accuracy.

## Limitations

- Only five photographs were used for the final test.
- Reference nuclei require further expert verification.
- Segmentation overlap does not establish biological correctness.
- Results do not directly prove pollution exposure.

## Next stage

The predicted nuclei will be used as seeds for watershed,
which proposes boundaries between touching cells.
