# Experiment C — Normal/Anomaly Classification

## Objective

After segmenting frog blood cells and extracting
their measurements, I investigated whether the
photographs could be classified as:

- 0: Normal
- 1: Anomaly

The labels come from the supplied Excel workbook.
They describe photographs, not individual cells.

## Models

### Multiple-Instance Learning (MIL)

MIL processes collections of cropped cells from a
photograph and combines their information into one
photograph-level prediction.

The initial five-image test achieved 60.0% accuracy.

In the larger evaluation, MIL achieved 47.3%
accuracy on 205 usable photographs.

### Logistic Regression

Logistic Regression uses numerical measurements
of cell morphology to predict the photograph label.

The keep-pink five-fold experiment achieved:

- Accuracy: 63.5%
- Balanced accuracy: 67.7%
- ROC-AUC: 0.724

### Random Forest

Random Forest uses multiple decision trees to
classify photographs using the extracted features.

The keep-pink five-fold experiment achieved:

- Accuracy: 70.6%
- Balanced accuracy: 68.7%
- ROC-AUC: 0.792

## Classification comparison

| Model | Images | Accuracy | Balanced accuracy | AUC |
| --- | ---: | ---: | ---: | ---: |
| MIL, larger evaluation | 205 | 47.3% | 48.0% | 0.497 |
| Logistic Regression, keep-pink | 211 | 63.5% | 67.7% | 0.724 |
| Random Forest, keep-pink | 211 | 70.6% | 68.7% | 0.792 |

The Logistic Regression and Random Forest results
come from five-fold evaluations on the 211
photographs outside the original annotated set.

## Random Forest confusion matrix

| Actual / Predicted | Normal | Anomaly |
| --- | ---: | ---: |
| Normal | 43 | 25 |
| Anomaly | 37 | 106 |

Random Forest correctly classified 149 of
211 photographs.

## Important limitations

These results are exploratory.

Some Excel labels may need biological verification.
Segmentation and watershed errors may affect the
measurements used by the classifiers.

The dataset has more anomalous photographs than
normal photographs.

An always-anomaly classifier would achieve
67.8% accuracy on the same 211 photographs.

These predictions do not establish the cause
of a blood-cell abnormality or prove pollution exposure.

## Files

- mil/: Multiple-Instance Learning scripts and results
- logistic_regression/: Logistic Regression scripts and results
- random_forest/: Random Forest script and results

## Next stage

Day 7 will add the final research report,
a guide to the repository and the study limitations.
