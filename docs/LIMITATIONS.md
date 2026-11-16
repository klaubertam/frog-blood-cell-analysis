# Research Limitations and Future Work

## Segmentation

The Experiment C segmentation model achieved
promising overlap with prepared reference masks.

However, the final segmentation test contained
only five photographs.

Some reference nuclei were proposed automatically
during mask preparation. Expert verification
remains necessary.

## Watershed

Overlapping cells can produce merged or incomplete
boundaries.

Some very small regions may be segmentation
fragments rather than complete erythrocytes.

These errors can affect morphological measurements.

## Classification

The labels are assigned to photographs, not
individual cells.

Some Excel labels may need biological verification.

The dataset contains more anomalous than normal
photographs.

Repeated model development on the available data
limits the strength of the evaluation.

## Future work

- Review reference masks with a biology specialist
- Verify image-level labels independently
- Improve cell extraction quality
- Investigate additional morphological features
- Evaluate using independent animal-level groups
- Test on an untouched external dataset

## Scientific interpretation

The current system predicts labels from the
supplied workbook.

It does not establish pollutant exposure or
identify the biological cause of abnormalities.
