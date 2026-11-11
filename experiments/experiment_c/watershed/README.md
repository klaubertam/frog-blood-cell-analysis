# Experiment C — Watershed and Feature Extraction

## Objective

After training the multiclass U-Net, I needed to
separate individual blood cells and measure their
morphological characteristics.

## How it works

1. U-Net predicts cytoplasm, nucleus and pink regions.
2. Predicted nuclei are used as watershed seeds.
3. Watershed proposes individual cell boundaries.
4. Quality rules reject some unsuitable candidates.
5. The retained cell regions are measured.

## Morphological measurements

The extraction records:

- Cell and nucleus areas
- Cell and nucleus major/minor axes
- Cell elongation (axis ratio)
- Cell solidity and eccentricity
- Nucleus-to-cell area measurements

These values describe the predicted cell regions.
They are not manually verified biological measurements.

Areas and lengths are measured on the resized model
grid and are not calibrated to micrometres.

## Pink-region comparison

I tested two extraction settings:

- Exclude: remove cells adjacent to pink regions.
- Keep: retain cells adjacent to pink regions.

The exclude setting left six of the 211
outside-original-30 photographs without retained cells.

The keep setting produced usable cell bags for all
211 outside-original-30 photographs.

The files in the results folder are from the
keep-pink setting.

## Results files

- cell_features.csv: Measurements for retained cells.
- instance_quality.csv: Accepted and rejected candidates.
- bag_quality.csv: Cell counts and extraction status
  for each photograph.

## Limitations

Some objects may be staining artifacts, fragments,
or incorrectly separated overlapping cells.

These potential errors can affect the measurements
used by downstream classifiers.

Expert annotation review remains necessary.

## Next stage

Day 6 will introduce Multiple-Instance Learning,
Logistic Regression and Random Forest for
photograph-level normal/anomaly classification.
