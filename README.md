# Frog Blood-Cell Analysis

## Research overview

This project investigates automated segmentation and
classification of frog blood-cell microscope images.

The dataset contains 241 original photographs.

## Methods

- U-Net for blood-cell segmentation
- Watershed for separating touching cells
- Multiple-Instance Learning (MIL)
- Logistic Regression
- Random Forest

## Research experiments

- Experiment A: Whole-cell segmentation using 30 masks
- Experiment B: Whole-cell segmentation using 60 masks
- Experiment C: Multiclass segmentation and image classification

## Current findings

Experiment C achieved a test foreground Dice score
of 0.9043 on five held-out segmentation images.

Random Forest achieved 70.6% classification accuracy
in the exploratory 211-image five-fold evaluation.

## Repository status

Completed research work is being documented and
published in stages.

## Data availability

Original research images and labels are not included
by default. Their public release requires permission.
