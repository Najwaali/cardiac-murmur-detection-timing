# Cardiac Murmur Detection and Timing Classification Using Deep Learning

This repository contains the implementation code for a two-stage deep learning pipeline for cardiac murmur detection and timing classification from phonocardiogram (PCG) signals.

## Dataset

This project uses the CirCor DigiScope Phonocardiogram Dataset v1.0.3 available from PhysioNet:

https://physionet.org/content/circor-heart-sound/1.0.3/

The dataset is not redistributed in this repository. Users should download it directly from PhysioNet and update the dataset path in the configuration files.

## Project Structure

- `train_phase1.py`: trains the murmur detection model.
- `train_phase2.py`: trains the systolic/diastolic timing classifier.
- `inference_pipeline.py`: runs the full two-stage inference pipeline.
- `preprocessing.py`: filtering, normalization, windowing, and segment extraction.
- `evaluation.py`: accuracy, F1-score, confusion matrix, threshold sweep, and cross-validation evaluation.
- `splits/`: patient-level fixed split and cross-validation fold files.
- `results/`: reported numerical results and ablation tables.
- `figures/`: figures used in the report.

## Method Summary

Phase 1 performs patient-level murmur detection using 5-second raw PCG windows.  
Phase 2 classifies cardiac phase segments as systolic or diastolic using a 1-D CNN architecture with self-attention.

## Reproducibility

The patient-level split files are provided to reproduce the fixed-split and cross-validation results reported in the paper.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
