# Separating Interval Duration from Murmur Acoustics in Deep Learning-Based Phonocardiogram Analysis

This repository contains the implementation code associated with the study:

**“Separating Interval Duration from Murmur Acoustics in Deep Learning-Based Phonocardiogram Analysis.”**

The project investigates whether strong phonocardiogram (PCG) classification performance reflects murmur-related acoustic information or unintended shortcut features such as cardiac-interval duration and zero-padding structure.

The repository therefore includes both the original systolic-versus-diastolic formulation and the revised shortcut-controlled murmur-bearing (MB) versus non-murmur-bearing (NMB) analysis.

## Dataset

This project uses the **CirCor DigiScope Phonocardiogram Dataset v1.0.3**, available from PhysioNet:

https://physionet.org/content/circor-heart-sound/1.0.3/

The dataset is not redistributed in this repository. Users should download it directly from PhysioNet and update the dataset path in the relevant scripts or configuration files.

The PhysioNet/CinC 2016 heart sound database is additionally used for cross-dataset evaluation of the cardiac-phase segmentation model.

## Project Structure

### Baseline Murmur Detection

- `src/phase1_data_loading.py`  
  Loads the CirCor dataset, preprocesses PCG recordings, slices 5-second windows, and creates patient-level train/validation/test splits.

- `src/train_phase1_fixed_ensemble.py`  
  Trains the Phase 1 murmur-detection model using a fixed patient-level split and a 5-model ensemble.

- `src/phase1_cross_validation.py`  
  Performs 5-fold stratified patient-level cross-validation for the Phase 1 murmur-detection model.

### Original Systolic-versus-Diastolic Formulation

- `src/train_phase2_timing.py`  
  Trains the original systolic-versus-diastolic classifier using variable-duration cardiac-phase intervals padded or truncated to 1 second.

- `src/phase2_mobilenet1d_baseline.py`  
  Trains the MobileNet1D baseline used for comparison with the original Phase 2 classifier.

- `src/combined_pipeline_inference.py`  
  Runs the original combined inference pipeline.

- `src/combined_pipeline_ablation.py`  
  Performs ablation analysis for the original combined pipeline.

### Shortcut Diagnosis

This analysis investigates whether the original systolic-versus-diastolic classifier relies on interval duration or zero-padding structure rather than murmur acoustics.

The shortcut analysis includes:

- duration-only systolic-versus-diastolic classification,
- an all-zero input sanity check,
- Grad-CAM analysis of the original zero-padded classifier,
- analysis of activation around the true-signal endpoint and padding onset.

### Duration-Blind MB-versus-NMB Classification

The revised formulation distinguishes **murmur-bearing (MB)** from **non-murmur-bearing (NMB)** cardiac intervals.

To remove direct access to interval duration and padding position, the acoustic model uses fixed **100-ms real-signal crops** with:

- no zero padding,
- no temporal resampling,
- no signal tiling.

The revised analysis includes:

- fixed-crop MB-versus-NMB acoustic classification,
- duration-only baseline evaluation,
- acoustic-duration probability fusion,
- patient-level paired analysis,
- murmur-grade analysis,
- auscultation-location analysis,
- most-audible-location analysis.

### Grad-CAM Analysis

Grad-CAM is used to compare model attention before and after removal of the identified shortcut.

The final analysis includes:

- patient-first averaging of Grad-CAM profiles,
- patient-level bootstrap 95% confidence intervals,
- comparison of the original padded model with the fixed-crop model,
- start-edge, interior, and end-edge activation-density analysis,
- MB-versus-NMB Grad-CAM/envelope alignment analysis.

For the fixed-crop Grad-CAM analysis, segments with negligible total Grad-CAM activation (`<= 1e-8`) are excluded from the interpretability subset.

### Cardiac-Phase Segmentation

The repository also includes the EmissionCNN-HSMM cardiac-phase segmentation framework used to estimate S1, systole, S2, and diastole from raw PCG recordings.

The segmentation model is evaluated:

- in-domain on CirCor,
- cross-dataset on PhysioNet/CinC 2016.

## Method Summary

The study consists of four main stages:

1. **Baseline murmur detection**  
   Patient-level murmur presence is predicted from 5-second PCG windows.

2. **Original systolic-versus-diastolic classification**  
   Cardiac-phase intervals are classified as systolic or diastolic using variable-duration signals padded to 1 second.

3. **Shortcut diagnosis and reformulation**  
   Duration-only classification, all-zero input testing, and Grad-CAM analysis show that the original task contains strong duration- and padding-related shortcut information.

4. **Duration-blind acoustic analysis**  
   The task is reformulated as MB-versus-NMB classification using fixed 100-ms PCG crops. Acoustic predictions are then combined with interval-duration information to evaluate whether the two sources provide complementary predictive value.

## Reproducibility

Patient-level split files should be used to reproduce the fixed-split and cross-validation experiments reported in the manuscript.

Where applicable, statistical analyses are performed at the patient level to avoid pseudoreplication from multiple cardiac intervals originating from the same patient.

Grad-CAM aggregate profiles are averaged within each patient before averaging across patients, and uncertainty is summarized using patient-level bootstrap confidence intervals.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
