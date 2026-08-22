# Separating Interval Duration from Murmur Acoustics in Deep Learning-Based Phonocardiogram Analysis

This repository contains the implementation code associated with the study:

**“Separating Interval Duration from Murmur Acoustics in Deep Learning-Based Phonocardiogram Analysis.”**

The project investigates whether strong phonocardiogram (PCG) classification performance reflects murmur-related acoustic information or unintended shortcut features such as cardiac-interval duration and zero-padding structure.

The repository therefore includes both the original systolic-versus-diastolic formulation and the revised shortcut-controlled murmur-bearing (MB) versus non-murmur-bearing (NMB) analysis.

## Dataset

This project uses the **CirCor DigiScope Phonocardiogram Dataset v1.0.3**, available from PhysioNet:

https://physionet.org/content/circor-heart-sound/1.0.3/

The dataset is not redistributed in this repository. Users should download it directly from PhysioNet and update the dataset path in the relevant scripts.

## Project Structure

### Phase 1: Murmur Detection

- `phase1_data_loading.py`  
  Loads and preprocesses CirCor PCG recordings and constructs patient-level data splits.

- `train_phase1_fixed_ensemble.py`  
  Trains the fixed-split Phase 1 murmur-detection ensemble.

- `phase1_cross_validation.py`  
  Performs patient-level 5-fold cross-validation for the murmur-detection model.

### Original Phase 2: Systolic-versus-Diastolic Classification

- `phase2_timing.py`  
  Implements the original systolic-versus-diastolic classification experiment using variable-duration cardiac intervals padded or truncated to 1 second.

- `phase2_mobilenet1d_baseline.py`  
  Implements the MobileNet1D baseline used for comparison with the original Phase 2 classifier.

### Shortcut-Controlled MB-versus-NMB Analysis

- `strategy_d_fixed_crop.py`  
  Implements the fixed 100-ms murmur-bearing versus non-murmur-bearing acoustic classifier using real-signal crops with no zero padding, temporal resampling, or signal tiling. The duration-only baseline is evaluated on the same cohort.

- `acoustic_duration_fusion.py`  
  Combines acoustic and interval-duration probabilities using logistic-regression fusion with patient-level out-of-fold training.

- `per_patient_analysis.py`  
  Performs paired within-patient comparisons of mean P(MB) for MB and NMB intervals for both the acoustic-only and fusion models.

- `robustness_analysis.py`  
  Evaluates acoustic-model performance across murmur grade, auscultation location, and the annotated most-audible location.

- `roc_and_probability_outputs.py`  
  Generates ROC curves for the duration-only, acoustic-only, and fusion models and saves held-out probabilities and summary metrics.

### Grad-CAM Analysis

- `gradcam_strategy_d.py`  
  Evaluates Grad-CAM activation density across the start edge, interior, and end edge of the fixed 100-ms Strategy D crops.

- `gradcam_mb_vs_nmb.py`  
  Compares Grad-CAM behavior between MB and NMB inputs using the MB-class logit for both groups. Analyses include activation-envelope correlation and high-versus-low energy activation contrast.

- `gradcam_original_padded.py`  
  Reconstructs the original 1-second zero-padded systolic-versus-diastolic model, rebuilds the eligible systolic murmur-bearing cohort, and computes old-model Grad-CAM maps including padding-only activation.

- `gradcam_before_after.py`  
  Generates the final before/after Grad-CAM comparison figure between the original padded classifier and the fixed-crop Strategy D classifier.

## Method Summary

The study consists of four main stages:

1. **Baseline murmur detection**  
   Patient-level murmur presence is predicted from PCG recordings using a fixed-split ensemble and patient-level cross-validation.

2. **Original systolic-versus-diastolic classification**  
   Cardiac-phase intervals are classified as systolic or diastolic using variable-duration signals padded or truncated to 1 second.

3. **Shortcut diagnosis and task reformulation**  
   Duration-only classification, zero-input testing, and Grad-CAM analysis are used to determine whether the original formulation contains strong duration- and padding-related shortcut information.

4. **Duration-controlled acoustic analysis**  
   The task is reformulated as murmur-bearing versus non-murmur-bearing classification using fixed 100-ms real-signal crops. Acoustic predictions are then combined with interval-duration information to evaluate whether the two sources provide complementary predictive information.

## Fixed-Crop Strategy

The shortcut-controlled acoustic model uses fixed **100-ms real-signal crops**.

The fixed-crop representation intentionally removes direct access to interval length and padding position by using:

- no zero padding,
- no temporal resampling,
- no repeated-signal tiling.

Training crops are sampled from the available cardiac interval, while deterministic centered crops are used for validation and test evaluation.

## Acoustic-Duration Fusion

The acoustic and duration models are combined using logistic-regression fusion.

Fusion training uses patient-level out-of-fold predictions so that the meta-classifier is not trained on in-sample predictions from the acoustic model.

Statistical uncertainty and pairwise model comparisons are evaluated using patient-level bootstrap resampling.

## Per-Patient Analysis

Within-patient analysis compares the mean predicted P(MB) for true MB and NMB intervals from the same patient.

Patients are included in the paired analysis when they contain at least:

- 3 MB intervals, and
- 3 NMB intervals.

Paired differences are evaluated using the Wilcoxon signed-rank test.

## Robustness Analysis

The fixed-crop acoustic classifier is additionally evaluated across:

- auscultation locations: AV, MV, PV, and TV,
- murmur grade,
- the annotated most-audible recording location.

These analyses use the held-out test cohort reconstructed with the corresponding metadata.

## Grad-CAM Analysis

Grad-CAM is used as a supportive interpretability analysis rather than as evidence of physiological localization.

For Strategy D, the 100-ms crop is divided into:

- start edge: 0–25 ms,
- interior: 25–75 ms,
- end edge: 75–100 ms.

The crop boundaries are generic input boundaries and should not be interpreted as S1 or S2 boundaries.

Grad-CAM profiles are aggregated using patient-first averaging. Bootstrap confidence intervals are generated by resampling patients.

For the original padded classifier, Grad-CAM profiles are aligned relative to the true-signal endpoint so that activation around the signal-end / padding-onset transition can be visualized.

The original padded-model cohort and the Strategy D test cohort are not fully patient matched, so the before/after Grad-CAM comparison is interpreted descriptively.

## Reproducibility

The scripts use patient-level splitting to prevent intervals or recordings from the same patient from appearing across training and test partitions.

Where multiple intervals originate from one patient, statistical analyses are performed at the patient level whenever appropriate to reduce pseudoreplication.

The revised analysis scripts depend on outputs or objects generated by earlier stages. A typical execution order is:

```text
strategy_d_fixed_crop.py
        |
        +--> acoustic_duration_fusion.py
        |       |
        |       +--> per_patient_analysis.py
        |       +--> roc_and_probability_outputs.py
        |
        +--> robustness_analysis.py
        |
        +--> gradcam_strategy_d.py
        |       |
        |       +--> gradcam_mb_vs_nmb.py
        |
        +--> old-model setup / checkpoint loading
                |
                +--> gradcam_original_padded.py
                        |
                        +--> gradcam_before_after.py

The Grad-CAM scripts are intended to be executed sequentially in the same Python session. In particular:

- `strategy_d_fixed_crop.py` creates the reconstructed Strategy D dataset and fixed-crop model objects.
- `gradcam_strategy_d.py` creates the patient-level Strategy D edge/interior Grad-CAM statistics.
- `gradcam_mb_vs_nmb.py` creates the MB-versus-NMB Grad-CAM results used in the final comparison.
- The original Phase 2 checkpoint is then loaded and the old padded-model Grad-CAM object is created.
- `gradcam_original_padded.py` reconstructs the eligible original padded-model cohort and creates `old_all_df`. 
- `gradcam_before_after.py` uses the original and fixed-crop Grad-CAM results to generate the final before/after figure.

Because these analysis scripts share in-memory objects, they should be run in the order shown above within the same analysis session.
