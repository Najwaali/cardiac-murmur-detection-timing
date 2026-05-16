# Cardiac Murmur Detection and Timing Classification Using Deep Learning

This repository contains the implementation code for a two-stage deep learning pipeline for cardiac murmur detection and timing classification from phonocardiogram (PCG) signals.

## Dataset

This project uses the CirCor DigiScope Phonocardiogram Dataset v1.0.3 available from PhysioNet:

https://physionet.org/content/circor-heart-sound/1.0.3/

The dataset is not redistributed in this repository. Users should download it directly from PhysioNet and update the dataset path in the configuration files.


## Project Structure

- `src/phase1_data_loading.py`: loads the CirCor dataset, preprocesses PCG recordings, slices 5-second windows, and creates patient-level train/validation/test splits.
- `src/train_phase1_fixed_ensemble.py`: trains the Phase 1 murmur detection model using a fixed patient-level split and a 5-model ensemble.
- `scr/phase1_cross_validation.py`:performs 5-fold patinet-level 
- `src/train_phase2_timing.py`: trains the Phase 2 systolic/diastolic timing classifier using cardiac phase segments.
- `src/phase2_mobilenet1d_baseline.py`: trains the MobileNet1D
- `src/combined_pipeline_inference.py`
- `src/combined_pipeline_ablation.py`
- `requirements.txt`: lists the Python packages required to run the code.
- 
- 
## Method Summary

Phase 1 performs patient-level murmur detection using 5-second raw PCG windows.  
Phase 2 classifies cardiac phase segments as systolic or diastolic using a 1-D CNN architecture with self-attention.

## Reproducibility

The patient-level split files are provided to reproduce the fixed-split and cross-validation results reported in the paper.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
## Running the Code

Phase 1 data loading:

```bash
python src/phase1_data_loading.py
```
```bash
python src/phase1_cross_validation.py
```

```bash
python src/train_phase1_fixed_ensemble.py
```


```bash
python src/train_phase2_timing.py
```

```bash
python src/phase2_mobilenet1d_baseline.py
```

```bash
python src/combined_pipeline_inference.py
```

```bash
python src/combined_pipeline_ablation.py
```
