


import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

# =========================================================
# 1. MOUNT DRIVE + CONFIG
# =========================================================
BASE_PATH = "path/to/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
DATA_PATH = os.path.join(BASE_PATH, "training_data")
CSV_PATH  = os.path.join(BASE_PATH, "training_data.csv")


SAMPLE_RATE    = 4000
WINDOW_SEC     = 5.0
WINDOW_LEN     = int(SAMPLE_RATE * WINDOW_SEC)   # 20,000 samples
MIN_WINDOW_LEN = int(WINDOW_LEN * 0.5)

BATCH_SIZE   = 64     # larger batch — GPU handles it well for 1D
NUM_EPOCHS   = 50     # more than ResNet18 since 1D trains faster
LR           = 1e-4
RANDOM_STATE = 42

PATIENT_THRESHOLD = 0.40
PATIENT_METHOD    = "mean"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DATA_PATH exists:", os.path.exists(DATA_PATH))
print("CSV_PATH  exists:", os.path.exists(CSV_PATH))
print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# =========================================================
# 2. SEED
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# =========================================================
# 3. AUDIO PREPROCESSING
#    Paper uses low-pass filter at 1000Hz then downsample to 2000Hz
#    We keep 4000Hz but apply bandpass as in our baseline
# =========================================================
def bandpass_filter(signal, lowcut=25, highcut=800, fs=4000, order=4):
    """
    Paper filters below 1000Hz. We use 800Hz cutoff
    (murmur frequency range) for consistency with Phase 2.
    """
    nyquist = 0.5 * fs
    low  = lowcut  / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def standardize(signal):
    """Paper standardizes: (x - mean) / std per recording."""
    mu  = np.mean(signal)
    std = np.std(signal) + 1e-8
    return ((signal - mu) / std).astype(np.float32)

def load_full_audio(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    y    = bandpass_filter(y)
    y    = standardize(y)
    return y

def slice_into_windows(y):
    windows, starts = [], []
    start = 0
    while start < len(y):
        seg = y[start : start + WINDOW_LEN]
        if len(seg) >= MIN_WINDOW_LEN:
            if len(seg) < WINDOW_LEN:
                seg = np.pad(seg, (0, WINDOW_LEN - len(seg)))
            windows.append(seg.astype(np.float32))
            starts.append(start / SAMPLE_RATE)
        start += WINDOW_LEN
    return windows, starts

# =========================================================
# 4. READ CSV + BUILD SEGMENT DATAFRAME
# =========================================================
df = pd.read_csv(CSV_PATH)
print("\nCSV shape:", df.shape)

df = df[df["Murmur"].isin(["Absent", "Present"])].copy()
df["label"] = df["Murmur"].map({"Absent": 0, "Present": 1})

print("Label distribution:")
print(df["Murmur"].value_counts())

all_files = os.listdir(DATA_PATH)
records   = []
skipped   = 0

print("\nSlicing WAV files into 5-second windows...")

for _, row in df.iterrows():
    pid    = row["Patient ID"]
    prefix = str(pid)
    wavs   = [f for f in all_files
               if f.startswith(prefix + "_") and f.endswith(".wav")]

    for wav in wavs:
        fpath = os.path.join(DATA_PATH, wav)
        try:
            y       = load_full_audio(fpath)
            windows, _ = slice_into_windows(y)
        except Exception:
            skipped += 1
            continue

        for window in windows:
            records.append({
                "patient_id": pid,
                "label":      int(row["label"]),
                "audio":      window
            })

data_df = pd.DataFrame(records)

print(f"\n===== SEGMENTATION SUMMARY =====")
print(f"Total segments:  {len(data_df):,}")
print(f"Unique patients: {data_df['patient_id'].nunique()}")
print(f"Class counts:")
print(data_df["label"].value_counts())

# =========================================================
# 5. TRAIN / VAL / TEST SPLIT
# =========================================================
patient_df = data_df.groupby("patient_id")["label"].first().reset_index()

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tv_idx, te_idx = next(gss_test.split(
    patient_df["patient_id"].values,
    patient_df["label"].values,
    patient_df["patient_id"].values
))

test_pats = set(patient_df.iloc[te_idx]["patient_id"].values)
tv_pat_df = patient_df.iloc[tv_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, va_idx = next(gss_val.split(
    tv_pat_df["patient_id"].values,
    tv_pat_df["label"].values,
    tv_pat_df["patient_id"].values
))

train_pats = set(tv_pat_df.iloc[tr_idx]["patient_id"].values)
val_pats   = set(tv_pat_df.iloc[va_idx]["patient_id"].values)

train_df = data_df[data_df["patient_id"].isin(train_pats)].reset_index(drop=True)
val_df   = data_df[data_df["patient_id"].isin(val_pats)].reset_index(drop=True)
test_df  = data_df[data_df["patient_id"].isin(test_pats)].reset_index(drop=True)

print(f"\n===== SPLIT SUMMARY =====")
print(f"Train: {len(train_df):,} segments | {len(train_pats)} patients")
print(f"Val:   {len(val_df):,} segments  | {len(val_pats)} patients")
print(f"Test:  {len(test_df):,} segments  | {len(test_pats)} patients")

os.makedirs("splits", exist_ok=True)

pd.DataFrame({"patient_id": list(train_pats), "split": "train"}).to_csv(
    "splits/phase1_train_patients.csv", index=False
)

pd.DataFrame({"patient_id": list(val_pats), "split": "val"}).to_csv(
    "splits/phase1_val_patients.csv", index=False
)

pd.DataFrame({"patient_id": list(test_pats), "split": "test"}).to_csv(
    "splits/phase1_test_patients.csv", index=False
)

print("\nSaved patient-level split files to: splits/")
