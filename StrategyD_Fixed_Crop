"""
Strategy D: fixed 100-ms murmur-bearing (MB) versus non-murmur-bearing (NMB)
classification for the shortcut-controlled analysis.

This script:
1. Loads CirCor DigiScope recordings and murmur metadata.
2. Constructs MB/NMB labels using cardiac phase and annotated murmur location.
3. Retains intervals >=100 ms and applies fixed 100-ms real-signal crops.
4. Trains the duration-blind acoustic classifier using patient-disjoint splits.
5. Evaluates a duration-only baseline on the same cohort.
6. Reports test accuracy, balanced accuracy, macro F1, and AUROC.

The dataset is not redistributed with this repository. Set BASE_PATH below to
the local CirCor DigiScope v1.0.3 directory before running.
"""

# =============================================================
# STRATEGY D: FIXED 100-ms MURMUR-BEARING VS NON-MURMUR-BEARING CLASSIFICATION
# Duration-blind acoustic model used in the revised manuscript
# Fixed 100-ms real-signal crops; no padding, resampling, or tiling
# =============================================================

import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score,
                              balanced_accuracy_score,
                              roc_auc_score,
                              classification_report)
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── Config ────────────────────────────────────────────────────
BASE_PATH  = "path/to/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
DATA_PATH  = os.path.join(BASE_PATH, "training_data")
CSV_PATH   = os.path.join(BASE_PATH, "training_data.csv")

SAMPLE_RATE  = 4000
CROP_MS      = 100
CROP_LEN     = int(CROP_MS * SAMPLE_RATE / 1000)   # 400 samples
MIN_DUR_S    = CROP_MS / 1000                        # 0.10s
SYSTOLE_LABEL  = 2
DIASTOLE_LABEL = 4
BATCH_SIZE     = 128
NUM_EPOCHS     = 50
LR             = 1e-4
RANDOM_STATE   = 42

print(f"Crop length: {CROP_MS}ms = {CROP_LEN} samples")
print(f"Min interval duration: {MIN_DUR_S}s")

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(RANDOM_STATE)

# ── Preprocessing ─────────────────────────────────────────────
def bandpass_filter(signal, lowcut=25, highcut=800,
                    fs=4000, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq],
                  btype="band")
    return filtfilt(b, a, signal)

def standardize(signal):
    mu  = np.mean(signal)
    std = np.std(signal) + 1e-8
    return ((signal - mu) / std).astype(np.float32)

def load_and_preprocess(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    y    = bandpass_filter(y)
    y    = standardize(y)
    return y

def extract_raw_segment(signal, start_sec, end_sec):
    start = int(start_sec * SAMPLE_RATE)
    end   = int(end_sec   * SAMPLE_RATE)
    seg   = signal[start:end]
    return seg if len(seg) >= CROP_LEN else None

def center_crop(seg):
    """Deterministic — used for val/test."""
    start = (len(seg) - CROP_LEN) // 2
    return standardize(
        seg[start:start+CROP_LEN].astype(np.float32))

def random_crop(seg):
    """Random — used during training augmentation."""
    if len(seg) == CROP_LEN:
        start = 0
    else:
        start = np.random.randint(0, len(seg) - CROP_LEN + 1)
    return standardize(
        seg[start:start+CROP_LEN].astype(np.float32))

# ── Patient metadata ──────────────────────────────────────────
df         = pd.read_csv(CSV_PATH)
df_present = df[df["Murmur"] == "Present"].copy()

def get_murmur_phase(row):
    has_sys = pd.notna(row["Systolic murmur timing"])
    has_dia = pd.notna(row["Diastolic murmur timing"])
    if has_sys and has_dia: return "both"
    elif has_sys:           return "systolic"
    elif has_dia:           return "diastolic"
    else:                   return "unknown"

df_present["murmur_phase"] = df_present.apply(
    get_murmur_phase, axis=1)
df_use = df_present[df_present["murmur_phase"].isin(
    ["systolic","diastolic"])].copy()

patient_info = {}
for _, row in df_use.iterrows():
    pid   = str(row["Patient ID"])
    phase = row["murmur_phase"]
    locs  = str(row["Murmur locations"]) \
            if pd.notna(row["Murmur locations"]) else ""
    patient_info[pid] = {
        "phase":     phase,
        "locations": set(locs.split("+")) if locs else set()
    }

# ── Extract segments — store RAW intervals ────────────────────
# Key: store raw segment, apply crop in Dataset
# This allows random crop during training each epoch
all_files = os.listdir(DATA_PATH)
records   = []
stats     = {"mb": 0, "nmb": 0, "failed": 0}

print(f"\nExtracting segments (min dur={CROP_MS}ms)...")

for pid, info in patient_info.items():
    murmur_phase = info["phase"]
    murmur_locs  = info["locations"]
    wav_files    = sorted([f for f in all_files
                           if f.startswith(pid+"_")
                           and f.endswith(".wav")])
    for wav_file in wav_files:
        rec_loc       = wav_file.replace(".wav","").split("_")[-1]
        at_murmur_loc = rec_loc in murmur_locs
        wav_path      = os.path.join(DATA_PATH, wav_file)
        tsv_path      = wav_path.replace(".wav", ".tsv")
        if not os.path.exists(tsv_path):
            continue
        try:
            signal = load_and_preprocess(wav_path)
        except Exception:
            stats["failed"] += 1
            continue

        tsv = pd.read_csv(tsv_path, sep='\t', header=None,
                          names=["start","end","label"])
        for _, row in tsv.iterrows():
            lbl = int(row["label"])
            if lbl not in [SYSTOLE_LABEL, DIASTOLE_LABEL]:
                continue
            start_sec = float(row["start"])
            end_sec   = float(row["end"])
            dur       = end_sec - start_sec
            if dur < MIN_DUR_S:
                continue

            seg_raw = extract_raw_segment(
                signal, start_sec, end_sec)
            if seg_raw is None:
                continue

            seg_phase = ("systolic" if lbl == SYSTOLE_LABEL
                         else "diastolic")
            is_mb = (seg_phase == murmur_phase
                     and at_murmur_loc)
            label = 1 if is_mb else 0

            records.append({
                "patient_id":     pid,
                "label":          label,
                "seg_phase":      seg_phase,
                "at_murmur_loc":  at_murmur_loc,
                "duration_sec":   dur,
                "raw_audio":      seg_raw   # store raw, crop in Dataset
            })
            if is_mb: stats["mb"] += 1
            else:     stats["nmb"] += 1

data_d = pd.DataFrame(records)
print(f"\nExtraction complete:")
print(f"  Total    : {len(data_d):,}")
print(f"  MB       : {stats['mb']:,}")
print(f"  NMB      : {stats['nmb']:,}")
print(f"  Patients : {data_d['patient_id'].nunique()}")
print(f"  MB ratio : {stats['mb']/(stats['mb']+stats['nmb']):.1%}")
print(f"  Duration mean: {data_d['duration_sec'].mean():.3f}s")

# ── Duration-only baseline on this cohort ─────────────────────
print(f"\n{'='*55}")
print("DURATION-ONLY BASELINE (≥100ms cohort)")
print(f"{'='*55}")

patient_df_d = data_d.groupby(
    "patient_id")["label"].first().reset_index()
gss_test = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tv_idx, te_idx = next(gss_test.split(
    patient_df_d["patient_id"].values,
    patient_df_d["label"].values,
    patient_df_d["patient_id"].values
))
test_pats_d  = set(
    patient_df_d.iloc[te_idx]["patient_id"].values)
tv_df_d      = patient_df_d.iloc[tv_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, va_idx = next(gss_val.split(
    tv_df_d["patient_id"].values,
    tv_df_d["label"].values,
    tv_df_d["patient_id"].values
))
train_pats_d = set(tv_df_d.iloc[tr_idx]["patient_id"].values)
val_pats_d   = set(tv_df_d.iloc[va_idx]["patient_id"].values)

train_d = data_d[data_d["patient_id"].isin(
    train_pats_d)].reset_index(drop=True)
val_d   = data_d[data_d["patient_id"].isin(
    val_pats_d)].reset_index(drop=True)
test_d  = data_d[data_d["patient_id"].isin(
    test_pats_d)].reset_index(drop=True)

print(f"Train: {len(train_d):,} | {len(train_pats_d)} patients")
print(f"Val  : {len(val_d):,}  | {len(val_pats_d)} patients")
print(f"Test : {len(test_d):,}  | {len(test_pats_d)} patients")

# Duration baseline on same split
dur_X_tr = train_d["duration_sec"].values.reshape(-1,1)
dur_y_tr = train_d["label"].values
dur_X_te = test_d["duration_sec"].values.reshape(-1,1)
dur_y_te = test_d["label"].values

lr_dur = LogisticRegression(max_iter=1000)
lr_dur.fit(dur_X_tr, dur_y_tr)
dur_prob = lr_dur.predict_proba(dur_X_te)[:,1]
dur_pred = lr_dur.predict(dur_X_te)

print(f"\nDuration-only baseline (≥100ms cohort):")
print(f"  Balanced acc : "
      f"{balanced_accuracy_score(dur_y_te,dur_pred):.4f}")
print(f"  AUROC        : "
      f"{roc_auc_score(dur_y_te,dur_prob):.4f}")

# ── Dataset with on-the-fly cropping ─────────────────────────
class CropDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        seg   = row["raw_audio"].copy()

        # Apply crop
        if self.augment:
            y = random_crop(seg)   # random position each epoch
        else:
            y = center_crop(seg)   # deterministic for val/test

        # Additional augmentation on the cropped window
        if self.augment:
            if np.random.rand() < 0.60:
                sig_rms   = np.sqrt(np.mean(y**2)) + 1e-8
                snr_db    = np.random.uniform(5.0, 30.0)
                noise_rms = sig_rms / (10**(snr_db/20.0))
                y = y + (np.random.randn(len(y)).astype(
                    np.float32) * noise_rms)
            if np.random.rand() < 0.40:
                y = (y * np.random.uniform(
                    0.85, 1.15)).astype(np.float32)
            y = standardize(y)

        return torch.FloatTensor(y).unsqueeze(0), label

def make_loader(df, augment=False, shuffle=False):
    ds = CropDataset(df, augment=augment)
    if shuffle:
        labels  = df["label"].values
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(weights),
            len(weights), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=2,
                          pin_memory=True)
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=2,
                      pin_memory=True)

# ── Model ─────────────────────────────────────────────────────
class TPEBlock(nn.Module):
    def __init__(self, in_channels=1, filters=(32,64,128),
                 kernel=6, stride=2, dropout=0.3):
        super().__init__()
        layers, ch = [], in_channels
        for f in filters:
            layers += [nn.Conv1d(ch, f, kernel_size=kernel,
                                 stride=stride,
                                 padding=kernel//2),
                       nn.BatchNorm1d(f), nn.ReLU(),
                       nn.MaxPool1d(2,2), nn.Dropout(dropout)]
            ch = f
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RosaNetBlock(nn.Module):
    def __init__(self, in_channels=128, hidden=64,
                 dropout=0.3):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(in_channels, hidden*2, kernel_size=3,
                      dilation=d, padding=d)
            for d in [2,4,6]])
        self.fuse_convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 1) for _ in [2,4,6]])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        outs = []
        for dc, fc in zip(self.dilated_convs, self.fuse_convs):
            o    = dc(x)
            gate = torch.sigmoid(o[:, :o.shape[1]//2])
            mask = F.relu(o[:, o.shape[1]//2:])
            outs.append(self.dropout(fc(gate * mask)))
        return torch.cat(outs, dim=1)

class TemporalSelfAttention(nn.Module):
    def __init__(self, channels=192, num_heads=4,
                 dropout=0.1):
        super().__init__()
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_proj    = nn.Linear(channels, channels)
        self.k_proj    = nn.Linear(channels, channels)
        self.v_proj    = nn.Linear(channels, channels)
        self.out_proj  = nn.Linear(channels, channels)
        self.dropout   = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(
            torch.randn(1,1,channels))
    def forward(self, x):
        B, C, L = x.shape
        x   = x.permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        S   = L + 1
        Q = self.q_proj(x).view(
            B,S,self.num_heads,
            self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(
            B,S,self.num_heads,
            self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(
            B,S,self.num_heads,
            self.head_dim).transpose(1,2)
        attn = F.softmax(
            (Q@K.transpose(-2,-1))*self.scale, dim=-1)
        attn = self.dropout(attn)
        out  = (attn@V).transpose(1,2).contiguous().view(
            B,S,C)
        return self.out_proj(out)[:, 0, :]

class Phase2_MB_Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.tpe       = TPEBlock()
        self.rosa      = RosaNetBlock()
        self.attention = TemporalSelfAttention()
        self.classifier = nn.Sequential(
            nn.LayerNorm(192), nn.Dropout(dropout),
            nn.Linear(192, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.tpe(x)
        x = self.rosa(x)
        x = self.attention(x)
        return self.classifier(x)

# ── Train / eval functions ────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for inputs, labels in loader:
        inputs  = inputs.to(DEVICE)
        logits  = model(inputs)
        probs   = torch.softmax(logits, dim=1)[:, 1]
        preds   = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

# ── Train ─────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("TRAINING: Strategy D — Fixed 100-ms MB-vs-NMB Crop")
print(f"{'='*55}")

seed_everything(RANDOM_STATE)
model_d   = Phase2_MB_Classifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()   # simplified — no alpha issue
optimizer = torch.optim.AdamW(
    model_d.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

train_loader = make_loader(train_d, augment=True, shuffle=True)
val_loader   = make_loader(val_d,   augment=False)

os.makedirs("models", exist_ok=True)
SAVE_D      = "models/phase2_mb_strategy_d.pth"
best_val_f1 = -1
no_improve  = 0
PATIENCE    = 10

for epoch in range(NUM_EPOCHS):
    train_one_epoch(model_d, train_loader, criterion, optimizer)
    scheduler.step()

    val_true, val_pred, _ = evaluate(model_d, val_loader)
    val_f1  = f1_score(val_true, val_pred,
                       average="macro", zero_division=0)
    val_acc = accuracy_score(val_true, val_pred)
    val_bal = balanced_accuracy_score(val_true, val_pred)

    print(f"Ep {epoch+1:02d}/{NUM_EPOCHS} | "
          f"ValAcc={val_acc:.4f} | "
          f"ValF1={val_f1:.4f} | "
          f"ValBal={val_bal:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model_d.state_dict(), SAVE_D)
        print(f"  Saved (ValF1={best_val_f1:.4f})")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

# ── Test evaluation ───────────────────────────────────────────
model_d.load_state_dict(
    torch.load(SAVE_D, map_location=DEVICE))
model_d.eval()

test_loader  = make_loader(test_d, augment=False)
true_d, pred_d, prob_d = evaluate(model_d, test_loader)

print(f"\n{'='*55}")
print("TEST RESULTS: Strategy D — Fixed 100-ms MB-vs-NMB Crop")
print(f"{'='*55}")
print(f"Segments evaluated : {len(true_d):,}")
print(f"Accuracy           : {accuracy_score(true_d,pred_d):.4f}")
print(f"Balanced accuracy  : "
      f"{balanced_accuracy_score(true_d,pred_d):.4f}")
print(f"Macro F1           : "
      f"{f1_score(true_d,pred_d,average='macro',zero_division=0):.4f}")
print(f"AUROC              : "
      f"{roc_auc_score(true_d,prob_d):.4f}")
print(f"\nClassification report:")
print(classification_report(
    true_d, pred_d,
    target_names=["Non-murmur-bearing","Murmur-bearing"],
    digits=4, zero_division=0))

# All-zero sanity check
zero_input = torch.zeros(1, 1, CROP_LEN).to(DEVICE)
with torch.no_grad():
    p_mb = torch.softmax(
        model_d(zero_input), dim=1)[0, 1].item()
print(f"All-zero {CROP_MS}ms input P(MB): {p_mb:.4f}")
print(f"  (model cannot observe original duration — "
      f"input is always {CROP_LEN} real samples)")

# ── Final comparison ──────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL COMPARISON — FIXED-CROP ACOUSTIC AND DURATION BASELINES")
print(f"{'='*60}")
print(f"{'Method':<42} {'BalAcc':>7} {'AUROC':>7} {'F1':>7}")
print("-"*60)
print(f"{'Duration-only (≥100ms cohort)':<42} "
      f"{balanced_accuracy_score(dur_y_te,dur_pred):>7.4f} "
      f"{roc_auc_score(dur_y_te,dur_prob):>7.4f} {'—':>7}")
print(f"{'Strategy D — 100ms crop (≥100ms segs)':<42} "
      f"{balanced_accuracy_score(true_d,pred_d):>7.4f} "
      f"{roc_auc_score(true_d,prob_d):>7.4f} "
      f"{f1_score(true_d,pred_d,average='macro',zero_division=0):>7.4f}")
