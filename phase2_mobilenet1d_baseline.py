
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)

# ── config (same as Phase 2 v3) ──────────────────────────
SAMPLE_RATE    = 4000
TARGET_LEN     = int(SAMPLE_RATE * 1.0)
BATCH_SIZE     = 128
NUM_EPOCHS     = 50
LR             = 1e-4
RANDOM_STATE   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# ── dataset (reuse from Phase 2 — data_df must be in scope)
class CardiacPhaseDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["timing_label"])
        y     = row["audio"].copy()
        if self.augment:
            if np.random.rand() < 0.4:
                y += np.random.randn(len(y)).astype(np.float32)*0.005
            if np.random.rand() < 0.4:
                y = (y * np.random.uniform(0.85,1.15)).astype(np.float32)
            if np.random.rand() < 0.3:
                shift = int(len(y)*np.random.uniform(-0.1,0.1))
                y = np.roll(y, shift).astype(np.float32)
            mu = np.mean(y); std = np.std(y)+1e-8
            y  = ((y-mu)/std).astype(np.float32)
        return torch.FloatTensor(y).unsqueeze(0), label

# =========================================================
# MobileNet1D ARCHITECTURE
# Depthwise separable convolutions reduce parameters while
# maintaining representational power. Expansion factor=6
# follows MobileNetV2 convention.
# =========================================================
class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for 1D signals.
    Replaces standard Conv1d with:
      1. Depthwise conv: one filter per input channel
      2. Pointwise conv: 1×1 conv to mix channels
    Parameter reduction: k×Cin×Cout → k×Cin + Cin×Cout
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1):
        super().__init__()
        self.depthwise  = nn.Conv1d(in_ch, in_ch, kernel,
                                     stride=stride,
                                     padding=padding,
                                     groups=in_ch)
        self.pointwise  = nn.Conv1d(in_ch, out_ch, 1)
        self.bn         = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu6(self.bn(self.pointwise(self.depthwise(x))))


class InvertedResidual1d(nn.Module):
    """
    MobileNetV2 inverted residual block adapted for 1D.
    Expand → Depthwise → Project.
    """
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_residual = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            # Expand
            nn.Conv1d(in_ch, mid_ch, 1),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv1d(mid_ch, mid_ch, 3, stride=stride,
                      padding=1, groups=mid_ch),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv1d(mid_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch)
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_residual else out


class MobileNet1D(nn.Module):
    """
    1D MobileNetV2-inspired architecture for PCG classification.
    Input: (B, 1, 4000) — 1-second PCG segment at 4kHz
    Output: (B, num_classes)

    Architecture follows MobileNetV2 channel progression
    scaled down for 1D audio:
      Conv → Inverted Residuals → GlobalAvgPool → Classifier
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        # Initial conv: downsample aggressively
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True)
        )

        # Inverted residual blocks
        # (in_ch, out_ch, stride, n_blocks)
        cfg = [
            (32,  16,  1, 1),
            (16,  24,  2, 2),
            (24,  32,  2, 3),
            (32,  64,  2, 4),
            (64,  96,  1, 3),
            (96,  160, 2, 3),
            (160, 320, 1, 1),
        ]

        blocks = []
        for in_ch, out_ch, stride, n in cfg:
            for i in range(n):
                s = stride if i == 0 else 1
                blocks.append(InvertedResidual1d(in_ch, out_ch,
                                                  stride=s))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv1d(320, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU6(inplace=True)
        )

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ── build and count params ────────────────────────────────
model_mobile = MobileNet1D(num_classes=2).to(DEVICE)
total_params = sum(p.numel() for p in model_mobile.parameters()
                   if p.requires_grad)
print(f"\nMobileNet1D parameters: {total_params:,}")
print(f"v3 parameters:          383,906")
print(f"ResNet18 parameters:    ~11,000,000")
del model_mobile  # rebuild with seed later

# =========================================================
# SPLIT (same as Phase 2 v3 — data_df must be in scope)
# =========================================================
patient_df_p2 = data_df.groupby("patient_id")["timing_label"]\
                        .first().reset_index()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                         random_state=RANDOM_STATE)
tv_idx, te_idx = next(gss.split(
    patient_df_p2["patient_id"].values,
    patient_df_p2["timing_label"].values,
    patient_df_p2["patient_id"].values))
test_pats = set(patient_df_p2.iloc[te_idx]["patient_id"].values)
tv_df     = patient_df_p2.iloc[tv_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2,
                          random_state=RANDOM_STATE)
tr_idx, va_idx = next(gss2.split(
    tv_df["patient_id"].values,
    tv_df["timing_label"].values,
    tv_df["patient_id"].values))
train_pats = set(tv_df.iloc[tr_idx]["patient_id"].values)
val_pats   = set(tv_df.iloc[va_idx]["patient_id"].values)

train_df_m = data_df[data_df["patient_id"].isin(train_pats)]\
             .reset_index(drop=True)
val_df_m   = data_df[data_df["patient_id"].isin(val_pats)]\
             .reset_index(drop=True)
test_df_m  = data_df[data_df["patient_id"].isin(test_pats)]\
             .reset_index(drop=True)

# dataloaders
train_ds = CardiacPhaseDataset(train_df_m, augment=True)
val_ds   = CardiacPhaseDataset(val_df_m,   augment=False)
test_ds  = CardiacPhaseDataset(test_df_m,  augment=False)

labels   = train_df_m["timing_label"].values
counts   = np.bincount(labels)
weights  = 1.0 / counts[labels]
sampler  = WeightedRandomSampler(torch.DoubleTensor(weights),
                                  len(weights), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=2,
                          pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2,
                          pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2,
                          pin_memory=True)

# =========================================================
# TRAIN MobileNet1D
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, targets):
        ce = self.ce(logits, targets); pt = torch.exp(-ce)
        return (self.alpha * ((1-pt)**self.gamma) * ce).mean()

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    for inputs, labels in loader:
        preds = torch.argmax(model(inputs.to(DEVICE)), dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="macro",
                   zero_division=0)
    return acc, f1, np.array(labels_all), np.array(preds_all)

seed_everything(RANDOM_STATE)
model_m    = MobileNet1D(num_classes=2).to(DEVICE)
criterion  = FocalLoss(alpha=0.80, gamma=2.0)
optimizer  = torch.optim.AdamW(model_m.parameters(),
                                lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

best_vf1  = -1
best_path = "models/phase2/best_mobilenet1d_phase2.pth"

no_imp    = 0

print(f"\nTraining MobileNet1D ({total_params:,} params)...")
for epoch in range(NUM_EPOCHS):
    train_one_epoch(model_m, train_loader, criterion, optimizer)
    val_acc, val_f1, _, _ = evaluate(model_m, val_loader)
    scheduler.step()

    if val_f1 > best_vf1:
        best_vf1 = val_f1
        torch.save(model_m.state_dict(), best_path)
        no_imp = 0
        print(f"  Ep {epoch+1:02d} | ValF1: {val_f1:.4f} ✅")
    else:
        no_imp += 1
        if no_imp >= 10:
            print(f"  Early stop at epoch {epoch+1}")
            break

# ── test ──────────────────────────────────────────────────
model_m.load_state_dict(torch.load(best_path, map_location=DEVICE))
test_acc, test_f1, test_labels, test_preds = evaluate(
    model_m, test_loader)

print(f"\n===== MobileNet1D RESULTS =====")
print(f"Accuracy : {test_acc*100:.2f}%")
print(f"Macro F1 : {test_f1:.4f}")
print(f"Params   : {total_params:,}")
print(f"\nClassification Report:")
print(classification_report(test_labels, test_preds,
      target_names=["Systolic","Diastolic"], digits=4))

print(f"""
===== UPDATED TABLE II =====
Model                          Accuracy    F1      Params

MobileNet1D (lightweight)      {test_acc*100:.2f}%    {test_f1:.4f}    {total_params:,}
