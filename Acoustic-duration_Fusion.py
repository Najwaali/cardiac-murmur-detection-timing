"""
Acoustic-duration fusion for the shortcut-controlled MB-vs-NMB analysis.

This script builds leakage-controlled patient-level out-of-fold (OOF)
predictions for the fixed-crop acoustic model and the duration-only model,
fits a logistic-regression fusion model, and evaluates it on the held-out
test cohort.

It also computes patient-level bootstrap confidence intervals and pairwise
AUROC differences.

Run this script after `strategy_d_fixed_crop.py`, or import the required
objects from that module.
"""

from torch.utils.data import DataLoader

@torch.no_grad()
def get_acoustic_probs(model, df, batch_size=256,
                       augment=False):
    ds = CropDataset(df, augment=augment)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    probs, labels = [], []

    model.eval()

    for inputs, lbl in loader:
        p = torch.softmax(
            model(inputs.to(DEVICE)), dim=1
        )[:, 1]

        probs.extend(p.cpu().numpy())
        labels.extend(lbl.numpy())

    return np.array(labels), np.array(probs)

print("get_acoustic_probs ready")

# =============================================================
# FINAL CORRECTED FUSION — BOTH ACOUSTIC AND DURATION OOF
# Nested inner/outer split, clean duration OOF, patient bootstrap
# =============================================================

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score
)

# This script depends on objects created by strategy_d_fixed_crop.py:
# CropDataset, make_loader, evaluate, train_one_epoch,
# Phase2_MB_Classifier, model_d, SAVE_D, DEVICE, RANDOM_STATE,
# train_d, val_d, test_d, lr_dur, LR, and seed_everything.

# Reload best Strategy D checkpoint
model_d.load_state_dict(torch.load(SAVE_D, map_location=DEVICE))
model_d.eval()
print("Strategy D model reloaded.")

# ── Rebuild test probabilities ────────────────────────────────
test_loader_det = make_loader(test_d, augment=False)
test_true, _, test_acoustic = evaluate(model_d, test_loader_det)

test_duration = lr_dur.predict_proba(
    test_d["duration_sec"].values.reshape(-1, 1)
)[:, 1]

test_pids = test_d["patient_id"].values
dur_pred_te = lr_dur.predict(
    test_d["duration_sec"].values.reshape(-1, 1)
)

print(f"Test segments  : {len(test_true):,}")
print(f"Acoustic AUROC : {roc_auc_score(test_true, test_acoustic):.4f}")
print(f"Duration AUROC : {roc_auc_score(test_true, test_duration):.4f}")

# ── Train/validation combined cohort ──────────────────────────
trainval_d = pd.concat([train_d, val_d]).reset_index(drop=True)
trainval_pids = trainval_d["patient_id"].unique()

# Patient-level shuffle
rng_split = np.random.default_rng(RANDOM_STATE)
pids_shuffled = trainval_pids.copy()
rng_split.shuffle(pids_shuffled)

N_FOLDS = 5
kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Initialize with NaN to detect missing OOF predictions
oof_probs = np.full(len(trainval_d), np.nan)
oof_dur = np.full(len(trainval_d), np.nan)
oof_labels = trainval_d["label"].values

print(
    "\nBuilding patient-level out-of-fold predictions "
    "(acoustic + duration, nested split)..."
)

os.makedirs("models/fusion_oof", exist_ok=True)

for fold, (outer_tr_idx, outer_va_idx) in enumerate(
    kf.split(pids_shuffled)
):
    outer_tr_pats = set(pids_shuffled[outer_tr_idx])
    outer_va_pats = set(pids_shuffled[outer_va_idx])

    outer_tr_df = trainval_d[
        trainval_d["patient_id"].isin(outer_tr_pats)
    ].reset_index(drop=True)

    outer_va_df = trainval_d[
        trainval_d["patient_id"].isin(outer_va_pats)
    ].reset_index(drop=True)

    outer_va_mask = trainval_d["patient_id"].isin(outer_va_pats)

    # ── Duration OOF predictions ──────────────────────────────
    fold_dur_lr = LogisticRegression(max_iter=1000)
    fold_dur_lr.fit(
        outer_tr_df["duration_sec"].values.reshape(-1, 1),
        outer_tr_df["label"].values
    )

    fold_dur_prob = fold_dur_lr.predict_proba(
        outer_va_df["duration_sec"].values.reshape(-1, 1)
    )[:, 1]

    oof_dur[outer_va_mask] = fold_dur_prob

    # ── Inner patient-level split for acoustic early stopping ─
    inner_pids = outer_tr_df["patient_id"].unique().copy()
    rng_inner = np.random.default_rng(RANDOM_STATE + fold)
    rng_inner.shuffle(inner_pids)

    cut = int(len(inner_pids) * 0.80)
    inner_tr_pats = set(inner_pids[:cut])
    inner_va_pats = set(inner_pids[cut:])

    inner_tr_df = outer_tr_df[
        outer_tr_df["patient_id"].isin(inner_tr_pats)
    ].reset_index(drop=True)

    inner_va_df = outer_tr_df[
        outer_tr_df["patient_id"].isin(inner_va_pats)
    ].reset_index(drop=True)

    # ── Train fold acoustic model ─────────────────────────────
    seed_everything(RANDOM_STATE + fold)

    fold_model = Phase2_MB_Classifier().to(DEVICE)
    fold_optim = torch.optim.AdamW(
        fold_model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
    fold_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        fold_optim,
        T_max=30,
        eta_min=1e-6
    )
    fold_crit = torch.nn.CrossEntropyLoss()

    inner_tr_loader = make_loader(
        inner_tr_df,
        augment=True,
        shuffle=True
    )
    inner_va_loader = make_loader(
        inner_va_df,
        augment=False
    )

    best_f1 = -1
    no_imp = 0
    fold_path = f"models/fusion_oof/oof_fold_{fold + 1}.pth"

    for epoch in range(30):
        train_one_epoch(
            fold_model,
            inner_tr_loader,
            fold_crit,
            fold_optim
        )
        fold_sched.step()

        va_t, va_p, _ = evaluate(
            fold_model,
            inner_va_loader
        )
        vf1 = f1_score(
            va_t,
            va_p,
            average="macro",
            zero_division=0
        )

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(
                fold_model.state_dict(),
                fold_path
            )
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 8:
                break

    fold_model.load_state_dict(
        torch.load(
            fold_path,
            map_location=DEVICE
        )
    )
    fold_model.eval()

    # OOF acoustic predictions on outer held-out patients
    _, oof_p = get_acoustic_probs(
        fold_model,
        outer_va_df
    )
    oof_probs[outer_va_mask] = oof_p

    print(
        f"  Fold {fold + 1}/{N_FOLDS} | "
        f"outer_va={outer_va_mask.sum()} segments | "
        f"inner_f1={best_f1:.4f}"
    )

# Verify no missing predictions
assert not np.isnan(oof_probs).any(), \
    "Missing acoustic OOF predictions"

assert not np.isnan(oof_dur).any(), \
    "Missing duration OOF predictions"

print("All OOF predictions complete; no patient leakage.")

# ── Train fusion model on OOF predictions ────────────────────
X_fusion_tr = np.column_stack(
    [oof_probs, oof_dur]
)
y_fusion_tr = oof_labels

fusion_lr = LogisticRegression(max_iter=1000)
fusion_lr.fit(X_fusion_tr, y_fusion_tr)

print("\nFusion coefficients:")
print(
    f"  Acoustic weight : "
    f"{fusion_lr.coef_[0][0]:.4f}"
)
print(
    f"  Duration weight : "
    f"{fusion_lr.coef_[0][1]:.4f}"
)

# ── Evaluate on held-out test set ─────────────────────────────
X_fusion_te = np.column_stack(
    [test_acoustic, test_duration]
)

fusion_prob = fusion_lr.predict_proba(
    X_fusion_te
)[:, 1]

fusion_pred = fusion_lr.predict(
    X_fusion_te
)

fusion_auc = roc_auc_score(
    test_true,
    fusion_prob
)
fusion_bal = balanced_accuracy_score(
    test_true,
    fusion_pred
)
fusion_f1 = f1_score(
    test_true,
    fusion_pred,
    average="macro",
    zero_division=0
)

d_auc = roc_auc_score(
    test_true,
    test_duration
)
a_auc = roc_auc_score(
    test_true,
    test_acoustic
)

print(
    f"\nFusion AUROC        : "
    f"{fusion_auc:.4f}"
)
print(
    f"Fusion balanced acc : "
    f"{fusion_bal:.4f}"
)
print(
    f"Fusion macro F1     : "
    f"{fusion_f1:.4f}"
)

# ── Patient-level bootstrap ───────────────────────────────────
N_BOOTSTRAP = 2000
print(
    f"\nRunning patient-level bootstrap "
    f"({N_BOOTSTRAP} resamples)..."
)

unique_pids = np.unique(test_pids)
n_pat = len(unique_pids)
rng_boot = np.random.default_rng(RANDOM_STATE)

b_dur, b_aco, b_fus = [], [], []
b_fd, b_fa, b_ad = [], [], []

for _ in range(N_BOOTSTRAP):
    boot_pats = rng_boot.choice(
        unique_pids,
        n_pat,
        replace=True
    )

    idx = np.concatenate([
        np.where(test_pids == p)[0]
        for p in boot_pats
    ])

    y_b = test_true[idx]

    if len(np.unique(y_b)) < 2:
        continue

    auc_d = roc_auc_score(
        y_b,
        test_duration[idx]
    )
    auc_a = roc_auc_score(
        y_b,
        test_acoustic[idx]
    )
    auc_f = roc_auc_score(
        y_b,
        fusion_prob[idx]
    )

    b_dur.append(auc_d)
    b_aco.append(auc_a)
    b_fus.append(auc_f)

    b_fd.append(auc_f - auc_d)
    b_fa.append(auc_f - auc_a)
    b_ad.append(auc_a - auc_d)


def ci95(arr):
    return np.percentile(
        np.asarray(arr),
        [2.5, 97.5]
    )


def pval_gt0(diffs):
    """
    One-sided bootstrap p-value for testing whether
    the observed AUROC difference is greater than zero.
    """
    diffs = np.asarray(diffs)
    return (
        (diffs <= 0).sum() + 1
    ) / (
        len(diffs) + 1
    )


# ── Final table ───────────────────────────────────────────────
print("\n" + "=" * 70)
print(
    "FINAL RESULTS — PATIENT-LEVEL OOF FUSION "
    "+ "AND BOOTSTRAP"
)
print("=" * 70)

print(
    f"\n{'Model':<28} "
    f"{'AUROC':>7} "
    f"{'95% CI':>22} "
    f"{'BalAcc':>8} "
    f"{'F1':>7}"
)
print("-" * 75)

dur_ci = ci95(b_dur)
aco_ci = ci95(b_aco)
fus_ci = ci95(b_fus)

print(
    f"{'Duration only':<28} "
    f"{d_auc:>7.4f} "
    f"[{dur_ci[0]:.4f}, {dur_ci[1]:.4f}] "
    f"{balanced_accuracy_score(test_true, dur_pred_te):>8.4f} "
    f"{f1_score(test_true, dur_pred_te, average='macro', zero_division=0):>7.4f}"
)

aco_pred = (
    test_acoustic >= 0.5
).astype(int)

print(
    f"{'Acoustic only (D)':<28} "
    f"{a_auc:>7.4f} "
    f"[{aco_ci[0]:.4f}, {aco_ci[1]:.4f}] "
    f"{balanced_accuracy_score(test_true, aco_pred):>8.4f} "
    f"{f1_score(test_true, aco_pred, average='macro', zero_division=0):>7.4f}"
)

print(
    f"{'Fusion (Acoustic + Duration)':<28} "
    f"{fusion_auc:>7.4f} "
    f"[{fus_ci[0]:.4f}, {fus_ci[1]:.4f}] "
    f"{fusion_bal:>8.4f} "
    f"{fusion_f1:>7.4f}"
)

fd_ci = ci95(b_fd)
fa_ci = ci95(b_fa)
ad_ci = ci95(b_ad)

print("\n" + "=" * 70)
print(
    "PAIRWISE AUROC COMPARISONS "
    "(patient-level bootstrap)"
)
print("=" * 70)

print("\nFusion vs Duration:")
print(
    f"  Difference : "
    f"{fusion_auc - d_auc:+.4f}"
)
print(
    f"  95% CI     : "
    f"[{fd_ci[0]:+.4f}, {fd_ci[1]:+.4f}]"
)
print(
    f"  p          : "
    f"{pval_gt0(b_fd):.4f}"
)

print("\nFusion vs Acoustic:")
print(
    f"  Difference : "
    f"{fusion_auc - a_auc:+.4f}"
)
print(
    f"  95% CI     : "
    f"[{fa_ci[0]:+.4f}, {fa_ci[1]:+.4f}]"
)
print(
    f"  p          : "
    f"{pval_gt0(b_fa):.4f}"
)

print("\nAcoustic vs Duration:")
print(
    f"  Difference : "
    f"{a_auc - d_auc:+.4f}"
)
print(
    f"  95% CI     : "
    f"[{ad_ci[0]:+.4f}, {ad_ci[1]:+.4f}]"
)
print(
    f"  p          : "
    f"{pval_gt0(b_ad):.4f}"
)
