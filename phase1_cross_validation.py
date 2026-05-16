
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)

# ── config ───────────────────────────────────────────────
SAMPLE_RATE  = 4000
WINDOW_LEN   = int(SAMPLE_RATE * 5.0)
BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LR           = 1e-4
PATIENCE     = 15
N_SPLITS     = 5
MODEL_SEED   = 42          # fixed model init for reproducibility
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print(f"5-fold StratifiedGroupKFold CV | Model seed: {MODEL_SEED}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ── dataset ───────────────────────────────────────────────
class PCGRawDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        y     = row["audio"].copy()
        if self.augment:
            if np.random.rand() < 0.4:
                y += np.random.randn(len(y)).astype(np.float32) * 0.005
            if np.random.rand() < 0.4:
                y = (y * np.random.uniform(0.85, 1.15)).astype(np.float32)
            if np.random.rand() < 0.4:
                shift = int(len(y) * np.random.uniform(-0.08, 0.08))
                y = np.roll(y, shift).astype(np.float32)
            mu = np.mean(y); std = np.std(y) + 1e-8
            y  = ((y - mu) / std).astype(np.float32)
        return torch.FloatTensor(y).unsqueeze(0), label

# ── model (identical to fixed-split version) ──────────────
class TPEBlock(nn.Module):
    def __init__(self, in_channels=1, filters=(32,64,128),
                 kernel=12, stride=3, dropout=0.3):
        super().__init__()
        layers, ch = [], in_channels
        for f in filters:
            layers += [nn.Conv1d(ch, f, kernel_size=kernel,
                                 stride=stride, padding=kernel//2),
                       nn.BatchNorm1d(f), nn.ReLU(),
                       nn.MaxPool1d(2,2), nn.Dropout(dropout)]
            ch = f
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RosaNetBlock(nn.Module):
    def __init__(self, in_channels=128, hidden=64, dropout=0.3):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(in_channels, hidden*2, kernel_size=3,
                      dilation=d, padding=d) for d in [2,4,6]])
        self.fuse_convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 1) for _ in [2,4,6]])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        outs = []
        for dc, fc in zip(self.dilated_convs, self.fuse_convs):
            o = dc(x)
            gate = torch.sigmoid(o[:, :o.shape[1]//2])
            mask = F.relu(o[:, o.shape[1]//2:])
            outs.append(self.dropout(fc(gate * mask)))
        return torch.cat(outs, dim=1)

class MorshedInspired1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tpe  = TPEBlock()
        self.rosa = RosaNetBlock()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(192, 64),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 2))
    def forward(self, x):
        return self.classifier(self.pool(self.rosa(self.tpe(x))).squeeze(-1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
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
def get_probs(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    for inputs, labels in loader:
        p = torch.softmax(model(inputs.to(DEVICE)), dim=1)[:, 1]
        probs_all.extend(p.cpu().numpy())
        labels_all.extend(labels.numpy())
    return np.array(probs_all), np.array(labels_all)

def sweep_threshold(true_labels, probs):
    best_f, best_t = -1, 0.5
    for thr in np.arange(0.10, 0.91, 0.01):
        pred = (probs >= thr).astype(int)
        f    = f1_score(true_labels, pred, zero_division=0)
        if f > best_f:
            best_f, best_t = f, thr
    return best_t, best_f

# =========================================================
# 5-FOLD CROSS-VALIDATION
# data_df and patient_df must be in scope from Phase 1 setup
# =========================================================
print("\nBuilding patient-level arrays for CV...")

# patient-level arrays
pat_ids    = patient_df["patient_id"].values
pat_labels = patient_df["label"].values

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=42)

fold_results     = []
all_test_true    = []   # pooled across folds for bootstrap CI
all_test_probs   = []
all_test_pids    = []

print(f"\n{'='*55}")
print(f"5-FOLD STRATIFIED GROUP CROSS-VALIDATION")
print(f"{'='*55}\n")

for fold_idx, (trainval_idx, test_idx) in enumerate(
        sgkf.split(pat_ids, pat_labels, groups=pat_ids)):

    print(f"─── FOLD {fold_idx+1}/5 ───────────────────────────────")

    test_pats    = set(pat_ids[test_idx])
    trainval_ids = pat_ids[trainval_idx]
    trainval_lbl = pat_labels[trainval_idx]

    # inner split: 80% train / 20% val from trainval
    inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                       random_state=42 + fold_idx)
    tr_idx_inner, va_idx_inner = next(
        inner_sgkf.split(trainval_ids, trainval_lbl,
                          groups=trainval_ids))

    train_pats = set(trainval_ids[tr_idx_inner])
    val_pats   = set(trainval_ids[va_idx_inner])

    train_df_f = data_df[data_df["patient_id"].isin(train_pats)].reset_index(drop=True)
    val_df_f   = data_df[data_df["patient_id"].isin(val_pats)].reset_index(drop=True)
    test_df_f  = data_df[data_df["patient_id"].isin(test_pats)].reset_index(drop=True)

    print(f"  Train: {len(train_pats)} | Val: {len(val_pats)} | "
          f"Test: {len(test_pats)} patients")

    # dataloaders
    train_ds = PCGRawDataset(train_df_f, augment=True)
    val_ds   = PCGRawDataset(val_df_f,   augment=False)
    test_ds  = PCGRawDataset(test_df_f,  augment=False)

    lbl_s  = train_df_f["label"].values
    cnt_s  = np.bincount(lbl_s)
    wts_s  = 1.0 / cnt_s[lbl_s]
    samp_s = WeightedRandomSampler(torch.DoubleTensor(wts_s),
                                    len(wts_s), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=samp_s, num_workers=2,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=True)

    # train
    seed_everything(MODEL_SEED)
    model     = MorshedInspired1DCNN().to(DEVICE)
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_vf1  = -1
    best_path = (f"/content/drive/MyDrive/"
                 f"cv_phase1_fold{fold_idx+1}.pth")
    no_imp    = 0

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        vp, vl = get_probs(model, val_loader)
        vdf    = val_df_f.copy(); vdf["prob"] = vp
        vpp    = vdf.groupby("patient_id")["prob"].mean()
        vpl    = vdf.groupby("patient_id")["label"].first()
        vr     = vpp.to_frame().join(vpl)
        _, val_f1 = sweep_threshold(vr["label"].values,
                                     vr["prob"].values)

        if val_f1 > best_vf1:
            best_vf1 = val_f1
            torch.save(model.state_dict(), best_path)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # evaluate on test fold
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # threshold from val
    vp_f, _ = get_probs(model, val_loader)
    vdf_f   = val_df_f.copy(); vdf_f["prob"] = vp_f
    vpp_f   = vdf_f.groupby("patient_id")["prob"].mean()
    vpl_f   = vdf_f.groupby("patient_id")["label"].first()
    vr_f    = vpp_f.to_frame().join(vpl_f)
    best_t, _ = sweep_threshold(vr_f["label"].values,
                                 vr_f["prob"].values)

    # test predictions
    tp, _  = get_probs(model, test_loader)
    tdf    = test_df_f.copy(); tdf["prob"] = tp
    tpp    = tdf.groupby("patient_id")["prob"].mean()
    tpl    = tdf.groupby("patient_id")["label"].first()
    tr     = tpp.to_frame().join(tpl)

    pat_probs = tr["prob"].values
    pat_true  = tr["label"].values
    pat_pids  = tr.index.values
    pat_preds = (pat_probs >= best_t).astype(int)

    acc_f = accuracy_score(pat_true, pat_preds)
    f1_f  = f1_score(pat_true, pat_preds, zero_division=0)

    print(f"  Fold {fold_idx+1} | Acc: {acc_f:.4f} | "
          f"F1: {f1_f:.4f} | Thr: {best_t:.2f} | "
          f"ValF1: {best_vf1:.4f}")

    fold_results.append({"fold": fold_idx+1, "acc": acc_f,
                          "f1": f1_f, "thr": best_t})
    all_test_true.extend(pat_true.tolist())
    all_test_probs.extend(pat_probs.tolist())
    all_test_pids.extend(pat_pids.tolist())

# =========================================================
# AGGREGATE RESULTS
# =========================================================
all_test_true  = np.array(all_test_true)
all_test_probs = np.array(all_test_probs)



# per-fold stats
accs = [r["acc"] for r in fold_results]
f1s  = [r["f1"]  for r in fold_results]
mean_acc = np.mean(accs); std_acc = np.std(accs, ddof=1)
mean_f1  = np.mean(f1s);  std_f1  = np.std(f1s,  ddof=1)
n        = len(accs)
t95      = scipy_stats.t.ppf(0.975, df=n-1)
ci_acc   = t95 * std_acc / np.sqrt(n)

# ── BOOTSTRAP CI ─────────────────────────────────────────
print("\nComputing bootstrap CI (2000 resamples)...")
rng = np.random.default_rng(42)
boot_accs, boot_f1s = [], []
for _ in range(2000):
    idx  = rng.choice(len(all_test_true), len(all_test_true),
                       replace=True)
    pred = (all_test_probs[idx] >= 0.50).astype(int)
    boot_accs.append(accuracy_score(all_test_true[idx], pred))
    boot_f1s.append(f1_score(all_test_true[idx], pred,
                              zero_division=0))
boot_acc_ci = np.percentile(boot_accs, [2.5, 97.5])
boot_f1_ci  = np.percentile(boot_f1s,  [2.5, 97.5])



# =========================================================
# FINAL REPORT
# =========================================================
print(f"\n{'='*55}")
print("5-FOLD CV RESULTS SUMMARY")
print(f"{'='*55}")
print(f"\nPer-fold results:")
for r in fold_results:
    print(f"  Fold {r['fold']} | Acc: {r['acc']:.4f} | "
          f"F1: {r['f1']:.4f} | Thr: {r['thr']:.2f}")

print(f"\nCross-validated mean ± std:")
print(f"  Accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"  t-CI 95% : [{(mean_acc-ci_acc)*100:.2f}%, "
      f"{(mean_acc+ci_acc)*100:.2f}%]")
print(f"  F1       : {mean_f1:.4f} ± {std_f1:.4f}")


print(f"\nConfusion Matrix (pooled):")
print(confusion_matrix(all_test_true, pooled_preds))
print(f"\nClassification Report (pooled):")
print(classification_report(all_test_true, pooled_preds,
      target_names=["Absent","Present"], digits=4))

print(f"""

