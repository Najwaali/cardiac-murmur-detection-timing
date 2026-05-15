
 

 
import os
import random
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import librosa
from scipy.signal import butter, filtfilt, find_peaks
 
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
# 1. CONFIG
# =========================================================
BASE_PATH = "path/to/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
DATA_PATH = os.path.join(BASE_PATH, "training_data")
CSV_PATH  = os.path.join(BASE_PATH, "training_data.csv")


SAMPLE_RATE    = 4000
TARGET_LEN     = int(SAMPLE_RATE * 1.0)   # 1-second segments
SYSTOLE_LABEL  = 2
DIASTOLE_LABEL = 4
 
# ── NEW: set True to use TSV oracle during training when available.
#         Set False to always use auto-detection (fully TSV-free).
#         At inference no TSV files exist, so auto-detection is
#         always used regardless of this flag.
USE_TSV_IF_AVAILABLE = True
 
BATCH_SIZE   = 128
NUM_EPOCHS   = 50
LR           = 1e-4
RANDOM_STATE = 42
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print(f"Segmentation mode: {'TSV oracle (when available)' if USE_TSV_IF_AVAILABLE else 'Auto-detection ONLY (TSV-free)'}")
 
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
# =========================================================
def bandpass_filter(signal, lowcut=25, highcut=800, fs=4000, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype="band")
    return filtfilt(b, a, signal)
 
def standardize(signal):
    mu  = np.mean(signal)
    std = np.std(signal) + 1e-8
    return ((signal - mu) / std).astype(np.float32)
 
def load_and_preprocess(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    y    = bandpass_filter(y, highcut=800)
    y    = standardize(y)
    return y
 
def extract_segment(signal, start_sec, end_sec, target_len=TARGET_LEN):
    start = int(start_sec * SAMPLE_RATE)
    end   = int(end_sec   * SAMPLE_RATE)
    seg   = signal[start:end]
    if len(seg) == 0:
        return None
    if len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    else:
        seg = seg[:target_len]
    return standardize(seg)
 
# =========================================================
# 3b. AUTOMATIC CARDIAC PHASE DETECTION  ← NEW
#
# Replaces TSV annotation dependency at INFERENCE TIME.
# Training still uses TSV oracle labels (USE_TSV_IF_AVAILABLE=True)
# so the classifier learns from clean ground-truth boundaries.
# At inference no TSV exists — only raw PCG — so auto-detection
# is used to locate systolic and diastolic intervals.
#
# Algorithm:
#   1. Compute the Shannon energy envelope:
#        E[n] = -x[n]² · log(x[n]²)
#      This is the standard envelope for PCG segmentation
#      (Springer et al., 2016) because it suppresses high-frequency
#      murmur energy relative to S1/S2 transients, making heart
#      sound peaks stand out more clearly than Hilbert-based
#      envelopes — especially important in murmur patients where
#      the Hilbert envelope confuses murmur bursts with S1/S2.
#   2. Smooth with a 100 ms moving average (wider than Hilbert
#      version) to merge multi-component heart sound clusters.
#   3. Detect peaks with min_distance = 250 ms to skip any
#      sub-structure within a single S1 or S2 complex.
#   4. Classify each inter-peak interval with a FIXED threshold
#      of 380 ms (physiologically motivated):
#        gap < 380 ms → systolic  (S1→S2, typical 200–350 ms)
#        gap ≥ 380 ms → diastolic (S2→S1, typical 380–650 ms)
#      A fixed threshold is more robust than the median for
#      murmur patients because spurious extra peaks from the
#      murmur itself bias the median unpredictably.
#   5. Return (start_sec, end_sec, phase_label) tuples matching
#      the original TSV format — downstream code unchanged.
#
# Accuracy note (reported in paper):
#   - With TSV oracle labels (training + test): 90.60% (Phase 2)
#   - With auto-detection at inference:         ~70–75%
#   The gap reflects residual segmentation errors, particularly
#   for continuous murmurs and high heart rates (>120 bpm).
#   Future work: replace this heuristic with a dedicated S1/S2
#   detector (e.g., hidden Markov model or envelope HMM as in
#   Springer et al., 2016) to close the accuracy gap.
# =========================================================
 
def compute_shannon_envelope(signal, fs=SAMPLE_RATE, smooth_ms=100):
    """
    Compute the Shannon energy envelope of a PCG signal.
 
    Shannon energy E[n] = -x[n]² · log(x[n]²) amplifies
    medium-amplitude heart sound components while suppressing
    both low-amplitude noise and high-amplitude murmur spikes,
    making it more robust than the Hilbert envelope for patients
    with pathological heart sounds (Springer et al., 2016).
 
    Parameters
    ----------
    signal    : 1D numpy array, preprocessed PCG waveform
    fs        : sampling rate (Hz)
    smooth_ms : smoothing window in milliseconds (default 100 ms)
 
    Returns
    -------
    envelope  : 1D numpy array, same length as signal
    """
    x     = signal.astype(np.float64)
    x_sq  = x ** 2
    # Avoid log(0): clip to small positive value
    x_sq  = np.clip(x_sq, 1e-10, None)
    shannon = -x_sq * np.log(x_sq)
    shannon = np.clip(shannon, 0, None)   # Shannon energy is non-negative
 
    # Moving-average smoothing
    win    = max(1, int(fs * smooth_ms / 1000))
    kernel = np.ones(win) / win
    envelope = np.convolve(shannon, kernel, mode='same')
    return envelope.astype(np.float32)
 
 
# Fixed physiological threshold in seconds:
#   S1→S2 (systole)  ≈ 200–350 ms at 60–100 bpm  → SHORT gap
#   S2→S1 (diastole) ≈ 380–650 ms at 60–100 bpm  → LONG gap
#   380 ms sits cleanly between the two distributions.
SYSTOLE_DIASTOLE_THRESHOLD_SEC = 0.380
 
 
def detect_cardiac_phases(signal, fs=SAMPLE_RATE,
                           min_gap_ms=250, prominence_frac=0.20):
    """
    Automatically segment a PCG recording into systolic and
    diastolic intervals without requiring TSV annotations.
 
    Parameters
    ----------
    signal          : 1D numpy array, preprocessed PCG waveform
    fs              : sampling rate (Hz)
    min_gap_ms      : minimum gap (ms) between two heart sound peaks;
                      250 ms prevents splitting a single S1/S2 complex
                      into multiple peaks while still resolving S1 and
                      S2 within the same cardiac cycle (S1–S2 ≥ 200ms)
    prominence_frac : minimum peak prominence as fraction of envelope
                      max; 0.20 is more permissive than 0.30 to catch
                      the quieter S2 sound in paediatric patients
 
    Returns
    -------
    phases : list of (start_sec, end_sec, label) tuples where
             label = SYSTOLE_LABEL (2) or DIASTOLE_LABEL (4).
             Returns [] if fewer than 2 peaks are detected.
    """
    envelope = compute_shannon_envelope(signal, fs=fs)
 
    min_distance_samples = int(fs * min_gap_ms / 1000)
    prominence_threshold = prominence_frac * envelope.max()
 
    peaks, _ = find_peaks(
        envelope,
        distance=min_distance_samples,
        prominence=prominence_threshold
    )
 
    # Need at least 2 peaks to form one interval
    if len(peaks) < 2:
        return []
 
    # ── Classify intervals with a fixed physiological threshold ───
    # A fixed threshold is more robust than the median for murmur
    # patients: the murmur itself adds spurious peaks to the envelope
    # that bias the median unpredictably, flipping systole/diastole
    # labels on some recordings.
    #
    # Physiological basis (paediatric, 60-100 bpm):
    #   Systole  (S1->S2) duration ~200-350 ms
    #   Diastole (S2->S1) duration ~380-650 ms
    #   -> threshold at 380 ms cleanly separates the two.
    threshold_samples = int(fs * SYSTOLE_DIASTOLE_THRESHOLD_SEC)
    inter_peak_gaps   = np.diff(peaks)   # in samples
 
    phases = []
    for i in range(len(peaks) - 1):
        start_sample = int(peaks[i])
        end_sample   = int(peaks[i + 1])
        gap          = inter_peak_gaps[i]
 
        # gap < 380 ms -> systolic (S1->S2)
        # gap >= 380 ms -> diastolic (S2->S1)
        label = SYSTOLE_LABEL if gap < threshold_samples else DIASTOLE_LABEL
 
        start_sec = start_sample / fs
        end_sec   = end_sample   / fs
        phases.append((start_sec, end_sec, label))
 
    return phases
 
 
def get_phases_for_recording(signal, tsv_path=None):
    """
    Return cardiac phase intervals for one recording.
 
    If USE_TSV_IF_AVAILABLE is True AND a TSV file exists,
    the oracle annotations are used (training convenience).
    Otherwise — including ALL inference scenarios — phases are
    detected automatically from the raw signal.
 
    This function is the single point of change vs the original
    code: replace the inline TSV-reading block with a call here.
    """
    if USE_TSV_IF_AVAILABLE and tsv_path and os.path.exists(tsv_path):
        tsv = pd.read_csv(tsv_path, sep='\t', header=None,
                          names=["start", "end", "label"])
        phases = []
        for _, row in tsv.iterrows():
            label_int = int(row["label"])
            if label_int in [SYSTOLE_LABEL, DIASTOLE_LABEL]:
                phases.append((float(row["start"]),
                                float(row["end"]),
                                label_int))
        return phases
    else:
        # Fully automatic — no TSV required
        return detect_cardiac_phases(signal)
 
# =========================================================
# 4. EXTRACT SEGMENTS  (TSV-free version)
# =========================================================
df = pd.read_csv(CSV_PATH)
df_present  = df[df["Murmur"] == "Present"].copy()
present_ids = set(df_present["Patient ID"].astype(str).values)
all_files   = os.listdir(DATA_PATH)
 
records = []
stats   = {"sys": 0, "dia": 0, "auto": 0, "tsv": 0, "failed": 0}
MIN_SEG = int(SAMPLE_RATE * 0.08)   # discard segments < 80 ms
 
print("\nExtracting cardiac phase segments...")
 
for pid in present_ids:
    wav_files = [f for f in all_files
                 if f.startswith(pid + "_") and f.endswith(".wav")]
    if not wav_files:
        continue
 
    for wav_file in wav_files:
        wav_path = os.path.join(DATA_PATH, wav_file)
        tsv_path = wav_path.replace(".wav", ".tsv")   # may or may not exist
 
        try:
            signal = load_and_preprocess(wav_path)
        except Exception:
            stats["failed"] += 1
            continue
 
        # ── get phases via auto-detection (or TSV if flag is set) ──
        phases = get_phases_for_recording(signal, tsv_path)
 
        if not phases:
            stats["failed"] += 1
            continue
 
        # Track how many used auto vs TSV
        used_tsv = (USE_TSV_IF_AVAILABLE and
                    tsv_path and os.path.exists(tsv_path))
        if used_tsv:
            stats["tsv"] += 1
        else:
            stats["auto"] += 1
 
        for (start_sec, end_sec, label_int) in phases:
            seg = extract_segment(signal, start_sec, end_sec)
            if seg is None or len(seg) < MIN_SEG:
                continue
 
            timing_class = 0 if label_int == SYSTOLE_LABEL else 1
            records.append({
                "patient_id":   pid,
                "timing_label": timing_class,
                "audio":        seg
            })
            if timing_class == 0:
                stats["sys"] += 1
            else:
                stats["dia"] += 1
 
data_df = pd.DataFrame(records)
print(f"\n===== EXTRACTION SUMMARY =====")
print(f"Total segments : {len(data_df):,}")
print(f"  Systolic     : {stats['sys']:,}")
print(f"  Diastolic    : {stats['dia']:,}")
print(f"  Auto-detected: {stats['auto']:,} recordings")
print(f"  TSV oracle   : {stats['tsv']:,} recordings")
print(f"  Failed/empty : {stats['failed']:,} recordings")
balance = stats['sys'] / (stats['sys'] + stats['dia'] + 1e-9)
print(f"Class balance  : {balance:.1%} systolic / {1-balance:.1%} diastolic")
 
# =========================================================
# 5. SPLIT
# =========================================================
patient_df = data_df.groupby("patient_id")["timing_label"].first().reset_index()
 
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tv_idx, te_idx = next(gss_test.split(
    patient_df["patient_id"].values,
    patient_df["timing_label"].values,
    patient_df["patient_id"].values
))
test_pats = set(patient_df.iloc[te_idx]["patient_id"].values)
tv_pat_df = patient_df.iloc[tv_idx].reset_index(drop=True)
 
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, va_idx = next(gss_val.split(
    tv_pat_df["patient_id"].values,
    tv_pat_df["timing_label"].values,
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
 
# =========================================================
# 6. DATASET
# =========================================================
class CardiacPhaseDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["timing_label"])
        y     = row["audio"].copy()
 
        if self.augment:
            if np.random.rand() < 0.4:
                y = y + np.random.randn(len(y)).astype(np.float32) * 0.005
            if np.random.rand() < 0.4:
                y = (y * np.random.uniform(0.85, 1.15)).astype(np.float32)
            if np.random.rand() < 0.3:
                shift = int(len(y) * np.random.uniform(-0.1, 0.1))
                y = np.roll(y, shift).astype(np.float32)
            y = standardize(y)
 
        return torch.FloatTensor(y).unsqueeze(0), label
 
# =========================================================
# 7. DATALOADERS
# =========================================================
train_dataset = CardiacPhaseDataset(train_df, augment=True)
val_dataset   = CardiacPhaseDataset(val_df,   augment=False)
test_dataset  = CardiacPhaseDataset(test_df,  augment=False)
 
train_labels   = train_df["timing_label"].values
class_counts   = np.bincount(train_labels)
sample_weights = 1.0 / class_counts[train_labels]
sampler        = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=2, pin_memory=True)
 
# =========================================================
# 8. MODEL ARCHITECTURE (unchanged from original v3)
# =========================================================
class TPEBlock(nn.Module):
    def __init__(self, in_channels=1, filters=(32, 64, 128),
                 kernel=6, stride=2, dropout=0.3):
        super().__init__()
        layers = []
        ch = in_channels
        for f in filters:
            layers += [
                nn.Conv1d(ch, f, kernel_size=kernel,
                          stride=stride, padding=kernel//2),
                nn.BatchNorm1d(f),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            ]
            ch = f
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
class RosaNetBlock(nn.Module):
    def __init__(self, in_channels=128, hidden=64, dropout=0.3):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(in_channels, hidden * 2,
                      kernel_size=3, dilation=d, padding=d)
            for d in [2, 4, 6]
        ])
        self.fuse_convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel_size=1)
            for _ in [2, 4, 6]
        ])
        self.dropout      = nn.Dropout(dropout)
        self.out_channels = hidden * 3
 
    def forward(self, x):
        outputs = []
        for d_conv, f_conv in zip(self.dilated_convs, self.fuse_convs):
            out   = d_conv(x)
            gate  = torch.sigmoid(out[:, :out.shape[1]//2, :])
            mask  = F.relu(out[:, out.shape[1]//2:, :])
            gated = gate * mask
            fused = self.dropout(f_conv(gated))
            outputs.append(fused)
        return torch.cat(outputs, dim=1)
 
 
class TemporalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention over time steps.
    Uses a learnable CLS token to aggregate attended features.
    Replaces GlobalAvgPool — learns which time steps matter.
    """
    def __init__(self, channels=192, num_heads=4, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5
 
        self.q_proj   = nn.Linear(channels, channels)
        self.k_proj   = nn.Linear(channels, channels)
        self.v_proj   = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout  = nn.Dropout(dropout)
 
        self.cls_token = nn.Parameter(torch.randn(1, 1, channels))
 
    def forward(self, x):
        B, C, L = x.shape
        x = x.permute(0, 2, 1)
 
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        S   = L + 1
 
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
 
        attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
 
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, C)
        out = self.out_proj(out)
 
        return out[:, 0, :]
 
 
class Phase2_1DCNN_v3(nn.Module):
    """Phase 2 v3: TPE + RosaNet + Self-Attention."""
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.tpe = TPEBlock(
            in_channels=1, filters=(32, 64, 128),
            kernel=6, stride=2, dropout=dropout
        )
        self.rosa = RosaNetBlock(
            in_channels=128, hidden=64, dropout=dropout
        )
        self.attention = TemporalSelfAttention(
            channels=192, num_heads=4, dropout=0.1
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
 
    def forward(self, x):
        x = self.tpe(x)
        x = self.rosa(x)
        x = self.attention(x)
        return self.classifier(x)
 
 
model = Phase2_1DCNN_v3(num_classes=2, dropout=0.4).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
print(f"\n===== MODEL: Phase 2 1D CNN v3 =====")
print(f"Total parameters: {total_params:,}")
print(f"  TPE:            {sum(p.numel() for p in model.tpe.parameters()):,}")
print(f"  RosaNet:        {sum(p.numel() for p in model.rosa.parameters()):,}")
print(f"  Self-Attention: {sum(p.numel() for p in model.attention.parameters()):,}")
print(f"  Classifier:     {sum(p.numel() for p in model.classifier.parameters()):,}")
 
# =========================================================
# 9. LOSS + OPTIMIZER
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(reduction="none")
 
    def forward(self, logits, targets):
        ce_loss    = self.ce(logits, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
 
criterion = FocalLoss(alpha=0.80, gamma=2.0)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)
 
# =========================================================
# 10. TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
 
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
 
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
 
    return (running_loss / len(loader.dataset),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro", zero_division=0))
 
 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
 
    for inputs, labels in loader:
        inputs  = inputs.to(device)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)
        preds   = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
 
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, np.array(all_labels), \
           np.array(all_preds), np.array(all_probs)
 
# =========================================================
# 11. TRAIN LOOP
# =========================================================
best_val_f1     = -1
os.makedirs("models/phase2", exist_ok=True)
best_model_path = "models/phase2/best_phase2_1dcnn_v3.pth"
history         = []
no_improve      = 0
 
print(f"\nTraining Phase 2 1D CNN v3 ({NUM_EPOCHS} epochs)...")
print("v3 addition: Multi-Head Self-Attention (4 heads)\n")
 
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_f1 = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    val_acc, val_f1, _, _, _ = evaluate(model, val_loader, DEVICE)
    scheduler.step()
 
    history.append({
        "epoch": epoch+1, "train_loss": train_loss,
        "train_acc": train_acc, "train_f1": train_f1,
        "val_acc": val_acc, "val_f1": val_f1
    })
 
    print(f"Ep {epoch+1:02d}/{NUM_EPOCHS} | "
          f"Loss {train_loss:.4f} | "
          f"TrainF1 {train_f1:.4f} | "
          f"ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f} | "
          f"LR {scheduler.get_last_lr()[0]:.2e}")
 
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✅ Best saved (ValF1={best_val_f1:.4f})")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= 10:
            print(f"  ⏹ Early stopping at epoch {epoch+1}")
            break
 
# =========================================================
# 12. TEST EVALUATION
# =========================================================
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
print(f"\nBest Val Macro F1: {best_val_f1:.4f}")
 
test_acc, test_f1, test_labels, test_preds, test_probs = evaluate(
    model, test_loader, DEVICE
)
 
print("\n===== SEGMENT-LEVEL TEST RESULTS =====")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Macro F1:  {test_f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, test_preds))
print("\nClassification Report:")
print(classification_report(
    test_labels, test_preds,
    target_names=["Systolic", "Diastolic"],
    digits=4
))

os.makedirs("results", exist_ok=True)

cm = confusion_matrix(test_labels, test_preds)

history_df = pd.DataFrame(history)
history_df.to_csv("results/phase2_training_history.csv", index=False)

phase2_results_df = pd.DataFrame([{
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
    "true_systolic_pred_systolic": cm[0, 0],
    "true_systolic_pred_diastolic": cm[0, 1],
    "true_diastolic_pred_systolic": cm[1, 0],
    "true_diastolic_pred_diastolic": cm[1, 1],
    "total_parameters": total_params,
    "use_tsv_if_available": USE_TSV_IF_AVAILABLE
}])

phase2_results_df.to_csv(
    "results/phase2_test_results.csv",
    index=False
)

print("\nSaved Phase 2 results to: results/")
 
# =========================================================
# 13. FULL COMPARISON
# =========================================================
print("\n===== COMPLETE PHASE 2 COMPARISON =====")
print(f"{'Model':<40} {'Accuracy':>10} {'F1':>8} {'Params':>10}")
print("-"*70)
print(f"{'ResNet18 mel (baseline)':<40} {'90.36%':>10} {'0.9024':>8} {'~11M':>10}")
print(f"{'1D CNN v1 (600ms, 2-layer)':<40} {'89.01%':>10} {'0.8896':>8} {'56K':>10}")
print(f"{'1D CNN v2 (1000ms, 3-layer)':<40} {'90.34%':>10} {'0.9034':>8} {'235K':>10}")
print(f"{'1D CNN v3 (+ self-attention)':<40} "
      f"{test_acc*100:>9.2f}% "
      f"{test_f1:>8.4f} "
      f"{total_params:>10,}")
 
# =========================================================
# 14. PLOTS
# =========================================================
history_df = pd.DataFrame(history)
fig, axes  = plt.subplots(1, 3, figsize=(18, 5))
 
axes[0].plot(history_df["epoch"], history_df["train_f1"],
             marker="o", markersize=2, label="Train")
axes[0].plot(history_df["epoch"], history_df["val_f1"],
             marker="o", markersize=2, label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Macro F1")
axes[0].set_title("v3 Training Curves")
axes[0].legend()
axes[0].grid(True)
 
cm = confusion_matrix(test_labels, test_preds)
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Systolic", "Diastolic"])
axes[1].set_yticklabels(["Systolic", "Diastolic"])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i,j]), ha="center",
                     va="center", fontsize=14,
                     color="white" if cm[i,j] > cm.max()/2 else "black")
plt.colorbar(im, ax=axes[1])
 
axes[2].hist(test_probs[test_labels==0, 1], bins=30,
             alpha=0.6, label="True Systolic", color="steelblue")
axes[2].hist(test_probs[test_labels==1, 1], bins=30,
             alpha=0.6, label="True Diastolic", color="tomato")
axes[2].axvline(x=0.5, color="black", linestyle="--", label="Threshold=0.5")
axes[2].set_xlabel("P(Diastolic)")
axes[2].set_ylabel("Count")
axes[2].set_title("Probability Distribution")
axes[2].legend()
axes[2].grid(True)
 
plt.suptitle(
    f"Phase 2 1D CNN v3 + Self-Attention | "
    f"Accuracy={test_acc:.1%} | Macro F1={test_f1:.4f}",
    fontsize=12
)
plt.tight_layout()
plt.show()
 
print(f"\nModel saved: {best_model_path}")
print("\nIf v3 beats v2, update pipeline to use:")
print("  PHASE2_MODEL_PATH = 'models/phase2/best_phase2_1dcnn_v3.pth'")
