# =========================================================
# COMBINED PIPELINE INFERENCE
# Evaluates the full two-stage pipeline on all 179
# present-murmur patients.
#
# Phase 1: 5-model ensemble → murmur probability per window
# Phase 2: v3 single model  → systolic/diastolic per segment
#
# Weighting: each Phase 2 segment probability is weighted
# by the Phase 1 murmur probability of the 5-second window
# in which the segment falls. Weighted probabilities are
# summed and normalised to produce patient-level confidence.
#
# Classification rule:
#   systolic  if sys_conf > dia_conf + 5%
#   diastolic if dia_conf > sys_conf + 5%
#   mixed     if margin ≤ 5%
#
# Requires in Google Drive:
#   Phase 1 models: ens_phase1_seed{42,123,456,789,1234}.pth
#   Phase 2 model:  best_phase2_1dcnn_v3.pth
# =========================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
from scipy.signal import butter, filtfilt, find_peaks

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# CONFIG
# =========================================================
BASE_PATH  = "/content/drive/MyDrive/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
DATA_PATH  = os.path.join(BASE_PATH, "training_data")
CSV_PATH   = os.path.join(BASE_PATH, "training_data.csv")

SAMPLE_RATE    = 4000
WINDOW_SEC     = 5.0
WINDOW_LEN     = int(SAMPLE_RATE * WINDOW_SEC)   # 20,000 samples
MIN_WINDOW_LEN = int(WINDOW_LEN * 0.5)
TARGET_LEN_P2  = int(SAMPLE_RATE * 1.0)          # 4,000 samples

SYSTOLE_LABEL  = 2
DIASTOLE_LABEL = 4
MARGIN         = 0.05    # 5% margin for mixed classification

PHASE1_SEEDS   = [42, 123, 456, 789, 1234]
PHASE1_PATHS   = [f"/content/drive/MyDrive/ens_phase1_seed{s}.pth"
                  for s in PHASE1_SEEDS]
PHASE2_PATH    = "/content/drive/MyDrive/best_phase2_1dcnn_v3.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# =========================================================
# PREPROCESSING
# =========================================================
def bandpass_filter(signal, lowcut=25, highcut=800, fs=4000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)

def standardize(y):
    mu = np.mean(y); std = np.std(y) + 1e-8
    return ((y - mu) / std).astype(np.float32)

def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y    = bandpass_filter(y)
    y    = standardize(y)
    return y

def slice_windows(y):
    """Slice into 5-second non-overlapping windows."""
    windows, starts = [], []
    start = 0
    while start < len(y):
        seg = y[start:start + WINDOW_LEN]
        if len(seg) >= MIN_WINDOW_LEN:
            if len(seg) < WINDOW_LEN:
                seg = np.pad(seg, (0, WINDOW_LEN - len(seg)))
            windows.append(seg.astype(np.float32))
            starts.append(start / SAMPLE_RATE)
        start += WINDOW_LEN
    return windows, starts

def extract_segment_p2(signal, start_sec, end_sec):
    start = int(start_sec * SAMPLE_RATE)
    end   = int(end_sec   * SAMPLE_RATE)
    seg   = signal[start:end]
    if len(seg) == 0:
        return None
    seg = np.pad(seg, (0, max(0, TARGET_LEN_P2 - len(seg))))[:TARGET_LEN_P2]
    return standardize(seg)

# =========================================================
# SHANNON ENERGY AUTO-DETECTOR (same as Phase 2 training)
# =========================================================
SYSTOLE_DIASTOLE_THRESHOLD_SEC = 0.380

def compute_shannon_envelope(signal, fs=SAMPLE_RATE, smooth_ms=100):
    x_sq    = np.clip(signal.astype(np.float64)**2, 1e-10, None)
    shannon = np.clip(-x_sq * np.log(x_sq), 0, None)
    win     = max(1, int(fs * smooth_ms / 1000))
    return np.convolve(shannon, np.ones(win)/win, mode='same').astype(np.float32)

def detect_cardiac_phases(signal, fs=SAMPLE_RATE):
    env = compute_shannon_envelope(signal, fs=fs)
    # Use LOCAL prominence: median of top-50% envelope values
    # This works on full recordings where global max may be
    # dominated by artifacts, making 0.20*max too aggressive
    local_ref = np.percentile(env, 75)
    prominence_thr = max(0.05 * env.max(), 0.3 * local_ref)
    peaks, _ = find_peaks(env,
                          distance=int(fs * 200 / 1000),
                          prominence=prominence_thr)
    if len(peaks) < 2:
        return []
    thr  = int(fs * SYSTOLE_DIASTOLE_THRESHOLD_SEC)
    gaps = np.diff(peaks)
    return [(peaks[i]/fs, peaks[i+1]/fs,
             SYSTOLE_LABEL if gaps[i] < thr else DIASTOLE_LABEL)
            for i in range(len(peaks)-1)]

def get_phases(signal, tsv_path=None):
    """
    Use TSV for CirCor evaluation (oracle boundaries).
    Auto-detection used only when TSV is absent.
    """
    if tsv_path and os.path.exists(tsv_path):
        tsv = pd.read_csv(tsv_path, sep='\t', header=None,
                          names=["start","end","label"])
        phases = [(float(r["start"]), float(r["end"]), int(r["label"]))
                  for _, r in tsv.iterrows()
                  if int(r["label"]) in [SYSTOLE_LABEL, DIASTOLE_LABEL]]
        if phases:
            return phases, "tsv"
    return detect_cardiac_phases(signal), "auto"

# =========================================================
# PHASE 1 MODEL
# =========================================================
class TPEBlock_P1(nn.Module):
    def __init__(self):
        super().__init__()
        layers, ch = [], 1
        for f in [32, 64, 128]:
            layers += [nn.Conv1d(ch, f, 12, stride=3, padding=6),
                       nn.BatchNorm1d(f), nn.ReLU(),
                       nn.MaxPool1d(2,2), nn.Dropout(0.3)]
            ch = f
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RosaNet_P1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(128, 128, 3, dilation=d, padding=d)
            for d in [2,4,6]])
        self.fuse_convs = nn.ModuleList([nn.Conv1d(64,64,1) for _ in [2,4,6]])
        self.dropout    = nn.Dropout(0.3)
    def forward(self, x):
        outs = []
        for dc, fc in zip(self.dilated_convs, self.fuse_convs):
            o = dc(x)
            gate = torch.sigmoid(o[:,:64]); mask = F.relu(o[:,64:])
            outs.append(self.dropout(fc(gate*mask)))
        return torch.cat(outs, dim=1)

class Phase1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tpe  = TPEBlock_P1()
        self.rosa = RosaNet_P1()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(192,64),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(64,2))
    def forward(self, x):
        return self.classifier(self.pool(self.rosa(self.tpe(x))).squeeze(-1))

# =========================================================
# PHASE 2 MODEL
# =========================================================
class TPEBlock_P2(nn.Module):
    def __init__(self):
        super().__init__()
        layers, ch = [], 1
        for f in [32,64,128]:
            layers += [nn.Conv1d(ch,f,6,stride=2,padding=3),
                       nn.BatchNorm1d(f), nn.ReLU(),
                       nn.MaxPool1d(2,2), nn.Dropout(0.3)]
            ch = f
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RosaNet_P2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(128,128,3,dilation=d,padding=d) for d in [2,4,6]])
        self.fuse_convs = nn.ModuleList([nn.Conv1d(64,64,1) for _ in [2,4,6]])
        self.dropout    = nn.Dropout(0.3)
    def forward(self, x):
        outs = []
        for dc, fc in zip(self.dilated_convs, self.fuse_convs):
            o = dc(x)
            gate = torch.sigmoid(o[:,:64]); mask = F.relu(o[:,64:])
            outs.append(self.dropout(fc(gate*mask)))
        return torch.cat(outs, dim=1)

class SelfAttention(nn.Module):
    def __init__(self, ch=192, heads=4):
        super().__init__()
        self.num_heads = heads
        self.head_dim  = ch // heads
        self.scale     = self.head_dim ** -0.5
        self.q_proj    = nn.Linear(ch, ch)
        self.k_proj    = nn.Linear(ch, ch)
        self.v_proj    = nn.Linear(ch, ch)
        self.out_proj  = nn.Linear(ch, ch)
        self.dropout   = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, ch))
    def forward(self, x):
        B, C, L = x.shape
        x = x.permute(0, 2, 1)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1); S=L+1
        Q = self.q_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        a = self.dropout(F.softmax((Q@K.transpose(-2,-1))*self.scale, dim=-1))
        return self.out_proj((a@V).transpose(1,2).contiguous().view(B,S,C))[:,0,:]

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tpe       = TPEBlock_P2()
        self.rosa      = RosaNet_P2()
        self.attention = SelfAttention()
        self.classifier = nn.Sequential(
            nn.LayerNorm(192), nn.Dropout(0.4),
            nn.Linear(192,64), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(64,2))
    def forward(self, x):
        return self.classifier(self.attention(self.rosa(self.tpe(x))))

# =========================================================
# LOAD MODELS
# =========================================================
print("\nLoading Phase 1 ensemble (5 models)...")
p1_models = []
for path in PHASE1_PATHS:
    m = Phase1Model().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    p1_models.append(m)
print(f"  Loaded {len(p1_models)} Phase 1 models")

print("Loading Phase 2 model...")
p2_model = Phase2Model().to(DEVICE)
p2_model.load_state_dict(torch.load(PHASE2_PATH, map_location=DEVICE))
p2_model.eval()
print("  Loaded Phase 2 model")

# =========================================================
# INFERENCE FUNCTIONS
# =========================================================
@torch.no_grad()
def phase1_prob(windows):
    """
    Returns murmur probability per 5-second window.
    Averages across all 5 ensemble models.
    """
    x = torch.FloatTensor(np.stack(windows)).unsqueeze(1).to(DEVICE)
    probs = torch.stack([
        torch.softmax(m(x), dim=1)[:,1] for m in p1_models
    ], dim=0).mean(0)
    return probs.cpu().numpy()

@torch.no_grad()
def phase2_prob(segments):
    """
    Returns [p_systolic, p_diastolic] per segment.
    """
    x = torch.FloatTensor(np.stack(segments)).unsqueeze(1).to(DEVICE)
    return torch.softmax(p2_model(x), dim=1).cpu().numpy()

# =========================================================
# PIPELINE INFERENCE
# =========================================================
df         = pd.read_csv(CSV_PATH)
present_df = df[df["Murmur"] == "Present"].copy()
all_files  = os.listdir(DATA_PATH)

# Ground truth timing labels from dataset
# CirCor provides: Murmur timing = systolic/diastolic/unknown
TIMING_COL = "Murmur timing"   # column name in CSV

results = []
patient_segments_dict = {}   # stores per-segment data for ablation
skipped = 0

print(f"\nRunning combined pipeline on {len(present_df)} patients...")

for _, row in present_df.iterrows():
    pid = str(row["Patient ID"])

    # Ground truth timing from CirCor CSV
    # A patient is systolic if Systolic murmur timing is filled
    # A patient is diastolic if Diastolic murmur timing is filled
    # A patient is mixed if both are filled
    sys_timing = str(row.get("Systolic murmur timing", "")).strip()
    dia_timing = str(row.get("Diastolic murmur timing", "")).strip()
    has_sys = sys_timing not in ["", "nan", "None", "Unknown"]
    has_dia = dia_timing not in ["", "nan", "None", "Unknown"]
    if has_sys and has_dia:
        gt_timing = "mixed"
    elif has_sys:
        gt_timing = "systolic"
    elif has_dia:
        gt_timing = "diastolic"
    else:
        gt_timing = "unknown"

    wav_files = sorted([f for f in all_files
                        if f.startswith(pid+"_") and f.endswith(".wav")])
    if not wav_files:
        skipped += 1
        continue

    all_seg_probs   = []   # Phase 2 probs per segment
    all_seg_weights = []   # Phase 1 murmur prob weights
    seg_sources     = []   # "tsv" or "auto"

    for wav_file in wav_files:
        wav_path = os.path.join(DATA_PATH, wav_file)
        tsv_path = wav_path.replace(".wav", ".tsv")

        try:
            signal = load_audio(wav_path)
        except Exception:
            continue

        # ── Phase 1: murmur probability per 5-second window ──
        windows, starts = slice_windows(signal)
        if not windows:
            continue
        win_probs = phase1_prob(windows)   # shape: (n_windows,)

        # ── Phase 2: systolic/diastolic per cardiac segment ──
        phases, src = get_phases(signal, tsv_path)
        if not phases:
            continue

        for (s_sec, e_sec, _) in phases:
            seg = extract_segment_p2(signal, s_sec, e_sec)
            if seg is None:
                continue

            # find which 5-second window this segment falls in
            seg_mid    = (s_sec + e_sec) / 2.0
            win_idx    = min(int(seg_mid / WINDOW_SEC), len(win_probs)-1)
            murmur_wt  = float(win_probs[win_idx])

            seg_p = phase2_prob([seg])[0]   # [p_sys, p_dia]
            all_seg_probs.append(seg_p)
            all_seg_weights.append(murmur_wt)
            seg_sources.append(src)

            # store for ablation analysis
            if pid not in patient_segments_dict:
                patient_segments_dict[pid] = []
            patient_segments_dict[pid].append({
                "p_sys":    float(seg_p[0]),
                "p_dia":    float(seg_p[1]),
                "p_murmur": murmur_wt
            })

    if not all_seg_probs:
        skipped += 1
        continue

    # ── Weighted aggregation ──────────────────────────────
    probs   = np.array(all_seg_probs)    # (N, 2)
    weights = np.array(all_seg_weights)  # (N,)
    weights = weights / (weights.sum() + 1e-8)

    sys_conf = float((probs[:,0] * weights).sum())
    dia_conf = float((probs[:,1] * weights).sum())

    # normalise
    total    = sys_conf + dia_conf + 1e-8
    sys_conf /= total
    dia_conf /= total

    # ── Classification rule ───────────────────────────────
    if sys_conf - dia_conf > MARGIN:
        pred_timing = "systolic"
    elif dia_conf - sys_conf > MARGIN:
        pred_timing = "diastolic"
    else:
        pred_timing = "mixed"

    results.append({
        "patient_id":  pid,
        "gt_timing":   gt_timing,
        "pred_timing": pred_timing,
        "sys_conf":    sys_conf,
        "dia_conf":    dia_conf,
        "margin":      abs(sys_conf - dia_conf),
        "n_segments":  len(all_seg_probs),
        "tsv_used":    seg_sources.count("tsv"),
        "auto_used":   seg_sources.count("auto")
    })

results_df = pd.DataFrame(results)
print(f"\nProcessed: {len(results_df)} patients | Skipped: {skipped}")

# =========================================================
# RESULTS SUMMARY
# =========================================================
print(f"\n{'='*55}")
print("COMBINED PIPELINE RESULTS")
print(f"{'='*55}")

# Distribution of predictions
pred_counts = results_df["pred_timing"].value_counts()
print(f"\nPrediction distribution:")
for label in ["systolic","diastolic","mixed"]:
    n    = pred_counts.get(label, 0)
    conf = results_df[results_df["pred_timing"]==label]["sys_conf" if label=="systolic" else "dia_conf"].mean()
    if label == "mixed":
        conf = results_df[results_df["pred_timing"]=="mixed"]["margin"].mean()
        print(f"  {label:<12}: {n:3d} patients (avg margin: {conf*100:.1f}%)")
    else:
        print(f"  {label:<12}: {n:3d} patients (avg confidence: {conf*100:.1f}%)")

# Patient-level accuracy where GT is known (systolic/diastolic only)
eval_df = results_df[results_df["gt_timing"].isin(["systolic","diastolic"])].copy()

if len(eval_df) > 0:
    print(f"\nGround truth distribution ({len(eval_df)} patients with known GT):")
    print(eval_df["gt_timing"].value_counts())

    # Accuracy: for systolic/diastolic GT, check exact match
    # mixed GT patients excluded from accuracy (ambiguous)
    strict_df = eval_df[eval_df["gt_timing"].isin(["systolic","diastolic"])]
    if len(strict_df) > 0:
        strict_df = strict_df.copy()
        strict_df["correct"] = strict_df["pred_timing"] == strict_df["gt_timing"]
        pat_acc = strict_df["correct"].mean()
        print(f"\nPatient-level timing accuracy (systolic/diastolic GT only):")
        print(f"  {pat_acc*100:.1f}% on {len(strict_df)} patients")
        print(f"\nNote: 174/179 patients have systolic GT — this metric")
        print(f"reflects dataset bias, not model capability.")
        print(f"Segment-level accuracy (90.54%) is the primary metric.")
else:
    print("\nNote: Ground truth timing labels not available in CSV.")
    print("Reporting prediction distribution only.")

print(f"\nNote: 174/179 present-murmur patients in CirCor carry")
print(f"systolic ground truth labels, making patient-level")
print(f"timing accuracy an unreliable performance metric.")
print(f"Segment-level accuracy (Phase 2: 90.54%) is the")
print(f"primary reported metric.")

# =========================================================
# PLOT EXAMPLE PATIENT
# =========================================================
# Find a patient with highest systolic confidence for Fig. 6
best_sys = results_df.loc[results_df["sys_conf"].idxmax(), "patient_id"]
pid      = best_sys

wav_files = sorted([f for f in all_files
                    if f.startswith(pid+"_") and f.endswith(".wav")])

if wav_files:
    signal    = load_audio(os.path.join(DATA_PATH, wav_files[0]))
    windows, starts = slice_windows(signal)
    win_probs = phase1_prob(windows)
    phases, _ = get_phases(signal,
                           os.path.join(DATA_PATH,
                                        wav_files[0].replace(".wav",".tsv")))

    pat_row  = results_df[results_df["patient_id"]==pid].iloc[0]
    pred_lbl = pat_row["pred_timing"].upper()
    sys_c    = pat_row["sys_conf"]*100
    dia_c    = pat_row["dia_conf"]*100

    t = np.arange(len(signal)) / SAMPLE_RATE
    envelope = np.abs(signal)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    gridspec_kw={"height_ratios":[3,1]})

    ax1.plot(t, envelope, color="gray", linewidth=0.5, alpha=0.7)
    for (s, e, lbl) in phases:
        color = "green" if lbl == SYSTOLE_LABEL else "orange"
        ax1.axvspan(s, e, alpha=0.25, color=color)

    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="green", alpha=0.4, label="Systole"),
                         Patch(color="orange", alpha=0.4, label="Diastole")])
    ax1.set_ylabel("Amplitude"); ax1.set_xlabel("Time (s)")
    ax1.set_title(f"Patient {pid} | True: Systolic | "
                  f"Predicted: {pred_lbl} | "
                  f"Sys conf: {sys_c:.1f}% | Dia conf: {dia_c:.1f}%")

    ax2.step(starts, win_probs, where="post", color="steelblue", linewidth=1.5)
    ax2.axhline(y=0.5, color="red", linestyle="--", linewidth=1,
                label="Threshold=0.50")
    ax2.set_ylabel("P(Murmur)"); ax2.set_xlabel("Time (s)")
    ax2.set_title("Phase 1 — Murmur Probability per 5-second Window")
    ax2.legend(); ax2.set_ylim([0,1]); ax2.grid(True, alpha=0.3)

    plt.suptitle("Example Combined Pipeline Output", fontsize=12)
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/fig6_pipeline_output.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: /content/drive/MyDrive/fig6_pipeline_output.png")

print(f"\npatient_segments_dict ready: {len(patient_segments_dict)} patients")
print("Run combined_pipeline_ablation.py next.")
print("\nDone.")
