# =========================================================
# COMBINED PIPELINE — AGGREGATION ABLATION + IMPROVED TIMING
# For Q1 journal submission
#
# WHAT THIS ADDS:
#   1. Murmur-positive filtering: only segments where
#      Phase 1 probability >= threshold contribute to timing
#   2. Confidence filtering: only high-confidence Phase 2
#      predictions are counted
#   3. Ablation table: shows how counts change under
#      different aggregation strategies
#
# Run AFTER combined_pipeline_inference.py has run and
# results_df is in scope. OR re-run inference here.
# =========================================================

# patient_segments_dict is built by combined_pipeline_inference.py
# Run that first then run this in the same Colab session.
# =========================================================

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, find_peaks
import librosa
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────
BASE_PATH  = "/content/drive/MyDrive/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
DATA_PATH  = os.path.join(BASE_PATH, "training_data")
CSV_PATH   = os.path.join(BASE_PATH, "training_data.csv")
SAMPLE_RATE    = 4000
WINDOW_SEC     = 5.0
WINDOW_LEN     = int(SAMPLE_RATE * WINDOW_SEC)
MIN_WINDOW_LEN = int(WINDOW_LEN * 0.5)
TARGET_LEN_P2  = int(SAMPLE_RATE * 1.0)
SYSTOLE_LABEL  = 2
DIASTOLE_LABEL = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── reuse model definitions from combined_pipeline_inference
# (paste Phase1Model and Phase2Model classes here)
# ... [same classes as combined_pipeline_inference.py]

# ── aggregation function ──────────────────────────────────
def aggregate_patient_timing(seg_list,
                              murmur_thr=0.50,
                              timing_conf_thr=0.60,
                              min_valid=3,
                              mixed_margin=0.05):
    """
    seg_list: list of dicts with keys:
        p_sys, p_dia, p_murmur
    Returns: label, confidence, details
    """
    sys_scores, dia_scores = [], []
    for seg in seg_list:
        p_murmur = seg["p_murmur"]
        p_sys    = seg["p_sys"]
        p_dia    = seg["p_dia"]
        conf     = max(p_sys, p_dia)

        if p_murmur < murmur_thr:
            continue
        if conf < timing_conf_thr:
            continue

        sys_scores.append(p_murmur * p_sys)
        dia_scores.append(p_murmur * p_dia)

    n_valid = len(sys_scores)
    if n_valid < min_valid:
        return "uncertain", 0.0, {"n_valid": n_valid}

    s = np.mean(sys_scores)
    d = np.mean(dia_scores)
    total = s + d + 1e-8
    sc = s / total; dc = d / total
    margin = abs(sc - dc)

    if margin < mixed_margin:
        label = "mixed"; conf = max(sc, dc)
    elif sc > dc:
        label = "systolic"; conf = sc
    else:
        label = "diastolic"; conf = dc

    return label, conf, {"n_valid": n_valid, "sys_conf": sc,
                          "dia_conf": dc, "margin": margin}

# ── ablation settings ─────────────────────────────────────
ABLATION_SETTINGS = [
    {"name": "All segments (original)",
     "murmur_thr": 0.00, "timing_conf_thr": 0.00,
     "mixed_margin": 0.05, "min_valid": 1},
    {"name": "Murmur-positive only (p1≥0.50)",
     "murmur_thr": 0.50, "timing_conf_thr": 0.00,
     "mixed_margin": 0.05, "min_valid": 1},
    {"name": "Murmur-positive + confident (p2≥0.60)",
     "murmur_thr": 0.50, "timing_conf_thr": 0.60,
     "mixed_margin": 0.05, "min_valid": 3},
    {"name": "High-confidence (p2≥0.70)",
     "murmur_thr": 0.50, "timing_conf_thr": 0.70,
     "mixed_margin": 0.10, "min_valid": 3},
]

# ── run ablation ──────────────────────────────────────────
# patient_segments_dict: {pid: [{"p_sys":, "p_dia":, "p_murmur":}, ...]}
# Build this from your inference run — see combined_pipeline_inference.py
# Here we assume it is already in scope as patient_segments_dict

print("\n" + "="*65)
print("AGGREGATION ABLATION TABLE")
print("="*65)
print(f"\n{'Strategy':<45} {'Sys':>5} {'Dia':>5} "
      f"{'Mix':>5} {'Unc':>5} {'Conf':>7}")
print("-"*65)

ablation_results = []

for setting in ABLATION_SETTINGS:
    sys_n = dia_n = mix_n = unc_n = 0
    confs = []

    for pid, segs in patient_segments_dict.items():
        label, conf, _ = aggregate_patient_timing(
            segs,
            murmur_thr      = setting["murmur_thr"],
            timing_conf_thr = setting["timing_conf_thr"],
            mixed_margin    = setting["mixed_margin"],
            min_valid       = setting["min_valid"])

        if label == "systolic":   sys_n += 1
        elif label == "diastolic": dia_n += 1
        elif label == "mixed":     mix_n += 1
        else:                      unc_n += 1

        if conf > 0: confs.append(conf)

    mean_conf = np.mean(confs) if confs else 0
    ablation_results.append({
        "name": setting["name"],
        "systolic": sys_n, "diastolic": dia_n,
        "mixed": mix_n, "uncertain": unc_n,
        "mean_conf": mean_conf})

    print(f"  {setting['name']:<43} {sys_n:>5} {dia_n:>5} "
          f"{mix_n:>5} {unc_n:>5} {mean_conf*100:>6.1f}%")

print("="*65)
print(f"\nNote: 174/179 patients have systolic GT.")
print("Patient-level timing distribution should be interpreted as exploratory because the CirCor timing labels are highly imbalanced.")
print("Murmur-positive filtering reduces noise from non-murmur")
print("intervals dominating the patient-level aggregation.")



# ── plot confidence distribution per strategy ─────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, setting, res in zip(axes, ABLATION_SETTINGS,
                              ablation_results):
    confs_sys = []; confs_dia = []
    for pid, segs in patient_segments_dict.items():
        label, conf, _ = aggregate_patient_timing(
            segs,
            murmur_thr      = setting["murmur_thr"],
            timing_conf_thr = setting["timing_conf_thr"],
            mixed_margin    = setting["mixed_margin"],
            min_valid       = setting["min_valid"])
        if label == "systolic":    confs_sys.append(conf)
        elif label == "diastolic": confs_dia.append(conf)

    ax.hist(confs_sys, bins=15, alpha=0.6,
            color="steelblue", label="Systolic")
    ax.hist(confs_dia, bins=15, alpha=0.6,
            color="tomato", label="Diastolic")
    ax.set_title(setting["name"][:30], fontsize=8)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Patient-Level Timing Confidence by Aggregation Strategy",
             fontsize=11)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/fig_aggregation_ablation.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved: fig_aggregation_ablation.png")
