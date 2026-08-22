"""
Final Fig. 5: Grad-CAM before/after shortcut removal.

This script generates the four-panel figure comparing:
(A) the original zero-padded systolic-vs-diastolic model,
(B) the fixed 100-ms Strategy D MB-vs-NMB model,
(C) representative old-model Grad-CAM examples, and
(D) representative Strategy D examples.

IMPORTANT
---------
This is a plotting/aggregation script, not a standalone training script.
Before running it, the following objects must already exist:

- old_all_df:
    DataFrame from the original padded-model Grad-CAM analysis containing
    patient_id, true_samp, true_dur, padding_only, and cam_up.
- mb_all:
    DataFrame from `gradcam_mb_vs_nmb.py` containing patient_id and cam_norm
    for Strategy D MB segments.
- CROP_LEN:
    fixed-crop input length from `strategy_d_fixed_crop.py`.

The old-model profiles are endpoint-aligned and max-normalized within the
displayed window. Strategy D profiles are normalized so that the mean
activation over the 100-ms crop equals 1. Absolute y-axis magnitudes should
therefore not be compared across the two panels.
"""

# =============================================================
# FINAL GRAD-CAM FIGURE — UNMATCHED COHORTS, STATED CLEARLY
# Endpoint-aligned old model + crop-mean-normalized Strategy D
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon

TARGET_LEN_OLD = 4000
EDGE_OLD_MS    = 50    # ms
EDGE_NEW_MS    = 25    # ms
SAMPLE_RATE    = 4000

# ── Old model: endpoint-aligned profiles ─────────────────────
# Align each CAM so that signal endpoint = time 0
# x-axis: negative = inside signal, positive = padding
WINDOW_MS  = 150   # show 150ms before and after signal end
WINDOW_SAM = int(WINDOW_MS * SAMPLE_RATE / 1000)

print("Building endpoint-aligned old model profiles...")
old_aligned = []   # each entry: (WINDOW_SAM*2,) array

for _, row in old_all_df.iterrows():
    cam    = row["cam_up"]
    ts     = int(row["true_samp"])

    # Extract window centred on signal endpoint
    start  = max(0, ts - WINDOW_SAM)
    end    = min(TARGET_LEN_OLD, ts + WINDOW_SAM)

    # Pad so every profile is exactly 2*WINDOW_SAM long
    left_pad  = max(0, WINDOW_SAM - ts)
    right_pad = max(0, (ts + WINDOW_SAM) - TARGET_LEN_OLD)

    segment   = cam[start:end]
    aligned   = np.pad(segment, (left_pad, right_pad),
                        constant_values=0)[:2*WINDOW_SAM]

    # Normalise: max over the aligned window
    amax = aligned.max()
    if amax > 0:
        aligned = aligned / amax

    old_aligned.append({
        "patient_id":   row["patient_id"],
        "padding_only": row["padding_only"],
        "cam_aligned":  aligned
    })

old_aligned_df = pd.DataFrame(old_aligned)

# Patient-level average of aligned profiles
old_pat_aligned = []
for pid in old_aligned_df["patient_id"].unique():
    pat = np.stack(
        old_aligned_df[old_aligned_df["patient_id"]==pid
                       ]["cam_aligned"].values)
    old_pat_aligned.append(pat.mean(axis=0))

old_pat_aligned  = np.stack(old_pat_aligned)
old_mean_aligned = old_pat_aligned.mean(axis=0)
old_n_pat        = len(old_pat_aligned)

# Bootstrap 95% CI
rng   = np.random.default_rng(42)
boots = []
for _ in range(500):
    idx = rng.choice(old_n_pat, old_n_pat, replace=True)
    boots.append(old_pat_aligned[idx].mean(axis=0))
boots     = np.stack(boots)
old_ci_lo = np.percentile(boots, 2.5,  axis=0)
old_ci_hi = np.percentile(boots, 97.5, axis=0)

# x-axis centred on signal endpoint
t_aligned_ms = np.linspace(-WINDOW_MS, WINDOW_MS,
                             2*WINDOW_SAM)

n_all     = len(old_all_df)
n_padding = int(old_all_df["padding_only"].sum())
n_pat_pad = old_all_df[
    old_all_df["padding_only"]==True]["patient_id"].nunique()

print(f"Old model aligned: {old_n_pat} patients")

# ── Strategy D: crop-mean-normalized profiles ─────────────────
# Normalize each cam_norm so its crop-wide mean = 1
# This makes y-axis correspond to reported density values
print("Building Strategy D crop-mean-normalized profiles...")
new_pat_cams = []
for pid in mb_all["patient_id"].unique():
    pid_cams = []
    for cam_raw in mb_all[mb_all["patient_id"]==pid
                          ]["cam_norm"].values:
        crop_mean = cam_raw.mean()
        if crop_mean > 0:
            pid_cams.append(cam_raw / crop_mean)
        else:
            pid_cams.append(cam_raw)
    new_pat_cams.append(np.stack(pid_cams).mean(axis=0))

new_pat_cams   = np.stack(new_pat_cams)
new_mean_pat   = new_pat_cams.mean(axis=0)
new_n_patients = len(new_pat_cams)

boots_new = []
for _ in range(500):
    idx = rng.choice(new_n_patients, new_n_patients, replace=True)
    boots_new.append(new_pat_cams[idx].mean(axis=0))
boots_new = np.stack(boots_new)
new_ci_lo = np.percentile(boots_new, 2.5,  axis=0)
new_ci_hi = np.percentile(boots_new, 97.5, axis=0)

t_new_ms = np.arange(400) / SAMPLE_RATE * 1000  # 0-100ms

print(f"Strategy D: {new_n_patients} patients")
print(f"\nStrategy D mean profile range: "
      f"{new_mean_pat.min():.3f} – {new_mean_pat.max():.3f}")

# ── Build figure ──────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.52, wspace=0.38)

# ── Panel A: Old model endpoint-aligned ───────────────────────
ax_a = fig.add_subplot(gs[0, 0])

ax_a.fill_between(t_aligned_ms, old_ci_lo, old_ci_hi,
                   alpha=0.25, color="tomato",
                   label="Bootstrap 95% CI")
ax_a.plot(t_aligned_ms, old_mean_aligned,
           color="darkred", lw=2,
           label=f"Patient mean (n={old_n_pat})")

# Signal end at x=0
ax_a.axvline(0, color="purple", linestyle="--", lw=2,
              label="Signal end / padding onset")

# Shade regions
ax_a.axvspan(-WINDOW_MS, 0, alpha=0.05, color="steelblue",
              label="True signal region")
ax_a.axvspan(0, WINDOW_MS, alpha=0.05, color="grey",
              label="Padding region")

# End-boundary region
ax_a.axvspan(-EDGE_OLD_MS, 0, alpha=0.12, color="orange",
              label=f"End edge ({EDGE_OLD_MS}ms)")

ax_a.set_xlabel("Time relative to signal end (ms)",
                  fontsize=10)
ax_a.set_ylabel("Normalised Grad-CAM\n(max-normalised)",
                  fontsize=9)
ax_a.set_title(
    f"(A) Original zero-padded model\n"
    f"Sys vs dia | {n_all} segs / {old_n_pat} patients\n"
    f"{n_padding} segs ({n_padding/n_all:.1%}) activation "
    f"in padding only",
    fontsize=8, fontweight="bold")
ax_a.legend(fontsize=7, loc="upper left")
ax_a.set_xlim(-WINDOW_MS, WINDOW_MS)
ax_a.set_ylim(bottom=0)
ax_a.grid(True, alpha=0.3)

# Annotate spike
ax_a.text(5, old_mean_aligned.max()*0.7,
           "Strong activation near\nsignal end / padding onset",
           fontsize=8, color="purple",
           bbox=dict(boxstyle="round,pad=0.2",
                     facecolor="lavender", alpha=0.85))

# ── Panel B: Strategy D crop-mean-normalized ──────────────────
ax_b = fig.add_subplot(gs[0, 1])

ax_b.fill_between(t_new_ms, new_ci_lo, new_ci_hi,
                   alpha=0.25, color="steelblue",
                   label="Bootstrap 95% CI")
ax_b.plot(t_new_ms, new_mean_pat,
           color="steelblue", lw=2,
           label=f"Patient mean (n={new_n_patients})")

# Uniform baseline
ax_b.axhline(1.0, color="black", linestyle="--",
              alpha=0.4, lw=1.2, label="Uniform (= 1.0)")

ax_b.axvline(EDGE_NEW_MS, color="green",
              linestyle=":", lw=1.5,
              label="Start edge (25ms)")
ax_b.axvline(75, color="orange", linestyle=":", lw=1.5,
              label="End edge (75ms)")
ax_b.axvspan(0,  EDGE_NEW_MS, alpha=0.10, color="green")
ax_b.axvspan(75, 100, alpha=0.10, color="orange")

# Region labels
y_top = max(new_ci_hi.max(), 1.5)
ax_b.text(12.5, y_top*0.88, "Start\nedge",
           ha="center", fontsize=8, color="darkgreen")
ax_b.text(50,   y_top*0.88, "Interior",
           ha="center", fontsize=8, color="steelblue")
ax_b.text(87.5, y_top*0.88, "End\nedge",
           ha="center", fontsize=8, color="darkorange")

ax_b.set_xlabel("Time (ms)", fontsize=10)
ax_b.set_ylabel("Normalised Grad-CAM\n(crop mean = 1)",
                  fontsize=9)
ax_b.set_title(
    f"(B) Fixed-crop model (Strategy D)\n"
    f"MB vs NMB | 100ms | {len(mb_all)} segs / "
    f"{new_n_patients} patients",
    fontsize=8, fontweight="bold")
ax_b.legend(fontsize=7, loc="upper right")
ax_b.set_xlim(0, 100)
ax_b.set_ylim(0, y_top)
ax_b.grid(True, alpha=0.3)

ax_b.text(50, y_top*0.42,
           "Interior/boundary = 0.978\n"
           "20/34 patients | p = 1.000\n"
           "No systematic edge preference",
           ha="center", fontsize=8,
           bbox=dict(boxstyle="round,pad=0.3",
                     facecolor="lightyellow", alpha=0.85))

# ── Panel C: Old model examples ───────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

# Show one padding-only and two true-signal examples
pad_rows = old_all_df[old_all_df["padding_only"]==True]
sig_rows = old_all_df[~old_all_df["padding_only"]]
ex_pad   = pad_rows.iloc[0]
ex_sig1  = sig_rows.iloc[0]
ex_sig2  = sig_rows.iloc[5]

t_old_ms = np.arange(TARGET_LEN_OLD) / SAMPLE_RATE * 1000

for ex, label, color in [
    (ex_pad,
     f"Padding-only ({ex_pad['true_dur']*1000:.0f}ms)",
     "purple"),
    (ex_sig1,
     f"Signal activation ({ex_sig1['true_dur']*1000:.0f}ms)",
     "darkred"),
    (ex_sig2,
     f"Signal activation ({ex_sig2['true_dur']*1000:.0f}ms)",
     "firebrick"),
]:
    cam_ex  = ex["cam_up"]
    dur_ms  = ex["true_dur"] * 1000
    cam_max = cam_ex.max()
    cam_n   = cam_ex / cam_max if cam_max > 0 else cam_ex
    xlim_ex = min(dur_ms * 3, 500)
    mask_ex = t_old_ms <= xlim_ex
    ax_c.plot(t_old_ms[mask_ex], cam_n[mask_ex],
               color=color, lw=1.4, alpha=0.9,
               label=label)
    ax_c.axvline(dur_ms, color=color,
                  linestyle="--", lw=0.9, alpha=0.7)

ax_c.set_xlabel("Time (ms)", fontsize=10)
ax_c.set_ylabel("Normalised Grad-CAM", fontsize=9)
ax_c.set_title("(C) Old model: representative examples\n"
                "Dashed = signal end / padding onset",
                fontsize=9, fontweight="bold")
ax_c.legend(fontsize=7)
ax_c.set_xlim(0, 500)
ax_c.set_ylim(bottom=0)
ax_c.grid(True, alpha=0.3)

# ── Panel D: Strategy D examples ──────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

rng_ex   = np.random.default_rng(42)
ex_idxs  = rng_ex.choice(len(mb_all), size=3, replace=False)
colors_d = ["steelblue", "dodgerblue", "cornflowerblue"]

for i, ex_i in enumerate(ex_idxs):
    cam_raw  = mb_all.iloc[ex_i]["cam_norm"]
    crop_mean = cam_raw.mean()
    cam_n    = cam_raw / crop_mean if crop_mean > 0 else cam_raw
    ax_d.plot(t_new_ms, cam_n, color=colors_d[i],
               lw=1.4, alpha=0.85, label=f"Ex {i+1}")

ax_d.axhline(1.0, color="black", linestyle="--",
              alpha=0.4, lw=1.2, label="Uniform (= 1.0)")
ax_d.axvline(EDGE_NEW_MS, color="green",
              linestyle=":", lw=1.2)
ax_d.axvline(75, color="orange", linestyle=":", lw=1.2)
ax_d.axvspan(0,  EDGE_NEW_MS, alpha=0.08, color="green")
ax_d.axvspan(75, 100, alpha=0.08, color="orange")
ax_d.set_xlabel("Time (ms)", fontsize=10)
ax_d.set_ylabel("Normalised Grad-CAM\n(crop mean = 1)",
                  fontsize=9)
ax_d.set_title("(D) Strategy D: representative examples\n"
                "No padding — no systematic crop-edge preference",
                fontsize=9, fontweight="bold")
ax_d.legend(fontsize=7)
ax_d.set_xlim(0, 100)
ax_d.set_ylim(bottom=0)
ax_d.grid(True, alpha=0.3)

plt.suptitle(
    "Fig. 5 — Grad-CAM: Before and After Shortcut Removal\n"
    "Top: patient-level mean ± bootstrap 95% CI  |  "
    "Bottom: representative examples\n"
    "Note: old model (133 patients, full dataset) and "
    "fixed-crop model (35 patients, test set) are not "
    "patient-matched cohorts",
    fontsize=10, fontweight="bold")

from pathlib import Path
Path("figures").mkdir(exist_ok=True)
save_path = "figures/gradcam_before_after_final.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved: {save_path}")
print(f"\nFINAL CAPTION NUMBERS:")
print(f"Old model  : {n_all} segs / {old_n_pat} patients")
print(f"  Padding-only: {n_padding} ({n_padding/n_all:.1%}) / "
      f"{n_pat_pad} patients")
print(f"  True-signal : {n_all-n_padding} / "
      f"{old_all_df[~old_all_df['padding_only']]['patient_id'].nunique()} patients")
print(f"Strategy D : {len(mb_all)} segs / {new_n_patients} patients")
print(f"Cohort overlap: 25/35 Strategy D patients in old cohort")
print(f"Limitation: cohorts not matched — state in caption")
