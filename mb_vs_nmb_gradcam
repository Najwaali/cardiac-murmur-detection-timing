"""
MB-versus-NMB Grad-CAM contrast for Strategy D.

This script compares Grad-CAM behavior between murmur-bearing (MB) and
non-murmur-bearing (NMB) fixed 100-ms test crops.

Key design choices:
1. The MB class logit (class=1) is used as the Grad-CAM target for both MB
   and NMB inputs so that the explained class is held constant.
2. All eligible test segments are analyzed in the primary analysis.
3. Correctly classified segments are analyzed separately as an exploratory
   secondary analysis.
4. Patient-level paired Wilcoxon tests are the primary statistical tests.
5. Segment-level Mann-Whitney U tests are reported descriptively only.
6. The stable energy metric is the difference between mean Grad-CAM
   activation in high-energy and low-energy samples:
       energy_contrast = high_cam - low_cam

This script depends on objects defined by the Strategy D and Grad-CAM scripts:
SAMPLE_RATE, CROP_LEN, DEVICE, model_d, test_d, center_crop, and gradcam.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from scipy.stats import mannwhitneyu, wilcoxon


# =============================================================
# Configuration
# =============================================================

EDGE = int(0.025 * SAMPLE_RATE)  # 25 ms


# =============================================================
# Prepare MB and NMB test crops
# =============================================================

test_mb = (
    test_d[test_d["label"] == 1]
    .copy()
    .reset_index(drop=True)
)

test_nmb = (
    test_d[test_d["label"] == 0]
    .copy()
    .reset_index(drop=True)
)

test_mb_audio = np.stack([
    center_crop(row["raw_audio"])
    for _, row in test_mb.iterrows()
])

test_nmb_audio = np.stack([
    center_crop(row["raw_audio"])
    for _, row in test_nmb.iterrows()
])


# =============================================================
# Model predictions for subgroup masks
# =============================================================

with torch.no_grad():
    mb_preds = torch.argmax(
        model_d(
            torch.FloatTensor(
                test_mb_audio[:, np.newaxis, :]
            ).to(DEVICE)
        ),
        dim=1
    ).cpu().numpy()

    nmb_preds = torch.argmax(
        model_d(
            torch.FloatTensor(
                test_nmb_audio[:, np.newaxis, :]
            ).to(DEVICE)
        ),
        dim=1
    ).cpu().numpy()

mb_correct_mask = mb_preds == 1
nmb_correct_mask = nmb_preds == 0

print(
    f"MB segments total/correct  : "
    f"{len(test_mb):,}/{mb_correct_mask.sum():,}"
)
print(
    f"NMB segments total/correct : "
    f"{len(test_nmb):,}/{nmb_correct_mask.sum():,}"
)


# =============================================================
# Grad-CAM metrics
# =============================================================

def compute_envelope(signal):
    """Return the Hilbert amplitude envelope."""
    return np.abs(hilbert(signal))


def get_gradcam_metrics(
    audio_array,
    indices,
    pid_array,
    target_class=1
):
    """
    Compute Grad-CAM metrics for selected segments.

    The same target class (MB logit, class=1) is used for both MB and NMB
    segments so that only the input group changes.

    Segments with negligible total Grad-CAM activation (<=1e-8) are excluded.
    """
    results = []

    for idx in indices:
        audio = audio_array[idx]
        pid = pid_array[idx]

        inp = (
            torch.FloatTensor(audio)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        cam = gradcam.generate(
            inp,
            target_class=target_class
        )

        cam_up = np.interp(
            np.linspace(0, 1, CROP_LEN),
            np.linspace(0, 1, len(cam)),
            cam
        )

        total = cam_up.sum()

        if total <= 1e-8:
            continue

        cam_norm = cam_up / total
        envelope = compute_envelope(audio)

        # Metric 1: Grad-CAM / amplitude-envelope correlation
        if (
            np.std(cam_norm) < 1e-12
            or np.std(envelope) < 1e-12
        ):
            corr = np.nan
        else:
            corr = np.corrcoef(
                cam_norm,
                envelope
            )[0, 1]

        # Metric 2: High-energy minus low-energy activation
        median_env = np.median(envelope)
        high_mask = envelope >= median_env
        low_mask = ~high_mask

        high_cam = (
            cam_norm[high_mask].mean()
            if high_mask.sum() > 0 else np.nan
        )
        low_cam = (
            cam_norm[low_mask].mean()
            if low_mask.sum() > 0 else np.nan
        )

        energy_contrast = high_cam - low_cam

        results.append({
            "patient_id": pid,
            "corr_env": corr,
            "energy_contrast": energy_contrast,
            "high_cam": high_cam,
            "low_cam": low_cam,
            "cam_norm": cam_norm
        })

    return pd.DataFrame(results)


# =============================================================
# Primary analysis: all eligible test segments
# =============================================================

print(
    "\nComputing Grad-CAM for all test segments "
    "(target = MB logit)..."
)

mb_all_idx = np.arange(len(test_mb))
nmb_all_idx = np.arange(len(test_nmb))

mb_all = get_gradcam_metrics(
    test_mb_audio,
    mb_all_idx,
    test_mb["patient_id"].values,
    target_class=1
)

nmb_all = get_gradcam_metrics(
    test_nmb_audio,
    nmb_all_idx,
    test_nmb["patient_id"].values,
    target_class=1
)


# =============================================================
# Exploratory analysis: correctly classified segments only
# =============================================================

print(
    "Computing Grad-CAM for correctly classified segments only..."
)

mb_cor = get_gradcam_metrics(
    test_mb_audio,
    np.where(mb_correct_mask)[0],
    test_mb["patient_id"].values,
    target_class=1
)

nmb_cor = get_gradcam_metrics(
    test_nmb_audio,
    np.where(nmb_correct_mask)[0],
    test_nmb["patient_id"].values,
    target_class=1
)

print(
    f"\nAll segments       : "
    f"MB={len(mb_all):,}, NMB={len(nmb_all):,}"
)
print(
    f"Correct-only subset: "
    f"MB={len(mb_cor):,}, NMB={len(nmb_cor):,}"
)


# =============================================================
# Statistical comparison
# =============================================================

def report_contrast(mb_df, nmb_df, label=""):
    print("\n" + "=" * 64)
    print(f"MB vs NMB GRAD-CAM CONTRAST — {label}")
    print("=" * 64)

    for metric, desc in [
        (
            "corr_env",
            "Grad-CAM / envelope correlation"
        ),
        (
            "energy_contrast",
            "High-energy minus low-energy activation"
        )
    ]:
        mb_vals = (
            mb_df[metric]
            .dropna()
            .values
        )
        nmb_vals = (
            nmb_df[metric]
            .dropna()
            .values
        )

        _, pval_seg = mannwhitneyu(
            mb_vals,
            nmb_vals,
            alternative="two-sided"
        )

        print(f"\n{desc}:")
        print(
            f"  MB  (n={len(mb_vals):,}): "
            f"{mb_vals.mean():.4f} ± "
            f"{mb_vals.std():.4f}"
        )
        print(
            f"  NMB (n={len(nmb_vals):,}): "
            f"{nmb_vals.mean():.4f} ± "
            f"{nmb_vals.std():.4f}"
        )
        print(
            f"  Difference: "
            f"{mb_vals.mean() - nmb_vals.mean():+.4f}"
        )
        print(
            f"  Segment-level MWU p: "
            f"{pval_seg:.4f} "
            "(descriptive only)"
        )

        # Patient-level paired analysis: primary statistic
        mb_pat = (
            mb_df
            .groupby("patient_id")[metric]
            .mean()
            .dropna()
        )
        nmb_pat = (
            nmb_df
            .groupby("patient_id")[metric]
            .mean()
            .dropna()
        )

        common = sorted(
            set(mb_pat.index)
            & set(nmb_pat.index)
        )

        if len(common) >= 5:
            mb_c = mb_pat.loc[common].values
            nmb_c = nmb_pat.loc[common].values

            _, pval_pat = wilcoxon(
                mb_c,
                nmb_c
            )

            print(
                f"  Patient-level Wilcoxon "
                f"(n={len(common)} paired patients):"
            )
            print(
                f"    MB  : "
                f"{mb_c.mean():.4f} ± "
                f"{mb_c.std():.4f}"
            )
            print(
                f"    NMB : "
                f"{nmb_c.mean():.4f} ± "
                f"{nmb_c.std():.4f}"
            )
            print(
                f"    p   : "
                f"{pval_pat:.4f}  [PRIMARY]"
            )
        else:
            _, pval_pat = mannwhitneyu(
                mb_pat.values,
                nmb_pat.values,
                alternative="two-sided"
            )

            print(
                "  Patient-level unpaired Mann-Whitney U:"
            )
            print(
                f"    MB patients  : "
                f"{len(mb_pat)}"
            )
            print(
                f"    NMB patients : "
                f"{len(nmb_pat)}"
            )
            print(
                f"    p            : "
                f"{pval_pat:.4f}  [PRIMARY]"
            )


report_contrast(
    mb_all,
    nmb_all,
    "ALL SEGMENTS"
)

report_contrast(
    mb_cor,
    nmb_cor,
    "CORRECTLY CLASSIFIED ONLY"
)


# =============================================================
# Mean activation profiles
# =============================================================

def mean_profile(df):
    cams = np.stack(
        df["cam_norm"].values
    )
    return (
        cams.mean(axis=0),
        cams.std(axis=0)
    )


mb_mean, mb_std = mean_profile(mb_all)
nmb_mean, nmb_std = mean_profile(nmb_all)

t = (
    np.arange(CROP_LEN)
    / SAMPLE_RATE
    * 1000
)


# =============================================================
# Figure
# =============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

# Plot 1: envelope-correlation distribution
axes[0].hist(
    mb_all["corr_env"].dropna(),
    bins=30,
    alpha=0.6,
    label="MB",
    density=True
)
axes[0].hist(
    nmb_all["corr_env"].dropna(),
    bins=30,
    alpha=0.6,
    label="NMB",
    density=True
)
axes[0].set_xlabel(
    "Grad-CAM / envelope correlation"
)
axes[0].set_ylabel("Density")
axes[0].set_title(
    "Activation-Envelope Correlation\n"
    "MB vs NMB"
)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: stable energy-contrast distribution
axes[1].hist(
    mb_all["energy_contrast"].dropna(),
    bins=30,
    alpha=0.6,
    label="MB",
    density=True
)
axes[1].hist(
    nmb_all["energy_contrast"].dropna(),
    bins=30,
    alpha=0.6,
    label="NMB",
    density=True
)
axes[1].set_xlabel(
    "High-energy minus low-energy activation"
)
axes[1].set_ylabel("Density")
axes[1].set_title(
    "Energy Contrast\n"
    "MB vs NMB"
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: mean activation profiles
axes[2].plot(
    t,
    mb_mean,
    label="MB",
    linewidth=2
)
axes[2].fill_between(
    t,
    mb_mean - mb_std,
    mb_mean + mb_std,
    alpha=0.2
)
axes[2].plot(
    t,
    nmb_mean,
    label="NMB",
    linewidth=2
)
axes[2].fill_between(
    t,
    nmb_mean - nmb_std,
    nmb_mean + nmb_std,
    alpha=0.2
)
axes[2].axvline(
    EDGE / SAMPLE_RATE * 1000,
    linestyle=":",
    linewidth=1.5,
    label="Start edge (25 ms)"
)
axes[2].axvline(
    (CROP_LEN - EDGE) / SAMPLE_RATE * 1000,
    linestyle=":",
    linewidth=1.5,
    label="End edge (75 ms)"
)
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel(
    "Mean normalized Grad-CAM"
)
axes[2].set_title(
    "Mean Activation Profile\n"
    "MB vs NMB"
)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.suptitle(
    "MB vs NMB Grad-CAM Contrast — Strategy D\n"
    "Target: MB logit (class=1) for both groups",
    fontsize=11
)

plt.tight_layout()

os.makedirs("figures", exist_ok=True)
FIG_PATH = "figures/gradcam_mb_vs_nmb.png"

plt.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved: {FIG_PATH}")
