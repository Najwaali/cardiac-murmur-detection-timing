"""
ROC curves and frozen probability outputs for the shortcut-controlled
MB-vs-NMB experiments.

This script uses already-computed held-out test probabilities from:
- the duration-only baseline,
- the fixed-crop acoustic model (Strategy D), and
- the acoustic-duration fusion model.

It:
1. verifies the three held-out AUROC values,
2. generates the final ROC comparison figure,
3. saves the held-out labels, probabilities, and patient IDs, and
4. writes a compact JSON file containing the principal ROC statistics.

The script intentionally does not recompute model predictions. Run it after
`strategy_d_fixed_crop.py` and `acoustic_duration_fusion.py`, or import the
required arrays and bootstrap results from those modules.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# Required objects from previous analysis scripts:
# test_acoustic, test_duration, fusion_prob, test_d,
# b_fd, ci95, pval_gt0

OUTPUT_DIR = Path("results")
FIGURE_DIR = Path("figures")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


# =============================================================
# Frozen held-out probabilities
# =============================================================

acoustic_prob = test_acoustic.copy()
duration_prob = test_duration.copy()
fusion_test_prob = fusion_prob.copy()

y_test = test_d["label"].values
patient_ids = test_d["patient_id"].values


# =============================================================
# Verification
# =============================================================

auc_acoustic = roc_auc_score(
    y_test,
    acoustic_prob
)
auc_duration = roc_auc_score(
    y_test,
    duration_prob
)
auc_fusion = roc_auc_score(
    y_test,
    fusion_test_prob
)

print("\nHeld-out test AUROCs:")
print(
    f"  Acoustic : "
    f"{auc_acoustic:.4f}"
)
print(
    f"  Duration : "
    f"{auc_duration:.4f}"
)
print(
    f"  Fusion   : "
    f"{auc_fusion:.4f}"
)


# =============================================================
# Fusion-versus-duration bootstrap result
# =============================================================

fusion_vs_duration_ci = ci95(b_fd)
fusion_vs_duration_p = pval_gt0(b_fd)
fusion_vs_duration_diff = (
    auc_fusion - auc_duration
)

print("\nFusion versus duration:")
print(
    f"  AUROC difference : "
    f"{fusion_vs_duration_diff:+.4f}"
)
print(
    f"  95% CI           : "
    f"[{fusion_vs_duration_ci[0]:+.4f}, "
    f"{fusion_vs_duration_ci[1]:+.4f}]"
)
print(
    f"  p                : "
    f"{fusion_vs_duration_p:.4f}"
)


# =============================================================
# ROC curves
# =============================================================

fpr_duration, tpr_duration, _ = roc_curve(
    y_test,
    duration_prob
)
fpr_acoustic, tpr_acoustic, _ = roc_curve(
    y_test,
    acoustic_prob
)
fpr_fusion, tpr_fusion, _ = roc_curve(
    y_test,
    fusion_test_prob
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    fpr_duration,
    tpr_duration,
    linewidth=2,
    label=(
        "Duration only "
        f"(AUROC = {auc_duration:.4f})"
    )
)
ax.plot(
    fpr_acoustic,
    tpr_acoustic,
    linewidth=2,
    label=(
        "Acoustic only "
        f"(AUROC = {auc_acoustic:.4f})"
    )
)
ax.plot(
    fpr_fusion,
    tpr_fusion,
    linewidth=2.5,
    label=(
        "Acoustic + Duration "
        f"(AUROC = {auc_fusion:.4f})"
    )
)
ax.plot(
    [0, 1],
    [0, 1],
    "--",
    linewidth=1,
    alpha=0.5,
    label="Random classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(
    "ROC Curves: Duration, Acoustic, and Fusion Models\n"
    "Murmur-Bearing vs Non-Murmur-Bearing Intervals"
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(
    loc="lower right"
)

ax.annotate(
    (
        f"Fusion vs duration: "
        f"{fusion_vs_duration_diff:+.4f}\n"
        f"95% CI "
        f"[{fusion_vs_duration_ci[0]:+.4f}, "
        f"{fusion_vs_duration_ci[1]:+.4f}]\n"
        f"p = {fusion_vs_duration_p:.4f}"
    ),
    xy=(0.30, 0.82),
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.3",
        alpha=0.8
    )
)

plt.tight_layout()

roc_path = (
    FIGURE_DIR
    / "fusion_roc_figure.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nSaved ROC figure: "
    f"{roc_path}"
)


# =============================================================
# Save held-out labels and probabilities
# =============================================================

np.save(
    OUTPUT_DIR / "test_true_labels.npy",
    y_test
)
np.save(
    OUTPUT_DIR / "test_acoustic_probs.npy",
    acoustic_prob
)
np.save(
    OUTPUT_DIR / "test_duration_probs.npy",
    duration_prob
)
np.save(
    OUTPUT_DIR / "test_fusion_probs.npy",
    fusion_test_prob
)
np.save(
    OUTPUT_DIR / "test_patient_ids.npy",
    patient_ids
)


# =============================================================
# Save compact metrics summary
# =============================================================

metrics = {
    "acoustic_auroc": float(
        auc_acoustic
    ),
    "duration_auroc": float(
        auc_duration
    ),
    "fusion_auroc": float(
        auc_fusion
    ),
    "fusion_vs_duration_diff": float(
        fusion_vs_duration_diff
    ),
    "fusion_vs_duration_ci_lo": float(
        fusion_vs_duration_ci[0]
    ),
    "fusion_vs_duration_ci_hi": float(
        fusion_vs_duration_ci[1]
    ),
    "fusion_vs_duration_p": float(
        fusion_vs_duration_p
    ),
    "n_test_segments": int(
        len(y_test)
    ),
    "n_test_patients": int(
        len(np.unique(patient_ids))
    )
}

metrics_path = (
    OUTPUT_DIR
    / "roc_probability_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=2
    )

print(
    f"Saved metrics: "
    f"{metrics_path}"
)

print(
    "\nSaved probability arrays to "
    "the results/ directory."
)
