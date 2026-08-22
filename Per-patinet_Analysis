"""
Per-patient paired analysis for the shortcut-controlled MB-vs-NMB experiment.

This script evaluates whether the fixed-crop acoustic model and the
acoustic-duration fusion model assign higher mean P(MB) to true
murmur-bearing (MB) intervals than to non-murmur-bearing (NMB) intervals
within the same patient.

Patients are included only when they contain at least three MB and three NMB
test intervals. Paired differences are evaluated with the Wilcoxon signed-rank
test.

Run after `strategy_d_fixed_crop.py` and `acoustic_duration_fusion.py`, or
import the required objects from those modules.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# Required objects:
# test_d, model_d, make_loader, evaluate, lr_dur, fusion_lr


# =============================================================
# Test-set probabilities
# =============================================================

test_loader_det = make_loader(
    test_d,
    augment=False
)

test_true, _, test_acoustic = evaluate(
    model_d,
    test_loader_det
)

test_duration = lr_dur.predict_proba(
    test_d["duration_sec"].values.reshape(-1, 1)
)[:, 1]

test_pids = test_d["patient_id"].values
test_labels = test_d["label"].values

X_fusion_te = np.column_stack([
    test_acoustic,
    test_duration
])

fusion_prob = fusion_lr.predict_proba(
    X_fusion_te
)[:, 1]

print(f"Test segments : {len(test_true):,}")
print(f"Test patients : {len(np.unique(test_pids))}")


# =============================================================
# Helper
# =============================================================

def build_patient_pairs(
    probabilities,
    value_name,
    min_mb=3,
    min_nmb=3
):
    """
    Compute within-patient mean P(MB) for MB and NMB intervals.

    Parameters
    ----------
    probabilities : array-like
        Segment-level predicted MB probabilities.
    value_name : str
        Prefix used for output-column names.
    min_mb : int
        Minimum number of MB intervals required per patient.
    min_nmb : int
        Minimum number of NMB intervals required per patient.
    """
    rows = []

    for pid in np.unique(test_pids):
        patient_mask = test_pids == pid
        labels = test_labels[patient_mask]

        mb_mask = labels == 1
        nmb_mask = labels == 0

        if (
            mb_mask.sum() < min_mb
            or nmb_mask.sum() < min_nmb
        ):
            continue

        probs = probabilities[patient_mask]

        mb_mean = probs[mb_mask].mean()
        nmb_mean = probs[nmb_mask].mean()

        rows.append({
            "patient_id": pid,
            f"{value_name}_mb": mb_mean,
            f"{value_name}_nmb": nmb_mean,
            "gap": mb_mean - nmb_mean,
            "mb_n": int(mb_mask.sum()),
            "nmb_n": int(nmb_mask.sum())
        })

    return pd.DataFrame(rows)


def report_patient_analysis(
    patient_df,
    mb_column,
    nmb_column,
    label
):
    """Print paired patient-level summary and Wilcoxon test."""

    _, p_value = wilcoxon(
        patient_df[mb_column].values,
        patient_df[nmb_column].values
    )

    print("\n" + "=" * 64)
    print(label)
    print("=" * 64)
    print(
        f"Qualifying patients       : "
        f"{len(patient_df)}"
    )
    print(
        f"Mean P(MB) on MB segments : "
        f"{patient_df[mb_column].mean():.4f} ± "
        f"{patient_df[mb_column].std():.4f}"
    )
    print(
        f"Mean P(MB) on NMB segments: "
        f"{patient_df[nmb_column].mean():.4f} ± "
        f"{patient_df[nmb_column].std():.4f}"
    )
    print(
        f"Mean within-patient gap   : "
        f"{patient_df['gap'].mean():.4f} ± "
        f"{patient_df['gap'].std():.4f}"
    )
    print(
        f"Patients with MB > NMB    : "
        f"{(patient_df['gap'] > 0).sum()}/"
        f"{len(patient_df)}"
    )
    print(
        f"Wilcoxon p                : "
        f"{p_value:.6g}"
    )

    return p_value


# =============================================================
# Acoustic-only paired analysis
# =============================================================

pat_acoustic = build_patient_pairs(
    test_acoustic,
    value_name="acoustic"
)

pval_acoustic = report_patient_analysis(
    pat_acoustic,
    mb_column="acoustic_mb",
    nmb_column="acoustic_nmb",
    label="PER-PATIENT ACOUSTIC PAIRED ANALYSIS"
)


# =============================================================
# Fusion paired analysis
# =============================================================

pat_fusion = build_patient_pairs(
    fusion_prob,
    value_name="fusion"
)

pval_fusion = report_patient_analysis(
    pat_fusion,
    mb_column="fusion_mb",
    nmb_column="fusion_nmb",
    label="PER-PATIENT FUSION PAIRED ANALYSIS"
)


# =============================================================
# Optional output tables
# =============================================================

from pathlib import Path

Path("results").mkdir(exist_ok=True)

pat_acoustic.to_csv(
    "results/per_patient_acoustic.csv",
    index=False
)

pat_fusion.to_csv(
    "results/per_patient_fusion.csv",
    index=False
)

print(
    "\nSaved patient-level tables to the results/ directory."
)
