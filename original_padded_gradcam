"""
Original padded-model Grad-CAM preparation for Fig. 5.

This script rebuilds `old_all_df`, the DataFrame required by
`gradcam_before_after.py`, and reports overlap between the original padded-model
cohort and the fixed-crop Strategy D cohort.

It assumes the following objects are already available:

- correct_idx:
    indices of eligible correctly classified systolic murmur-bearing segments
    from the original padded model.
- mb_sys_df:
    metadata table aligned with `mb_sys_audio`.
- mb_sys_audio:
    1-second zero-padded inputs for the original model.
- gradcam_old:
    GradCAM1D object attached to the original padded model.
- DEVICE:
    PyTorch device.
- SAMPLE_RATE:
    sampling rate in Hz.
- mb_all:
    Strategy D Grad-CAM DataFrame from `gradcam_mb_vs_nmb.py`.
- gc_pat:
    patient-level Strategy D edge/interior Grad-CAM table from
    `gradcam_strategy_d.py`.

The resulting `old_all_df` contains:
patient_id, true_dur, true_samp, padding_only, and cam_up.
"""

import numpy as np
import pandas as pd
import torch


# =============================================================
# Configuration
# =============================================================

TARGET_LEN_OLD = 4000
EDGE_OLD = int(0.050 * SAMPLE_RATE)  # 50 ms


# =============================================================
# Rebuild old-model Grad-CAM table
# =============================================================

print(
    f"Rebuilding old_all_df from "
    f"{len(correct_idx):,} eligible segments..."
)

old_records_all = []

for count, seg_idx in enumerate(correct_idx):
    row = mb_sys_df.iloc[seg_idx]
    audio = mb_sys_audio[seg_idx]

    true_dur = float(
        row["duration_sec"]
    )
    true_samp = min(
        int(true_dur * SAMPLE_RATE),
        TARGET_LEN_OLD
    )

    if true_samp <= 2 * EDGE_OLD:
        continue

    inp = (
        torch.FloatTensor(audio)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    cam = gradcam_old.generate(
        inp,
        target_class=0
    )

    cam_up = np.interp(
        np.linspace(
            0,
            1,
            TARGET_LEN_OLD
        ),
        np.linspace(
            0,
            1,
            len(cam)
        ),
        cam
    )

    cam_up = np.maximum(
        cam_up,
        0
    )

    cam_in_signal = cam_up[
        :true_samp
    ].sum()

    cam_in_padding = cam_up[
        true_samp:
    ].sum()

    padding_only = (
        cam_in_signal <= 1e-8
        and cam_in_padding > 1e-8
    )

    old_records_all.append({
        "patient_id": str(
            row["patient_id"]
        ),
        "true_dur": true_dur,
        "true_samp": true_samp,
        "padding_only": padding_only,
        "cam_up": cam_up
    })

    if (count + 1) % 300 == 0:
        print(
            f"  Processed "
            f"{count + 1}/{len(correct_idx)}"
        )


old_all_df = pd.DataFrame(
    old_records_all
)


# =============================================================
# Old-model cohort summary
# =============================================================

n_all = len(old_all_df)

n_padding = int(
    old_all_df[
        "padding_only"
    ].sum()
)

n_signal = int(
    (
        ~old_all_df[
            "padding_only"
        ]
    ).sum()
)

n_pat_all = old_all_df[
    "patient_id"
].nunique()

n_pat_pad = old_all_df.loc[
    old_all_df[
        "padding_only"
    ],
    "patient_id"
].nunique()

n_pat_signal = old_all_df.loc[
    ~old_all_df[
        "padding_only"
    ],
    "patient_id"
].nunique()

print("\nold_all_df rebuilt:")
print(
    f"  Total segments      : "
    f"{n_all:,}"
)
print(
    f"  Patients            : "
    f"{n_pat_all}"
)
print(
    f"  Padding-only        : "
    f"{n_padding:,} "
    f"({n_padding / n_all:.1%}) / "
    f"{n_pat_pad} patients"
)
print(
    f"  True-signal active  : "
    f"{n_signal:,} / "
    f"{n_pat_signal} patients"
)


# =============================================================
# Cohort overlap with Strategy D
# =============================================================

old_ids = set(
    old_all_df[
        "patient_id"
    ].astype(str)
)

new_ids = set(
    mb_all[
        "patient_id"
    ].astype(str)
)

overlap_ids = (
    old_ids & new_ids
)

print("\nPatient overlap:")
print(
    f"  Old-model patients  : "
    f"{len(old_ids)}"
)
print(
    f"  Strategy D patients : "
    f"{len(new_ids)}"
)
print(
    f"  Overlap             : "
    f"{len(overlap_ids)}"
)
print(
    f"  New but not old     : "
    f"{sorted(new_ids - old_ids)}"
)
print(
    f"  Old but not new     : "
    f"{len(old_ids - new_ids)}"
)


# =============================================================
# Strategy D 34-vs-35 consistency check
# =============================================================

gc_pat_ids = set(
    gc_pat[
        "patient_id"
    ].astype(str)
)

mb_all_ids = set(
    mb_all[
        "patient_id"
    ].astype(str)
)

print("\nStrategy D patient-count check:")
print(
    f"  mb_all patients : "
    f"{len(mb_all_ids)}"
)
print(
    f"  gc_pat patients : "
    f"{len(gc_pat_ids)}"
)
print(
    f"  Present in mb_all but "
    f"not gc_pat: "
    f"{sorted(mb_all_ids - gc_pat_ids)}"
)
