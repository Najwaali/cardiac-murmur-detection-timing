"""
Rebuild the original padded-model Grad-CAM cohort used for the
before/after comparison figure.

This script:

1. Reconstructs 1-second zero-padded systolic murmur-bearing inputs
   from `data_d`.
2. Runs the original padded systolic-versus-diastolic model to identify
   eligible correctly classified systolic segments with duration >= 0.15 s.
3. Computes Grad-CAM maps for those segments.
4. Builds `old_all_df`, containing:
      patient_id, true_dur, true_samp, padding_only, cam_up
5. Reports cohort overlap with the fixed-crop Strategy D Grad-CAM cohort.

Required objects already in memory:

- data_d:
    full Strategy D source DataFrame containing raw_audio, label,
    seg_phase, duration_sec, and patient_id.
- model_old:
    trained original systolic-versus-diastolic model.
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

The resulting `old_all_df` is used by `gradcam_before_after.py`.
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
# Rebuild old padded-model inputs and eligible cohort
# =============================================================

def standardize(signal):
    mu = np.mean(signal)
    std = np.std(signal) + 1e-8
    return ((signal - mu) / std).astype(np.float32)


def make_padded_segment(seg_raw, target_len=TARGET_LEN_OLD):
    """
    Reproduce the original Phase 2 input:
    zero-pad or truncate to 1 second.
    """
    seg = seg_raw.copy()

    if len(seg) < target_len:
        seg = np.pad(
            seg,
            (0, target_len - len(seg))
        )
    else:
        seg = seg[:target_len]

    return standardize(seg)


# Use all available murmur-bearing systolic intervals
# from the reconstructed Strategy D source dataset.
mb_sys_df = data_d[
    (data_d["label"] == 1) &
    (data_d["seg_phase"] == "systolic")
].copy().reset_index(drop=True)

print(
    f"MB systolic segments: "
    f"{len(mb_sys_df):,}"
)
print(
    f"Patients: "
    f"{mb_sys_df['patient_id'].nunique()}"
)


# Recreate 1-second inputs used by the original padded model
mb_sys_audio = np.stack([
    make_padded_segment(
        row["raw_audio"]
    )
    for _, row in mb_sys_df.iterrows()
])

print(
    f"Old-model input shape: "
    f"{mb_sys_audio.shape}"
)


# Identify correctly classified systolic intervals
with torch.no_grad():
    sys_preds = torch.argmax(
        model_old(
            torch.FloatTensor(
                mb_sys_audio[
                    :, np.newaxis, :
                ]
            ).to(DEVICE)
        ),
        dim=1
    ).cpu().numpy()


# Final old-model Grad-CAM cohort:
# correctly classified systolic MB intervals
# with duration >= 150 ms
eligible_mask = (
    (sys_preds == 0) &
    (
        mb_sys_df[
            "duration_sec"
        ].values >= 0.15
    )
)

correct_idx = np.where(
    eligible_mask
)[0]

print(
    f"Correctly classified + "
    f"duration >= 0.15 s: "
    f"{len(correct_idx):,}"
)

print(
    f"Patients represented: "
    f"{mb_sys_df.iloc[correct_idx]"
    f"['patient_id'].nunique()}"
)
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
