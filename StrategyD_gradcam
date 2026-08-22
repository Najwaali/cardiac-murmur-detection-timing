"""
Grad-CAM analysis for Strategy D: fixed 100-ms MB-vs-NMB classifier.

This script analyzes correctly classified murmur-bearing (MB) test segments
from the fixed-crop acoustic model. The 100-ms crop is divided into:

- start edge: 0-25 ms
- interior: 25-75 ms
- end edge: 75-100 ms

Because Strategy D uses only real PCG samples, the crop edges are generic crop
limits and do not correspond to physiological S1/S2 boundaries or zero padding.

The analysis reports segment-level and patient-level activation densities and
uses a paired Wilcoxon signed-rank test to compare patient-level interior and
boundary density.

Run this script after `strategy_d_fixed_crop.py`, or import the required
objects from that module.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import wilcoxon

# Required objects from strategy_d_fixed_crop.py:
# SAMPLE_RATE, CROP_LEN, CROP_MS, DEVICE, model_d, test_d, center_crop

EDGE = int(0.025 * SAMPLE_RATE)  # 25 ms = 100 samples at 4 kHz

print(f"Crop    : {CROP_LEN} samples = {CROP_MS} ms")
print(f"Edge    : {EDGE} samples = 25 ms")
print(
    f"Interior: {CROP_LEN - 2 * EDGE} samples = "
    f"{(CROP_LEN - 2 * EDGE) / SAMPLE_RATE * 1000:.0f} ms"
)


# =============================================================
# Grad-CAM implementation
# =============================================================

class GradCAM1D:
    def __init__(self, model, target_module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_module.register_forward_hook(
            lambda module, inputs, output:
            setattr(self, "activations", output.detach())
        )

        target_module.register_full_backward_hook(
            lambda module, grad_input, grad_output:
            setattr(self, "gradients", grad_output[0].detach())
        )

    def generate(self, inp, target_class):
        self.model.eval()
        inp = inp.requires_grad_(True)

        output = self.model(inp)

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / cam.max()

        return cam


# Use the full RosaNet block as the Grad-CAM target module.
gradcam = GradCAM1D(model_d, model_d.rosa)


# =============================================================
# Correctly classified MB test segments
# =============================================================

test_mb = (
    test_d[test_d["label"] == 1]
    .copy()
    .reset_index(drop=True)
)

test_mb_audio = np.stack([
    center_crop(row["raw_audio"])
    for _, row in test_mb.iterrows()
])

with torch.no_grad():
    mb_preds = torch.argmax(
        model_d(
            torch.FloatTensor(
                test_mb_audio[:, np.newaxis, :]
            ).to(DEVICE)
        ),
        dim=1
    ).cpu().numpy()

correct_mask = mb_preds == 1
correct_idx = np.where(correct_mask)[0]

print(f"\nMB segments in test      : {len(test_mb):,}")
print(f"Correctly classified MB  : {len(correct_idx):,}")


# =============================================================
# Grad-CAM metrics
# =============================================================

results = []

for idx in correct_idx:
    audio = test_mb_audio[idx]
    pid = test_mb["patient_id"].values[idx]

    inp = (
        torch.FloatTensor(audio)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    cam = gradcam.generate(
        inp,
        target_class=1
    )

    cam_up = np.interp(
        np.linspace(0, 1, CROP_LEN),
        np.linspace(0, 1, len(cam)),
        cam
    )

    total = cam_up.sum()

    # Exclude negligible Grad-CAM maps from the interpretability subset.
    if total <= 1e-8:
        continue

    start_act = cam_up[:EDGE].sum()
    interior_act = cam_up[EDGE:-EDGE].sum()
    end_act = cam_up[-EDGE:].sum()

    start_dur = EDGE / CROP_LEN
    interior_dur = (CROP_LEN - 2 * EDGE) / CROP_LEN
    end_dur = EDGE / CROP_LEN

    start_frac = start_act / total
    interior_frac = interior_act / total
    end_frac = end_act / total

    start_density = start_frac / start_dur
    interior_density = interior_frac / interior_dur
    end_density = end_frac / end_dur

    boundary_density = (
        start_density + end_density
    ) / 2.0

    density_gap = (
        interior_density - boundary_density
    )

    results.append({
        "patient_id": pid,
        "start_frac": start_frac,
        "interior_frac": interior_frac,
        "end_frac": end_frac,
        "start_density": start_density,
        "interior_density": interior_density,
        "end_density": end_density,
        "boundary_density": boundary_density,
        "density_gap": density_gap
    })

gc_df = pd.DataFrame(results)

print(
    f"Valid Grad-CAM segments  : {len(gc_df):,} "
    f"of {len(correct_idx):,} correctly classified MB segments"
)


# =============================================================
# Patient-level aggregation
# =============================================================

gc_pat = (
    gc_df
    .groupby("patient_id")[
        [
            "start_density",
            "interior_density",
            "end_density",
            "boundary_density",
            "density_gap"
        ]
    ]
    .mean()
    .reset_index()
)

n_pat = len(gc_pat)

_, pval_w = wilcoxon(
    gc_pat["interior_density"].values,
    gc_pat["boundary_density"].values
)

print("\n" + "=" * 60)
print("GRAD-CAM — STRATEGY D (100-ms crop, no padding)")
print("=" * 60)
print(
    "Regions: start edge 0-25 ms | "
    "interior 25-75 ms | end edge 75-100 ms"
)
print(
    "Note: crop edges are fixed input limits, "
    "not physiological S1/S2 boundaries."
)

print(f"\nSEGMENT LEVEL (n={len(gc_df):,}):")
print("  Activation fraction:")
print(
    f"    Start edge : "
    f"{gc_df['start_frac'].mean():.3f} ± "
    f"{gc_df['start_frac'].std():.3f}"
)
print(
    f"    Interior   : "
    f"{gc_df['interior_frac'].mean():.3f} ± "
    f"{gc_df['interior_frac'].std():.3f}"
)
print(
    f"    End edge   : "
    f"{gc_df['end_frac'].mean():.3f} ± "
    f"{gc_df['end_frac'].std():.3f}"
)

print("  Activation density:")
print(
    f"    Start edge : "
    f"{gc_df['start_density'].mean():.3f} ± "
    f"{gc_df['start_density'].std():.3f}"
)
print(
    f"    Interior   : "
    f"{gc_df['interior_density'].mean():.3f} ± "
    f"{gc_df['interior_density'].std():.3f}"
)
print(
    f"    End edge   : "
    f"{gc_df['end_density'].mean():.3f} ± "
    f"{gc_df['end_density'].std():.3f}"
)
print(
    "  Segments interior > boundary: "
    f"{(gc_df['density_gap'] > 0).mean():.1%}"
)

print(f"\nPATIENT LEVEL (n={n_pat}):")
print(
    f"  Start-edge density   : "
    f"{gc_pat['start_density'].mean():.3f} ± "
    f"{gc_pat['start_density'].std():.3f}"
)
print(
    f"  Interior density     : "
    f"{gc_pat['interior_density'].mean():.3f} ± "
    f"{gc_pat['interior_density'].std():.3f}"
)
print(
    f"  End-edge density     : "
    f"{gc_pat['end_density'].mean():.3f} ± "
    f"{gc_pat['end_density'].std():.3f}"
)

interior_mean = gc_pat["interior_density"].mean()
boundary_mean = gc_pat["boundary_density"].mean()

print(
    f"  Interior/boundary ratio: "
    f"{interior_mean / boundary_mean:.3f}"
)
print(
    f"  Patients interior > boundary: "
    f"{(gc_pat['density_gap'] > 0).sum()}/{n_pat}"
)
print(
    f"  Wilcoxon p: {pval_w:.4g}"
)


# =============================================================
# Patient-level density ratios
# =============================================================

start_mean = gc_pat["start_density"].mean()
end_mean = gc_pat["end_density"].mean()

print("\nDensity ratios (patient-level):")
print(
    f"  Interior / Start    : "
    f"{interior_mean / start_mean:.3f}"
)
print(
    f"  Interior / End      : "
    f"{interior_mean / end_mean:.3f}"
)
print(
    f"  Interior / Boundary : "
    f"{interior_mean / boundary_mean:.3f}"
)


# =============================================================
# Representative Grad-CAM examples
# =============================================================

rng = np.random.default_rng(42)

plot_idxs = rng.choice(
    correct_idx,
    size=min(6, len(correct_idx)),
    replace=False
)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8)
)
axes = axes.flatten()

for panel_idx, idx in enumerate(plot_idxs):
    audio = test_mb_audio[idx]

    inp = (
        torch.FloatTensor(audio)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    cam = gradcam.generate(
        inp,
        target_class=1
    )

    cam_up = np.interp(
        np.linspace(0, 1, CROP_LEN),
        np.linspace(0, 1, len(cam)),
        cam
    )

    t = (
        np.arange(CROP_LEN)
        / SAMPLE_RATE
        * 1000
    )

    ax = axes[panel_idx]

    ax.plot(
        t,
        audio,
        alpha=0.6,
        linewidth=0.8,
        label="PCG"
    )

    ax.fill_between(
        t,
        0,
        cam_up * np.abs(audio).max(),
        alpha=0.4,
        label="Grad-CAM"
    )

    ax.axvline(
        EDGE / SAMPLE_RATE * 1000,
        linestyle=":",
        linewidth=1.5,
        label="Start edge (25 ms)"
    )

    ax.axvline(
        (CROP_LEN - EDGE) / SAMPLE_RATE * 1000,
        linestyle=":",
        linewidth=1.5,
        label="End edge (75 ms)"
    )

    ax.axvspan(
        0,
        EDGE / SAMPLE_RATE * 1000,
        alpha=0.08
    )

    ax.axvspan(
        (CROP_LEN - EDGE) / SAMPLE_RATE * 1000,
        CROP_LEN / SAMPLE_RATE * 1000,
        alpha=0.08
    )

    ax.set_xlabel("Time (ms)")
    ax.set_title(
        f"MB example {panel_idx + 1}"
    )

    if panel_idx == 0:
        ax.legend(fontsize=7)

    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Grad-CAM: Murmur-Bearing Segments (Strategy D)\n"
    "Fixed 100-ms real-signal crops",
    fontsize=11
)

plt.tight_layout()

os.makedirs("figures", exist_ok=True)
FIG_PATH = "figures/gradcam_strategy_d.png"

plt.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved: {FIG_PATH}")
