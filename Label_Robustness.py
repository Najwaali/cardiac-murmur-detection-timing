"""
Robustness analysis for the fixed 100-ms MB-vs-NMB acoustic model.

This script rebuilds the held-out Strategy D test cohort while preserving
recording-location, murmur-grade, and most-audible-location metadata. It then
evaluates the acoustic model across:

1. Aortic (AV), mitral (MV), pulmonic (PV), and tricuspid (TV) locations.
2. Murmur grade I/VI versus grade II/VI or higher.
3. The annotated most-audible recording location.

The rebuilt cohort is checked against the original held-out test set before
analysis.

Run after `strategy_d_fixed_crop.py` and `acoustic_duration_fusion.py`, or
import the required objects from those modules.
"""

import os

# =============================================================
# MURMUR GRADE AND RECORDING-LOCATION ROBUSTNESS ANALYSIS
# Rebuilds the held-out Strategy D test cohort with metadata
# =============================================================

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ── Rebuild test_d with actual recording location ─────────────
all_files = os.listdir(DATA_PATH)

df         = pd.read_csv(CSV_PATH)
df_present = df[df["Murmur"] == "Present"].copy()
df_present["murmur_phase"] = df_present.apply(
    get_murmur_phase, axis=1)
df_use = df_present[df_present["murmur_phase"].isin(
    ["systolic","diastolic"])].copy()

patient_meta = {}
for _, row in df_use.iterrows():
    pid   = str(row["Patient ID"])
    phase = row["murmur_phase"]
    locs  = str(row["Murmur locations"]) \
            if pd.notna(row["Murmur locations"]) else ""
    mal   = str(row["Most audible location"]) \
            if pd.notna(row["Most audible location"]) else ""
    grade = str(row["Systolic murmur grading"]) \
            if pd.notna(row["Systolic murmur grading"]) else ""
    patient_meta[pid] = {
        "phase":     phase,
        "locations": set(locs.split("+")) if locs else set(),
        "mal":       mal,
        "grade":     grade
    }

test_pat_set = set(test_d["patient_id"].unique())
records_test = []

print(f"Re-extracting {len(test_pat_set)} test patients "
      f"with location metadata...")

for pid in test_pat_set:
    if pid not in patient_meta:
        continue
    info         = patient_meta[pid]
    murmur_phase = info["phase"]
    murmur_locs  = info["locations"]
    mal          = info["mal"]
    grade        = info["grade"]

    wav_files = sorted([f for f in all_files
                        if f.startswith(pid+"_")
                        and f.endswith(".wav")])
    for wav_file in wav_files:
        rec_loc       = wav_file.replace(".wav","").split("_")[-1]
        at_murmur_loc = rec_loc in murmur_locs
        wav_path      = os.path.join(DATA_PATH, wav_file)
        tsv_path      = wav_path.replace(".wav", ".tsv")
        if not os.path.exists(tsv_path):
            continue
        try:
            signal = load_and_preprocess(wav_path)
        except Exception:
            continue

        tsv = pd.read_csv(tsv_path, sep='\t', header=None,
                          names=["start","end","label"])
        for _, row in tsv.iterrows():
            lbl = int(row["label"])
            if lbl not in [SYSTOLE_LABEL, DIASTOLE_LABEL]:
                continue
            start_sec = float(row["start"])
            end_sec   = float(row["end"])
            dur       = end_sec - start_sec
            if dur < MIN_DUR_S:
                continue
            seg_raw = extract_raw_segment(
                signal, start_sec, end_sec)
            if seg_raw is None:
                continue

            seg_phase = ("systolic" if lbl == SYSTOLE_LABEL
                         else "diastolic")
            is_mb = (seg_phase == murmur_phase
                     and at_murmur_loc)
            label = 1 if is_mb else 0

            records_test.append({
                "patient_id":     pid,
                "label":          label,
                "seg_phase":      seg_phase,
                "at_murmur_loc":  at_murmur_loc,
                "location":       rec_loc,
                "is_mal":         rec_loc == mal,
                "grade":          grade,
                "duration_sec":   dur,
                "raw_audio":      seg_raw
            })

test_d_full = pd.DataFrame(records_test)
print(f"\nExtraction complete:")
print(f"  Segments  : {len(test_d_full):,}")
print(f"  Locations : {test_d_full['location'].value_counts().to_dict()}")
print(f"  Grades    : {test_d_full['grade'].value_counts().to_dict()}")
print(f"  MAL segs  : {test_d_full['is_mal'].sum():,} "
      f"({test_d_full['is_mal'].mean():.1%})")

# ── Rebuild check ─────────────────────────────────────────────
print(f"\nREBUILD CHECK")
print(f"Original test segments : {len(test_d)}")
print(f"Rebuilt  test segments : {len(test_d_full)}")
print(f"Original patients      : {test_d['patient_id'].nunique()}")
print(f"Rebuilt  patients      : {test_d_full['patient_id'].nunique()}")
assert len(test_d_full) == len(test_d), \
    "Rebuilt test set does not match original!"
assert set(test_d_full["patient_id"]) == \
    set(test_d["patient_id"]), \
    "Patient sets do not match!"
print("Rebuild check passed.")

# ── Get probabilities ─────────────────────────────────────────
test_loader_full = make_loader(test_d_full, augment=False)
test_true_f, _, test_aco_f = evaluate(model_d, test_loader_full)
test_dur_f = lr_dur.predict_proba(
    test_d_full["duration_sec"].values.reshape(-1,1))[:,1]
X_fus_f    = np.column_stack([test_aco_f, test_dur_f])
test_fus_f = fusion_lr.predict_proba(X_fus_f)[:,1]

test_d_full["acoustic_prob"] = test_aco_f
test_d_full["fusion_prob"]   = test_fus_f
test_d_full["true_label"]    = test_true_f

print(f"\nVerification — AUROC on rebuilt test set:")
print(f"  Acoustic : {roc_auc_score(test_true_f, test_aco_f):.4f}")
print(f"  Fusion   : {roc_auc_score(test_true_f, test_fus_f):.4f}")

# ── Analysis 2A: Per-location sensitivity ─────────────────────
print(f"\n{'='*60}")
print("ANALYSIS 2A: PER-LOCATION SENSITIVITY")
print(f"{'='*60}")
print(f"{'Location':<10} {'N_segs':>7} {'N_MB':>6} "
      f"{'N_NMB':>6} {'N_pat':>6} {'AUROC':>7} {'BalAcc':>8}")
print("-"*57)

for loc in ["AV","MV","PV","TV"]:
    mask = test_d_full["location"] == loc
    sub  = test_d_full[mask]
    if len(sub) == 0 or sub["true_label"].nunique() < 2:
        print(f"{loc:<10} — insufficient data")
        continue
    auc   = roc_auc_score(sub["true_label"],
                           sub["acoustic_prob"])
    bal   = balanced_accuracy_score(
        sub["true_label"],
        (sub["acoustic_prob"]>=0.5).astype(int))
    n_pat = sub["patient_id"].nunique()
    n_mb  = (sub["true_label"]==1).sum()
    n_nmb = (sub["true_label"]==0).sum()
    print(f"{loc:<10} {len(sub):>7} {n_mb:>6} "
          f"{n_nmb:>6} {n_pat:>6} {auc:>7.4f} {bal:>8.4f}")

# ── Analysis 2B: Higher-grade sensitivity ─────────────────────
print(f"\n{'='*60}")
print("ANALYSIS 2B: HIGHER-GRADE SENSITIVITY")
print("Higher grade = II/VI or III/VI (excludes I/VI)")
print(f"{'='*60}")

grade_map = {
    "All test patients": np.ones(len(test_d_full), dtype=bool),
    "Grade I/VI":        test_d_full["grade"] == "I/VI",
    "Grade II/VI+":      test_d_full["grade"].isin(
                             ["II/VI","III/VI"])
}

for label, mask in grade_map.items():
    sub = test_d_full[mask]
    if len(sub) == 0 or sub["true_label"].nunique() < 2:
        print(f"\n{label}: insufficient data")
        continue
    auc   = roc_auc_score(sub["true_label"],
                           sub["acoustic_prob"])
    bal   = balanced_accuracy_score(
        sub["true_label"],
        (sub["acoustic_prob"]>=0.5).astype(int))
    n_pat = sub["patient_id"].nunique()
    n_mb  = (sub["true_label"]==1).sum()
    n_nmb = (sub["true_label"]==0).sum()
    print(f"\n{label}:")
    print(f"  Segments : {len(sub):,} "
          f"(MB={n_mb}, NMB={n_nmb}, patients={n_pat})")
    print(f"  AUROC    : {auc:.4f}")
    print(f"  BalAcc   : {bal:.4f}")

    # Per-patient paired within grade group
    pids_g = sub["patient_id"].values
    aco_g  = sub["acoustic_prob"].values
    lbl_g  = sub["true_label"].values
    gaps_g = []
    for pid in np.unique(pids_g):
        m    = pids_g == pid
        mb_m = lbl_g[m] == 1
        nm_m = lbl_g[m] == 0
        if mb_m.sum() >= 3 and nm_m.sum() >= 3:
            gaps_g.append(
                aco_g[m][mb_m].mean() -
                aco_g[m][nm_m].mean())
    if len(gaps_g) >= 5:
        _, pg = wilcoxon(gaps_g)
        print(f"  Per-patient gap : {np.mean(gaps_g):.4f} "
              f"(n={len(gaps_g)} patients, p={pg:.4f})")
    else:
        print(f"  Per-patient: only {len(gaps_g)} "
              f"qualifying patients — report descriptively")

# ── Analysis 2C: Most audible location only ───────────────────
print(f"\n{'='*60}")
print("ANALYSIS 2C: MOST AUDIBLE LOCATION ONLY")
print(f"{'='*60}")

sub_mal = test_d_full[test_d_full["is_mal"]]
print(f"Segments at most audible location : {len(sub_mal):,}")
print(f"MB  : {(sub_mal['true_label']==1).sum():,}")
print(f"NMB : {(sub_mal['true_label']==0).sum():,}")
print(f"Patients : {sub_mal['patient_id'].nunique()}")

if sub_mal["true_label"].nunique() > 1:
    auc_mal = roc_auc_score(sub_mal["true_label"],
                             sub_mal["acoustic_prob"])
    bal_mal = balanced_accuracy_score(
        sub_mal["true_label"],
        (sub_mal["acoustic_prob"]>=0.5).astype(int))
    print(f"AUROC    : {auc_mal:.4f}")
    print(f"BalAcc   : {bal_mal:.4f}")

    pids_m = sub_mal["patient_id"].values
    aco_m  = sub_mal["acoustic_prob"].values
    lbl_m  = sub_mal["true_label"].values
    gaps_m = []
    for pid in np.unique(pids_m):
        m    = pids_m == pid
        mb_m = lbl_m[m] == 1
        nm_m = lbl_m[m] == 0
        if mb_m.sum() >= 3 and nm_m.sum() >= 3:
            gaps_m.append(
                aco_m[m][mb_m].mean() -
                aco_m[m][nm_m].mean())
    if len(gaps_m) >= 5:
        _, pm = wilcoxon(gaps_m)
        print(f"Per-patient gap : {np.mean(gaps_m):.4f} "
              f"(n={len(gaps_m)} patients, p={pm:.4f})")
    else:
        print(f"Per-patient: {len(gaps_m)} qualifying patients"
              f" — report descriptively")
