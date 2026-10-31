"""
Merge NHANES August 2021-August 2023 (suffix _L) component files into one
analysis dataset for PHQ-9 vs diabetes / hypertension.

Put the .xpt files (DEMO_L.xpt, DPQ_L.xpt, ...) in DATA_DIR, then run:
    python merge_nhanes.py
Output: nhanes_analysis.csv (one row per participant) and a sample-flow printout.

Check variable names against the NHANES codebooks for the _L cycle; a few
(especially BP medication items) can differ from older cycles.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("C:/Users/premc/OneDrive/Documents/mind_score_all_scripts_v2/MindScore-Dataset")          # folder holding the .xpt files
OUT_FILE = "nhanes_analysis.csv"

# ---------------------------------------------------------------- 1. load
FILES = [
    "DEMO_L",   # demographics + survey design/weights (base file)
    "DPQ_L",    # PHQ-9
    "DIQ_L",    # diabetes questionnaire
    "GHB_L",    # HbA1c
    "GLU_L",    # fasting glucose (+ fasting weight WTSAFPRP)
    "BPXO_L",   # blood pressure exam
    "BPQ_L",    # BP questionnaire
    "BMX_L",    # body measures
    "SMQ_L",    # smoking
    "ALQ_L",    # alcohol
    "PAQ_L",    # physical activity
    "SLQ_L",    # sleep
    "TCHOL_L",  # total cholesterol
    "HDL_L",    # HDL cholesterol
]


def read_xpt(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.xpt"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping")
        return None
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    df["SEQN"] = df["SEQN"].astype(int)
    assert df["SEQN"].is_unique, f"{name}: SEQN not unique"
    print(f"{name:8s} rows={len(df):6d} cols={df.shape[1]}")
    return df


frames = {n: read_xpt(n) for n in FILES}
frames = {n: d for n, d in frames.items() if d is not None}

# ---------------------------------------------------------------- 2. merge
# Left join from DEMO_L so no participant is silently dropped.
nh = frames.pop("DEMO_L")
for name, df in frames.items():
    dup = [c for c in df.columns if c in nh.columns and c != "SEQN"]
    if dup:                                   # avoid _x/_y columns
        df = df.rename(columns={c: f"{c}_{name}" for c in dup})
    nh = nh.merge(df, on="SEQN", how="left", validate="one_to_one")

print("\nMerged shape:", nh.shape)

# ---------------------------------------------------------------- 3. PHQ-9
phq_items = [f"DPQ0{i}0" for i in range(1, 10)]          # DPQ010..DPQ090
phq = nh[phq_items].replace({7: np.nan, 9: np.nan})       # refused / don't know
nh["phq9"] = phq.sum(axis=1, min_count=9)                 # NaN unless all 9 answered

nh["phq9_cat"] = pd.cut(
    nh["phq9"],
    bins=[-0.1, 4, 9, 14, 19, 27],
    labels=["0-4 minimal", "5-9 mild", "10-14 moderate",
            "15-19 mod. severe", "20-27 severe"],
)
# Optional collapsed version (severe group is usually small):
nh["phq9_cat4"] = nh["phq9_cat"].replace({"20-27 severe": "15-27 mod.severe/severe",
                                          "15-19 mod. severe": "15-27 mod.severe/severe"})

# ---------------------------------------------------------------- 4. diabetes
dx = nh["DIQ010"].eq(1)                                   # told by doctor
insulin = nh["DIQ050"].eq(1) if "DIQ050" in nh else False
pills = nh["DIQ070"].eq(1) if "DIQ070" in nh else False
a1c = nh["LBXGH"].ge(6.5) if "LBXGH" in nh else False
fpg = nh["LBXGLU"].ge(126) if "LBXGLU" in nh else False   # fasting subsample only

nh["diabetes_dx"] = np.where(nh["DIQ010"].isin([1, 2, 3]), dx.astype(float), np.nan)
nh["diabetes_any"] = (dx | insulin | pills | a1c | fpg).astype(float)
no_info = nh["DIQ010"].isna() & nh.get("LBXGH", pd.Series(np.nan, index=nh.index)).isna()
nh.loc[no_info, "diabetes_any"] = np.nan

# ---------------------------------------------------------------- 5. hypertension
sbp_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in nh]
dbp_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in nh]


def mean_bp(cols):
    """Mean of readings 2-3 if available, otherwise reading 1."""
    later = nh[cols[1:]].mean(axis=1)
    return later.fillna(nh[cols[0]])


nh["sbp"] = mean_bp(sbp_cols)
nh["dbp"] = mean_bp(dbp_cols)

bp_high = (nh["sbp"].ge(130) | nh["dbp"].ge(80))          # 2017 ACC/AHA
bp_high_old = (nh["sbp"].ge(140) | nh["dbp"].ge(90))      # older cutoff
told_htn = nh["BPQ020"].eq(1)
bp_meds = nh["BPQ050A"].eq(1) if "BPQ050A" in nh else False   # verify in codebook

nh["htn_any"] = (bp_high | told_htn | bp_meds).astype(float)
nh["htn_any_140_90"] = (bp_high_old | told_htn | bp_meds).astype(float)
no_bp_info = nh["sbp"].isna() & nh["BPQ020"].isna()
nh.loc[no_bp_info, ["htn_any", "htn_any_140_90"]] = np.nan

# ---------------------------------------------------------------- 6. covariates
nh["age"] = nh["RIDAGEYR"]
nh["female"] = nh["RIAGENDR"].eq(2).astype(float)
nh["race_eth"] = nh["RIDRETH3"]
nh["educ"] = nh["DMDEDUC2"].replace({7: np.nan, 9: np.nan})
nh["pir"] = nh["INDFMPIR"]
nh["bmi"] = nh["BMXBMI"]
nh["waist"] = nh["BMXWAIST"] if "BMXWAIST" in nh else np.nan
nh["sleep_hrs"] = nh["SLD012"] if "SLD012" in nh else np.nan
nh["ever_smoker"] = nh["SMQ020"].map({1: 1.0, 2: 0.0})
nh["tchol"] = nh["LBXTC"] if "LBXTC" in nh else np.nan
nh["hdl"] = nh["LBDHDD"] if "LBDHDD" in nh else np.nan

# ---------------------------------------------------------------- 7. sample flow
adults = nh["age"] >= 18
has_phq = nh["phq9"].notna()
print("\nSample flow")
print("All participants            :", len(nh))
print("Adults (>=18)               :", adults.sum())
print("  + complete PHQ-9          :", (adults & has_phq).sum())
print("  + diabetes status known   :", (adults & has_phq & nh['diabetes_any'].notna()).sum())
print("  + hypertension known      :", (adults & has_phq & nh['htn_any'].notna()).sum())

# Keep everyone in the file; filter at analysis time and use design-based
# domain (subpopulation) estimation rather than dropping rows beforehand.
nh["analysis_adult_phq"] = (adults & has_phq).astype(int)

nh.to_csv(OUT_FILE, index=False)
print(f"\nSaved {OUT_FILE}")
