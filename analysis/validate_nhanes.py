"""
Validate the merged NHANES file (nhanes_analysis.csv from merge_nhanes.py).

Usage:
    python validate_nhanes.py                 # uses nhanes_analysis.csv
    python validate_nhanes.py my_file.csv

Prints PASS / WARN / FAIL lines and writes validation_report.txt.
No plotting; it only needs pandas and numpy.
"""
import sys
from io import StringIO

import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "nhanes_analysis.csv"
df = pd.read_csv(PATH)

report = StringIO()
n_fail = 0
n_warn = 0


def say(level, msg):
    global n_fail, n_warn
    if level == "FAIL":
        n_fail += 1
    elif level == "WARN":
        n_warn += 1
    line = f"[{level}] {msg}"
    print(line)
    report.write(line + "\n")


def section(title):
    line = f"\n=== {title} ==="
    print(line)
    report.write(line + "\n")


def need(cols):
    """Return the columns from `cols` that exist; FAIL for any missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        say("FAIL", f"missing columns: {missing}")
    return [c for c in cols if c in df.columns]


# ------------------------------------------------------------ 1. structure
section("1. Structure")
say("PASS", f"loaded {PATH}: {df.shape[0]} rows x {df.shape[1]} columns")

if "SEQN" in df.columns:
    if df["SEQN"].is_unique:
        say("PASS", "SEQN is unique (one row per participant)")
    else:
        say("FAIL", f"SEQN has {df['SEQN'].duplicated().sum()} duplicates (bad merge)")
else:
    say("FAIL", "SEQN column not found")

# The 2021-2023 cycle has roughly 11,900 participants in DEMO_L.
if not (10000 <= len(df) <= 13000):
    say("WARN", f"{len(df)} rows is outside the expected ~11,900 for DEMO_L (Aug 2021-Aug 2023)")
else:
    say("PASS", "row count is in the expected range for this cycle")

dup_suffix = [c for c in df.columns if c.endswith(("_x", "_y"))]
if dup_suffix:
    say("WARN", f"columns with _x/_y suffixes (duplicate names across files): {dup_suffix}")

# ------------------------------------------------------------ 2. required columns
section("2. Required columns")
required = ["SEQN", "RIDAGEYR", "RIAGENDR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR",
            "phq9", "phq9_cat", "diabetes_any", "htn_any", "bmi"]
present = need(required)
if len(present) == len(required):
    say("PASS", "all key analysis columns present")

inputs_expected = {
    "BPQ020": "hypertension 'told by doctor' (BPQ_L not merged? htn_any then uses measured BP only)",
    "BPQ050A": "BP medication (BPQ_L not merged or variable renamed)",
    "DIQ010": "diabetes 'told by doctor'",
    "LBXGH": "HbA1c (GHB_L)",
}
for c, why in inputs_expected.items():
    if c not in df.columns:
        say("WARN", f"{c} not in file: {why}")

# ------------------------------------------------------------ 3. missingness
section("3. Missingness (percent missing; adults only)")
adults = df[df["RIDAGEYR"] >= 18] if "RIDAGEYR" in df.columns else df
say("PASS", f"adults (>=18): {len(adults)}")

check_cols = ["phq9", "diabetes_any", "diabetes_dx", "htn_any", "sbp", "dbp",
              "LBXGH", "LBXGLU", "bmi", "pir", "educ", "ever_smoker",
              "sleep_hrs", "WTMEC2YR"]
rows = []
for c in check_cols:
    if c in adults.columns:
        pct = adults[c].isna().mean() * 100
        rows.append((c, pct))
        lvl = "PASS" if pct < 10 else ("WARN" if pct < 40 else "FAIL")
        say(lvl, f"{c:14s} {pct:5.1f}% missing")
    else:
        say("WARN", f"{c} not in file")

# Fasting glucose is only collected on a fasting subsample, so high missingness is expected.
if "LBXGLU" in adults.columns and adults["LBXGLU"].isna().mean() > 0.4:
    say("PASS", "LBXGLU is mostly missing, which is expected (fasting subsample only)")

# ------------------------------------------------------------ 4. PHQ-9
section("4. PHQ-9")
if "phq9" in df.columns:
    p = df["phq9"].dropna()
    if p.empty:
        say("FAIL", "phq9 is entirely missing (check DPQ_L merge / item names)")
    else:
        if p.min() < 0 or p.max() > 27:
            say("FAIL", f"phq9 outside 0-27: min={p.min()}, max={p.max()}")
        else:
            say("PASS", f"phq9 within 0-27 (min={p.min():.0f}, max={p.max():.0f}, mean={p.mean():.2f})")
        item_cols = [f"DPQ0{i}0" for i in range(1, 10)]
        if all(c in df.columns for c in item_cols):
            recomputed = df[item_cols].replace({7: np.nan, 9: np.nan}).sum(axis=1, min_count=9)
            same = np.isclose(recomputed.fillna(-1), df["phq9"].fillna(-1)).all()
            say("PASS" if same else "FAIL", "phq9 matches a recomputation from items DPQ010-DPQ090")
            bad_codes = df[item_cols].isin([7, 9]).sum().sum()
            say("PASS", f"{bad_codes} item responses coded 7/9 were set to missing in the score")
        else:
            say("WARN", "PHQ-9 item columns not found, cannot recompute")

    if "phq9_cat" in df.columns:
        counts = df["phq9_cat"].value_counts(dropna=False).sort_index()
        print("\nPHQ-9 category counts (all participants):")
        print(counts.to_string())
        report.write("\nPHQ-9 category counts:\n" + counts.to_string() + "\n")
        sev = df["phq9_cat"].astype(str).str.contains("20-27").sum()
        if 0 < sev < 30:
            say("WARN", f"only {sev} people in the 20-27 group; use the collapsed phq9_cat4")

# ------------------------------------------------------------ 5. outcomes
section("5. Outcomes (unweighted; adults with complete PHQ-9)")
sub = adults[adults["phq9"].notna()] if "phq9" in adults.columns else adults
for col, lo, hi, label in [("diabetes_any", 0.08, 0.25, "diabetes (any)"),
                           ("htn_any", 0.30, 0.65, "hypertension (any)")]:
    if col in sub.columns:
        vals = sub[col].dropna().unique()
        if not set(vals).issubset({0, 1, 0.0, 1.0}):
            say("FAIL", f"{col} has values other than 0/1: {vals}")
            continue
        prev = sub[col].mean()
        lvl = "PASS" if lo <= prev <= hi else "WARN"
        say(lvl, f"{label}: unweighted prevalence {prev*100:.1f}% (typical US adult range ~{lo*100:.0f}-{hi*100:.0f}%)")
        if sub[col].sum() < 100:
            say("WARN", f"{col}: fewer than 100 events, limits modelling")

if {"diabetes_dx", "diabetes_any"} <= set(sub.columns):
    contradict = ((sub["diabetes_dx"] == 1) & (sub["diabetes_any"] == 0)).sum()
    say("PASS" if contradict == 0 else "FAIL",
        f"diagnosed diabetes but diabetes_any=0: {contradict} rows (should be 0)")

# ------------------------------------------------------------ 6. plausible ranges
section("6. Value ranges")
ranges = {
    "RIDAGEYR": (0, 80), "bmi": (10, 90), "sbp": (60, 260), "dbp": (30, 150),
    "LBXGH": (3, 18), "LBXGLU": (30, 600), "pir": (0, 5), "sleep_hrs": (1, 16),
}
for c, (lo, hi) in ranges.items():
    if c in df.columns and df[c].notna().any():
        mn, mx = df[c].min(), df[c].max()
        ok = mn >= lo and mx <= hi
        say("PASS" if ok else "WARN", f"{c:10s} min={mn:.1f} max={mx:.1f} (expected within {lo}-{hi})")

if {"sbp", "dbp"} <= set(df.columns):
    inverted = (df["dbp"] > df["sbp"]).sum()
    say("PASS" if inverted == 0 else "WARN", f"diastolic > systolic in {inverted} rows")

# ------------------------------------------------------------ 7. survey design
section("7. Survey design variables")
for c in ["SDMVPSU", "SDMVSTRA", "WTMEC2YR"]:
    if c in df.columns:
        n_na = df[c].isna().sum()
        say("PASS" if n_na == 0 or c == "WTMEC2YR" else "WARN", f"{c}: {n_na} missing")
if "WTMEC2YR" in df.columns:
    w = df["WTMEC2YR"]
    exam = w.notna() & (w > 0)
    say("PASS", f"{exam.sum()} participants have a positive exam weight (MEC-examined)")
    in_analysis = (df.get("analysis_adult_phq", pd.Series(0, index=df.index)) == 1)
    zero_w = (in_analysis & ~exam).sum()
    if zero_w:
        say("WARN", f"{zero_w} analysis-eligible adults have no exam weight; they drop out of weighted models")
    print(f"  Sum of exam weights: {w.sum():,.0f} (should approximate the US civilian population)")
if {"SDMVSTRA", "SDMVPSU"} <= set(df.columns):
    n_strata = df["SDMVSTRA"].nunique()
    n_psu = df.groupby("SDMVSTRA")["SDMVPSU"].nunique()
    say("PASS", f"{n_strata} strata, PSUs per stratum: min={n_psu.min()}, max={n_psu.max()}")

# ------------------------------------------------------------ 8. sample flow
section("8. Analysis sample flow")
flow = []
flow.append(("All participants", len(df)))
ad = df["RIDAGEYR"] >= 18
flow.append(("Adults >=18", int(ad.sum())))
m = ad & df["phq9"].notna()
flow.append(("+ complete PHQ-9", int(m.sum())))
if "WTMEC2YR" in df.columns:
    m = m & df["WTMEC2YR"].gt(0)
    flow.append(("+ positive exam weight", int(m.sum())))
for col, lab in [("diabetes_any", "+ diabetes status known"), ("htn_any", "+ hypertension status known")]:
    if col in df.columns:
        flow.append((lab, int((m & df[col].notna()).sum())))
cov = [c for c in ["bmi", "pir", "educ", "ever_smoker"] if c in df.columns]
if cov:
    flow.append((f"+ complete covariates {cov}", int((m & df[cov].notna().all(axis=1)).sum())))
for lab, n in flow:
    print(f"  {lab:55s} {n:6d}")
    report.write(f"  {lab:55s} {n:6d}\n")

# ------------------------------------------------------------ summary
section("Summary")
print(f"FAIL: {n_fail}   WARN: {n_warn}")
report.write(f"\nFAIL: {n_fail}   WARN: {n_warn}\n")
with open("validation_report.txt", "w") as f:
    f.write(report.getvalue())
print("Report saved to validation_report.txt")
