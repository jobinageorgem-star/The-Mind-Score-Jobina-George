"""Check that the browser predictor (predict.js + models.json) matches scikit-learn exactly."""
import json
import sys

import numpy as np
import pandas as pd

from export_models_for_web import RACE, build, fit_all

PATH = sys.argv[1]
df = pd.read_csv(PATH)
d, feats, phq, w_raw = build(df)
w = w_raw / w_raw.mean()

full = feats.assign(phq9=phq)
full = full.fillna(full.median())                    # complete rows, so the imputer is a no-op
rng = np.random.default_rng(0)
idx = rng.choice(len(full), 100, replace=False)
case_X = full.iloc[idx].reset_index(drop=True)

cases = []
for _, r in case_X.iterrows():
    race = next(n for n in RACE.values() if r[f"race: {n}"] == 1) if any(r[f"race: {n}"] == 1 for n in RACE.values()) else "NH White"
    cases.append({"age": r["age"], "female": r["female"], "race": race, "pir": r["income_poverty_ratio"],
                  "educ": r["education (1-5)"], "smoked": r["ever_smoked"], "alcohol": r["alcohol (0-2)"],
                  "activity": r["leisure_activity_min_wk"], "sleep": r["sleep_hours"], "bmi": r["bmi"],
                  "phq9": r["phq9"]})

expected = {}
for oname, col in [("diabetes", "diabetes_any"), ("hypertension", "htn_any")]:
    y_all = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
    keep = ~np.isnan(y_all)
    y, ww = y_all[keep].astype(int), w[keep]
    for sname, cols in [("base", list(feats.columns)), ("phq", list(feats.columns) + ["phq9"])]:
        Xk = feats.assign(phq9=phq).loc[keep, cols].reset_index(drop=True)   # same input as the export (with NaNs)
        _, fitted = fit_all(Xk, y, ww)
        Xc = case_X[cols].to_numpy()
        sc, lr = fitted["lr"]
        expected[f"{oname}|{sname}"] = {
            "lr": lr.predict_proba(sc.transform(Xc))[:, 1].tolist(),
            "dt": fitted["dt"].predict_proba(Xc)[:, 1].tolist(),
            "rf": fitted["rf"].predict_proba(Xc)[:, 1].tolist(),
            "gb": fitted["gb"].predict_proba(Xc)[:, 1].tolist(),
        }

json.dump(cases, open("parity_cases.json", "w"))
json.dump(expected, open("parity_expected.json", "w"))
print("wrote parity files for", len(cases), "people")
