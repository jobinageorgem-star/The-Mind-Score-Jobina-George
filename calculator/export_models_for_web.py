"""
Fit the NHANES risk models and export them as compact JSON for the browser calculator.

Models (each fitted without and with PHQ-9, for diabetes and hypertension):
  logistic regression, decision tree, random forest, gradient boosting (scikit-learn).
Survey weights (WTMEC2YR) are used when fitting. Predictors are the same as in
nhanes_prediction.py.

Also computes, for the page, cross-validated survey-weighted classification measures
(accuracy, sensitivity/recall, specificity, precision, F1) with and without PHQ-9. A predicted
risk at or above the weighted prevalence of the training fold counts as a predicted case.

Usage:  python export_models_for_web.py [nhanes_analysis.csv] [out.json]
Needs:  numpy, pandas, scikit-learn
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PATH = sys.argv[1] if len(sys.argv) > 1 else "nhanes_analysis.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models.json"
SEED = 42

# ----------------------------------------------------------------- features
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
        4: "NH Black", 6: "NH Asian", 7: "Other/multiracial"}
UNIT = {"D": 7.0, "W": 1.0, "M": 12 / 52, "Y": 1 / 52}


def build(df):
    w_all = df["WTMEC2YR"].fillna(0).to_numpy(float)
    dom = ((df["RIDAGEYR"] >= 18) & df["phq9"].notna()).to_numpy() & (w_all > 0)
    d = df.loc[dom].reset_index(drop=True)
    w = w_all[dom]

    def num(col):
        return pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index)

    race = num("race_eth").round()
    f = pd.DataFrame(index=d.index)
    f["age"] = num("age")
    f["female"] = num("female")
    for code, name in RACE.items():
        f[f"race: {name}"] = np.where(race.isna(), np.nan, (race == code).astype(float))
    f["income_poverty_ratio"] = num("pir")
    f["education (1-5)"] = num("educ").round()
    f["ever_smoked"] = num("ever_smoker")

    a111, a121 = num("ALQ111").round(), num("ALQ121").round()
    acode = a121.where(~a121.isin([77, 99])).mask(a111 == 2, 0)
    alc = pd.Series(np.nan, index=d.index)
    alc[acode == 0] = 0
    alc[(acode >= 6) & (acode <= 10)] = 1
    alc[(acode >= 1) & (acode <= 5)] = 2
    f["alcohol (0-2)"] = alc

    def weekly(q, u, m):
        qq = num(q).round().mask(lambda s: s >= 77)
        per_week = d[u].map(UNIT) if u in d.columns else pd.Series(np.nan, index=d.index)
        mm = num(m).mask(lambda s: s >= 777)
        return (qq * per_week * mm).mask(qq == 0, 0.0)

    mvpa = weekly("PAD790Q", "PAD790U", "PAD800") + 2 * weekly("PAD810Q", "PAD810U", "PAD820")
    f["leisure_activity_min_wk"] = mvpa.clip(lower=0, upper=2000)
    f["sleep_hours"] = num("sleep_hrs")
    f["bmi"] = num("bmi")
    return d, f, num("phq9"), w


def weighted_confusion_metrics(y, p, w, threshold):
    """Survey-weighted accuracy, sensitivity (recall), specificity, precision and F1."""
    y = np.asarray(y); w = np.asarray(w, float)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = w[(pred == 1) & (y == 1)].sum(); fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum(); tn = w[(pred == 0) & (y == 0)].sum()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else float("nan")
    return {"acc": (tp + tn) / (tp + tn + fp + fn), "sens": sens, "spec": spec, "prec": prec, "f1": f1}


# ------------------------------------------------- cross-validated AUC (same models)
def light_models():
    return {
        "lr": LogisticRegression(C=1.0, max_iter=2000),
        "dt": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=60, min_samples_leaf=25, max_features="sqrt",
                                     n_jobs=-1, random_state=SEED),
        "gb": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3,
                                         subsample=0.8, min_samples_leaf=20, random_state=SEED),
    }


def cv_auc(sets, y, w, n_splits=5, n_repeats=3):
    """Weighted AUC per fold for each model and feature set, on identical folds."""
    res = {m: {s: [] for s in sets} for m in light_models()}
    cmres = {m: {s: [] for s in sets} for m in light_models()}
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    ref = next(iter(sets.values()))
    n_tr = n_te = 0
    for tr, te in rskf.split(ref, y):
        n_tr, n_te = len(tr), len(te)
        for sname, X in sets.items():
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
            Xtr, Xte = imp.transform(X.iloc[tr]), imp.transform(X.iloc[te])
            sc = StandardScaler().fit(Xtr)
            for mname, model in light_models().items():
                if mname == "lr":
                    model.fit(sc.transform(Xtr), y[tr], sample_weight=w[tr])
                    p = model.predict_proba(sc.transform(Xte))[:, 1]
                else:
                    model.fit(Xtr, y[tr], sample_weight=w[tr])
                    p = model.predict_proba(Xte)[:, 1]
                res[mname][sname].append(roc_auc_score(y[te], p, sample_weight=w[te]))
                thr = float(np.average(y[tr], weights=w[tr]))
                cmres[mname][sname].append(weighted_confusion_metrics(y[te], p, w[te], thr))
    out = {}
    for m, d in res.items():
        a, b = np.array(d["base"]), np.array(d["phq"])
        diff = b - a
        k = len(diff)
        se = np.sqrt((1 / k + n_te / n_tr) * diff.var(ddof=1))      # Nadeau-Bengio correction
        tc = stats.t.ppf(0.975, k - 1)
        p = 2 * stats.t.sf(abs(diff.mean() / se), k - 1) if se > 0 else 1.0
        out[m] = {"base": round(float(a.mean()), 4), "phq": round(float(b.mean()), 4),
                  "delta": round(float(diff.mean()), 4),
                  "lo": round(float(diff.mean() - tc * se), 4), "hi": round(float(diff.mean() + tc * se), 4),
                  "p": round(float(p), 3),
                  "cm": {sn: {k: round(float(np.nanmean([f[k] for f in cmres[m][sn]])), 3)
                              for k in ("acc", "sens", "spec", "prec", "f1")} for sn in sets}}
    return out


# ------------------------------------------------------------------- export
def r(x, nd=5):
    return [round(float(v), nd) for v in x]


def tree_json(t, kind):
    """kind='prob' -> leaf value = P(class 1); kind='raw' -> regression leaf (log-odds step)."""
    tr = t.tree_
    if kind == "prob":
        val = tr.value[:, 0, :]
        v = val[:, 1] / val.sum(axis=1)
    else:
        v = tr.value[:, 0, 0]
    leaf = tr.children_left == -1
    return {"f": [int(a) for a in np.where(leaf, -1, tr.feature)],
            "t": r(np.where(leaf, 0, tr.threshold), 6),
            "l": [int(a) for a in tr.children_left],
            "r": [int(a) for a in tr.children_right],
            "v": r(v, 5)}


def fit_all(X, y, w):
    """Returns (json_dict, fitted objects for later checking)."""
    imp = SimpleImputer(strategy="median").fit(X)
    Xi = imp.transform(X)
    out, fitted = {}, {}

    sc = StandardScaler().fit(Xi)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xi), y, sample_weight=w)
    out["lr"] = {"mean": r(sc.mean_, 6), "scale": r(sc.scale_, 6),
                 "coef": r(lr.coef_[0], 6), "intercept": round(float(lr.intercept_[0]), 6)}
    fitted["lr"] = (sc, lr)

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=SEED).fit(Xi, y, sample_weight=w)
    out["dt"] = tree_json(dt, "prob")
    fitted["dt"] = dt

    rf = RandomForestClassifier(n_estimators=60, min_samples_leaf=25, max_features="sqrt",
                                n_jobs=-1, random_state=SEED).fit(Xi, y, sample_weight=w)
    out["rf"] = [tree_json(t, "prob") for t in rf.estimators_]
    fitted["rf"] = rf

    gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, subsample=0.8,
                                    min_samples_leaf=20, random_state=SEED).fit(Xi, y, sample_weight=w)
    trees = [e[0] for e in gb.estimators_]
    first = Xi[:1]
    init = float(gb.decision_function(first)[0] - gb.learning_rate * sum(t.predict(first)[0] for t in trees))
    out["gb"] = {"init": round(init, 6), "lr": gb.learning_rate, "trees": [tree_json(t, "raw") for t in trees]}
    fitted["gb"] = gb

    out["medians"] = r(imp.statistics_, 4)
    fitted["imp"] = imp
    return out, fitted


if __name__ == "__main__":
    df = pd.read_csv(PATH)
    d, feats, phq, w_raw = build(df)
    w = w_raw / w_raw.mean()
    print(f"Adults with PHQ-9: {len(d)}")

    outcomes = {"diabetes": "diabetes_any", "hypertension": "htn_any"}
    sets = {"base": feats, "phq": feats.assign(phq9=phq)}
    bundle = {"outcomes": {}, "features": {k: list(v.columns) for k, v in sets.items()}}

    for oname, col in outcomes.items():
        y_all = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        keep = ~np.isnan(y_all)
        y, ww = y_all[keep].astype(int), w[keep]
        prev = float(np.average(y, weights=w_raw[keep]))
        bundle["outcomes"][oname] = {"prevalence": round(prev, 4), "n": int(keep.sum()),
                                     "cases": int(y.sum()), "models": {}}
        for sname, X in sets.items():
            out, _ = fit_all(X.loc[keep].reset_index(drop=True), y, ww)
            bundle["outcomes"][oname]["models"][sname] = out
        Xs = {k: v.loc[keep].reset_index(drop=True) for k, v in sets.items()}
        bundle["outcomes"][oname]["cv"] = cv_auc(Xs, y, ww / ww.mean())
        print(f"{oname}: n={keep.sum()}, weighted prevalence {prev:.1%}; cv:",
              {m: (v["base"], v["phq"]) for m, v in bundle["outcomes"][oname]["cv"].items()})

    bundle["ranges"] = {c: [round(float(feats[c].quantile(.01)), 2), round(float(feats[c].quantile(.99)), 2)]
                        for c in ["age", "bmi", "income_poverty_ratio", "sleep_hours", "leisure_activity_min_wk"]}
    with open(OUT, "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print(f"Wrote {OUT}")
