"""
NHANES 2021-2023: does PHQ-9 help predict diabetes / hypertension, and which variables matter?

Models: regularised logistic regression, decision tree, random forest, XGBoost.
Each is fitted with and without PHQ-9 (same folds), using survey weights (WTMEC2YR)
for fitting and for the evaluation metrics.

Outputs: weighted AUC and Brier score (prediction_cv_summary.csv) and, new in this version,
survey-weighted classification measures from the confusion matrix (accuracy, sensitivity,
specificity, precision, F1) in classification_measures_summary.csv.

Usage:   python nhanes_prediction.py [path_to_nhanes_analysis.csv]
Requires: numpy, pandas, scipy, scikit-learn  (+ xgboost for the XGBoost model)
          pip install -U scikit-learn xgboost
"""
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    import xgboost
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

PATH = r"C:\Users\premc\OneDrive\Documents\mind_score_all_scripts_v2\analysis\nhanes_analysis.csv"
SEED = 42
N_SPLITS, N_REPEATS = 5, 3           # 15 train/test splits per model
N_PERM_REPEATS = 30                  # permutation-importance repeats
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"scikit-learn {sklearn.__version__}",
      f"| xgboost {xgboost.__version__}" if HAVE_XGB else
      "| xgboost NOT installed (XGBoost model will be skipped: pip install -U xgboost)")
if not HAVE_XGB:
    print("\n*** WARNING: XGBoost is missing, so the XGBoost rows will NOT be produced. "
          "Install it and rerun before using these results in the report. ***\n")

# ====================================================================== data
df = pd.read_csv(PATH)
w_all = df["WTMEC2YR"].fillna(0).to_numpy(float)
dom = ((df["RIDAGEYR"] >= 18) & df["phq9"].notna()).to_numpy() & (w_all > 0)
d = df.loc[dom].reset_index(drop=True)          # adults with a PHQ-9 and an exam weight
w_dom = w_all[dom]
print(f"Analysis population: {len(d)} adults with PHQ-9")


def num(col):
    return pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index)


RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
        4: "NH Black", 6: "NH Asian", 7: "Other/multiracial"}
race_code = num("race_eth").round()

feat = pd.DataFrame(index=d.index)
feat["age"] = num("age")
feat["female"] = num("female")
for code, name in RACE.items():
    feat[f"race: {name}"] = np.where(race_code.isna(), np.nan, (race_code == code).astype(float))
feat["income_poverty_ratio"] = num("pir")
feat["education (1-5)"] = num("educ").round()
feat["ever_smoked"] = num("ever_smoker")

# Alcohol (ALQ_L): lifetime abstainers skip ALQ121 -> code as 0. 0 none, 1 low (<=3 d/mo), 2 regular (>=1 d/wk)
a111, a121 = num("ALQ111").round(), num("ALQ121").round()
acode = a121.where(~a121.isin([77, 99])).mask(a111 == 2, 0)
alc = pd.Series(np.nan, index=d.index)
alc[acode == 0] = 0
alc[(acode >= 6) & (acode <= 10)] = 1
alc[(acode >= 1) & (acode <= 5)] = 2
feat["alcohol (0-2)"] = alc

# Leisure activity (PAQ_L): minutes/week, vigorous counted twice
UNIT = {"D": 7.0, "W": 1.0, "M": 12 / 52, "Y": 1 / 52}


def weekly_minutes(q, u, m):
    qq = num(q).round().mask(lambda s: s >= 77)
    per_week = d[u].map(UNIT) if u in d.columns else pd.Series(np.nan, index=d.index)
    mm = num(m).mask(lambda s: s >= 777)
    return (qq * per_week * mm).mask(qq == 0, 0.0)


mvpa = weekly_minutes("PAD790Q", "PAD790U", "PAD800") + 2 * weekly_minutes("PAD810Q", "PAD810U", "PAD820")
feat["leisure_activity_min_wk"] = mvpa.clip(lower=0, upper=2000)
feat["sleep_hours"] = num("sleep_hrs")
feat["bmi"] = num("bmi")

X_BASE = feat
X_PHQ = feat.assign(phq9=num("phq9"))
FEATURE_SETS = {"Base (no PHQ-9)": X_BASE, "Base + PHQ-9": X_PHQ}

OUTCOMES = [("Diabetes", "diabetes_any"), ("Hypertension", "htn_any")]
# Leakage guard: diabetes_any / htn_any are DEFINED from HbA1c, glucose, blood pressure,
# diagnosis and medication items, so none of those may be used as predictors.


# ===================================================================== models
def make_models():
    """Fresh, regularised models. Trees/forest use min_samples_leaf to limit overfitting."""
    imp = lambda: SimpleImputer(strategy="median")        # fitted inside each training fold
    models = {
        "Logistic regression": Pipeline([("imp", imp()), ("scale", StandardScaler()),
                                         ("clf", LogisticRegression(C=1.0, max_iter=2000))]),
        "Decision tree": Pipeline([("imp", imp()),
                                   ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                                                  random_state=SEED))]),
        "Random forest": Pipeline([("imp", imp()),
                                   ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=20,
                                                                  max_features="sqrt", n_jobs=-1,
                                                                  random_state=SEED))]),
    }
    if HAVE_XGB:   # handles missing values natively, so no imputer
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)
    return models


def fit(model, X, y, w):
    if isinstance(model, Pipeline):
        return model.fit(X, y, clf__sample_weight=w)
    return model.fit(X, y, sample_weight=w)


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def corrected_paired_t(diffs, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for repeated CV (accounts for overlap)."""
    k = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    se = np.sqrt((1 / k + n_test / n_train) * var)
    t = mean / se if se > 0 else np.nan
    tc = stats.t.ppf(0.975, k - 1)
    return mean, mean - tc * se, mean + tc * se, 2 * stats.t.sf(abs(t), k - 1)


def weighted_confusion_metrics(y, p, w, threshold):
    """Survey-weighted accuracy, sensitivity (recall), specificity, precision and F1.

    A predicted risk >= threshold counts as a predicted case. Every count is a weighted count.
    """
    y = np.asarray(y)
    w = np.asarray(w, float)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    tn = w[(pred == 0) & (y == 0)].sum()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else np.nan
    return dict(accuracy=(tp + tn) / (tp + tn + fp + fn), sensitivity=sens,
                specificity=spec, precision=prec, f1=f1)


# ===================================================== Q5: with vs without PHQ-9
cv_rows, summary_rows = [], []
for out_label, out_col in OUTCOMES:
    y_all = pd.to_numeric(d[out_col], errors="coerce").to_numpy(float)
    keep = ~np.isnan(y_all)
    y = y_all[keep].astype(int)
    w = w_dom[keep] / w_dom[keep].mean()          # weights rescaled to mean 1
    sets = {k: v.loc[keep].reset_index(drop=True) for k, v in FEATURE_SETS.items()}
    print(f"\n=== {out_label}: n={len(y)}, cases={int(y.sum())} ({y.mean():.1%} unweighted) ===")

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for fold, (tr, te) in enumerate(rskf.split(sets["Base + PHQ-9"], y)):
        for fs_name, X in sets.items():
            for m_name, model in make_models().items():
                fit(model, X.iloc[tr], y[tr], w[tr])
                p = proba(model, X.iloc[te])
                thr = np.average(y[tr], weights=w[tr])    # threshold = weighted prevalence in training fold
                cm = weighted_confusion_metrics(y[te], p, w[te], thr)
                cv_rows.append(dict(outcome=out_label, model=m_name, features=fs_name, fold=fold,
                                    auc=roc_auc_score(y[te], p, sample_weight=w[te]),
                                    brier=brier_score_loss(y[te], p, sample_weight=w[te]),
                                    **cm, n_train=len(tr), n_test=len(te)))
        print(f"  fold {fold + 1}/{N_SPLITS * N_REPEATS} done", end="\r")
    print()

cv = pd.DataFrame(cv_rows)
cv.to_csv("prediction_cv_fold_results.csv", index=False)

for out_label, _ in OUTCOMES:
    for m_name in cv.model.unique():
        sub = cv[(cv.outcome == out_label) & (cv.model == m_name)]
        a = sub[sub.features == "Base (no PHQ-9)"].sort_values("fold")
        b = sub[sub.features == "Base + PHQ-9"].sort_values("fold")
        diff, lo, hi, p = corrected_paired_t(b.auc.to_numpy() - a.auc.to_numpy(),
                                             int(a.n_train.iloc[0]), int(a.n_test.iloc[0]))
        summary_rows.append({
            "Outcome": out_label, "Model": m_name,
            "AUC base": f"{a.auc.mean():.3f} ({a.auc.std():.3f})",
            "AUC +PHQ-9": f"{b.auc.mean():.3f} ({b.auc.std():.3f})",
            "Delta AUC (95% CI)": f"{diff:+.4f} ({lo:+.4f} to {hi:+.4f})",
            "p": "<0.001" if p < 0.001 else f"{p:.3f}",
            "Brier base": f"{a.brier.mean():.4f}", "Brier +PHQ-9": f"{b.brier.mean():.4f}"})

summary = pd.DataFrame(summary_rows)
summary.to_csv("prediction_cv_summary.csv", index=False)

# Classification measures (threshold = weighted prevalence), mean over folds, with vs without PHQ-9
cm_cols = ["accuracy", "sensitivity", "specificity", "precision", "f1"]
cm_summary = (cv.groupby(["outcome", "model", "features"])[cm_cols].mean().round(3)
              .unstack("features"))
cm_summary.columns = [f"{m} | {f}" for m, f in cm_summary.columns]
cm_summary = cm_summary.reset_index()
cm_summary.to_csv("classification_measures_summary.csv", index=False)
print("\nSurvey-weighted classification measures (threshold = weighted prevalence; mean over folds)")
print(cm_summary.to_string(index=False))
print("\nCross-validated performance (5-fold x 3 repeats; survey-weighted AUC, mean (SD) over folds)")
print("Delta AUC = AUC(with PHQ-9) - AUC(without); CI/p use the Nadeau-Bengio corrected t-test.\n")
print(summary.to_string(index=False))

# ============================================== Q4: which variables matter?
print("\n\n=== Variable importance (held-out 30% test set, model with PHQ-9) ===")
for out_label, out_col in OUTCOMES:
    y_all = pd.to_numeric(d[out_col], errors="coerce").to_numpy(float)
    keep = ~np.isnan(y_all)
    y = y_all[keep].astype(int)
    w = w_dom[keep] / w_dom[keep].mean()
    X = X_PHQ.loc[keep].reset_index(drop=True)
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, w, test_size=0.30, stratify=y, random_state=SEED)

    imp_cols, models = {}, make_models()
    for m_name, model in models.items():
        fit(model, Xtr, ytr, wtr)
        auc_te = roc_auc_score(yte, proba(model, Xte), sample_weight=wte)
        pi = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=N_PERM_REPEATS,
                                    random_state=SEED, sample_weight=wte, n_jobs=1)
        imp_cols[m_name] = pd.Series(pi.importances_mean, index=X.columns)
        print(f"  {out_label} / {m_name}: held-out weighted AUC = {auc_te:.3f}")

    imp = pd.DataFrame(imp_cols)                                   # drop in AUC when shuffled
    imp["mean rank"] = imp.rank(ascending=False).mean(axis=1)
    imp = imp.sort_values("mean rank")
    imp.round(4).to_csv(f"feature_importance_{out_label.lower()}.csv")
    print(f"\n{out_label}: permutation importance (drop in weighted AUC when the variable is shuffled; "
          f"higher = more important)")
    print(imp.round(4).to_string())

    lr = models["Logistic regression"]
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=X.columns)
    odds = np.exp(coefs).sort_values(ascending=False).round(3)
    print(f"\n{out_label}: logistic regression, odds ratio per 1 SD increase (all variables standardised)")
    print(odds.to_string())

    tree = models["Decision tree"].named_steps["clf"]
    print(f"\n{out_label}: decision tree rules (depth <= 3 shown)")
    print(export_text(tree, feature_names=list(X.columns), max_depth=3))

print("\nSaved: prediction_cv_summary.csv, classification_measures_summary.csv, prediction_cv_fold_results.csv, feature_importance_*.csv")
