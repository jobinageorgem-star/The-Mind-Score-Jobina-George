"""
NHANES 2021-2023: survey-weighted logistic models of diabetes / hypertension on PHQ-9
(continuous, categorical, trend) with stepwise adjustment. Single self-contained file.

Usage:  python nhanes_models.py [path_to_nhanes_analysis.csv]
Requires: numpy, pandas, scipy.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats


def _fit_logit(X, y, w, max_iter=50, tol=1e-10):
    """Weighted logistic regression by Newton-Raphson (point estimates only)."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(X @ beta)))
        grad = X.T @ (w * (y - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta
    raise RuntimeError("Logistic regression did not converge "
                       "(check for perfect separation or collinear predictors).")


class SurveyDesign:
    def __init__(self, df, strata="SDMVSTRA", psu="SDMVPSU", weight="WTMEC2YR"):
        for c in (strata, psu, weight):
            if c not in df.columns:
                raise KeyError(f"column {c!r} not found in data")
        self.df = df.reset_index(drop=True)
        d = self.df
        self.strata_col, self.psu_col, self.weight_col = strata, psu, weight

        if d[[strata, psu]].isna().any().any():
            raise ValueError("Missing values in strata/PSU. Every row needs both.")

        # Weight 0 or NaN = not in the exam sample; row stays in the design.
        self.w = d[weight].fillna(0.0).to_numpy(float)
        if (self.w < 0).any():
            raise ValueError("Negative weights found.")

        # PSU numbers (1, 2) repeat across strata, so a cluster = (stratum, PSU).
        self.cluster = d.groupby([strata, psu]).ngroup().to_numpy()
        stratum = d.groupby(strata).ngroup().to_numpy()
        cs = (pd.DataFrame({"cluster": self.cluster, "stratum": stratum})
              .drop_duplicates().sort_values("cluster"))
        self.cl_stratum = cs["stratum"].to_numpy()
        self.n_clusters = len(self.cl_stratum)
        self.n_strata = len(np.unique(self.cl_stratum))
        self.psu_per_stratum = np.bincount(self.cl_stratum)
        self.df_design = self.n_clusters - self.n_strata

    # ------------------------------------------------------------------ info
    def summary(self):
        print("Survey design")
        print(f"  rows in design        : {len(self.df)}")
        print(f"  strata ({self.strata_col})  : {self.n_strata}")
        print(f"  PSUs ({self.psu_col}, nested): {self.n_clusters}")
        print(f"  PSUs per stratum      : min={self.psu_per_stratum.min()}, "
              f"max={self.psu_per_stratum.max()}")
        print(f"  design df (PSU-strata): {self.df_design}")
        print(f"  rows with weight > 0  : {int((self.w > 0).sum())}")
        print(f"  sum of weights        : {self.w.sum():,.0f}")
        if (self.psu_per_stratum < 2).any():
            print("  WARNING: some strata have a single PSU (lonely PSU); "
                  "those strata contribute no variance.")

    # -------------------------------------------------------------- variance
    def _vcov_total(self, u):
        """Variance of the weighted total of linearized scores u (n x p)."""
        u = np.asarray(u, float)
        if u.ndim == 1:
            u = u[:, None]
        tot = np.zeros((self.n_clusters, u.shape[1]))
        np.add.at(tot, self.cluster, u)           # sum within each PSU
        V = np.zeros((u.shape[1], u.shape[1]))
        for h in range(self.n_strata):
            idx = np.where(self.cl_stratum == h)[0]
            n_h = len(idx)
            if n_h < 2:
                continue
            t = tot[idx] - tot[idx].mean(axis=0)  # center within stratum
            V += n_h / (n_h - 1) * (t.T @ t)
        return V

    def _tcrit(self, level):
        return stats.t.ppf(1 - (1 - level) / 2, self.df_design)

    # ----------------------------------------------------------- estimators
    def mean(self, y, domain=None, level=0.95):
        """Weighted mean / proportion of y in a domain, with design-based SE/CI.

        y      : array-like (NaN = missing)
        domain : boolean mask (same length as design) or None
        Proportions (all values 0/1) get a logit-scale CI, as in Stata.
        """
        y = np.asarray(y, float)
        dom = (self.w > 0) & ~np.isnan(y)
        if domain is not None:
            dom &= np.asarray(domain, bool)
        if dom.sum() == 0:
            return dict(estimate=np.nan, se=np.nan, lo=np.nan, hi=np.nan, n=0, pop=0.0)

        wd = np.where(dom, self.w, 0.0)
        yy = np.where(dom, y, 0.0)
        W = wd.sum()
        m = (wd * yy).sum() / W
        z = wd * (yy - m) / W                      # linearized ratio estimator
        se = float(np.sqrt(self._vcov_total(z)[0, 0]))
        tc = self._tcrit(level)

        binary = set(np.unique(y[dom])) <= {0.0, 1.0}
        if binary and 0 < m < 1 and se > 0:
            se_logit = se / (m * (1 - m))
            lg = np.log(m / (1 - m))
            lo, hi = (1 / (1 + np.exp(-(lg - tc * se_logit))),
                      1 / (1 + np.exp(-(lg + tc * se_logit))))
        else:
            lo, hi = m - tc * se, m + tc * se
        return dict(estimate=m, se=se, lo=lo, hi=hi, n=int(dom.sum()), pop=W)

    def mean_by(self, y, group, domain=None, order=None, level=0.95):
        """Weighted mean/proportion of y within each level of `group`."""
        group = pd.Series(np.asarray(group, dtype=object))
        levels = order if order is not None else sorted(group.dropna().unique())
        rows = []
        for g in levels:
            in_g = (group == g).to_numpy()
            if domain is not None:
                in_g = in_g & np.asarray(domain, bool)
            r = self.mean(y, domain=in_g, level=level)
            r["group"] = g
            rows.append(r)
        table = pd.DataFrame(rows).set_index("group")
        return table[["n", "pop", "estimate", "se", "lo", "hi"]]

    def logit(self, y, X, domain=None, add_const=True, level=0.95):
        """Survey-weighted logistic regression with linearized (sandwich) SEs.

        X: DataFrame of predictors (dummy-code categoricals first, e.g. pd.get_dummies).
        Rows outside `domain` or with missing y/X keep their place in the design
        but get weight 0, so the variance stays correct.
        Returns a table of coefficients, odds ratios, CIs and p-values.
        """
        X = pd.DataFrame(X).reset_index(drop=True).astype(float)
        if add_const:
            X.insert(0, "const", 1.0)
        names = list(X.columns)
        y = np.asarray(y, float)

        ok = (self.w > 0) & ~np.isnan(y) & ~X.isna().any(axis=1).to_numpy()
        if domain is not None:
            ok &= np.asarray(domain, bool)
        if ok.sum() <= len(names):
            raise ValueError("Too few complete observations for this model.")

        Xm = np.where(ok[:, None], X.to_numpy(), 0.0)
        wd = np.where(ok, self.w, 0.0)
        yy = np.where(ok, y, 0.0)

        beta = _fit_logit(Xm[ok], yy[ok], wd[ok])

        p = 1 / (1 + np.exp(-(Xm @ beta)))
        scores = (wd * (yy - p))[:, None] * Xm        # weighted score contributions
        B = self._vcov_total(scores)                  # "meat"
        A = (Xm * (wd * p * (1 - p))[:, None]).T @ Xm  # "bread"
        Ainv = np.linalg.inv(A)
        cov = Ainv @ B @ Ainv
        se = np.sqrt(np.diag(cov))

        tc = self._tcrit(level)
        t = beta / se
        result = pd.DataFrame({
            "coef": beta, "se": se,
            "OR": np.exp(beta),
            "OR_lo": np.exp(beta - tc * se),
            "OR_hi": np.exp(beta + tc * se),
            "p": 2 * stats.t.sf(np.abs(t), self.df_design),
        }, index=names)
        result.attrs["n"] = int(ok.sum())
        return result



# ======================= Q3: ASSOCIATION MODELS =======================

csv_file = r"C:\Users\premc\OneDrive\Documents\mind_score_all_scripts_v2\analysis\nhanes_analysis.csv"

print(csv_file)

df = pd.read_csv(csv_file)

design = SurveyDesign(df, strata="SDMVSTRA", psu="SDMVPSU", weight="WTMEC2YR")
design.summary()
d = design.df.copy()

in_dom = ((d["RIDAGEYR"] >= 18) & d["phq9"].notna()).to_numpy() & (design.w > 0)

# ------------------------------------------------------------------ helpers
def fmt_p(p):
    if p is None or np.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def dummies(series, levels, ref):
    """Indicator columns for every level except `ref`; NaN where series is NaN."""
    out = {}
    s = pd.Series(series).reset_index(drop=True)
    for lev in levels:
        if lev == ref:
            continue
        out[lev] = np.where(s.isna(), np.nan, (s == lev).astype(float))
    return pd.DataFrame(out, index=range(len(s)))


def as_float(col):
    return pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index)


# ------------------------------------------------------------- derive variables
GROUPS = ["PHQ 0-4", "PHQ 5-9", "PHQ 10-14", "PHQ 15-27"]
d["phq_grp"] = pd.cut(d["phq9"], bins=[-0.1, 4, 9, 14, 27], labels=GROUPS).astype(object)
d["phq_trend"] = d["phq_grp"].map({g: i for i, g in enumerate(GROUPS)}).astype(float)

RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/multiracial"}
race = d["race_eth"].round().map(RACE)

EDUC = {1: "Less than high school", 2: "Less than high school", 3: "High school/GED",
        4: "Some college/AA", 5: "College graduate+"}
educ = d["educ"].round().map(EDUC)

# Alcohol (ALQ_L): lifetime abstainers skip ALQ121, so code them as 0 first.
# ALQ121 codes: 0 none past year; 1 every day ... 5 once a week; 6 2-3x/month ... 10 1-2x/year; 77/99 refused/DK.
alq111, alq121 = as_float("ALQ111").round(), as_float("ALQ121").round()
alc_code = alq121.where(~alq121.isin([77, 99]))
alc_code = alc_code.mask(alq111 == 2, 0)
alcohol = pd.Series(np.nan, index=d.index, dtype=object)
alcohol[alc_code == 0] = "No alcohol past year"
alcohol[(alc_code >= 6) & (alc_code <= 10)] = "Low (<= 3 days/month)"
alcohol[(alc_code >= 1) & (alc_code <= 5)] = "Regular (>= 1 day/week)"

# Physical activity (PAQ_L leisure-time): minutes/week of moderate + 2 x vigorous
UNIT = {"D": 7.0, "W": 1.0, "M": 12 / 52, "Y": 1 / 52}


def weekly_minutes(q_col, u_col, m_col):
    q = as_float(q_col).round()
    q = q.mask(q >= 77)
    per_week = d[u_col].map(UNIT) if u_col in d.columns else pd.Series(np.nan, index=d.index)
    mins = as_float(m_col)
    mins = mins.mask(mins >= 777)
    total = q * per_week * mins
    return total.mask(q == 0, 0.0)


mod = weekly_minutes("PAD790Q", "PAD790U", "PAD800")
vig = weekly_minutes("PAD810Q", "PAD810U", "PAD820")
mvpa = mod + 2 * vig                                   # NaN if either part is NaN
activity = pd.Series(np.nan, index=d.index, dtype=object)
activity[mvpa == 0] = "No leisure activity"
activity[(mvpa > 0) & (mvpa < 150)] = "Some (<150 min/week)"
activity[mvpa >= 150] = "Meets guideline (>=150 min/week)"

LEVELS = {
    "race": (list(RACE.values()), "Non-Hispanic White"),
    "educ": (["Less than high school", "High school/GED", "Some college/AA", "College graduate+"],
             "College graduate+"),
    "alc": (["No alcohol past year", "Low (<= 3 days/month)", "Regular (>= 1 day/week)"],
            "Low (<= 3 days/month)"),
    "act": (["No leisure activity", "Some (<150 min/week)", "Meets guideline (>=150 min/week)"],
            "Meets guideline (>=150 min/week)"),
}

# -------------------------------------------------------------- design matrices
EXPOSURE = {
    "cont": d[["phq9"]].reset_index(drop=True),
    "cat": dummies(d["phq_grp"], GROUPS, ref="PHQ 0-4"),
    "trend": d[["phq_trend"]].reset_index(drop=True),
}

demo = pd.concat([
    d[["age", "female"]].reset_index(drop=True),
    dummies(race, *LEVELS["race"]).add_prefix("Race: "),
], axis=1)

lifestyle = pd.concat([
    d[["pir"]].reset_index(drop=True).rename(columns={"pir": "Income-poverty ratio"}),
    dummies(educ, *LEVELS["educ"]).add_prefix("Education: "),
    d[["ever_smoker"]].reset_index(drop=True).rename(columns={"ever_smoker": "Ever smoked"}),
    dummies(alcohol, *LEVELS["alc"]).add_prefix("Alcohol: "),
    dummies(activity, *LEVELS["act"]).add_prefix("Activity: "),
    d[["sleep_hrs"]].reset_index(drop=True).rename(columns={"sleep_hrs": "Sleep hours"}),
], axis=1)

bmi = d[["bmi"]].reset_index(drop=True).rename(columns={"bmi": "BMI"})

MODELS = {
    "Model 1: unadjusted": [],
    "Model 2: + age, sex, race/ethnicity": [demo],
    "Model 3: + income, education, smoking, alcohol, activity, sleep": [demo, lifestyle],
    "Model 4: Model 3 + BMI": [demo, lifestyle, bmi],
}
MODEL_SHORT = {m: f"Model {i + 1}" for i, m in enumerate(MODELS)}

OUTCOMES = [("Diabetes (any definition)", "diabetes_any"),
            ("Hypertension (any definition)", "htn_any")]

# ------------------------------------------------------------------ fit models
long_rows = []      # PHQ-9 terms for every model
full_rows = []      # all terms (continuous PHQ-9) for Models 3 and 4

all_cov = pd.concat([demo, lifestyle, bmi], axis=1)
for out_label, out_col in OUTCOMES:
    y = d[out_col].to_numpy(float)
    # Same people in every model for this outcome, so ORs are comparable across models.
    common = in_dom & ~np.isnan(y) & ~all_cov.isna().any(axis=1).to_numpy()
    n_common, events = int(common.sum()), int(np.nansum(y[common]))
    print(f"\n{out_label}: complete-case sample n={n_common} ({events} cases)")

    for form, expo in EXPOSURE.items():
        for mname, blocks in MODELS.items():
            X = pd.concat([expo] + blocks, axis=1)
            try:
                res = design.logit(y, X, domain=common)
            except Exception as e:
                print(f"  [skipped] {out_label} / {form} / {mname}: {e}")
                continue
            for term in expo.columns:
                r = res.loc[term]
                if form == "cont":
                    label, b, se = "PHQ-9, per 1 point", r["coef"], r["se"]
                    long_rows.append((out_label, form, label, mname, r["OR"], r["OR_lo"], r["OR_hi"], r["p"], n_common))
                    tc = design._tcrit(0.95)
                    long_rows.append((out_label, form, "PHQ-9, per 5 points", mname,
                                      np.exp(5 * b), np.exp(5 * (b - tc * se)), np.exp(5 * (b + tc * se)),
                                      r["p"], n_common))
                elif form == "cat":
                    long_rows.append((out_label, form, f"{term} vs PHQ 0-4", mname,
                                      r["OR"], r["OR_lo"], r["OR_hi"], r["p"], n_common))
                else:
                    long_rows.append((out_label, form, "Per PHQ-9 group step (trend)", mname,
                                      r["OR"], r["OR_lo"], r["OR_hi"], r["p"], n_common))
            if form == "cont" and mname.startswith(("Model 3", "Model 4")):
                for term, r in res.iterrows():
                    if term == "const":
                        continue
                    full_rows.append((out_label, MODEL_SHORT[mname], term, r["OR"], r["OR_lo"], r["OR_hi"], r["p"]))

long = pd.DataFrame(long_rows, columns=["Outcome", "Form", "Term", "Model", "OR", "OR_lo", "OR_hi", "p", "n"])
long["Model"] = long["Model"].map(MODEL_SHORT)
long.to_csv("models_phq9_long.csv", index=False)
pd.DataFrame(full_rows, columns=["Outcome", "Model", "Term", "OR", "OR_lo", "OR_hi", "p"]).to_csv(
    "models_all_terms_continuous_phq9.csv", index=False)

# ------------------------------------------------------------------ print table
def cell(r):
    star = "*" if r.p < 0.05 else " "
    return f"{r.OR:.2f} ({r.OR_lo:.2f}-{r.OR_hi:.2f}){star}"


long["cell"] = long.apply(cell, axis=1)
order_terms = ["PHQ-9, per 1 point", "PHQ-9, per 5 points",
               "PHQ 5-9 vs PHQ 0-4", "PHQ 10-14 vs PHQ 0-4", "PHQ 15-27 vs PHQ 0-4",
               "Per PHQ-9 group step (trend)"]
wide = long.pivot_table(index=["Outcome", "Term"], columns="Model", values="cell", aggfunc="first")
wide = wide.reindex(pd.MultiIndex.from_product([[o[0] for o in OUTCOMES], order_terms],
                                               names=["Outcome", "Term"])).dropna(how="all")
print("\nSurvey-weighted odds ratios (95% CI); * = p<0.05")
print("Model 1 unadjusted | Model 2 + age, sex, race/ethnicity | "
      "Model 3 + income, education, smoking, alcohol, activity, sleep | Model 4 = Model 3 + BMI\n")
print(wide.to_string())
wide.to_csv("models_phq9_or_table.csv")

print("\nP-values for the per-1-point PHQ-9 term")
pv = long[long.Term == "PHQ-9, per 1 point"].pivot_table(index="Outcome", columns="Model", values="p")
print(pv.map(fmt_p).to_string())

print("\nNotes: logistic regression with survey weights (WTMEC2YR), strata and PSUs; "
      "CIs use t with design df = PSUs - strata. All models for an outcome use the same "
      "complete-case sample. BMI may lie on the pathway from depression to diabetes, so Model 3 "
      "(without BMI) and Model 4 (with BMI) are both shown. Cross-sectional data: associations, "
      "not causal effects.")
print("\nSaved: models_phq9_or_table.csv, models_phq9_long.csv, models_all_terms_continuous_phq9.csv")
