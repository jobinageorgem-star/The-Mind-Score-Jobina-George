"""
NHANES 2021-2023: Table 1 by PHQ-9 group with weighted characteristics, prevalence
of diabetes/hypertension and p-trend (single self-contained file).

Usage:  python nhanes_table1.py [path_to_nhanes_analysis.csv]
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

    def linreg(self, y, X, domain=None, add_const=True, level=0.95):
        """Survey-weighted linear regression with linearized (sandwich) SEs."""
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

        A = (Xm * wd[:, None]).T @ Xm
        beta = np.linalg.solve(A, (Xm * wd[:, None]).T @ yy)
        resid = yy - Xm @ beta
        scores = (wd * resid)[:, None] * Xm
        B = self._vcov_total(scores)
        Ainv = np.linalg.inv(A)
        cov = Ainv @ B @ Ainv
        se = np.sqrt(np.diag(cov))

        tc = self._tcrit(level)
        t = beta / se
        result = pd.DataFrame({
            "coef": beta, "se": se,
            "lo": beta - tc * se, "hi": beta + tc * se,
            "p": 2 * stats.t.sf(np.abs(t), self.df_design),
        }, index=names)
        result.attrs["n"] = int(ok.sum())
        return result



# ======================= TABLE 1: weighted characteristics by PHQ-9 group =======================

PATH = sys.argv[1] if len(sys.argv) > 1 else "nhanes_analysis.csv"
df = pd.read_csv(PATH)

# All rows stay in the design; subpopulation = adults with a complete PHQ-9.
design = SurveyDesign(df, strata="SDMVSTRA", psu="SDMVPSU", weight="WTMEC2YR")
design.summary()
d = design.df.copy()

in_dom = ((d["RIDAGEYR"] >= 18) & d["phq9"].notna()).to_numpy() & (design.w > 0)

GROUPS = ["PHQ 0-4", "PHQ 5-9", "PHQ 10-14", "PHQ 15-27"]
d["phq_grp"] = pd.cut(d["phq9"], bins=[-0.1, 4, 9, 14, 27], labels=GROUPS).astype(object)
grp = d["phq_grp"].to_numpy()
# ordinal score 0..3 used for the linear trend test (1 degree of freedom)
d["phq_trend"] = pd.Series(grp).map({g: i for i, g in enumerate(GROUPS)}).astype(float)

# Column domains: Overall + each PHQ-9 group (all inside the adult/PHQ-9 subpopulation)
domains = {"Overall": in_dom}
for g in GROUPS:
    domains[g] = in_dom & (grp == g)

RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other / multiracial"}
EDUC = {1: "Less than 9th grade", 2: "9th-11th grade", 3: "High school / GED",
        4: "Some college / AA", 5: "College graduate or above"}


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def trend_p(y, kind):
    """P-value for a linear trend across the ordered PHQ-9 groups."""
    X = d[["phq_trend"]]
    try:
        if kind == "cont":
            res = design.linreg(y, X, domain=in_dom)
        else:
            res = design.logit(y, X, domain=in_dom)
        return float(res.loc["phq_trend", "p"])
    except Exception:
        return np.nan


def cell(y, dom, pct):
    r = design.mean(y, domain=dom)
    if np.isnan(r["estimate"]):
        return ""
    scale = 100 if pct else 1
    return f"{r['estimate'] * scale:.1f} ({r['se'] * scale:.1f})"


rows = []          # (label, {column: text}, p_trend_text)


def add_header(label):
    rows.append((label, {}, "", True))


def add_row(label, y, pct, kind):
    y = np.asarray(y, float)
    cells = {c: cell(y, dom, pct) for c, dom in domains.items()}
    rows.append((label, cells, fmt_p(trend_p(y, kind)), False))


# ---- sample size rows
cells_n = {c: f"{int(dom.sum())}" for c, dom in domains.items()}
cells_pop = {c: f"{design.w[dom].sum() / 1e6:.1f}" for c, dom in domains.items()}
rows.append(("Unweighted n", cells_n, "", False))
rows.append(("Weighted population (millions)", cells_pop, "", False))

# ---- continuous variables: mean (SE)
add_header("Mean (SE)")
for label, col in [("Age, years", "age"), ("PHQ-9 score", "phq9"),
                   ("Income-to-poverty ratio", "pir"), ("BMI, kg/m2", "bmi"),
                   ("Sleep, hours/night", "sleep_hrs")]:
    if col in d.columns:
        add_row(label, d[col], pct=False, kind="cont")

# ---- binary and categorical variables: weighted % (SE)
add_header("Weighted % (SE)")
for label, col in [("Female", "female"), ("Ever smoked", "ever_smoker")]:
    if col in d.columns:
        add_row(label, d[col], pct=True, kind="bin")

for title, col, mapping in [("Race/ethnicity", "race_eth", RACE), ("Education", "educ", EDUC)]:
    if col in d.columns:
        rows.append((title, {}, "", True))
        for code, name in mapping.items():
            v = d[col].to_numpy(float)
            ind = np.where(np.isnan(v), np.nan, (v == code).astype(float))
            add_row("   " + name, ind, pct=True, kind="bin")

# ---- outcomes (Q1, Q2): weighted prevalence by PHQ-9 group
add_header("Outcomes: weighted prevalence % (SE)")
for label, col in [("Diabetes (any definition)", "diabetes_any"),
                   ("Diabetes (doctor-diagnosed only)", "diabetes_dx"),
                   ("Hypertension (any definition)", "htn_any"),
                   ("Hypertension (BP >=140/90 definition)", "htn_any_140_90")]:
    if col in d.columns:
        add_row(label, d[col], pct=True, kind="bin")

# ---- assemble and save
columns = ["Overall"] + GROUPS
table = pd.DataFrame(
    [[r[1].get(c, "") for c in columns] + [r[2]] for r in rows],
    index=[r[0] for r in rows],
    columns=columns + ["P-trend"],
)
print("\nTable 1. Characteristics of US adults by PHQ-9 group "
      "(NHANES Aug 2021-Aug 2023, survey-weighted)\n")
print(table.to_string())
table.to_csv("table1_by_phq9.csv", index_label="Characteristic")

# ---- adjusted trend in prevalence of the outcomes (extra, for Q1/Q2)
print("\nOdds ratio per PHQ-9 group step (linear trend across PHQ 0-4 -> 15-27)")
for col, lab in [("diabetes_any", "Diabetes"), ("htn_any", "Hypertension")]:
    res = design.logit(d[col], d[["phq_trend"]], domain=in_dom)
    r = res.loc["phq_trend"]
    print(f"  {lab:13s} unadjusted OR={r['OR']:.3f} (95% CI {r['OR_lo']:.3f}-{r['OR_hi']:.3f}), "
          f"p-trend={fmt_p(r['p'])}")
    res = design.logit(d[col], d[["phq_trend", "age", "female"]], domain=in_dom)
    r = res.loc["phq_trend"]
    print(f"  {'':13s} age/sex-adj OR={r['OR']:.3f} (95% CI {r['OR_lo']:.3f}-{r['OR_hi']:.3f}), "
          f"p-trend={fmt_p(r['p'])}")

print("\nNotes: values are weighted mean (SE) or weighted % (SE). P-trend = Wald test of a "
      "linear trend across the ordered PHQ-9 groups (survey-weighted regression, "
      "design df = PSUs - strata). For multi-category variables the p-trend on each "
      "line tests that category's share against PHQ-9 group. Missing values are excluded "
      "separately for each row.")
print("\nSaved table1_by_phq9.csv")
