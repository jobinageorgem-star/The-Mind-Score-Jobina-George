"""
NHANES 2021-2023: survey design set-up + first weighted estimates (single file).

Usage:  python nhanes_survey_run.py [path_to_nhanes_analysis.csv]
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


# ======================= RUN =======================
csv_file = r"C:\Users\premc\OneDrive\Documents\mind_score_all_scripts_v2\analysis\nhanes_analysis.csv"

print(csv_file)

df = pd.read_csv(csv_file)

# ------------------------------------------------------------------ design
# All rows stay in the design. Do NOT filter df before this point.
design = SurveyDesign(df, strata="SDMVSTRA", psu="SDMVPSU", weight="WTMEC2YR")
design.summary()
d = design.df                      # same rows, same order as the design

# ------------------------------------------------------------ subpopulation
# Adults (>=18) with a complete PHQ-9. Rows outside get weight 0 inside the
# estimators; they are never deleted.
adult_phq = ((d["RIDAGEYR"] >= 18) & d["phq9"].notna()).to_numpy()
in_dom = adult_phq & (design.w > 0)
print(f"\nSubpopulation: adults with PHQ-9 and an exam weight")
print(f"  unweighted n          : {int(in_dom.sum())}")
print(f"  weighted population   : {design.w[in_dom].sum():,.0f}")

# PHQ-9 groups (collapsed top two so the severe group is not tiny)
d = d.copy()                      # de-fragment (avoids a pandas PerformanceWarning)
d["phq_grp"] = pd.cut(d["phq9"], bins=[-0.1, 4, 9, 14, 27],
                      labels=["PHQ 0-4", "PHQ 5-9", "PHQ 10-14", "PHQ 15-27"])
order = ["PHQ 0-4", "PHQ 5-9", "PHQ 10-14", "PHQ 15-27"]

# ------------------------------------------------------- weighted prevalence
print("\nWeighted overall prevalence (adults with PHQ-9)")
for col, lab in [("diabetes_any", "Diabetes"), ("htn_any", "Hypertension")]:
    r = design.mean(d[col], domain=in_dom)
    print(f"  {lab:13s} {r['estimate']*100:5.1f}%  "
          f"(95% CI {r['lo']*100:.1f}-{r['hi']*100:.1f}), n={r['n']}")

print("\nWeighted prevalence by PHQ-9 group")
tables = {}
for col, lab in [("diabetes_any", "Diabetes"), ("htn_any", "Hypertension")]:
    t = design.mean_by(d[col], d["phq_grp"].astype(object), domain=in_dom, order=order)
    for c in ["estimate", "lo", "hi"]:
        t[c] = (t[c] * 100).round(1)
    t["se"] = (t["se"] * 100).round(2)
    t["pop"] = t["pop"].round(0)
    tables[lab] = t
    print(f"\n{lab}")
    print(t.to_string())
    t.to_csv(f"weighted_prevalence_{lab.lower()}_by_phq.csv")

# --------------------------------------------- first model: PHQ-9 per point
print("\nUnadjusted odds ratio per 1-point increase in PHQ-9")
for col, lab in [("diabetes_any", "Diabetes"), ("htn_any", "Hypertension")]:
    res = design.logit(d[col], d[["phq9"]], domain=in_dom)
    r = res.loc["phq9"]
    print(f"  {lab:13s} OR={r['OR']:.3f} (95% CI {r['OR_lo']:.3f}-{r['OR_hi']:.3f}), "
          f"p={r['p']:.4f}, n={res.attrs['n']}")

print("\nAge- and sex-adjusted")
Xadj = d[["phq9", "age", "female"]]
for col, lab in [("diabetes_any", "Diabetes"), ("htn_any", "Hypertension")]:
    res = design.logit(d[col], Xadj, domain=in_dom)
    r = res.loc["phq9"]
    print(f"  {lab:13s} OR={r['OR']:.3f} (95% CI {r['OR_lo']:.3f}-{r['OR_hi']:.3f}), "
          f"p={r['p']:.4f}, n={res.attrs['n']}")

print("\nDone. Tables saved as weighted_prevalence_*_by_phq.csv")
