"""Survey inference, count modelling and prediction of maternal child loss.

Primary analyses use survey-weighted descriptives and design-adjusted logistic
and Poisson regression. Unweighted count likelihood comparisons, calibrated
classifiers and raw log-odds SHAP provide complementary analyses.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from statsmodels.discrete.count_model import (ZeroInflatedNegativeBinomialP,
                                              ZeroInflatedPoisson)
from xgboost import XGBClassifier
import shap

import data_prep
import survey_utils as su

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "figures"
TAB = ROOT / "tables"

RNG = 42
BLUE, ORANGE, GREEN, RED = "#4c72b0", "#dd8452", "#55a868", "#c44e52"

# ==========================================================================
# Design-matrix definition: ONE place, used by every model.
# ==========================================================================
ZONE_LABELS = {1: "North Central", 2: "North East", 3: "North West",
               4: "South East", 5: "South South", 6: "South West"}

# term -> (publication label, reference category)
TERM_INFO = {
    "const": ("Intercept", ""),
    "CEB": ("Children ever born (per child)", "continuous"),
    "WB4": ("Maternal age (per year)", "continuous"),
    "MA11C": ("Age at first union (per year)", "continuous"),
    "wscore": ("Wealth index score (per unit)", "continuous"),
    "edu_Primary": ("Maternal education: Primary", "No education"),
    "edu_Junior Secondary": ("Maternal education: Junior secondary", "No education"),
    "edu_Senior Secondary": ("Maternal education: Senior secondary", "No education"),
    "edu_Tertiary": ("Maternal education: Higher/tertiary", "No education"),
    "marital_Living with partner": ("Marital status: Living with partner", "Currently married"),
    "marital_Not in union": ("Marital status: Not in union", "Currently married"),
    "polygynous_Yes": ("Polygynous union: Yes", "Monogamous union"),
    "residence_Rural": ("Place of residence: Rural", "Urban"),
    "zone_North East": ("Geopolitical zone: North East", "North Central"),
    "zone_North West": ("Geopolitical zone: North West", "North Central"),
    "zone_South East": ("Geopolitical zone: South East", "North Central"),
    "zone_South South": ("Geopolitical zone: South South", "North Central"),
    "zone_South West": ("Geopolitical zone: South West", "North Central"),
    "delivery_Maternity/other facility": ("Place of delivery: Maternity/other facility", "Hospital/clinic"),
    "delivery_Home": ("Place of delivery: Home", "Hospital/clinic"),
    "prenatal_care_Yes": ("Received prenatal care: Yes", "No"),
}

BINARY_CODING_KEY = {
    "polygynous_Yes": "1 = husband/partner has other wives (MA3=1); 0 = monogamous (MA3=2)",
    "residence_Rural": "1 = rural (HH6=2); 0 = urban (HH6=1)",
    "prenatal_care_Yes": "1 = received prenatal care (MN2=1); 0 = did not (MN2=2)",
    "delivery_Home": "1 = delivered at home (MN20 11/12); 0 = otherwise",
    "delivery_Maternity/other facility": "1 = maternity home/other facility (MN20 33/36/26); 0 = otherwise",
    "marital_Living with partner": "1 = living with partner (MA1=2); 0 = otherwise",
    "marital_Not in union": "1 = not in union (MA1=3); 0 = otherwise",
    "edu_Primary": "1 = primary education; 0 = otherwise",
    "edu_Junior Secondary": "1 = junior secondary; 0 = otherwise",
    "edu_Senior Secondary": "1 = senior secondary; 0 = otherwise",
    "edu_Tertiary": "1 = higher/tertiary; 0 = otherwise",
    "zone_North East": "1 = North East; 0 = otherwise",
    "zone_North West": "1 = North West; 0 = otherwise",
    "zone_South East": "1 = South East; 0 = otherwise",
    "zone_South South": "1 = South South; 0 = otherwise",
    "zone_South West": "1 = South West; 0 = otherwise",
}


def label_of(term):
    return TERM_INFO.get(term, (term, ""))[0]


def reference_of(term):
    ref = TERM_INFO.get(term, (term, ""))[1]
    return "" if ref == "continuous" else ref


def build_design(df, add_const=True):
    """Return the model matrix X with explicit, audited reference categories."""
    X = pd.DataFrame(index=df.index)
    X["CEB"] = df["CEB"].astype(float)
    X["WB4"] = df["WB4"].astype(float)
    X["MA11C"] = df["MA11C"].astype(float)
    X["wscore"] = df["wscore"].astype(float)

    for lvl in ["Primary", "Junior Secondary", "Senior Secondary", "Tertiary"]:
        X[f"edu_{lvl}"] = (df["welevel"].astype(str) == lvl).astype(float)

    ma1 = pd.to_numeric(df["MA1"], errors="coerce")
    X["marital_Living with partner"] = (ma1 == 2).astype(float)
    X["marital_Not in union"] = (ma1 == 3).astype(float)

    X["polygynous_Yes"] = (pd.to_numeric(df["MA3"], errors="coerce") == 1).astype(float)
    X["residence_Rural"] = (pd.to_numeric(df["HH6"], errors="coerce") == 2).astype(float)

    zone = pd.to_numeric(df["zone"], errors="coerce")
    for code in [2, 3, 4, 5, 6]:
        X[f"zone_{ZONE_LABELS[code]}"] = (zone == code).astype(float)

    pdel = pd.to_numeric(df["place_delivery"], errors="coerce")
    X["delivery_Maternity/other facility"] = (pdel == 2).astype(float)
    X["delivery_Home"] = (pdel == 3).astype(float)

    X["prenatal_care_Yes"] = (pd.to_numeric(df["MN2"], errors="coerce") == 1).astype(float)

    # Structurally empty indicators cannot be estimated and must not be reported
    # as coefficients. In this analytic sample MA1 = 3 ("not in union") is empty
    # because polygyny status (MA3) is only ascertained for partnered women.
    empty = [c for c in X.columns if X[c].nunique() < 2]
    if empty:
        X = X.drop(columns=empty)
        build_design.dropped_terms = empty
    else:
        build_design.dropped_terms = []

    if add_const:
        X.insert(0, "const", 1.0)
    return X


def _fmt_table(tbl, est_name):
    out = pd.DataFrame({
        "predictor": [label_of(t) for t in tbl.index],
        "reference_category": [reference_of(t) for t in tbl.index],
        est_name: tbl["estimate"],
        "CI_lower": tbl["est_ci_lower"],
        "CI_upper": tbl["est_ci_upper"],
        "p_value": tbl["p_value"],
        "design_df": tbl["design_df"],
    }, index=tbl.index)
    out.index.name = "term"
    return out


# ==========================================================================
# B. SURVEY-DESIGN AUDIT  /  C. WEIGHTED DESCRIPTIVES
# ==========================================================================
def survey_design_audit(df_all, cc, audit):
    print("=" * 72)
    print("B. SURVEY-DESIGN AUDIT")
    print("=" * 72)
    for v, d in audit["design_vars_present"].items():
        print(f"  {v:10s} present=YES  missing={d['n_missing']}  "
              f"n_unique={d['n_unique']}  range=[{d['min']:.4g}, {d['max']:.4g}]")
    print(f"  Zero-weight observations in analytic sample: "
          f"{int((cc['wmweight'] == 0).sum())}")
    print(f"  PSUs in analytic sample .......... {cc['PSU'].nunique()}")
    print(f"  Strata in analytic sample ........ {cc['stratum'].nunique()}")
    lonely = int((cc.groupby('stratum')['PSU'].nunique() < 2).sum())
    print(f"  Strata with a single PSU ......... {lonely}")

    print("\n" + "=" * 72)
    print("C. SURVEY-WEIGHTED DESCRIPTIVE ANALYSIS")
    print("=" * 72)
    rows = []

    def add(name, sample, col, kind="prop"):
        fn = su.weighted_proportion if kind == "prop" else su.weighted_mean
        r = fn(sample[col], sample["wmweight"], sample["PSU"], sample["stratum"])
        rows.append({"sample": name, "measure": col, "type": kind,
                     "weighted_estimate": r["estimate"], "se": r["se"],
                     "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                     "unweighted_estimate": float(np.nanmean(sample[col])),
                     "n": r["n"], "design_df": r["dof"]})
        return r

    parous = df_all[df_all["child_death"].notna()]
    r_all = add("All parous women", parous, "child_death")
    r_cc = add("Complete-case analytic", cc, "child_death")
    add("Complete-case analytic", cc, "child_deaths", kind="mean")

    deff = su.design_effect(cc["child_death"], cc["wmweight"],
                            cc["PSU"], cc["stratum"])

    print(f"  Maternal bereavement, all parous women (n={r_all['n']:,}):")
    print(f"     weighted {100*r_all['estimate']:.2f}% "
          f"(95% CI {100*r_all['ci_lower']:.2f}-{100*r_all['ci_upper']:.2f}); "
          f"unweighted {100*parous['child_death'].mean():.2f}%")
    print(f"  Maternal bereavement, complete-case sample (n={r_cc['n']:,}):")
    print(f"     weighted {100*r_cc['estimate']:.2f}% "
          f"(95% CI {100*r_cc['ci_lower']:.2f}-{100*r_cc['ci_upper']:.2f}); "
          f"unweighted {100*cc['child_death'].mean():.2f}%")
    print(f"  Design effect (DEFF) for bereavement prevalence: {deff:.2f}")

    Xd = build_design(cc, add_const=False)
    for c in Xd.columns:
        tmp = cc.copy()
        tmp[c] = Xd[c]
        kind = "prop" if set(np.unique(Xd[c])) <= {0.0, 1.0} else "mean"
        add("Complete-case analytic", tmp, c, kind=kind)

    desc = pd.DataFrame(rows)
    desc.to_csv(TAB / "T1_weighted_descriptives.csv", index=False)
    print("  -> tables/T1_weighted_descriptives.csv")
    return desc, r_all, r_cc, deff


# ==========================================================================
# D. COMPLETE-CASE REPRESENTATIVENESS
# ==========================================================================
def representativeness(df_all, cc):
    print("\n" + "=" * 72)
    print("D. COMPLETE-CASE REPRESENTATIVENESS")
    print("=" * 72)
    parous = df_all[df_all["child_death"].notna()].copy()
    parous["__inc"] = parous.index.isin(cc.index)
    inc, exc = parous[parous["__inc"]], parous[~parous["__inc"]]
    n_par = len(parous)
    print(f"  Parous women (CEB>=1) ............ {n_par:,}")
    print(f"  Included (complete case) ......... {len(inc):,} ({100*len(inc)/n_par:.1f}%)")
    print(f"  Excluded (missing analytic data).. {len(exc):,} ({100*len(exc)/n_par:.1f}%)")

    rows = []

    def comp(name, a_s, b_s, kind="prop"):
        fn = su.weighted_proportion if kind == "prop" else su.weighted_mean
        a = fn(a_s, inc["wmweight"], inc["PSU"], inc["stratum"])
        b = fn(b_s, exc["wmweight"], exc["PSU"], exc["stratum"])
        rows.append({"characteristic": name,
                     "included_observed_n": a["n"],
                     "excluded_observed_n": b["n"],
                     "included_weighted": a["estimate"],
                     "included_ci": f"({a['ci_lower']:.4f}, {a['ci_upper']:.4f})",
                     "excluded_weighted": b["estimate"],
                     "excluded_ci": f"({b['ci_lower']:.4f}, {b['ci_upper']:.4f})",
                     "absolute_difference": a["estimate"] - b["estimate"]})

    comp("Maternal bereavement (prevalence)", inc["child_death"], exc["child_death"])
    comp("Mean number of child deaths", inc["child_deaths"], exc["child_deaths"], "mean")
    comp("Mean children ever born", inc["CEB"], exc["CEB"], "mean")
    comp("Mean maternal age (years)", inc["WB4"], exc["WB4"], "mean")
    comp("Mean wealth index score", inc["wscore"], exc["wscore"], "mean")

    def indicator(series, value):
        # Unknown categories must not enter the denominator as observed zeros.
        return (series == value).astype(float).where(series.notna())

    comp("Rural residence", indicator(inc["HH6"], 2),
         indicator(exc["HH6"], 2))
    for lvl in data_prep.EDUCATION_LEVELS:
        comp(f"Education: {lvl}", indicator(inc["welevel"], lvl),
             indicator(exc["welevel"], lvl))
    for code, lab in ZONE_LABELS.items():
        comp(f"Zone: {lab}", indicator(inc["zone"], code),
             indicator(exc["zone"], code))

    rep = pd.DataFrame(rows)
    rep.to_csv(TAB / "T2_completecase_representativeness.csv", index=False)
    print("\n  Largest weighted differences (included - excluded):")
    top = rep.reindex(rep["absolute_difference"].abs().sort_values(
        ascending=False).index).head(8)
    for _, r in top.iterrows():
        print(f"    {r['characteristic']:38s} {r['included_weighted']:.4f} vs "
              f"{r['excluded_weighted']:.4f} (diff {r['absolute_difference']:+.4f})")
    print("  -> tables/T2_completecase_representativeness.csv")
    return rep


# ==========================================================================
# E. SURVEY-ADJUSTED LOGISTIC REGRESSION
# ==========================================================================
def survey_logistic(cc):
    print("\n" + "=" * 72)
    print("E. SURVEY-ADJUSTED LOGISTIC REGRESSION (maternal bereavement)")
    print("=" * 72)
    X = build_design(cc)
    y = cc["child_death"].astype(float)
    if build_design.dropped_terms:
        print("  Structurally empty terms removed (no observations): "
              + ", ".join(label_of(t) for t in build_design.dropped_terms))

    tbl, res, cov = su.design_adjusted_glm(
        y, X, cc["wmweight"], cc["PSU"], cc["stratum"],
        family=sm.families.Binomial())
    out = _fmt_table(tbl, "AOR")
    un = sm.GLM(y, X.astype(float), family=sm.families.Binomial()).fit()
    out["AOR_unweighted"] = np.exp(np.asarray(un.params))
    out.to_csv(TAB / "T3_survey_logistic_AOR.csv")

    print(f"  n = {len(cc):,}   PSUs = {cc['PSU'].nunique()}   "
          f"strata = {cc['stratum'].nunique()}   "
          f"design df = {int(tbl['design_df'].iloc[0])}")
    for t in out.index:
        r = out.loc[t]
        print(f"   {r['predictor'][:42]:43s} {r['AOR']:6.3f} "
              f"({r['CI_lower']:.2f}-{r['CI_upper']:.2f})  p={r['p_value']:.4f}")
    print("  -> tables/T3_survey_logistic_AOR.csv")
    forest_plot(out, "AOR", "Adjusted odds ratio (survey-adjusted)",
                FIG / "F3_logistic_forest.tiff")
    return out, X, y


def forest_plot(tbl, est_col, xlabel, path, top_n=12):
    d = tbl.drop(index=[i for i in ["const"] if i in tbl.index]).copy()
    d["__d"] = np.abs(np.log(d[est_col].clip(lower=1e-6)))
    d = d.sort_values("__d", ascending=False).head(top_n)[::-1]
    labels = [r["predictor"] + (f"  (ref: {r['reference_category']})"
              if r["reference_category"] else "") for _, r in d.iterrows()]
    ypos = np.arange(len(d))
    plt.figure(figsize=(9.5, 3 + 0.45 * len(d)))
    plt.errorbar(d[est_col], ypos,
                 xerr=[d[est_col] - d["CI_lower"], d["CI_upper"] - d[est_col]],
                 fmt="o", color=BLUE, ecolor="black", capsize=4)
    plt.yticks(ypos, labels, fontsize=9)
    plt.axvline(1, color="gray", ls="--")
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="tiff")
    plt.close()


# ==========================================================================
# F. COUNT-OUTCOME ANALYSIS
# ==========================================================================
def count_models(cc):
    print("\n" + "=" * 72)
    print("F. COUNT-OUTCOME ANALYSIS (number of child deaths)")
    print("=" * 72)
    X = build_design(cc)
    y = cc["child_deaths"].astype(int)
    n = len(y)
    obs_zeros = int((y == 0).sum())
    print("  Distribution of child deaths:")
    print("   ", y.value_counts().sort_index().to_dict())
    print(f"  Zeros: {obs_zeros:,} ({100*obs_zeros/n:.2f}%)   mean={y.mean():.4f}  "
          f"var={y.var():.4f}  var/mean={y.var()/y.mean():.3f}")

    ptbl, pres, _ = su.design_adjusted_glm(
        y, X, cc["wmweight"], cc["PSU"], cc["stratum"],
        family=sm.families.Poisson())
    pout = _fmt_table(ptbl, "IRR")
    pout.to_csv(TAB / "T4_survey_poisson_IRR.csv")
    wn = cc['wmweight'].to_numpy(dtype=float)
    wn = wn * n / wn.sum()
    survey_mu = np.asarray(pres.fittedvalues)
    conditional_dispersion = float(np.sum(
        wn * (y.to_numpy() - survey_mu) ** 2 / survey_mu) / (n - X.shape[1]))
    pd.DataFrame([{'analysis': 'Primary survey-weighted Poisson', 'n': n,
                   'mean_outcome_unweighted': float(y.mean()),
                   'marginal_variance_to_mean': float(y.var() / y.mean()),
                   'weighted_pearson_dispersion': conditional_dispersion,
                   'variance_estimator': 'Stratified PSU Taylor sandwich; original design registry',
                   'likelihood_comparison_permitted': False}]).to_csv(
        TAB / 'T13_survey_poisson_diagnostics.csv', index=False)
    print(f'  Survey Poisson weighted Pearson dispersion: {conditional_dispersion:.4f}')
    print("\n  Design-adjusted Poisson regression (wmweight + PSU + stratum):")
    for t in pout.index:
        r = pout.loc[t]
        print(f"   {r['predictor'][:42]:43s} {r['IRR']:6.3f} "
              f"({r['CI_lower']:.2f}-{r['CI_upper']:.2f})  p={r['p_value']:.4f}")

    print("\n  Likelihood-based model-fit analysis (unweighted):")
    Xf = X.astype(float)
    attempts = []

    def record_attempt(name, method, start, result=None, floor=-np.inf,
                       error="", warning_text=""):
        """Require explicit convergence, finite likelihood/parameters and nesting.

        Score/covariance diagnostics are recorded, not substituted for the
        optimizer's convergence flag. Missing covariance precludes Wald CIs.
        """
        row = {"model": name, "optimizer": method, "start": start,
               "count_spec": "full", "inflation_spec":
               "full" if name in ("ZIP", "ZINB") else "none",
               "optimizer_converged": False, "optimizer_flags": "",
               "finite_params": False, "finite_llf": False, "llf": np.nan,
               "nesting_floor_llf": floor, "nesting_pass": False,
               "alpha": np.nan, "score_norm": np.nan,
               "covariance_available": False, "diagnostic_errors": "",
               "error": error, "warnings": warning_text, "accepted": False,
               "selected": False}
        if result is not None:
            flags = getattr(result, "mle_retvals", {}) or {}
            row["optimizer_flags"] = repr(flags)
            row["optimizer_converged"] = bool(
                getattr(result, "converged", False) if name == "Poisson"
                else flags.get("converged", False))
            params = np.asarray(result.params, dtype=float)
            row["finite_params"] = bool(np.all(np.isfinite(params)))
            row["llf"] = float(result.llf)
            row["finite_llf"] = bool(np.isfinite(row["llf"]))
            row["nesting_pass"] = bool(row["finite_llf"]
                                        and row["llf"] >= floor - 1e-6)
            alpha_ok = True
            if name in ("NegBin", "ZINB"):
                row["alpha"] = float(params[-1])
                alpha_ok = bool(np.isfinite(row["alpha"]) and row["alpha"] > 0)
            diagnostics = []
            try:
                row["score_norm"] = float(np.linalg.norm(
                    result.model.score(params)))
            except Exception as exc:
                diagnostics.append(f"score: {type(exc).__name__}: {exc}")
            try:
                cov = np.asarray(result.cov_params(), dtype=float)
                row["covariance_available"] = bool(np.all(np.isfinite(cov)))
            except Exception as exc:
                diagnostics.append(f"covariance: {type(exc).__name__}: {exc}")
            row["diagnostic_errors"] = "; ".join(diagnostics)
            row["accepted"] = bool(row["optimizer_converged"]
                                    and row["finite_params"]
                                    and row["nesting_pass"] and alpha_ok)
            if not row["accepted"]:
                row["error"] = ("Rejected: convergence, finite parameters/llf, "
                                "positive alpha or nesting guard failed")
        attempts.append(row)
        return row["accepted"]

    def try_fit(name, ctor, floor, starts):
        best_r, best_m, best_attempt = None, None, None
        for start_name, start_params in starts:
            for method, maxiter in [("bfgs", 2000), ("lbfgs", 2000),
                                    ("newton", 300), ("nm", 8000)]:
                caught = []
                try:
                    kw = {} if start_params is None else {"start_params": start_params}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        r = ctor().fit(method=method, disp=0, maxiter=maxiter, **kw)
                    accepted = record_attempt(
                        name, method, start_name, r, floor,
                        warning_text="; ".join(str(w.message) for w in caught))
                    if accepted and (best_r is None or r.llf > best_r.llf):
                        best_r, best_m, best_attempt = r, method, len(attempts) - 1
                except Exception as exc:
                    record_attempt(name, method, start_name, floor=floor,
                                   error=f"{type(exc).__name__}: {exc}",
                                   warning_text="; ".join(str(w.message) for w in caught))
        if best_attempt is not None:
            attempts[best_attempt]["selected"] = True
        return best_r, best_m

    try:
        poisson = sm.GLM(y, Xf, family=sm.families.Poisson()).fit()
    except Exception as exc:
        record_attempt("Poisson", "IRLS", "default",
                       error=f"{type(exc).__name__}: {exc}")
        pd.DataFrame(attempts).to_csv(TAB / "T12_count_convergence.csv", index=False)
        raise RuntimeError("Poisson baseline estimation failed; see T12 audit") from exc
    if not record_attempt("Poisson", "IRLS", "default", poisson):
        pd.DataFrame(attempts).to_csv(TAB / "T12_count_convergence.csv", index=False)
        raise RuntimeError("Poisson baseline failed convergence/finite-value checks")
    attempts[-1]["selected"] = True
    models = {"Poisson": poisson}
    ref_llf = float(poisson.llf)
    beta_p = np.asarray(poisson.params)
    nb_res, nb_m = try_fit(
        "NegBin", lambda: sm.NegativeBinomial(y, Xf, loglike_method="nb2"),
        ref_llf, [("default", None), ("Poisson+alpha0.1", np.r_[beta_p, 0.1])])
    # NB2 dispersion is estimated jointly with the regression coefficients.
    models["NegBin"] = nb_res
    alpha_hat = float(np.asarray(nb_res.params)[-1]) if nb_res is not None else np.nan
    print(f"    NegBin: {nb_m or 'DID NOT CONVERGE'}; MLE alpha={alpha_hat:.4f}")

    inflation_start = np.zeros(Xf.shape[1])
    inflation_start[Xf.columns.get_loc("const")] = -10.0
    infl_exog = {"ZIP": Xf, "ZINB": Xf}
    zip_res, zip_m = try_fit(
        "ZIP", lambda: ZeroInflatedPoisson(
            endog=y, exog=Xf, exog_infl=Xf, inflation="logit"), ref_llf,
        [("default", None), ("near-Poisson", np.r_[inflation_start, beta_p])])
    models["ZIP"] = zip_res
    # ZINB nests every accepted NB2, ZIP and Poisson fit. Never simplify the
    # inflation equation to rescue convergence; failed full models are excluded.
    zinb_floor = max(float(m.llf) for m in models.values() if m is not None)
    zinb_starts = [("default", None),
                   ("near-Poisson", np.r_[inflation_start, beta_p, 0.1])]
    if nb_res is not None:
        zinb_starts.append(("near-NB", np.r_[inflation_start, nb_res.params]))
    if zip_res is not None:
        zinb_starts.append(("near-ZIP", np.r_[zip_res.params, 0.01]))
    zinb_res, zinb_m = try_fit(
        "ZINB", lambda: ZeroInflatedNegativeBinomialP(
            endog=y, exog=Xf, exog_infl=Xf, inflation="logit", p=2),
        zinb_floor, zinb_starts)
    models["ZINB"] = zinb_res
    pd.DataFrame(attempts).to_csv(TAB / "T12_count_convergence.csv", index=False)
    for name, method in [("ZIP", zip_m), ("ZINB", zinb_m)]:
        print(f"    {name}: {method or 'DID NOT CONVERGE - excluded'} (full inflation)")

    rows = []
    for name, m in models.items():
        if m is None:
            rows.append({"model": name, "converged": False, "n": n, "k": np.nan,
                         "llf": np.nan, "aic": np.nan, "bic": np.nan,
                         "dispersion": np.nan, "alpha": np.nan, "obs_zeros": obs_zeros,
                         "obs_zero_pct": 100 * obs_zeros / n,
                         "pred_zeros": np.nan, "pred_zero_pct": np.nan})
            continue
        k = len(np.asarray(m.params))
        llf = float(m.llf)
        pz = float(np.sum(pmf_matrix(name, m, Xf, infl_exog, alpha_hat, [0])[:, 0]))
        mean, variance = count_moments(name, m, Xf, infl_exog)
        valid_variance = np.all(np.isfinite(variance)) and np.all(variance > 0)
        disp = (float(np.sum((np.asarray(y) - mean) ** 2 / variance) / (n - k))
                if valid_variance and n > k else np.nan)
        alpha = float(np.asarray(m.params)[-1]) if name in ("NegBin", "ZINB") else np.nan
        rows.append({"model": name, "converged": True, "n": n, "k": k, "llf": llf,
                     "aic": 2 * k - 2 * llf, "bic": k * np.log(n) - 2 * llf,
                     "dispersion": disp, "alpha": alpha, "obs_zeros": obs_zeros,
                     "obs_zero_pct": 100 * obs_zeros / n,
                     "pred_zeros": pz, "pred_zero_pct": 100 * pz / n})
    fit = pd.DataFrame(rows).set_index("model")
    fit.to_csv(TAB / "T5_count_model_fit.csv")
    print("\n  Model-fit comparison:")
    print(fit[["converged", "k", "llf", "aic", "bic", "dispersion",
               "alpha", "obs_zero_pct", "pred_zero_pct"]].to_string())

    for name, m in models.items():
        if m is None:
            # Excluded fits have no reportable coefficients.
            pd.DataFrame(columns=["predictor", "reference_category", "IRR",
                                  "CI_lower", "CI_upper", "p_value"],
                         index=pd.Index([], name="term")).to_csv(
                TAB / f"T6_count_IRR_{name}.csv")
            continue
        try:
            p = m.params
            # Export count coefficients only, never inflation terms or alpha.
            idx = [i for i in Xf.columns if i in p.index]
            if idx != list(Xf.columns):
                raise ValueError(f"Unexpected count coefficient terms for {name}")
            try:
                ci = m.conf_int()
                pvalues = m.pvalues
            except Exception:
                ci = pd.DataFrame(np.nan, index=p.index, columns=[0, 1])
                pvalues = pd.Series(np.nan, index=p.index)
            t = pd.DataFrame({"IRR": np.exp(p[idx]),
                              "CI_lower": np.exp(ci.loc[idx, 0]),
                              "CI_upper": np.exp(ci.loc[idx, 1]),
                              "p_value": pvalues[idx]})
            t.insert(0, "reference_category", [reference_of(i) for i in t.index])
            t.insert(0, "predictor", [label_of(i) for i in t.index])
            t.index.name = "term"
            t.to_csv(TAB / f"T6_count_IRR_{name}.csv")
        except Exception as e:
            pd.DataFrame(columns=["predictor", "reference_category", "IRR",
                                  "CI_lower", "CI_upper", "p_value"],
                         index=pd.Index([], name="term")).to_csv(
                TAB / f"T6_count_IRR_{name}.csv")
            raise RuntimeError(f"Count coefficient export failed for {name}") from e

    best = fit[fit["converged"]]["aic"].idxmin()
    print(f"\n  Best-fitting likelihood-based model by AIC: {best}")
    count_figures(models, Xf, y, best, alpha_hat, infl_exog)
    forest_plot(pout, "IRR", "Incidence rate ratio (survey-adjusted Poisson)",
                FIG / "F5_count_IRR_forest.tiff")
    print("  -> tables/T4_survey_poisson_IRR.csv, T5_count_model_fit.csv, T6_count_IRR_*.csv")
    print("  -> tables/T12_count_convergence.csv (all attempts, flags and diagnostics)")
    return pout, fit, models, best


def count_moments(name, m, Xf, infl_exog):
    """Mean and full variance (including structural-zero mixing), NB2 basis."""
    par = np.asarray(m.params, dtype=float)
    if name in ("Poisson", "NegBin"):
        mu = np.asarray(m.predict(Xf), dtype=float)
        alpha = float(par[-1]) if name == "NegBin" else 0.0
        return mu, mu + alpha * mu ** 2
    n_infl = infl_exog[name].shape[1]
    pi = expit(np.asarray(infl_exog[name], dtype=float) @ par[:n_infl])
    mu = np.exp(np.asarray(Xf, dtype=float) @ par[n_infl:n_infl + Xf.shape[1]])
    alpha = float(par[-1]) if name == "ZINB" else 0.0
    mean = (1 - pi) * mu
    variance = (1 - pi) * (mu + alpha * mu ** 2) + pi * (1 - pi) * mu ** 2
    return mean, variance


def pmf_matrix(name, m, Xf, infl_exog, alpha_hat, ks):
    """Observation-level P(Y = k) for each k in ``ks``; returns n x len(ks)."""
    ks = list(ks)
    if name == "Poisson":
        mu = np.asarray(m.predict(Xf), dtype=float)
        return np.column_stack([stats.poisson.pmf(k, mu) for k in ks])
    if name == "NegBin":
        mu = np.asarray(m.predict(Xf), dtype=float)
        # alpha_hat argument retained for caller compatibility; the full MLE
        # result is the authoritative source of NB2 dispersion.
        r_ = 1.0 / float(np.asarray(m.params)[-1])
        return np.column_stack([stats.nbinom.pmf(k, r_, r_ / (r_ + mu)) for k in ks])

    # zero-inflated models: pi = P(structural zero), mu = latent count mean
    par = np.asarray(m.params, dtype=float)
    n_infl = infl_exog[name].shape[1]
    pi = expit(np.asarray(infl_exog[name], dtype=float) @ par[:n_infl])
    beta = par[n_infl:n_infl + Xf.shape[1]]
    mu = np.exp(np.asarray(Xf, dtype=float) @ beta)
    if name == "ZIP":
        base = np.column_stack([stats.poisson.pmf(k, mu) for k in ks])
    else:
        r_ = 1.0 / float(par[-1])
        base = np.column_stack([stats.nbinom.pmf(k, r_, r_ / (r_ + mu)) for k in ks])
    out = (1 - pi)[:, None] * base
    for i, k in enumerate(ks):
        if k == 0:
            out[:, i] = out[:, i] + pi
    return out


def count_figures(models, Xf, y, best, alpha_hat, infl_exog):
    max_k = 5
    ks = list(range(max_k + 1))
    labels = [str(k) if k < max_k else f"{max_k}+" for k in ks]
    obs = y.clip(upper=max_k).value_counts().reindex(ks, fill_value=0).values

    def expected(name, m):
        e = list(pmf_matrix(name, m, Xf, infl_exog, alpha_hat,
                            range(max_k)).sum(axis=0))
        e.append(max(len(y) - sum(e), 0.0))   # P(Y >= max_k) by complement
        return np.array(e)

    plt.figure(figsize=(10, 6))
    width, xs = 0.18, np.arange(len(ks))
    cols = [BLUE, ORANGE, GREEN, RED]
    i, exp_best = 0, None
    for name, m in models.items():
        if m is None:
            continue
        e = expected(name, m)
        if name == best:
            exp_best = e
        plt.bar(xs + (i - 1.5) * width, e, width, label=name,
                color=cols[i % 4], edgecolor="black")
        i += 1
    plt.bar(xs + 2 * width, obs, width, label="Observed", color="gray",
            edgecolor="black")
    plt.xticks(xs, labels)
    plt.xlabel("Number of child deaths"); plt.ylabel("Number of women")
    plt.title("Observed versus model-predicted distribution of child deaths")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "F4_count_obs_vs_pred.tiff", dpi=300, format="tiff")
    plt.close()

    if exp_best is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(labels, np.sqrt(obs) - np.sqrt(exp_best), color=BLUE,
                edgecolor="black")
        plt.axhline(0, color="k")
        plt.ylabel(r"$\sqrt{Observed}-\sqrt{Expected}$")
        plt.title(f"Hanging rootogram - {best}")
        plt.tight_layout()
        plt.savefig(FIG / f"F4b_rootogram_{best}.tiff", dpi=300, format="tiff")
        plt.close()


# ==========================================================================
# G/H. MACHINE LEARNING + CALIBRATION
# ==========================================================================
def calibration_metrics(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))
    slope = sm.GLM(y_true, sm.add_constant(lo),
                   family=sm.families.Binomial()).fit()
    inter = sm.GLM(y_true, np.ones((len(p), 1)),
                   family=sm.families.Binomial(), offset=lo).fit()
    return {"brier": float(brier_score_loss(y_true, p)),
            "calib_intercept": float(np.asarray(inter.params)[0]),
            "calib_slope": float(np.asarray(slope.params)[1])}


def full_metrics(y_true, p, thr=0.5):
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = {"roc_auc": float(roc_auc_score(y_true, p)),
         "pr_auc": float(average_precision_score(y_true, p)),
         "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
         "precision": float(precision_score(y_true, pred, zero_division=0)),
         "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
         "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
         "f1": float(f1_score(y_true, pred, zero_division=0)),
         "mcc": float(matthews_corrcoef(y_true, pred))}
    m.update(calibration_metrics(y_true, p))
    return m


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def select_calibrator(p_cal, ycal):
    """Select using pooled five-fold OOF Brier on calibration data ONLY.

    Base-model predictions are fixed: each calibrator sees only the training
    calibration folds. Refit the winner on all calibration observations. The
    signature deliberately excludes test predictions and outcomes.
    """
    p_cal = np.asarray(p_cal, dtype=float)
    ycal = np.asarray(ycal, dtype=int)
    if np.min(np.bincount(ycal, minlength=2)) < 5:
        raise ValueError("Calibration set needs at least five observations per class")
    oof = {method: np.full(len(ycal), np.nan) for method in ("Platt", "isotonic")}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    for train, valid in folds.split(p_cal.reshape(-1, 1), ycal):
        platt = LogisticRegression(random_state=RNG, max_iter=2000).fit(
            _logit(p_cal[train]).reshape(-1, 1), ycal[train])
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(
            p_cal[train], ycal[train])
        oof["Platt"][valid] = platt.predict_proba(
            _logit(p_cal[valid]).reshape(-1, 1))[:, 1]
        oof["isotonic"][valid] = isotonic.predict(p_cal[valid])
    scores = {method: float(brier_score_loss(ycal, prediction))
              for method, prediction in oof.items()}
    # Deterministic exact-tie rule: Platt (first candidate). No raw/test score
    # participates in choosing the calibration method.
    chosen = min(scores, key=scores.get)
    if chosen == "Platt":
        calibrator = LogisticRegression(random_state=RNG, max_iter=2000).fit(
            _logit(p_cal).reshape(-1, 1), ycal)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_cal, ycal)
    return chosen, calibrator, scores


def machine_learning(cc):
    print("\n" + "=" * 72)
    print("G. MACHINE-LEARNING ANALYSIS (prediction; unweighted primary)")
    print("=" * 72)
    X = build_design(cc, add_const=False)
    y = cc["child_death"].astype(int).to_numpy()
    w = cc["wmweight"].to_numpy()

    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, y, w, test_size=0.20, random_state=RNG, stratify=y)
    Xfit, Xcal, yfit, ycal = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=RNG, stratify=ytr)

    print(f"  Model-fitting n={len(yfit):,}  Calibration n={len(ycal):,}  "
          f"Test n={len(yte):,}")
    print(f"  Event rate  fit={yfit.mean():.4f}  calib={ycal.mean():.4f}  "
          f"test={yte.mean():.4f}")

    continuous = [c for c in X.columns if TERM_INFO[c][1] == "continuous"]
    # Fit scaling on ORIGINAL fitting observations, before neighbor search.
    # Standardize continuous columns only; dummy definitions remain unchanged.
    scaler = StandardScaler().fit(Xfit[continuous])

    def scaled(frame):
        out = frame.copy()
        out[continuous] = scaler.transform(frame[continuous])
        return out

    Xfit_scaled_res, yfit_res = SMOTE(random_state=RNG).fit_resample(
        scaled(Xfit), yfit)
    Xfit_res = Xfit_scaled_res.copy()
    Xfit_res[continuous] = scaler.inverse_transform(Xfit_scaled_res[continuous])
    # Ordinary SMOTE may interpolate fractional dummy values, including
    # mutually exclusive categories; this limits categorical interpretation.
    print(f"  SMOTE applied to the model-fitting split ONLY: {len(yfit):,} -> "
          f"{len(yfit_res):,} (event rate {yfit_res.mean():.3f})")
    print("  SMOTE NOT applied to the calibration split or the test set.")

    defs = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RNG), True),
        "Random Forest": (RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                 random_state=RNG, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RNG), False),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RNG,
                                  n_jobs=-1, tree_method="hist"), False),
    }

    results, rows, selection_rows = {}, [], []
    for name, (model, needs_scale) in defs.items():
        if needs_scale:
            Xa, Xb, Xc = Xfit_scaled_res, scaled(Xcal), scaled(Xte)
        else:
            Xa, Xb, Xc = Xfit_res, Xcal, Xte
        model.fit(Xa, yfit_res)
        p_cal = model.predict_proba(Xb)[:, 1]
        chosen, calibrator, cv_scores = select_calibrator(p_cal, ycal)
        selection_rows.append({
            "model": name, "n_calibration": len(ycal), "cv_folds": 5,
            "cv_seed": RNG, "selection_metric": "pooled OOF Brier",
            "Platt_cv_brier": cv_scores["Platt"],
            "isotonic_cv_brier": cv_scores["isotonic"],
            "selected_method": chosen, "refit_on_full_calibration": True,
            "test_outcomes_used_for_selection": False})
        # Method selection and full-calibration refit are complete before any
        # test evaluation. Test outcomes only quantify raw/final performance.
        p_te = model.predict_proba(Xc)[:, 1]
        p_final = (calibrator.predict_proba(_logit(p_te).reshape(-1, 1))[:, 1]
                   if chosen == "Platt" else calibrator.predict(p_te))
        raw = full_metrics(yte, p_te)
        final = full_metrics(yte, p_final)
        results[name] = {"model": model, "needs_scale": needs_scale,
                         "X_eval": Xc, "p_raw": p_te,
                         "p_final": p_final, "chosen": chosen,
                         "calibrator": calibrator, "calibration_cv_scores": cv_scores,
                         "metrics_raw": raw, "metrics_final": final}
        for tag, mm in [("uncalibrated", raw), (chosen, final)]:
            rows.append({"model": name, "calibration": tag, **mm,
                         "selected": tag == chosen})
        print(f"\n  {name}")
        print(f"    uncalibrated: AUC={raw['roc_auc']:.4f} PR-AUC={raw['pr_auc']:.4f} "
              f"Brier={raw['brier']:.4f} slope={raw['calib_slope']:.3f} "
              f"intercept={raw['calib_intercept']:+.3f}")
        print(f"    Calibration-only OOF Brier: Platt={cv_scores['Platt']:.4f}; "
              f"isotonic={cv_scores['isotonic']:.4f}")
        print(f"    -> reported : {chosen}; test Brier={final['brier']:.4f} "
              f"slope={final['calib_slope']:.3f} "
              f"intercept={final['calib_intercept']:+.3f}")

    pd.DataFrame(selection_rows).to_csv(TAB / "T11_calibration_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(TAB / "T7_ml_performance.csv", index=False)
    pd.DataFrame([{"model": k,
                   "roc_auc_weighted_test": float(roc_auc_score(
                       yte, v["p_final"], sample_weight=wte)),
                   "roc_auc_unweighted_test": v["metrics_final"]["roc_auc"]}
                  for k, v in results.items()]).to_csv(
        TAB / "T8_ml_weighted_sensitivity.csv", index=False)

    ml_figures(results, yte, X.columns)
    print("\n  -> tables/T7_ml_performance.csv, T8_ml_weighted_sensitivity.csv, "
          "T11_calibration_selection.csv")
    return results, (Xte, yte), pd.DataFrame(rows)


def ml_figures(results, yte, feat_names):
    plt.figure(figsize=(7.5, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(yte, r["p_final"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['metrics_final']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("1 - specificity"); plt.ylabel("Sensitivity")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "F6_roc_curves.tiff", dpi=300, format="tiff"); plt.close()

    plt.figure(figsize=(7.5, 6))
    for name, r in results.items():
        pr, rc, _ = precision_recall_curve(yte, r["p_final"])
        plt.plot(rc, pr, label=f"{name} (PR-AUC={r['metrics_final']['pr_auc']:.3f})")
    plt.axhline(yte.mean(), color="k", ls="--",
                label=f"Prevalence = {yte.mean():.3f}")
    plt.xlabel("Recall (sensitivity)"); plt.ylabel("Precision")
    plt.title("Precision-recall curves (internal test set)")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "F7_pr_curves.tiff", dpi=300, format="tiff"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, key, ttl in zip(axes, ["p_raw", "p_final"],
                            ["Before recalibration", "After recalibration"]):
        for name, r in results.items():
            pt, pp = calibration_curve(yte, r[key], n_bins=10, strategy="quantile")
            ax.plot(pp, pt, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", label="Ideal (45 degrees)")
        ax.set_xlabel("Mean predicted probability"); ax.set_title(ttl)
    axes[0].set_ylabel("Observed event frequency")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "F8_calibration.tiff", dpi=300, format="tiff"); plt.close()

    rf = results["Random Forest"]["model"]
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(
        ascending=False).head(10)[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh([label_of(c) for c in imp.index], imp.values, color=BLUE,
             edgecolor="black")
    plt.xlabel("Mean decrease in impurity")
    plt.title("Random forest: 10 most important predictors")
    plt.tight_layout()
    plt.savefig(FIG / "F9_rf_importance.tiff", dpi=300, format="tiff"); plt.close()


# ==========================================================================
# I. SHAP AUDIT
# ==========================================================================
def shap_audit(results, test):
    print("\n" + "=" * 72)
    print("I. SHAP AUDIT")
    print("=" * 72)
    Xte, yte = test
    rows = []
    for name in ["Gradient Boosting", "XGBoost"]:
        model = results[name]["model"]
        Xsrc = results[name]["X_eval"]
        idx = np.random.default_rng(RNG).choice(
            len(Xsrc), size=min(1000, len(Xsrc)), replace=False)
        Xs = Xsrc.iloc[idx]
        # Explain the base classifier's RAW log-odds, not the nonlinear
        # calibration mapping or contributions to calibrated probabilities.
        explainer = shap.TreeExplainer(model, model_output="raw")
        sv = np.asarray(explainer.shap_values(Xs, check_additivity=False))
        if sv.ndim == 3:
            sv = sv[..., 1]
        base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if base.size != 1:
            raise ValueError(f"Unexpected raw binary SHAP baseline for {name}: {base}")
        target_name = ("decision_function" if name == "Gradient Boosting"
                       else "predict(output_margin=True)")
        margin = np.asarray(model.decision_function(Xs) if name == "Gradient Boosting"
                            else model.predict(Xs, output_margin=True), dtype=float)
        reconstruction = base[0] + sv.sum(axis=1)
        max_error = float(np.max(np.abs(reconstruction - margin)))
        additivity_ok = bool(np.allclose(reconstruction, margin, atol=1e-4, rtol=1e-5))
        if not additivity_ok:
            raise RuntimeError(
                f"{name}: raw log-odds SHAP additivity failed ({max_error:g})")
        cols = list(Xs.columns)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=cols)
        ranks = imp.rank(ascending=False)
        top = imp.sort_values(ascending=False).head(10).index.tolist()

        for f in cols:
            x = Xs[f].to_numpy()
            s = sv[:, cols.index(f)]
            uniq = np.unique(x)
            binary = set(uniq) <= {0.0, 1.0} and len(uniq) == 2
            direction_status = "directional"
            if binary:
                m1, m0 = s[x == 1].mean(), s[x == 0].mean()
                contrast = m1 - m0
                if float(imp[f]) <= 1e-8 or abs(contrast) < 0.20 * float(imp[f]):
                    direction_status = "ambiguous"
                    direction = ("negligible / ambiguous contribution - no "
                                 "directional claim supported")
                else:
                    direction = ("presence of the category contributes to a HIGHER "
                                 "raw predicted log-odds of maternal bereavement"
                                 if contrast > 0 else
                                 "presence of the category contributes to a LOWER "
                                 "raw predicted log-odds of maternal bereavement")
                evidence = (f"mean SHAP | x=1: {m1:+.4f}; mean SHAP | x=0: {m0:+.4f}; "
                            f"contrast {contrast:+.4f}")
            else:
                r, _ = stats.spearmanr(x, s)
                shape = "approximately monotonic" if abs(r) > 0.9 else \
                        ("largely monotonic" if abs(r) > 0.6 else
                         "non-monotonic / heterogeneous")
                if not np.isfinite(r) or abs(r) <= 0.6 or float(imp[f]) <= 1e-8:
                    direction_status = "non-directional"
                    direction = ("weak / non-monotonic / heterogeneous contribution - "
                                 "no global directional claim supported")
                else:
                    direction = ("higher values tend to contribute to HIGHER raw log-odds"
                                 if r > 0 else
                                 "higher values tend to contribute to LOWER raw log-odds")
                evidence = f"Spearman(feature, SHAP) = {r:+.3f}; {shape}"
            if not additivity_ok:
                direction_status = "ambiguous"
                direction = "additivity failed - no directional claim supported"
            rows.append({"model": name, "feature": f,
                         "publication_label": label_of(f),
                         "reference_category": reference_of(f),
                         "coding_key": BINARY_CODING_KEY.get(f, "continuous predictor"),
                         "is_binary": bool(binary),
                         "mean_abs_shap": float(imp[f]),
                         "rank": int(ranks[f]), "direction": direction,
                         "direction_status": direction_status,
                         "output_scale": "raw log-odds (uncalibrated base classifier)",
                         "additivity_target": target_name,
                         "expected_value_raw": float(base[0]),
                         "additivity_n": len(Xs), "additivity_atol": 1e-4,
                         "additivity_rtol": 1e-5,
                         "additivity_max_abs_error": max_error,
                         "additivity_pass": additivity_ok,
                         "evidence": evidence, "in_top10": f in top})

        disp = Xs[top].copy()
        disp.columns = [label_of(c) for c in top]
        shap.summary_plot(sv[:, [cols.index(c) for c in top]], disp, show=False)
        plt.xlabel("SHAP value (raw log-odds; uncalibrated base classifier)")
        plt.tight_layout()
        plt.savefig(FIG / f"F10_shap_{name.replace(' ', '_')}.tiff",
                    dpi=300, format="tiff")
        plt.close()

    sa = pd.DataFrame(rows)
    sa.to_csv(TAB / "T9_shap_audit.csv", index=False)
    print("  Gradient Boosting - top 10 contributions (direction verified against coding):")
    for _, r in sa[(sa["model"] == "Gradient Boosting") & sa["in_top10"]] \
            .sort_values("rank").iterrows():
        print(f"    {r['rank']:>2}. {r['publication_label'][:42]:43s} "
              f"|SHAP|={r['mean_abs_shap']:.4f}")
        print(f"        coding: {r['coding_key']}")
        print(f"        {r['evidence']}")
    print("  -> tables/T9_shap_audit.csv, figures/F10_shap_*.tiff")
    return sa


# ==========================================================================
# J. CONSISTENCY CHECKLIST
# ==========================================================================
def consistency_checklist(logit_tbl, irr_tbl, shap_tbl):
    print("\n" + "=" * 72)
    print("J. CONSISTENCY CHECKLIST (regression vs SHAP vs coding)")
    print("=" * 72)
    gb = shap_tbl[shap_tbl["model"] == "Gradient Boosting"].set_index("feature")
    rows = []
    for t in logit_tbl.index:
        if t == "const":
            continue
        l = logit_tbl.loc[t]
        aor_dir = "higher" if l["AOR"] > 1 else "lower"
        shap_dir, agree = "", "n/a"
        if t in gb.index:
            d = gb.loc[t, "direction"]
            status = gb.loc[t].get("direction_status", "")
            if (status in ("ambiguous", "non-directional")
                    or not isinstance(d, str)
                    or not ("HIGHER" in d or "LOWER" in d)):
                shap_dir, agree = status or "ambiguous", "no directional claim from SHAP"
            else:
                shap_dir = "higher" if "HIGHER" in d else "lower"
                agree = ("consistent" if shap_dir == aor_dir
                         else "DIFFERS - investigate")
        rows.append({
            "term": t, "publication_label": l["predictor"],
            "label_mapped": label_of(t) != t,
            "reference_category": l["reference_category"],
            "coding_key": BINARY_CODING_KEY.get(t, "continuous predictor"),
            "AOR": l["AOR"], "AOR_CI": f"({l['CI_lower']:.2f}, {l['CI_upper']:.2f})",
            "AOR_p": l["p_value"], "AOR_direction": aor_dir,
            "IRR": irr_tbl.loc[t, "IRR"] if t in irr_tbl.index else np.nan,
            "IRR_CI": (f"({irr_tbl.loc[t,'CI_lower']:.2f}, "
                       f"{irr_tbl.loc[t,'CI_upper']:.2f})") if t in irr_tbl.index else "",
            "IRR_p": irr_tbl.loc[t, "p_value"] if t in irr_tbl.index else np.nan,
            "SHAP_direction": shap_dir, "AOR_vs_SHAP": agree,
            "significant_005": bool(l["p_value"] < 0.05)})
    chk = pd.DataFrame(rows)
    chk.to_csv(TAB / "T10_consistency_checklist.csv", index=False)
    n_diff = int((chk["AOR_vs_SHAP"] == "DIFFERS - investigate").sum())
    print(f"  Predictors audited ............................. {len(chk)}")
    print(f"  Every term mapped to a publication label ....... {bool(chk['label_mapped'].all())}")
    print(f"  Duplicate predictor labels ..................... "
          f"{int(chk['publication_label'].duplicated().sum())}")
    print(f"  Reference category shown as an estimate ........ 0 "
          f"(references are omitted by construction)")
    print(f"  Terms where AOR and SHAP directions differ ..... {n_diff}")
    for _, r in chk[chk["AOR_vs_SHAP"] == "DIFFERS - investigate"].iterrows():
        print(f"    - {r['publication_label']}: AOR={r['AOR']:.2f} ({r['AOR_direction']}) "
              f"vs SHAP ({r['SHAP_direction']}); p={r['AOR_p']:.3f}")
    print("  -> tables/T10_consistency_checklist.csv")
    return chk


# ==========================================================================
# figures
# ==========================================================================
def prevalence_figure(cc, r_cc):
    dist = cc["child_deaths"].astype(int).clip(upper=5).value_counts() \
             .reindex(range(6), fill_value=0)
    labels = ["0", "1", "2", "3", "4", "5+"]
    p, lo, hi = (100 * r_cc["estimate"], 100 * r_cc["ci_lower"],
                 100 * r_cc["ci_upper"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5),
                           gridspec_kw={"width_ratios": [2.5, 1]})
    ax[0].bar(labels, dist.values, color=BLUE, edgecolor="black")
    ax[0].set_title("Number of child deaths per woman")
    ax[0].set_xlabel("Number of child deaths"); ax[0].set_ylabel("Number of women")
    for i, v in enumerate(dist.values):
        ax[0].text(i, v + max(dist.values) * 0.01, f"{v:,}", ha="center", fontsize=9)
    ax[1].bar(["Maternal\nbereavement"], [p], color=ORANGE, edgecolor="black",
              width=0.45, yerr=[[p - lo], [hi - p]], capsize=8, ecolor="black")
    ax[1].set_ylim(0, max(hi * 1.35, 30))
    ax[1].set_ylabel("Survey-weighted prevalence (%)")
    ax[1].set_title("Maternal bereavement prevalence")
    ax[1].text(0, hi + 1.2, f"{p:.2f}%\n(95% CI {lo:.2f}-{hi:.2f})",
               ha="center", fontsize=10)
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "F1_prevalence_and_counts.tiff", dpi=300, format="tiff")
    plt.close()


def workflow_figure(audit, n_cc):
    fig, ax = plt.subplots(figsize=(10.5, 7.5)); ax.axis("off")
    steps = [
        f"Nigeria MICS 2021 women's file ({audit['raw_rows']:,} records; "
        f"{audit['interviewed_women']:,} with nonmissing CEB)",
        "Variable-specific codebook recoding (9 retained for MN5, CEB, CSURV)",
        f"Restrict to parous women, CEB >= 1 (n = {audit['parous_women_CEB_ge_1']:,})",
        "Outcomes: child deaths = CEB - CSURV; bereavement = 1 if >= 1 death",
        f"Complete-case analytic sample (n = {n_cc:,}); wmweight / PSU / stratum retained",
        "Survey-weighted descriptives and design-adjusted regression",
        "Count models: survey Poisson + unweighted Poisson / NB2 / ZIP / ZINB",
        "ML: 80/20 split, then calibration split; SMOTE on fitting split only",
        "Calibration-only 5-fold selection/refit; test evaluation + raw log-odds SHAP",
    ]
    ys = np.linspace(0.94, 0.06, len(steps))
    for i, (s, yy) in enumerate(zip(steps, ys)):
        ax.text(0.03, yy, s, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.45", fc="#f2f2f2", ec="black"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.06, yy - 0.052), xytext=(0.06, yy - 0.018),
                        arrowprops=dict(arrowstyle="->", lw=1.4))
    plt.tight_layout()
    plt.savefig(FIG / "F0_workflow.png", dpi=200)
    plt.savefig(FIG / "F0_workflow.tiff", dpi=300, format="tiff")
    plt.close()


# ==========================================================================
def main(data_path=None, output_dir=None):
    """Run all analyses using a separately obtained SPSS women's dataset."""
    global FIG, TAB
    data_path = pathlib.Path(data_path or ROOT / "Nig_Women_MICs_data.sav").resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            "SPSS input not found. Supply --data with the privately obtained dataset.")
    output_dir = pathlib.Path(output_dir or ROOT).resolve()
    FIG, TAB = output_dir / "figures", output_dir / "tables"
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df_all, meta, audit = data_prep.load_and_clean(data_path)
    su.set_design_registry(df_all.attrs['original_design'])
    cc = data_prep.complete_case(df_all)
    assert len(cc) == cc['child_deaths'].value_counts().sum()
    assert (cc['child_death'] == (cc['child_deaths'] >= 1)).all()
    assert not cc[data_prep.DESIGN_VARS].isna().any().any()
    pd.DataFrame(audit['recodes']).to_csv(TAB / 'T0_coding_audit.csv', index=False)
    n_par = audit["parous_women_CEB_ge_1"]

    print(f"Complete-case analytic sample .............. {len(cc):,}")
    print(f"Excluded for missing analytic information .. {n_par - len(cc):,}")
    print(f"Percentage of parous women retained ........ {100*len(cc)/n_par:.1f}%")
    print(f"Percentage of nonmissing-CEB proxy retained . "
          f"{100*len(cc)/audit['interviewed_women']:.1f}%\n")

    desc, r_all, r_cc, deff = survey_design_audit(df_all, cc, audit)
    rep = representativeness(df_all, cc)
    prevalence_figure(cc, r_cc)

    logit_tbl, Xl, yl = survey_logistic(cc)
    irr_tbl, fit, cmodels, best = count_models(cc)
    results, test, met = machine_learning(cc)
    sa = shap_audit(results, test)
    chk = consistency_checklist(logit_tbl, irr_tbl, sa)
    workflow_figure(audit, len(cc))

    summary = {
        "raw_records": audit["raw_rows"],
        "interviewed_women": audit["interviewed_women"],
        "interview_count_basis": audit["interview_count_basis"],
        "parous_women": n_par,
        "complete_case_n": int(len(cc)),
        "excluded_n": int(n_par - len(cc)),
        "pct_parous_retained": round(100 * len(cc) / n_par, 2),
        "pct_interviewed_retained": round(
            100 * len(cc) / audit["interviewed_women"], 2),
        "value9_retained_for": [r["variable"] for r in audit["recodes"]
                                if r["value_9_retained"]],
        "value9_missing_for": [r["variable"] for r in audit["recodes"]
                               if not r["value_9_retained"]],
        "missing_code_map": {r["variable"]: r["codes_treated_as_missing"]
                             for r in audit["recodes"]},
        "weighted_prevalence_pct": round(100 * r_cc["estimate"], 2),
        "weighted_prevalence_ci": [round(100 * r_cc["ci_lower"], 2),
                                   round(100 * r_cc["ci_upper"], 2)],
        "unweighted_prevalence_pct": round(100 * float(cc["child_death"].mean()), 2),
        "weighted_prevalence_all_parous_pct": round(100 * r_all["estimate"], 2),
        "design_effect": round(deff, 3),
        "design_df": int(logit_tbl["design_df"].iloc[0]),
        "n_psu": int(cc["PSU"].nunique()),
        "n_strata": int(cc["stratum"].nunique()),
        "mean_child_deaths": round(float(cc["child_deaths"].mean()), 4),
        "pct_zero_deaths": round(100 * float((cc["child_deaths"] == 0).mean()), 2),
        "best_count_model_by_aic": best,
        "count_model_fit": fit.reset_index().to_dict("records"),
        "ml_selected": {k: {"calibration_reported": v["chosen"],
                            **{kk: round(float(vv), 4)
                               for kk, vv in v["metrics_final"].items()}}
                        for k, v in results.items()},
    }
    with open(TAB / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("K. OUTPUTS WRITTEN")
    print("=" * 72)
    for p in sorted(TAB.glob("*")):
        print("  tables/" + p.name)
    for p in sorted(FIG.glob("*")):
        print("  figures/" + p.name)
    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path,
                        default=ROOT / "Nig_Women_MICs_data.sav",
                        help="Path to the separately obtained SPSS women's file.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT,
                        help="Parent directory for generated tables and figures.")
    args = parser.parse_args()
    main(args.data, args.output_dir)
