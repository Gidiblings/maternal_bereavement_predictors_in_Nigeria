"""Taylor-linearised descriptives and canonical GLMs for clustered surveys.

Under the ultimate-cluster assumption, the variance of PSU totals is

    V = sum_h  (n_h / (n_h - 1)) * sum_{c in h} (u_hc - ubar_h)(u_hc - ubar_h)'

where u_hc is the PSU-level total of the weighted score (or linearised
value) contributions. For GLMs with canonical links (logit for binomial,
log for Poisson) the score contribution of observation i is

    u_i = w_i * (y_i - mu_i) * x_i

and the sandwich estimator is  A^-1 V A^-1  with A^-1 = (X' W D X)^-1,
which statsmodels returns directly as `cov_params()` when the scale is
fixed at 1 (true for Binomial and Poisson families).

Degrees of freedom = (number of PSUs) - (number of strata), as in Stata.
Strata containing a single PSU ("lonely PSU") are centred on the grand
mean of all PSU totals rather than dropped (the Stata
`singleunit(centered)` option).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DESIGN_REGISTRY = None


def set_design_registry(design):
    """Preserve sampled PSU/stratum pairs, including zero-contribution domains."""
    global DESIGN_REGISTRY
    if design[['PSU', 'stratum']].isna().any().any():
        raise ValueError('Missing survey design identifiers')
    DESIGN_REGISTRY = pd.MultiIndex.from_frame(
        design[['stratum', 'PSU']].drop_duplicates().rename(
            columns={'stratum': '__str', 'PSU': '__psu'}))


def _design_ids(psu, strata, n):
    """Validate one design identifier per observation without dropping rows."""
    psu, strata = np.asarray(psu), np.asarray(strata)
    if psu.shape != (n,) or strata.shape != (n,):
        raise ValueError('Survey identifiers must be one-dimensional and match data')
    if pd.isna(psu).any() or pd.isna(strata).any():
        raise ValueError('Missing survey design identifiers')
    return psu, strata


def _descriptive_inputs(values, w, psu, strata, alpha, proportion=False):
    """Drop nonfinite outcomes only; reject invalid weights and identifiers."""
    values, w = np.asarray(values, dtype=float), np.asarray(w, dtype=float)
    if values.ndim != 1 or w.shape != values.shape:
        raise ValueError('Outcomes and weights must be matching one-dimensional arrays')
    psu, strata = _design_ids(psu, strata, len(values))
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Sampling weights must be finite and nonnegative')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    ok = np.isfinite(values)
    values, w, psu, strata = values[ok], w[ok], psu[ok], strata[ok]
    if not len(values):
        raise ValueError('No finite outcomes')
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Sampling weights must have a finite positive total')
    if proportion and ((values < 0).any() or (values > 1).any()):
        raise ValueError('Proportion outcomes must be between zero and one')
    return values, w, psu, strata


# ----------------------------------------------------------------------
# core variance machinery
# ----------------------------------------------------------------------
def taylor_variance(u, psu, strata):
    """Stratified ultimate-cluster variance of the total of ``u``.

    Parameters
    ----------
    u : (n, k) array of linearised / score contributions (already weighted)
    psu : (n,) primary sampling unit identifiers
    strata : (n,) stratum identifiers

    Returns
    -------
    V : (k, k) variance matrix of sum(u)
    dof : design degrees of freedom (n_psu - n_strata)
    """
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    if u.ndim != 2 or not all(u.shape) or not np.isfinite(u).all():
        raise ValueError('Taylor contributions must be a nonempty finite matrix')
    psu, strata = _design_ids(psu, strata, u.shape[0])
    k = u.shape[1]

    frame = pd.DataFrame(u)
    frame["__psu"] = np.asarray(psu)
    frame["__str"] = np.asarray(strata)

    # PSU-level totals
    clus = frame.groupby(["__str", "__psu"], sort=False)[list(range(k))].sum()
    if DESIGN_REGISTRY is not None:
        if not clus.index.isin(DESIGN_REGISTRY).all():
            raise ValueError('Analytical PSU not in original design registry')
        clus = clus.reindex(DESIGN_REGISTRY, fill_value=0.0)
    clus_arr = clus.to_numpy()
    stratum_of_cluster = clus.index.get_level_values("__str").to_numpy()

    grand_mean = clus_arr.mean(axis=0)

    V = np.zeros((k, k))
    n_psu = clus_arr.shape[0]
    strata_ids = np.unique(stratum_of_cluster)

    for h in strata_ids:
        rows = clus_arr[stratum_of_cluster == h]
        n_h = rows.shape[0]
        if n_h < 2:
            # lonely PSU -> centre on grand mean (Stata singleunit(centered))
            dev = rows - grand_mean
            V += dev.T @ dev
        else:
            dev = rows - rows.mean(axis=0)
            V += (n_h / (n_h - 1.0)) * (dev.T @ dev)

    dof = n_psu - len(strata_ids)
    if dof <= 0:
        raise ValueError('Survey design has no positive residual degrees of freedom')
    return V, dof


# ----------------------------------------------------------------------
# descriptive estimators
# ----------------------------------------------------------------------
def weighted_proportion(y, w, psu, strata, alpha=0.05):
    """Design-based weighted proportion with logit-transformed CI."""
    y, w, psu, strata = _descriptive_inputs(
        y, w, psu, strata, alpha, proportion=True)

    W = w.sum()
    p = float((w * y).sum() / W)

    # linearised value for a ratio estimator
    z = w * (y - p) / W
    V, dof = taylor_variance(z, psu, strata)
    se = float(np.sqrt(V[0, 0]))

    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    if 0 < p < 1 and se > 0:
        lo_l = np.log(p / (1 - p)) - tcrit * se / (p * (1 - p))
        hi_l = np.log(p / (1 - p)) + tcrit * se / (p * (1 - p))
        lo, hi = 1 / (1 + np.exp(-lo_l)), 1 / (1 + np.exp(-hi_l))
    else:
        lo, hi = max(p - tcrit * se, 0.0), min(p + tcrit * se, 1.0)

    return {
        "estimate": p, "se": se, "ci_lower": float(lo), "ci_upper": float(hi),
        "dof": int(dof), "n": int(len(y)), "sum_weights": float(W),
    }


def weighted_mean(x, w, psu, strata, alpha=0.05):
    """Design-based weighted mean with a t-based CI."""
    x, w, psu, strata = _descriptive_inputs(x, w, psu, strata, alpha)

    W = w.sum()
    m = float((w * x).sum() / W)
    z = w * (x - m) / W
    V, dof = taylor_variance(z, psu, strata)
    se = float(np.sqrt(V[0, 0]))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    return {
        "estimate": m, "se": se, "ci_lower": m - tcrit * se,
        "ci_upper": m + tcrit * se, "dof": int(dof), "n": int(len(x)),
    }


# ----------------------------------------------------------------------
# design-adjusted GLM
# ----------------------------------------------------------------------
def design_adjusted_glm(y, X, w, psu, strata, family, alpha=0.05,
                        exponentiate=True):
    """Fit a survey-weighted GLM with a stratified-cluster sandwich covariance.

    Only canonical links are supported (Binomial/logit, Poisson/log), for which
    the observation-level score is w_i (y_i - mu_i) x_i.

    Returns (results_table, fitted_statsmodels_result, design_cov).
    """
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    Xa = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    if not ((isinstance(family, sm.families.Binomial)
             and isinstance(family.link, sm.families.links.Logit))
            or (isinstance(family, sm.families.Poisson)
                and isinstance(family.link, sm.families.links.Log))):
        raise ValueError('Only canonical binomial/logit and Poisson/log supported')
    if (y.ndim != 1 or not y.size or Xa.ndim != 2 or not Xa.shape[1]
            or Xa.shape[0] != len(y) or w.shape != y.shape):
        raise ValueError('Model outcomes, matrix and weights must have matching rows')
    psu, strata = _design_ids(psu, strata, len(y))
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    if not np.isfinite(Xa).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite model data')
    if (not np.isfinite(w).all() or (w < 0).any()
            or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError('Invalid sampling weights')
    if np.linalg.matrix_rank(Xa) != Xa.shape[1]:
        raise ValueError('Rank-deficient model matrix')
    # normalise weights to mean 1 (point estimates unchanged; keeps the
    # statsmodels deviance/likelihood on the sample scale)
    w_norm = w * (len(w) / w.sum())

    res = sm.GLM(y, Xa, family=family, var_weights=w_norm).fit(scale=1.0)
    if not res.converged:
        raise RuntimeError('Survey estimating-equation GLM did not converge')

    mu = np.asarray(res.fittedvalues, dtype=float)
    u = (w_norm * (y - mu))[:, None] * Xa          # score contributions
    V_meat, dof = taylor_variance(u, psu, strata)
    A_inv = np.asarray(res.cov_params(), dtype=float)   # (X'WDX)^-1  (scale = 1)
    cov = A_inv @ V_meat @ A_inv

    beta = np.asarray(res.params, dtype=float)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    lo, hi = beta - tcrit * se, beta + tcrit * se

    names = list(X.columns) if hasattr(X, "columns") else \
        [f"x{i}" for i in range(Xa.shape[1])]

    tbl = pd.DataFrame(
        {"coef": beta, "se": se, "t": tval, "p_value": pval,
         "ci_lower": lo, "ci_upper": hi}, index=names)
    if exponentiate:
        tbl["estimate"] = np.exp(beta)
        tbl["est_ci_lower"] = np.exp(lo)
        tbl["est_ci_upper"] = np.exp(hi)
    tbl["design_df"] = dof
    return tbl, res, cov


def design_effect(y, w, psu, strata):
    """Ratio of the design-based variance to the SRS variance of a proportion."""
    r = weighted_proportion(y, w, psu, strata)
    p = r["estimate"]
    n = r["n"]
    srs_var = p * (1 - p) / n
    return float(r["se"] ** 2 / srs_var) if srs_var > 0 else np.nan
