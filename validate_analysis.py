"""Reproducible algebraic and saved-output QC; not external survey validation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import survey_utils as su


def _rejects_value_error(function, *args, **kwargs):
    """Require explicit rejection, not an incidental indexing/type error."""
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f'{function.__name__} accepted invalid input')


def synthetic_checks():
    """Run dataset-free checks, restoring the caller's survey registry."""
    previous_registry = su.DESIGN_REGISTRY
    try:
        registry = pd.DataFrame({'PSU': [1, 2, 3, 4], 'stratum': [1, 1, 2, 2]})
        su.set_design_registry(registry)
        # Zero-contribution domain PSUs retain the original design degrees of freedom.
        v, df = su.taylor_variance([1., 2.], [1, 3], [1, 2])
        assert np.allclose(v, [[5.]]) and df == 2
        x = pd.DataFrame({'const': np.ones(8), 'x': [-1., 0., 1., 2.] * 2})
        y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        w = np.arange(1., 9.)
        p = np.repeat([1, 2, 3, 4], 2)
        h = np.repeat([1, 2], 4)
        for family, outcome in [(sm.families.Binomial(), y),
                                (sm.families.Poisson(), y + np.arange(8) % 3)]:
            t, r, cov = su.design_adjusted_glm(outcome, x, w, p, h, family)
            t2, _, cov2 = su.design_adjusted_glm(outcome, x, w * 123, p, h, family)
            assert np.allclose(t.coef, t2.coef) and np.allclose(cov, cov2)
            # Independently assemble PSU scores and inverse expected information.
            mu = np.asarray(r.fittedvalues)
            wn = w * len(w) / w.sum()
            xa = x.to_numpy()
            score = (wn * (outcome - mu))[:, None] * xa
            z = np.array([score[p == c].sum(axis=0) for c in [1, 2, 3, 4]])
            meat = np.zeros((2, 2))
            for pair in [z[:2], z[2:]]:
                dev = pair - pair.mean(axis=0)
                meat += 2 * dev.T @ dev
            variance = mu * (1 - mu) if isinstance(family, sm.families.Binomial) else mu
            bread = np.linalg.inv(xa.T @ ((wn * variance)[:, None] * xa))
            assert np.allclose(cov, bread @ meat @ bread)
            assert (t.design_df == df).all()

        for estimator in (su.weighted_mean, su.weighted_proportion):
            result = estimator(y, w, p, h)
            scaled = estimator(y, w * 123, p, h)
            for key in ('estimate', 'se', 'ci_lower', 'ci_upper', 'dof', 'n'):
                assert np.isclose(result[key], scaled[key])
            dropped = estimator([0., np.nan, np.inf, 1.], [1.] * 4,
                                [1, 2, 3, 4], [1, 1, 2, 2])
            assert dropped['n'] == 2 and dropped['estimate'] == 0.5
            zero_weight = estimator([0., 1.], [0., 1.], [1, 2], [1, 1])
            assert zero_weight['estimate'] == 1.
            for values, weights, psus, strata in [
                    ([], [], [], []),
                    ([np.nan, np.inf], [1., 1.], [1, 2], [1, 1]),
                    ([0., 1.], [0., 0.], [1, 2], [1, 1]),
                    ([0., 1.], [-1., 2.], [1, 2], [1, 1]),
                    ([0., 1.], [np.nan, 1.], [1, 2], [1, 1]),
                    ([0., 1.], [np.inf, 1.], [1, 2], [1, 1]),
                    ([0., np.nan], [0., 1.], [1, 2], [1, 1]),
                    ([0., np.nan], [1., np.nan], [1, 2], [1, 1]),
                    ([0., 1.], [1.], [1, 2], [1, 1]),
                    ([[0., 1.]], [[1., 1.]], [1, 2], [1, 1]),
                    ([0., 1.], [1., 1.], [1], [1, 1]),
                    ([0., 1.], [1., 1.], [1, None], [1, 1]),
                    ([0., 1.], [1., 1.], [1, 2], [1, None]),
                    ([0., 1.], [1., 1.], [[1], [2]], [1, 1])]:
                _rejects_value_error(estimator, values, weights, psus, strata)
            _rejects_value_error(estimator, y, w, p, h, alpha=0)
        for values in ([-0.1, 1.], [0., 1.1]):
            _rejects_value_error(su.weighted_proportion, values, [1., 1.], [1, 2], [1, 1])
        for family in (sm.families.Binomial(link=sm.families.links.Log()),
                       sm.families.Poisson(link=sm.families.links.Identity()),
                       sm.families.Gaussian()):
            _rejects_value_error(su.design_adjusted_glm, y, x, w, p, h, family)
        for outcome, matrix, weights, psus, strata in [
            (y[:-1], x, w, p, h),
            (y, x.to_numpy()[:, 0], w, p, h),
            (y, x, np.zeros(8), p, h),
            (y, x, -w, p, h),
            (y, x, np.full(8, np.nan), p, h),
            (y, x, w, p[:-1], h),
            (y, x, w, p, [None] * 8)]:
            _rejects_value_error(su.design_adjusted_glm, outcome, matrix, weights,
                     psus, strata, sm.families.Binomial())
        for contributions, psus, strata in [
                ([], [], []), (1., [1], [1]),
                (np.ones((2, 1, 1)), [1, 2], [1, 1]),
                (np.empty((2, 0)), [1, 2], [1, 1]),
                ([np.nan, 1.], [1, 2], [1, 1]),
                ([np.inf, 1.], [1, 2], [1, 1]),
                ([0., 1.], [1], [1, 1]),
                ([0., 1.], [1, None], [1, 1]),
                ([0., 1.], [1, 2], [1, None]),
                ([0., 1.], [1, 2], [[1], [1]]),
                ([0., 1.], [1, 99], [1, 1])]:
            _rejects_value_error(su.taylor_variance, contributions, psus, strata)
        su.DESIGN_REGISTRY = None
        _rejects_value_error(su.taylor_variance, [1.], [1], [1])
    finally:
        su.DESIGN_REGISTRY = previous_registry


def _booleans(values):
    """Reject missing/ambiguous CSV flags rather than treating strings as truthy."""
    flags = values.astype(str).str.strip().str.lower()
    assert flags.isin(['true', 'false']).all(), 'Invalid boolean flags'
    return flags.eq('true')


def _check_cis(table, estimate):
    """Check available intervals; absent covariance may leave missing bounds."""
    est, lo, hi = table[[estimate, 'CI_lower', 'CI_upper']].to_numpy(dtype=float).T
    finite_lo = np.isfinite(lo)
    finite_hi = np.isfinite(hi)
    assert (lo[finite_lo] >= 0).all() and (hi[finite_hi] >= 0).all()
    lower_pair = finite_lo & np.isfinite(est)
    upper_pair = finite_hi & np.isfinite(est)
    bounds_pair = finite_lo & finite_hi
    assert (lo[lower_pair] <= est[lower_pair]).all(), 'Lower CI exceeds estimate'
    assert (est[upper_pair] <= hi[upper_pair]).all(), 'Upper CI below estimate'
    assert (lo[bounds_pair] <= hi[bounds_pair]).all(), 'Reversed confidence interval'


def check_outputs(output_dir):
    """Check exported tables in output_dir without reading data or fitting models."""
    output_dir = Path(output_dir)
    survey_tables = []
    for filename, estimate in [('T3_survey_logistic_AOR.csv', 'AOR'),
                               ('T4_survey_poisson_IRR.csv', 'IRR')]:
        table = pd.read_csv(output_dir / filename)
        assert not table.empty, f'{filename}: empty survey table'
        assert table.predictor.notna().all() and not table.predictor.duplicated().any()
        assert np.isfinite(table[estimate]).all(), f'{filename}: nonfinite estimates'
        _check_cis(table, estimate)
        survey_tables.append(table)
    design_df = pd.concat([table.design_df for table in survey_tables], ignore_index=True)
    assert np.isfinite(design_df).all() and (design_df > 0).all()
    assert (design_df == np.floor(design_df)).all() and design_df.nunique() == 1
    nterms = len(survey_tables[0])
    expected_k = {'Poisson': nterms, 'NegBin': nterms + 1,
                  'ZIP': 2 * nterms, 'ZINB': 2 * nterms + 1}
    fit = pd.read_csv(output_dir / 'T5_count_model_fit.csv').set_index('model')
    assert fit.index.is_unique and set(fit.index) == set(expected_k)
    assert np.isfinite(fit.n).all() and (fit.n > 0).all() and fit.n.nunique() == 1
    assert (fit.n == np.floor(fit.n)).all()
    accepted = _booleans(fit.converged)
    for name, row in fit.iterrows():
        table = pd.read_csv(output_dir / f'T6_count_IRR_{name}.csv')
        if not accepted.loc[name]:
            assert table.empty, f'{name}: failed model has reportable coefficients'
            assert row[['k', 'llf', 'aic', 'bic']].isna().all(), f'{name}: stale failed-model fit'
            continue
        assert row.k == expected_k[name], f'{name}: incorrect parameter count'
        assert np.isfinite([row.llf, row.aic, row.bic]).all()
        assert np.isclose(row.aic, 2 * row.k - 2 * row.llf), f'{name}: AIC mismatch'
        assert np.isclose(row.bic, row.k * np.log(row.n) - 2 * row.llf), f'{name}: BIC mismatch'
        if name in ('NegBin', 'ZINB'):
            assert np.isfinite(row.alpha) and row.alpha > 0, f'{name}: invalid dispersion'
        assert len(table) == nterms, f'{name}: missing count coefficients'
        assert table.predictor.notna().all() and not table.predictor.duplicated().any()
        assert np.isfinite(table.IRR).all(), f'{name}: nonfinite coefficients'
        _check_cis(table, 'IRR')
    # Compare likelihoods only when both nested models were accepted.
    for larger, smaller in [('NegBin', 'Poisson'), ('ZIP', 'Poisson'),
                            ('ZINB', 'NegBin'), ('ZINB', 'ZIP')]:
        if accepted.loc[larger] and accepted.loc[smaller]:
            assert fit.loc[larger, 'llf'] >= fit.loc[smaller, 'llf'] - 1e-6, \
                f'{larger}: likelihood below {smaller}'

    selection = pd.read_csv(output_dir / 'T11_calibration_selection.csv')
    assert not selection.empty and not selection.model.duplicated().any()
    assert not _booleans(selection.test_outcomes_used_for_selection).any()
    assert _booleans(selection.refit_on_full_calibration).all()
    assert (selection.cv_folds >= 2).all()
    assert selection.selection_metric.eq('pooled OOF Brier').all()
    scores = selection[['Platt_cv_brier', 'isotonic_cv_brier']].to_numpy(dtype=float)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    choices = np.where(scores[:, 0] <= scores[:, 1], 'Platt', 'isotonic')
    assert selection.selected_method.eq(choices).all(), 'Calibration not selected by CV Brier'
    shap = pd.read_csv(output_dir / 'T9_shap_audit.csv')
    assert not shap.empty and _booleans(shap.additivity_pass).all()


def main(argv=None):
    """Run synthetic checks by default; saved-output QC is opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='?',
                        const=Path(__file__).resolve().parent / 'tables',
                        help='also check this tables directory (flag alone: tables beside this script)')
    args = parser.parse_args(argv)
    synthetic_checks()
    print('PASS: synthetic survey validation, sandwich algebra and weight invariance.')
    if args.outputs is not None:
        check_outputs(args.outputs)
        print('PASS: saved-output count models, survey CIs, calibration CV and SHAP.')


if __name__ == '__main__':
    main()
