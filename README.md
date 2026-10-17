# Maternal child loss: survey inference and predictive modelling

Python analysis of maternal bereavement and child-death counts using the
2021 Nigeria Multiple Indicator Cluster Survey (MICS) women's file.
This repository distributes analytical code only. It contains no microdata,
results, rendered figures, manuscripts, notebooks or presentation generators.

## Pipeline

- Variable-specific SPSS cleaning and a shared complete-case sample of parous women.
- Outcomes: any child loss (at least one death) and number of child deaths,
  derived as children ever born minus children surviving.
- Survey-weighted prevalence, count means, design effect and included/excluded
  sample comparisons.
- Primary survey-adjusted logistic and Poisson regression using woman-level
  weights, strata and PSU-level Taylor-linearised covariance. The original
  sampled PSU registry is preserved for domain analysis.
- Secondary unweighted Poisson, NB2, zero-inflated Poisson (ZIP) and
  zero-inflated NB2 (ZINB): convergence audit, AIC/BIC, count-component
  coefficients, fitted count distributions and rootogram. All four models are
  attempted; unsuccessful fits are explicitly excluded rather than simplified.
- Logistic regression, random forest, gradient boosting and XGBoost classifiers.
  Stratified fitting/calibration/test splits, fitting-only preprocessing and
  SMOTE, calibration-only five-fold Platt/isotonic selection, and internal test
  evaluation with ROC, precision-recall and calibration plots.
- Gradient-boosting and XGBoost SHAP summaries on the uncalibrated classifiers'
  raw log-odds scale, with additivity checks and a predictor-coding audit.

All analytical plotting functions are included; generated images are not.
Zero-inflated-model coefficient tables report the count component, not the
inflation equation. The ZIP model is a count regression, not an ML classifier.

## Files

| File | Purpose |
| --- | --- |
| analysis_validation.py | Analysis entry point, model fitting, comparison and analytical plots |
| data_prep.py | SPSS import, explicit recoding, outcomes and complete-case selection |
| survey_utils.py | Survey-weighted descriptives and canonical-link GLM covariance |
| validate_analysis.py | Dataset-free numerical checks and optional exported-table checks |
| requirements.txt | Direct dependencies pinned to the tested environment |

## Input and execution

Use Python 3.12. Obtain the Nigeria MICS women's microdata independently through
the official MICS data-access process and comply with its access and reuse terms.
No data are downloaded by this code. Supply the study-specific SPSS file with
the required variable names; this is not a generic loader for every MICS release.
Recoding rules should be checked against the accompanying survey metadata before
using a different extract.

Install dependencies with `python -m pip install -r requirements.txt`.

Run dataset-free checks with `python validate_analysis.py`.

Run the complete pipeline with
`python analysis_validation.py --data /path/to/womens_file.sav --output-dir /path/to/results`.
Paths containing spaces must be quoted. Omitting `--data` looks for
Nig_Women_MICs_data.sav beside the script; omitting `--output-dir` writes beside
the script. Prefer a fresh output directory for each run. Full count-model
optimisation can take substantially longer than the classifier analyses.

Outputs are created in `tables` and `figures` below the chosen output directory.
After a run, check tables with
`python validate_analysis.py --outputs /path/to/results/tables`.
Run QC without Python's `-O` option: its numerical checks use assertions.

## Interpretation and scope

- The binary outcome is any loss, not specifically recurrent loss. Counts are
  retrospective deaths per woman; Poisson ratios are not person-time mortality
  rates because no exposure-time offset is fitted.
- Complete-case, cross-sectional associations do not establish causality.
  Survey weighting does not remove selection bias from missing analytical data.
  Nonmissing CEB is an interview-count proxy, not verified interview completion.
- The custom ultimate-cluster survey estimators have internal algebraic checks;
  this release does not claim external benchmarking against R or Stata.
- Unweighted likelihood comparisons are complementary and must not be used to
  select the primary survey inferential model. AIC/BIC describe in-sample fit,
  not held-out predictive performance.
- Ordinary SMOTE can interpolate categorical dummy variables. Row-level splits
  can share survey clusters. The holdout was previously inspected; reported
  results are internal evaluation, not external validation or deployment evidence.
- SHAP describes fitted-model contributions, not causal effects or calibrated
  probability changes. Logistic AORs and SHAP importance measure different things.

## Publication boundary

The Git allowlist includes only the four analytical modules and these minimal
repository support files. Data, virtual environments, caches, logs, generated
tables/figures, manuscript files and poster/graphical-abstract generators are
excluded. No participant-level records or fitted model objects are exported by
the pipeline. Review the staged file list before every publication; do not force
add ignored files.