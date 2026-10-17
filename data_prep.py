"""Variable-specific cleaning for the 2021 Nigeria MICS women's data.

Missing-code rules use SPSS value labels and explicit analytic classifications,
not code magnitude or cardinality: MN5 and MN4AN_2 use 98/99; welevel, MA1,
MA3, MN2 and MN4AU_1 use 9; MN20 uses 76/96/99 (unclassifiable delivery
settings). MN5 value 77 is undefined and retained pending codebook confirmation.
Value 9 is retained for MN5, MN4AN_2, CEB, CSURV and CDEAD.

CEB, CSURV, CDEAD, MA11C, WB4, wscore, HH6 and zone have no designated
missing codes. Numeric conversion failures and negative MA11C values become
missing. Education maps codes 0–4; MN2 retains only 1/2. Delivery settings map
MN20 codes 21/22/23/31/32, 33/36/26 and 11/12 to three categories; other
codes become missing. Analysis retains CEB >= 1 and derives child deaths as
CEB minus CSURV, setting negative differences to missing. Survey design
variables are wmweight, PSU and stratum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat

# --------------------------------------------------------------------------
# Explicit per-variable missing-code dictionary (the audit trail)
# --------------------------------------------------------------------------
MISSING_CODES = {
    "MN5":     [98, 99],
    "CEB":     [],
    "CSURV":   [],
    "CDEAD":   [],
    "welevel": [9],
    "MA1":     [9],
    "MA3":     [9],
    "MN2":     [9],
    "MN20":    [76, 96, 99],
    "MN4AU_1": [9],
    "MN4AN_2": [98, 99],
    "MA11C":   [],          # handled by a logical-range rule below
    "WB4":     [],
    "wscore":  [],
    "HH6":     [],
    "zone":    [],
}

EDUCATION_LEVELS = ["None", "Primary", "Junior Secondary",
                    "Senior Secondary", "Tertiary"]
EDUCATION_MAP = {0: "None", 1: "Primary", 2: "Junior Secondary",
                 3: "Senior Secondary", 4: "Tertiary"}

DESIGN_VARS = ["wmweight", "PSU", "stratum"]

ANALYTIC_VARS = ["child_death", "child_deaths", "CEB", "WB4", "welevel",
                 "MA1", "MA3", "MA11C", "wscore", "HH6", "zone", "MN2",
                 "place_delivery"]


def load_and_clean(data_path, verbose=True):
    """Return cleaned parous data, SPSS metadata and audit (three values).

    No complete-case restriction is applied.
    """
    df, meta = pyreadstat.read_sav(str(data_path), apply_value_formats=False)
    audit = {"raw_rows": len(df), "recodes": []}

    required_columns = [
        "CEB", "CSURV", "welevel", "MA1", "MA3", "MA11C", "wscore",
        "HH6", "zone", "MN2", "MN20", "WB4", *DESIGN_VARS,
    ]
    missing_columns = [var for var in required_columns if var not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns not found in dataset: {', '.join(missing_columns)}")
    if df[["PSU", "stratum"]].isna().any().any():
        raise ValueError("Survey design IDs PSU and stratum must not be missing")
    original_design = df[['PSU', 'stratum']].drop_duplicates().copy()

    audit["design_vars_present"] = {
        v: {"n_missing": int(df[v].isna().sum()),
            "n_unique": int(df[v].nunique()),
            "min": float(df[v].min()), "max": float(df[v].max())}
        for v in DESIGN_VARS}

    # Nonmissing CEB proxy; retain the legacy audit key for compatibility.
    audit["interviewed_women"] = int(df["CEB"].notna().sum())
    audit['interview_count_basis'] = 'Nonmissing CEB proxy; no interview-result field in supplied file'

    # ---- apply the explicit codebook rules ------------------------------
    for var, codes in MISSING_CODES.items():
        if var not in df.columns:
            continue
        df[var] = pd.to_numeric(df[var], errors="coerce")
        before = int(df[var].notna().sum())
        if codes:
            df.loc[df[var].isin(codes), var] = np.nan
        after = int(df[var].notna().sum())
        audit["recodes"].append({
            "variable": var,
            "label": meta.column_names_to_labels.get(var, ""),
            "codes_treated_as_missing": codes if codes else "none",
            "value_9_retained": 9 not in codes,
            "n_valid_before": before, "n_valid_after": after,
            "n_set_missing": before - after})

    # MA11C: logically impossible ages at first union
    if "MA11C" in df.columns:
        n_bad = int((df["MA11C"] < 0).sum())
        df.loc[df["MA11C"] < 0, "MA11C"] = np.nan
        audit["MA11C_negative_set_missing"] = n_bad

    # ---- restrict to parous women (CEB >= 1), the study population ------
    df = df[df["CEB"] >= 1].copy()
    audit["parous_women_CEB_ge_1"] = len(df)

    # ---- outcomes --------------------------------------------------------
    df["child_deaths"] = df["CEB"] - df["CSURV"]
    audit["impossible_CSURV_gt_CEB"] = int((df["child_deaths"] < 0).sum())
    df.loc[df["child_deaths"] < 0, "child_deaths"] = np.nan
    # consistency with the survey's own CDEAD variable
    if "CDEAD" in df.columns:
        paired = df["child_deaths"].notna() & df["CDEAD"].notna()
        audit["child_deaths_matches_CDEAD_pct"] = float(
            100 * (df.loc[paired, "child_deaths"] == df.loc[paired, "CDEAD"]).mean()
        ) if paired.any() else np.nan
    df["child_death"] = np.where(df["child_deaths"] >= 1, 1.0,
                                 np.where(df["child_deaths"] == 0, 0.0, np.nan))
    if (df['child_deaths'].dropna() % 1 != 0).any():
        raise ValueError('Noninteger child-death count')

    # ---- education -------------------------------------------------------
    if "welevel" in df.columns:
        df["welevel"] = pd.Categorical(
            pd.to_numeric(df["welevel"], errors="coerce").map(EDUCATION_MAP),
            categories=EDUCATION_LEVELS)

    # ---- place of delivery ----------------------------------------------
    if "MN20" not in df.columns:
        raise ValueError("MN20 not found in dataset")
    df["place_delivery"] = np.select(
        [df["MN20"].isin([21, 22, 23, 31, 32]),   # 1 hospital / clinic
         df["MN20"].isin([33, 36, 26]),           # 2 maternity / other facility
         df["MN20"].isin([11, 12])],              # 3 home
        [1.0, 2.0, 3.0], default=np.nan)

    # ---- prenatal care ---------------------------------------------------
    if "MN2" in df.columns:
        df["MN2"] = pd.to_numeric(df["MN2"], errors="coerce")
        df.loc[~df["MN2"].isin([1, 2]), "MN2"] = np.nan

    if verbose:
        print_audit(audit)
    df.attrs['original_design'] = original_design
    return df, meta, audit


def complete_case(df, features=None):
    """Single shared complete-case analytic sample (design vars retained)."""
    features = features or ANALYTIC_VARS
    keep = list(dict.fromkeys(features + DESIGN_VARS))
    cc = df[keep].dropna(subset=features).copy()
    return cc


def print_audit(audit):
    print("=" * 72)
    print("A. DATA-CODING AUDIT")
    print("=" * 72)
    print(f"Raw records in file ........................ {audit['raw_rows']:,}")
    print(f"Nonmissing CEB interview-count proxy ....... {audit['interviewed_women']:,}")
    print(f"Parous women (CEB >= 1) .................... {audit['parous_women_CEB_ge_1']:,}")
    print(f"Logically impossible CSURV > CEB ........... {audit['impossible_CSURV_gt_CEB']}")
    print(f"MA11C negative values set missing .......... {audit.get('MA11C_negative_set_missing', 0)}")
    if "child_deaths_matches_CDEAD_pct" in audit:
        print(f"(CEB-CSURV) agrees with MICS CDEAD .......... "
              f"{audit['child_deaths_matches_CDEAD_pct']:.2f}%")
    print("\nSurvey design variables:")
    for v, d in audit["design_vars_present"].items():
        print(f"  {v:10s} missing={d['n_missing']:<5} unique={d['n_unique']:<6} "
              f"range=[{d['min']:.4g}, {d['max']:.4g}]")
    print("\nPer-variable missing-code treatment:")
    hdr = f"  {'variable':9s} {'treated as missing':24s} {'9 kept':7s} {'n set missing':>13s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in audit["recodes"]:
        print(f"  {r['variable']:9s} {str(r['codes_treated_as_missing']):24s} "
              f"{str(r['value_9_retained']):7s} {r['n_set_missing']:>13,}")
    print()
