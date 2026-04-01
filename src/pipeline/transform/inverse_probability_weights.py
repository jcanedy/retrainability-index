import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

_CAT_COLS    = ["sex", "race", "funding_stream", "state", "program_year",
                "highest_educational_level", "employment_status"]
_NUM_COLS    = ["age"]
_BOOL_COLS   = ["low_income_status", "received_training"]
_TARGET_COLS = ["workforce_board_code"]


def compute_ipw(
    df: pl.DataFrame,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    bool_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    trim_quantile: float = 0.99,
) -> pl.DataFrame:
    """
    Estimate inverse probability weights for the calculable subsample.

    Fits a logistic regression to predict is_calculable = (index IS NOT NULL)
    from participant covariates, then returns a narrow DataFrame with two
    weight columns keyed by unique_id + program_year:

      ipw_simple     = 1 / P(calculable | X)
      ipw_stabilized = P(calculable) / P(calculable | X)

    Both are trimmed at trim_quantile of the calculable-row weights and set
    to NULL for non-calculable rows.
    """
    cat_cols    = cat_cols    or _CAT_COLS
    num_cols    = num_cols    or _NUM_COLS
    bool_cols   = bool_cols   or _BOOL_COLS
    target_cols = target_cols or _TARGET_COLS

    feature_cols = cat_cols + num_cols + bool_cols + target_cols

    # is_calculable is always defined as "index is not null" — never imputed
    is_calculable = df["index"].is_not_null().cast(pl.Int8).to_numpy()
    calculable_mask = is_calculable == 1

    # Narrow conversion: only covariate columns needed for the model
    pdf = df.select(feature_cols).to_pandas()

    # program_year: treat as a discrete category, not a linear trend
    pdf["program_year"] = pdf["program_year"].astype(str)

    # Polars nulls become Python None in object (string) columns.
    # Replace with np.nan so SimpleImputer(missing_values=np.nan) recognises them as missing.
    for col in cat_cols:
        pdf[col] = pdf[col].fillna(np.nan)

    # Boolean columns: convert to float (0.0 / 1.0 / nan) for consistent missing value handling.
    for col in bool_cols:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    # workforce_board_code stays numeric; coerce any non-numeric stragglers to nan.
    for col in target_cols:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("bool", SimpleImputer(strategy="most_frequent"), bool_cols),
        ("target", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # smooth="auto" shrinks rare boards toward the global mean — avoids perfect separation
            ("te", TargetEncoder(smooth="auto")),
        ]), target_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])
    model.fit(pdf, is_calculable)

    p_hat = model.predict_proba(pdf)[:, 1]
    p_marginal = float(is_calculable.mean())

    def _trim_and_null(weights: np.ndarray, name: str) -> pl.Series:
        threshold = np.quantile(weights[calculable_mask], trim_quantile)
        trimmed = np.clip(weights, None, threshold).astype(float)
        trimmed[~calculable_mask] = float("nan")
        return pl.Series(name, trimmed, dtype=pl.Float64).fill_nan(None)

    return df.select(["unique_id", "program_year"]).with_columns([
        _trim_and_null(1.0 / p_hat, "ipw_simple"),
        _trim_and_null(p_marginal / p_hat, "ipw_stabilized"),
        pl.Series("p_hat", p_hat, dtype=pl.Float64),
    ])


def _smd(
    x_calculable: np.ndarray,
    x_full: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Standardized mean difference between the calculable group and the full sample.

    Pooled SD is always unweighted so before/after values are on the same scale.
    weights, if provided, are applied only to the calculable group (IPW reweighting).
    """
    pooled_sd = np.sqrt((np.nanvar(x_calculable) + np.nanvar(x_full)) / 2)
    if pooled_sd < 1e-10:
        return 0.0
    mean_f = float(np.nanmean(x_full))
    if weights is not None:
        mean_t = float(np.average(x_calculable, weights=weights))
    else:
        mean_t = float(np.nanmean(x_calculable))
    return (mean_t - mean_f) / pooled_sd


def diagnostics(
    df: pl.DataFrame,
    weights_df: pl.DataFrame,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    bool_cols: list[str] | None = None,
) -> dict:
    """
    Compute standard IPW quality metrics.

    Args:
        df:          Full DataFrame from wioa_retrainability_index (covariates + index column).
        weights_df:  Output of compute_ipw() (unique_id, program_year, ipw_simple,
                     ipw_stabilized, p_hat).

    Returns dict with three keys:

        "smd"     — pl.DataFrame(covariate, smd_before, smd_after)
                    One row per covariate level (binary indicators for categoricals).
                    |smd_after| < 0.1 is the usual adequacy threshold.

        "ess"     — pl.DataFrame with one row:
                    n_total, n_calculable, p_calculable,
                    ess_simple, ess_stabilized
                    ESS = (Σw)² / Σw² — how much effective sample remains after weighting.

        "overlap" — pl.DataFrame(p_hat, is_calculable)
                    Propensity score for every record; plot as overlapping histograms
                    to check that calculable and non-calculable records share support.
    """
    cat_cols  = cat_cols  or _CAT_COLS
    num_cols  = num_cols  or _NUM_COLS
    bool_cols = bool_cols or _BOOL_COLS

    # Join covariates + index with weights on unique_id + program_year.
    # Deduplicate in case program_year is also in cat_cols.
    covariate_cols = cat_cols + num_cols + bool_cols
    select_cols = list(dict.fromkeys(["unique_id", "program_year", "index"] + covariate_cols))
    joined = (
        df.select(select_cols)
        .join(weights_df, on=["unique_id", "program_year"], how="left")
        .to_pandas()
    )

    is_calculable_mask = joined["index"].notna()
    calculable = joined[is_calculable_mask].copy()

    # ipw_stabilized weights for the calculable rows (trim already applied)
    w_stabilized = calculable["ipw_stabilized"].to_numpy().astype(float)

    # --- SMD ---
    records = []

    for col in num_cols + bool_cols:
        x_calc = pd.to_numeric(calculable[col], errors="coerce").to_numpy()
        x_full = pd.to_numeric(joined[col], errors="coerce").to_numpy()
        notna_c = ~np.isnan(x_calc)
        notna_f = ~np.isnan(x_full)
        records.append({
            "covariate": col,
            "smd_before": _smd(x_calc[notna_c], x_full[notna_f]),
            "smd_after":  _smd(x_calc[notna_c], x_full[notna_f], weights=w_stabilized[notna_c]),
        })

    for col in cat_cols:
        notna_c = calculable[col].notna()
        notna_f = joined[col].notna()
        dummies_calc = pd.get_dummies(calculable.loc[notna_c, col], prefix=col, dtype=float)
        dummies_full = pd.get_dummies(joined.loc[notna_f, col], prefix=col, dtype=float)
        all_levels = dummies_full.columns.union(dummies_calc.columns)
        dummies_calc = dummies_calc.reindex(columns=all_levels, fill_value=0.0)
        dummies_full = dummies_full.reindex(columns=all_levels, fill_value=0.0)
        w_cat = w_stabilized[notna_c.to_numpy()]
        for level in all_levels:
            records.append({
                "covariate": level,
                "smd_before": _smd(dummies_calc[level].to_numpy(), dummies_full[level].to_numpy()),
                "smd_after":  _smd(dummies_calc[level].to_numpy(), dummies_full[level].to_numpy(), weights=w_cat),
            })

    smd_df = pl.DataFrame(records, schema={"covariate": pl.String, "smd_before": pl.Float64, "smd_after": pl.Float64})

    # --- ESS ---
    w_simple = calculable["ipw_simple"].dropna().to_numpy().astype(float)
    w_stab   = calculable["ipw_stabilized"].dropna().to_numpy().astype(float)
    n_total      = len(joined)
    n_calculable = int(is_calculable_mask.sum())

    ess_df = pl.DataFrame([{
        "n_total":        n_total,
        "n_calculable":   n_calculable,
        "p_calculable":   round(n_calculable / n_total, 4),
        "ess_simple":     round(float(w_simple.sum() ** 2 / (w_simple ** 2).sum()), 1),
        "ess_stabilized": round(float(w_stab.sum() ** 2 / (w_stab ** 2).sum()), 1),
    }])

    # --- Overlap ---
    overlap_df = weights_df.select("p_hat").with_columns(
        (df["index"].is_not_null().cast(pl.Int8)).alias("is_calculable")
    )

    return {"smd": smd_df, "ess": ess_df, "overlap": overlap_df}
