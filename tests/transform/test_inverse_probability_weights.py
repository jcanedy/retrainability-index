import numpy as np
import polars as pl
import pytest
from pipeline.transform.inverse_probability_weights import compute_ipw, diagnostics


def _make_df(n_calculable: int = 60, n_non_calculable: int = 40, seed: int = 42) -> pl.DataFrame:
    """Minimal DataFrame with all required covariate columns and a partial index column."""
    rng = np.random.default_rng(seed)
    n = n_calculable + n_non_calculable

    index_vals = [float(rng.uniform(-0.5, 0.5)) for _ in range(n_calculable)] + [None] * n_non_calculable

    return pl.DataFrame({
        "unique_id": [str(i) for i in range(n)],
        "program_year": rng.choice([2019, 2020, 2021, 2022, 2023], size=n).tolist(),
        "index": index_vals,
        "sex": rng.choice(["Male", "Female", None], size=n).tolist(),
        "race": rng.choice(["White", "Black", "Hispanic", "Asian", None], size=n).tolist(),
        "funding_stream": rng.choice(["Adult", "Dislocated Worker", "Youth"], size=n).tolist(),
        "state": rng.choice(["CA", "TX", "NY", "FL"], size=n).tolist(),
        "highest_educational_level": rng.choice(["HS", "Some College", "Bachelor", None], size=n).tolist(),
        "employment_status": rng.choice(["Employed", "Unemployed", None], size=n).tolist(),
        "age": [float(v) if rng.random() > 0.1 else None for v in rng.integers(18, 65, size=n)],
        "low_income_status": rng.choice([True, False, None], size=n).tolist(),
        "received_training": rng.choice([True, False], size=n).tolist(),
        "workforce_board_code": rng.choice(list(range(1, 30)) + [None], size=n).tolist(),
    })


def test_output_columns():
    df = _make_df()
    result = compute_ipw(df)
    assert result.columns == ["unique_id", "program_year", "ipw_simple", "ipw_stabilized", "p_hat"]


def test_ipw_null_for_non_calculable():
    df = _make_df(n_calculable=60, n_non_calculable=40)
    result = compute_ipw(df)
    non_calculable_mask = df["index"].is_null()
    assert result.filter(non_calculable_mask)["ipw_simple"].is_null().all()
    assert result.filter(non_calculable_mask)["ipw_stabilized"].is_null().all()


def test_ipw_positive_for_calculable():
    df = _make_df()
    result = compute_ipw(df)
    calculable = result.filter(pl.col("ipw_simple").is_not_null())
    assert (calculable["ipw_simple"] > 0).all()
    assert (calculable["ipw_stabilized"] > 0).all()


def test_ipw_simple_vs_stabilized():
    """Stabilized weights should be proportionally smaller than simple weights by P(calculable)."""
    df = _make_df(n_calculable=60, n_non_calculable=40)
    result = compute_ipw(df)
    calculable = result.filter(pl.col("ipw_simple").is_not_null())
    p_marginal = 60 / 100
    # stabilized = simple * p_marginal (before trimming, approximately)
    ratio = calculable["ipw_stabilized"] / calculable["ipw_simple"]
    assert (ratio <= 1.0 + 1e-6).all(), "Stabilized weights should never exceed simple weights"
    assert ratio.mean() == pytest.approx(p_marginal, abs=0.15)


def test_ipw_trimming():
    """No weight should exceed the pre-trim 95th percentile when trim_quantile=0.95."""
    df = _make_df(n_calculable=80, n_non_calculable=20, seed=7)
    result_trimmed = compute_ipw(df, trim_quantile=0.95)
    result_untrimmed = compute_ipw(df, trim_quantile=1.0)

    for col in ("ipw_simple", "ipw_stabilized"):
        untrimmed_vals = result_untrimmed[col].drop_nulls().to_numpy()
        threshold = np.quantile(untrimmed_vals, 0.95)
        trimmed_vals = result_trimmed[col].drop_nulls().to_numpy()
        assert trimmed_vals.max() <= threshold + 1e-9, f"{col}: trimmed max exceeds 95th percentile threshold"


def test_ipw_missing_covariates():
    """Missing values in features should not raise errors (handled by SimpleImputer)."""
    df = _make_df()
    # Introduce additional nulls across multiple covariate columns
    df = df.with_columns([
        pl.when(pl.col("unique_id").cast(pl.Int64) % 5 == 0)
          .then(None)
          .otherwise(pl.col("sex"))
          .alias("sex"),
        pl.when(pl.col("unique_id").cast(pl.Int64) % 3 == 0)
          .then(None)
          .otherwise(pl.col("age"))
          .alias("age"),
        pl.when(pl.col("unique_id").cast(pl.Int64) % 7 == 0)
          .then(None)
          .otherwise(pl.col("workforce_board_code"))
          .alias("workforce_board_code"),
    ])
    result = compute_ipw(df)
    assert "ipw_simple" in result.columns
    assert "ipw_stabilized" in result.columns


# --- diagnostics() tests ---

def test_diagnostics_keys():
    df = _make_df()
    weights_df = compute_ipw(df)
    diag = diagnostics(df, weights_df)
    assert set(diag.keys()) == {"smd", "ess", "overlap"}


def test_diagnostics_smd_columns():
    df = _make_df()
    weights_df = compute_ipw(df)
    smd = diagnostics(df, weights_df)["smd"]
    assert smd.columns == ["covariate", "smd_before", "smd_after"]
    assert len(smd) > 0


def test_diagnostics_smd_after_closer_to_zero():
    """Weighted SMD should be smaller in absolute value than unweighted for most covariates."""
    df = _make_df(n_calculable=60, n_non_calculable=40, seed=0)
    weights_df = compute_ipw(df)
    smd = diagnostics(df, weights_df)["smd"].to_pandas()
    improved = (smd["smd_after"].abs() <= smd["smd_before"].abs() + 0.05).mean()
    assert improved >= 0.5, "IPW should reduce SMD for at least half of covariates"


def test_diagnostics_ess_columns():
    df = _make_df()
    weights_df = compute_ipw(df)
    ess = diagnostics(df, weights_df)["ess"]
    assert ess.columns == ["n_total", "n_calculable", "p_calculable", "ess_simple", "ess_stabilized"]
    assert ess["n_total"][0] == len(df)
    assert ess["n_calculable"][0] == df["index"].is_not_null().sum()


def test_diagnostics_ess_leq_n_calculable():
    """ESS should not exceed the number of calculable records."""
    df = _make_df()
    weights_df = compute_ipw(df)
    ess = diagnostics(df, weights_df)["ess"]
    n_calc = ess["n_calculable"][0]
    assert ess["ess_simple"][0] <= n_calc + 0.01
    assert ess["ess_stabilized"][0] <= n_calc + 0.01


def test_diagnostics_overlap_columns():
    df = _make_df()
    weights_df = compute_ipw(df)
    overlap = diagnostics(df, weights_df)["overlap"]
    assert overlap.columns == ["p_hat", "is_calculable"]
    assert len(overlap) == len(df)
    assert overlap["p_hat"].is_between(0, 1).all()
    assert set(overlap["is_calculable"].unique().to_list()).issubset({0, 1})
