from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import inverse_probability_weights
from pipeline.load import writers

PROJECT = "retraining-index"
DATASET = "staging"


@task
def task_ipw_read() -> pl.DataFrame:
    return readers.read_bigquery(PROJECT, DATASET, table="wioa_retrainability_index")


@task
def task_ipw_compute(df: pl.DataFrame) -> pl.DataFrame:
    return inverse_probability_weights.compute_ipw(df)


@task
def task_ipw_write(df: pl.DataFrame) -> None:
    writers.write_bigquery(df, PROJECT, DATASET, "wioa_ipw_weights", if_exists="replace", sink=True)


@task
def task_ipw_diagnostics(index_df: pl.DataFrame, weights_df: pl.DataFrame) -> dict:
    return inverse_probability_weights.diagnostics(index_df, weights_df)


@task
def task_ipw_diagnostics_write(diag: dict) -> None:
    writers.write_bigquery(diag["smd"], PROJECT, DATASET, "wioa_ipw_diagnostics_smd", if_exists="replace", sink=True)
    writers.write_bigquery(diag["ess"], PROJECT, DATASET, "wioa_ipw_diagnostics_ess", if_exists="replace", sink=True)


@flow()
def ipw_weights_pipeline() -> None:
    index_df   = task_ipw_read()
    weights_df = task_ipw_compute(index_df)
    task_ipw_write(weights_df)
    diag = task_ipw_diagnostics(index_df, weights_df)
    task_ipw_diagnostics_write(diag)


if __name__ == "__main__":
    ipw_weights_pipeline()
