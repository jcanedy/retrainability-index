from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import industries
from pipeline.load import writers

DATA_PATH = "gs://retrainability-index/raw/industries/"

@task
def task_industries_excel_read() -> pl.DataFrame:
    # data-source: https://www.bls.gov/emp/data/occupational-data.htm
    df = readers.read_excel(f"{DATA_PATH}national_employment_matrix.xlsx")
    return df

@task
def task_industries_normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = industries.normalize(df)
    return df

@task
def task_industries_filter(df: pl.DataFrame) -> pl.DataFrame:
    df = industries.filter(df)
    return df

@task
def task_industries_filter_to_sector(df: pl.DataFrame) -> pl.DataFrame:
    df = industries.filter_to_sector(df)
    return df

@task
def task_industries_filter_to_subsector(df: pl.DataFrame) -> pl.DataFrame:
    df = industries.filter_to_subsector(df)
    return df

@task
def task_industries_join(
    df: pl.DataFrame, 
    df_sector: pl.DataFrame,
    df_subsector: pl.DataFrame,
):
    df = industries.join_sector(df, df_sector)
    df = industries.join_subsector(df, df_subsector)

    return df

@task
def task_industries_write_bigquery(df: pl.DataFrame, table: str) -> None:
    writers.write_bigquery(df, "retraining-index", "staging", table, if_exists="replace")

@flow
def industries_pipeline() -> None:
    df = task_industries_excel_read()
    df = task_industries_normalize(df)
    df_sector = task_industries_filter_to_sector(df)
    df_subsector = task_industries_filter_to_subsector(df)
    df_industries = task_industries_filter(df)
    df_industries = task_industries_join(
        df_industries, 
        df_sector, 
        df_subsector
    )
    task_industries_write_bigquery(df_sector, "sectors")
    task_industries_write_bigquery(df_subsector, "subsectors")
    task_industries_write_bigquery(df_industries, "industries")

    return

if __name__ == "__main__":
    industries_pipeline.fn()