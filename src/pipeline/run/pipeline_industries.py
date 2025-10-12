from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import industries
from pipeline.load import writers

DATA_PATH = "data/raw/industries/"
DATA_OUTPUT_PATH = "data/processed/industries/"

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
def task_industries_write_parquet(df: pl.DataFrame) -> None:
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}industries.parquet", compression="zstd")

@flow
def occupations_pipeline() -> None:
    df = task_industries_excel_read()
    df = task_industries_normalize(df)
    df = task_industries_filter(df)
    df = task_industries_group_by_subsector(df)
    print(df.head(10))
    task_industries_write_parquet(df)

    return

if __name__ == "__main__":
    occupations_pipeline.fn()