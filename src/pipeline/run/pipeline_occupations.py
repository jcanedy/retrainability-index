from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import occupations
from pipeline.load import writers

DATA_PATH = "data/raw/occupations/"
DATA_OUTPUT_PATH = "data/processed/occupations/"

@task
def task_occupations_csv_read() -> pl.DataFrame:
    df = readers.read_excel(f"{DATA_PATH}soc_codes_2018.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 7 })
    return df

@task
def task_occupations_normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = occupations.normalize(df)
    return df

@task
def task_occupations_melt_occupation_levels(df: pl.DataFrame) -> pl.DataFrame:
    df = occupations.melt_occupation_levels(df)
    return df

@task
def task_occupation_write(df: pl.DataFrame):
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}occupations.parquet", compression="zstd")
    return 


@flow()
def occupations_pipeline() -> None:
    df = task_occupations_csv_read()
    df = task_occupations_normalize(df)
    df = task_occupations_melt_occupation_levels(df)
    task_occupation_write(df)

    return 

if __name__ == "__main__":
    occupations_pipeline()