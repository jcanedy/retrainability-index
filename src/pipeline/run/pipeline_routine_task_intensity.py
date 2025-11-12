from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import routine_task_intensity
from pipeline.load import writers

DATA_PATH = "data/raw/routine_task_intensity/"
DATA_OUTPUT_PATH = "data/processed/routine_task_intensity/"

@task
def task_routine_task_intensity_stata_read() -> pl.DataFrame:
    df = readers.read_stata(f"{DATA_PATH}rti_by_occupation_code.dta")
    return df

@task
def task_routine_task_intensity_normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = routine_task_intensity.normalize(df)
    return df

@task
def task_routine_task_intensity_join_industries(df: pl.DataFrame) -> pl.DataFrame:
    df = routine_task_intensity.join_industries(df)
    return df

@task
def task_routine_task_intensity_compute_industry(df: pl.DataFrame) -> pl.DataFrame:
    df = routine_task_intensity.compute_industry(df)
    return df

@task
def task_routine_task_intensity_compute_sector(df: pl.DataFrame) -> pl.DataFrame:
    df = routine_task_intensity.compute_sector(df)
    return df

@task
def task_routine_task_intensity_compute_subsector(df: pl.DataFrame) -> pl.DataFrame:
    df = routine_task_intensity.compute_subsector(df)
    return df

@task
def task_routine_task_intensity_write_parquet(df: pl.DataFrame, filename: str) -> None:
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}{filename}", use_pyarrow=True, compression="zstd")


@flow()
def routine_task_intensity_pipeline() -> None:
    df = task_routine_task_intensity_stata_read()
    df = task_routine_task_intensity_normalize(df)
    task_routine_task_intensity_write_parquet(df, "routine_task_intensity_occupation.parquet")
    
    df = task_routine_task_intensity_join_industries(df)
    df_industry = task_routine_task_intensity_compute_industry(df)
    df_subsector = task_routine_task_intensity_compute_subsector(df)
    df_sector = task_routine_task_intensity_compute_sector(df)
    
    task_routine_task_intensity_write_parquet(df_industry, "routine_task_intensity_industry.parquet")
    task_routine_task_intensity_write_parquet(df_sector, "routine_task_intensity_sector.parquet")
    task_routine_task_intensity_write_parquet(df_subsector, "routine_task_intensity_subsector.parquet")
    return 

if __name__ == "__main__":
    routine_task_intensity_pipeline.fn()