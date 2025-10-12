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
def task_routine_task_intensity_group_by_subsector(df: pl.DataFrame) -> pl.DataFrame:
    df = routine_task_intensity.group_by_subsector(df)
    return df


@flow()
def routine_task_intensity_pipeline() -> None:
    df = task_routine_task_intensity_stata_read()
    df = task_routine_task_intensity_normalize(df)
    df = task_routine_task_intensity_join_industries(df)
    df = task_routine_task_intensity_compute_industry(df)
    df = task_routine_task_intensity_group_by_subsector(df)
    print(df.head(10))
    return 

if __name__ == "__main__":
    routine_task_intensity_pipeline.fn()