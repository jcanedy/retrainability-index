from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import retrainability_index
from pipeline.load import writers
from typing import Literal

DATA_PATH = "data/raw/"
DATA_OUTPUT_PATH = "data/processed/retrainability_index/"

@task
def task_retrainability_index_read(use_sample: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Read performance records data.
    
    Args:
        use_sample: If True, read sample data; if False, read full data
    """
    suffix = "_sample" if use_sample else ""
    df = readers.read_parquet(
        f"data/processed/performance_records/performance_records{suffix}.parquet", 
        lazy=True
    )

    return df

@task
def task_retrainability_index_read_rti_subsector() -> pl.DataFrame | pl.LazyFrame:
    df = readers.read_parquet(
        f"data/processed/routine_task_intensity/routine_task_intensity_subsector.parquet",
        lazy=True
    )

    return df

@task
def task_retrainability_index_read_rti_industry() -> pl.DataFrame | pl.LazyFrame:
    df = readers.read_parquet(
        f"data/processed/routine_task_intensity/routine_task_intensity_industry.parquet",
        lazy=True
    )

    return df

@task
def task_retrainability_index_read_rti_occupation() -> pl.DataFrame | pl.LazyFrame:
    df = readers.read_parquet(
        f"data/processed/routine_task_intensity/routine_task_intensity_occupation.parquet",
        lazy=True
    )

    return df

@task
def task_retrainability_index_read_workforce_boards() -> pl.DataFrame:
    df = readers.read_parquet(
        f"data/processed/workforce_boards/workforce_boards_grouped.parquet",
        lazy=True
    ).drop([
        "state"
    ]).with_columns(pl.col("workforce_board_code").cast(pl.String))

    return df


@task
def task_retrainability_index_read_occupations() -> pl.DataFrame:
    df = readers.read_parquet(
        f"data/processed/occupations/occupations.parquet",
        lazy=True
    )

    return df

@task
def task_retrainability_index_compute_rti_diff(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    return retrainability_index.compute_routine_task_intensity_diff(df)

@task
def task_retrainability_index_compute_index(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    return retrainability_index.compute_index(df)

@task
def task_retrainability_index_join_rti(
    df: pl.LazyFrame | pl.DataFrame,
    df_rti_occupation: pl.LazyFrame | pl.DataFrame,
    df_rti_industry: pl.LazyFrame | pl.DataFrame,
    df_rti_subsector: pl.LazyFrame | pl.DataFrame
):
    df = retrainability_index.join_routine_task_intensity(
        df,
        df_rti_occupation,
        df_rti_industry,
        df_rti_subsector
    )

    return df

@task
def task_retrainability_index_join_workforce_boards(
   df: pl.DataFrame | pl.LazyFrame,
   df_workforce_boards: pl.DataFrame | pl.LazyFrame 
):
    df = retrainability_index.join_workforce_boards(df, df_workforce_boards)
    return df

@task
def task_retrainability_index_collect(lf: pl.LazyFrame) -> pl.DataFrame:

    df = lf.collect()

    return df

@task
def task_retrainability_index_write(
    df: pl.LazyFrame | pl.DataFrame,
    use_sample: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """Write retrainability index data.
    
    Args:
        df: DataFrame to write
        use_sample: If True, save with '_sample' suffix; if False, save without suffix
    """

    if isinstance(df, pl.LazyFrame):
        df = task_retrainability_index_collect(df)

    suffix = "_sample" if use_sample else ""
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}retrainability_index{suffix}.parquet", compression="zstd")

    return df


@flow()
def retrainability_index_pipeline(use_sample: bool = False) -> None:
    """Run the retrainability index pipeline.
    
    Args:
        use_sample: If True, process sample data and save with '_sample' suffix;
                   if False, process full data and save without suffix
    """
    df = task_retrainability_index_read(use_sample=use_sample)
    df_occupations = task_retrainability_index_read_occupations()
    df_workforce_boards = task_retrainability_index_read_workforce_boards()
    df_rti_subsector = task_retrainability_index_read_rti_subsector()
    df_rti_industry = task_retrainability_index_read_rti_industry()
    df_rti_occupation = task_retrainability_index_read_rti_occupation()

    df = task_retrainability_index_join_rti(
        df,
        df_rti_occupation,
        df_rti_industry,
        df_rti_subsector
    )

    df = task_retrainability_index_join_workforce_boards(df, df_workforce_boards)
    df = task_retrainability_index_compute_rti_diff(df)
    df = task_retrainability_index_compute_index(df)
    df = task_retrainability_index_write(df, use_sample=use_sample)

    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run retrainability index pipeline')
    parser.add_argument(
        '--use-sample', 
        action='store_true',
        help='Use sample data (with _sample suffix) instead of full data'
    )
    args = parser.parse_args()
    
    retrainability_index_pipeline(use_sample=args.use_sample)