import polars as pl
from pathlib import Path

def write_csv(
    df: pl.DataFrame, 
    path: str | Path,
    **kwargs
) -> None: 
    df.write_csv(path, **kwargs)


def write_parquet(
    df: pl.DataFrame, 
    path: str | Path,
    **kwargs
) -> None:
     df.write_parquet(path, **kwargs)