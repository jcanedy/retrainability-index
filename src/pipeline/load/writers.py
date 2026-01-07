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
    sink: bool=False,
    **kwargs
) -> None:
    if sink:
        df.sink_parquet(path, **kwargs)
        return
        
    df.write_parquet(path, **kwargs)