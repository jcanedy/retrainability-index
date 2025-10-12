"""
Readers: responsible for loading raw data from various sources
into memory (DataFrame, list of dicts, etc.).
"""

from pathlib import Path
from typing import Any
import polars as pl
import pandas as pd

def read_csv(
    path: str | Path,
    lazy: bool = False,
    **kwargs
) -> pl.DataFrame | pl.LazyFrame:
    """Read a CSV file into a DataFrame."""

    if lazy:
        return pl.scan_csv(path, **kwargs)

    return pl.read_csv(path, **kwargs)

def read_parquet(
    path: str | Path, 
    **kwargs: Any
) -> pl.DataFrame:
    """Read a Parquet dataset."""
    return pl.read_parquet(path, **kwargs)

def read_excel(
    path: str | Path,
    **kwargs: Any,
) -> pl.DataFrame:
    """Read an Excel file into a Polars DataFrame."""
    return pl.read_excel(path, **kwargs)

def read_stata(
    path: str | Path,
    **kwargs: Any
) -> pl.DataFrame:
    """Read a Stata file into a Polars DataFrame."""
    df = pd.read_stata(path, **kwargs)
    return pl.from_pandas(df)