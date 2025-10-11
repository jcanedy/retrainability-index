"""
Readers: responsible for loading raw data from various sources
into memory (DataFrame, list of dicts, etc.).
"""

from pathlib import Path
from typing import Any
import polars as pl

def read_csv(
    path: str | Path, 
    **kwargs
) -> pl.DataFrame:
    """Read a CSV file into a DataFrame."""
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