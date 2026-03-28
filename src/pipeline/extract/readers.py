"""
Readers: responsible for loading raw data from various sources
into memory (DataFrame, list of dicts, etc.).
"""

from pathlib import Path
from typing import Any
import hashlib
import os
import polars as pl

import pandas as pd
import tempfile
from google.cloud import storage
from google.cloud import bigquery



def _download_from_gcs(gcs_path: str) -> str:
    """
    Download a file from GCS to a temporary file and return the local path.

    Args:
        gcs_path: GCS path in format gs://bucket-name/path/to/file

    Returns:
        Path to the temporary file
    """
    if not gcs_path.startswith("gs://"):
        return gcs_path

    # Parse GCS path
    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""

    # Build a deterministic temp path so re-runs skip the download
    file_extension = Path(blob_name).suffix
    path_hash = hashlib.md5(gcs_path.encode()).hexdigest()[:8]
    temp_path = os.path.join(tempfile.gettempdir(), f"gcs_{path_hash}{file_extension}")

    if os.path.exists(temp_path):
        return temp_path

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(temp_path)

    return temp_path

def read_csv(
    path: str | Path,
    lazy: bool = False,
    **kwargs
) -> pl.DataFrame | pl.LazyFrame:
    """
    Read a CSV file into a DataFrame.

    Supports local paths and GCS paths (gs://bucket-name/path/to/file.csv).
    """
    local_path = _download_from_gcs(str(path))

    if lazy:
        return pl.scan_csv(local_path, **kwargs)

    return pl.read_csv(local_path, **kwargs)

def read_parquet(
    path: str | Path,
    lazy: bool = False,
    **kwargs: Any
) -> pl.DataFrame:
    """
    Read a Parquet dataset.

    Supports local paths and GCS paths (gs://bucket-name/path/to/file.parquet).
    """
    local_path = _download_from_gcs(str(path))

    if lazy:
        return pl.scan_parquet(local_path, **kwargs)

    return pl.read_parquet(local_path, **kwargs)

def read_excel(
    path: str | Path,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Read an Excel file into a Polars DataFrame.

    Supports local paths and GCS paths (gs://bucket-name/path/to/file.xlsx).
    """
    local_path = _download_from_gcs(str(path))
    return pl.read_excel(local_path, **kwargs)

def read_stata(
    path: str | Path,
    **kwargs: Any
) -> pl.DataFrame:
    """
    Read a Stata file into a Polars DataFrame.

    Supports local paths and GCS paths (gs://bucket-name/path/to/file.dta).
    """
    local_path = _download_from_gcs(str(path))
    return pl.from_pandas(pd.read_stata(local_path, **kwargs))


def read_bigquery(
    project: str,
    dataset: str,
    table: str,
    query: str | None = None,
) -> pl.DataFrame:
    """
    Read a BigQuery table into a Polars DataFrame.

    Args:
        project: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name (ignored if query is provided)
        query: Optional SQL query to run instead of SELECT *
    """
    client = bigquery.Client(project=project)
    if query is None:
        query = f"SELECT * FROM `{project}.{dataset}.{table}`"
    return pl.from_arrow(client.query(query).to_arrow(create_bqstorage_client=True))