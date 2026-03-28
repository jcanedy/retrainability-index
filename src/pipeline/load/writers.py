import polars as pl
from pathlib import Path
import tempfile
from google.cloud import bigquery
from google.cloud import bigquery

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


def write_bigquery(
    df: pl.DataFrame | pl.LazyFrame,
    project: str,
    dataset: str,
    table: str,
    if_exists: str = "fail",
    sink: bool = False,
    **kwargs
) -> None:
    """
    Write a Polars DataFrame or LazyFrame to BigQuery.

    Args:
        df: Polars DataFrame or LazyFrame to write
        project: GCP project ID (e.g., "my-project-id")
        dataset: BigQuery dataset name
        table: BigQuery table name
        if_exists: How to behave if the table already exists. Options:
            - "fail": Raise an error if the table exists
            - "replace": Drop the table before inserting new values
            - "append": Insert new values to the existing table
        sink: If True, streams data through a temporary parquet file to avoid
            materializing the full dataset in memory
        **kwargs: Additional arguments passed to BigQuery LoadJobConfig
    """
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"

    # Convert if_exists to BigQuery write disposition
    if if_exists == "replace":
        write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    elif if_exists == "append":
        write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    elif if_exists == "fail":
        write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
    else:
        raise ValueError(f"Invalid value for if_exists: {if_exists}. Must be 'fail', 'replace', or 'append'")

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=write_disposition,
        **kwargs
    )

    if sink:
        # Stream through temporary parquet file to avoid materializing in memory
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True, mode='wb') as tmp:
            # Handle both LazyFrame and DataFrame
            if isinstance(df, pl.LazyFrame):
                df.sink_parquet(tmp.name)
            else:
                df.write_parquet(tmp.name)

            with open(tmp.name, "rb") as source_file:
                job = client.load_table_from_file(
                    source_file,
                    table_id,
                    job_config=job_config
                )
                job.result()
    else:
        # Standard approach: materialize and load
        if isinstance(df, pl.LazyFrame):
            pandas_df = df.collect().to_pandas()
        else:
            pandas_df = df.to_pandas()

        job = client.load_table_from_dataframe(
            pandas_df,
            table_id,
            job_config=job_config
        )
        job.result()