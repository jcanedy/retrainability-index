from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import consumer_price_index
from pipeline.load import writers

DATA_PATH = "data/raw/consumer_price_index/"
DATA_OUTPUT_PATH = "data/processed/consumer_price_index/"

@task
def task_consumer_price_index_excel_read() -> pl.DataFrame:
    # data-source: https://www.bls.gov/emp/data/occupational-data.htm
    df = readers.read_excel(f"{DATA_PATH}consumer_price_index.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 11 })
    return df

@task
def task_consumer_price_index_normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = consumer_price_index.normalize(df)
    return df

@task
def task_consumer_price_index_write(df: pl.DataFrame, filename: str) -> None:
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}{filename}")

@flow
def consumer_price_index_pipeline() -> None:
    df = task_consumer_price_index_excel_read()
    df = task_consumer_price_index_normalize(df)
    task_consumer_price_index_write(df, "consumer_price_index.parquet")

    return

if __name__ == "__main__":
    consumer_price_index_pipeline.fn()