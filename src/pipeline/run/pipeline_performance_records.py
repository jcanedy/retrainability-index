from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import performance_records
from pipeline.load import writers

DATA_PATH = "gs://retrainability-index/raw/performance_records/"
DATA_OUTPUT_PATH = "temp/processed/performance_records/"

@task
def task_performance_records_csv_read(lazy=True) -> dict[str, pl.LazyFrame | pl.DataFrame]:
    dict_performance_records = {}

    # WIOA Individual Performance Records (Public Use Data)
    dict_performance_records["2024"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2024Q3_PUBLIC.csv", lazy=lazy)
    dict_performance_records["2023"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2023Q4_PUBLIC.csv", lazy=lazy)
    dict_performance_records["2022"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2022Q4_Public.csv", lazy=lazy)
    dict_performance_records["2021"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2021Q4_PUBLIC.csv", lazy=lazy)
    dict_performance_records["2020"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2020Q4_Public.csv", lazy=lazy)
    dict_performance_records["2019"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2019Q4_Public.csv", lazy=lazy)
    dict_performance_records["2018"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2018Q4_Public.csv", lazy=lazy)
    dict_performance_records["2017"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2017Q4_Public.csv", lazy=lazy)

    # WIASRD (Public Use Data)
    # dict_performance_records["2015"] = readers.read_csv(f"{DATA_PATH}PublicWIASRD2015Q4.csv", lazy=lazy)
    # dict_performance_records["2014"] = readers.read_csv(f"{DATA_PATH}PublicWIASRD2014q4.csv", lazy=lazy)
    # dict_performance_records["2013"] = readers.read_csv(f"{DATA_PATH}PublicWIASRD2013q4.csv", lazy=lazy)
    
    return dict_performance_records


@task
def task_performance_records_normalize(dict_lf: dict[str, pl.LazyFrame | pl.DataFrame]) -> pl.LazyFrame | pl.DataFrame:
    dict_lf_normalized = {}

    for year, lf in dict_lf.items():
        try:
            dict_lf_normalized[year] = performance_records.normalize(lf, year)
        except ValueError as e:
            print(f"ValueError: {e}")

    lf = pl.concat(dict_lf_normalized.values())

    # Ensure there is only one observation per program year with the same participant id
    # This will keep the entry in the most recent Performance Records file.
    # According to WIOA documenation, there should only be 1 unique_id in each program year. 

    lf = lf.unique(
        subset=["unique_id", "program_year"],
        keep="first"
    )
    
    return lf

@task
def task_compute_inflation_adjusted_wages(lf: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:

    lf = performance_records.compute_inflation_adjusted_wages(lf)

    return lf

@task
def task_performance_records_compute_additional_columns(lf: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:

    lf = performance_records.compute_industry_code(lf)
    lf = performance_records.compute_subsector_code(lf)
    lf = performance_records.compute_workforce_board_code(lf)
    lf = performance_records.compute_funding_stream(lf)
    lf = performance_records.compute_mean_wages(lf)
    lf = performance_records.compute_program_duration(lf)

    return lf

@task
def task_performance_records_filter(lf: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:

    lf = performance_records.filter(lf)

    return lf

@task
def task_performance_records_write(lf: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    
    @task
    def task_performance_records_collect(lf: pl.LazyFrame) -> pl.DataFrame:
        return lf.collect(engine="streaming")

    if (isinstance(lf, pl.LazyFrame)):
        df = task_performance_records_collect(lf)
    else:
        df = lf
    
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}performance_records.parquet", compression="zstd")

    return df

@task
def task_performance_records_write_bigquery(df: pl.DataFrame) -> None:

    writers.write_bigquery(
        df,
        "retraining-index",
        "staging",
        "wioa_performance_records",
        if_exists="replace",
        sink=True
    )

    return


@task
def task_performance_records_write_sample(df: pl.DataFrame) -> pl.DataFrame:
    
    df_sample = performance_records.sample(df)

    writers.write_parquet(df_sample, f"{DATA_OUTPUT_PATH}performance_records_sample.parquet", compression="zstd")

    return df_sample

@task
def task_performance_records_compute_count_by_state(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    
    df = (
        df
        .group_by(
            pl.col("state"),
            pl.col("program_year")
        )
        .agg(
            pl.len().alias("count_total")
        )
    )

    return df

@task
def task_performance_records_write_count_by_state(lf: pl.LazyFrame | pl.DataFrame) -> None:

    writers.write_parquet(lf, f"{DATA_OUTPUT_PATH}count_by_state.parquet", sink=True, compression="zstd")

    return


@flow
def performance_records_pipeline() -> None:
    dict_performance_records = task_performance_records_csv_read(lazy=True)
    lf_performance_records = task_performance_records_normalize(dict_performance_records)

    # lf_count_by_state = task_performance_records_compute_count_by_state(lf_performance_records)
    # df_count_by_state = task_performance_records_write_count_by_state(lf_count_by_state)

    lf_performance_records = task_compute_inflation_adjusted_wages(lf_performance_records)
    lf_performance_records = task_performance_records_compute_additional_columns(lf_performance_records)
    
    lf_performance_records = task_performance_records_filter(lf_performance_records)

    task_performance_records_write_bigquery(lf_performance_records)
    
    return 

if __name__ == "__main__":
    performance_records_pipeline.fn()