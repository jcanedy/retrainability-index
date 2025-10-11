from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import performance_records
from pipeline.load import writers

DATA_PATH = "data/raw/performance_records/"
DATA_OUTPUT_PATH = "data/processed/performance_records/"

@task
def task_performance_records_csv_read() -> dict[str, pl.LazyFrame]:
    dict_performance_records = {}

    # WIOA Individual Performance Records (Public Use Data)
    dict_performance_records["2024"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2024Q3_PUBLIC.csv", lazy=True)
    dict_performance_records["2023"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2023Q4_PUBLIC.csv", lazy=True)
    dict_performance_records["2022"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2022Q4_Public.csv", lazy=True)
    dict_performance_records["2021"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2021Q4_PUBLIC.csv", lazy=True)
    dict_performance_records["2020"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2020Q4_Public.csv", lazy=True)
    dict_performance_records["2019"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2019Q4_Public.csv", lazy=True)
    dict_performance_records["2018"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2018Q4_Public.csv", lazy=True)
    dict_performance_records["2017"] = readers.read_csv(f"{DATA_PATH}WIOAPerformanceRecords_PY2017Q4_Public.csv", lazy=True)

    # WIASRD (Public Use Data)
    dict_performance_records["2015"] = readers.read_csv(f"{DATA_PATH}PublicWIASRD2015Q4.csv", lazy=True)
    dict_performance_records["2014"] = readers.read_csv(f"{DATA_PATH}PublicWIASRD2014q4.csv", lazy=True)
    dict_performance_records["2013"] = readers.read_csv(f"{DATA_PATH}PublicWIASRD2013q4.csv", lazy=True)
    
    return dict_performance_records


@task
def task_performance_records_normalize(dict_lf: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    dict_lf_normalized = {}

    for year, lf in dict_lf.items():
        try:
            dict_lf_normalized[year] = performance_records.normalize(lf, year)
        except ValueError as e:
            print(f"ValueError: {e}")

    lf = pl.concat(dict_lf_normalized.values())
    
    return lf

@task
def task_performance_records_compute_additional_columns(lf: pl.LazyFrame) -> pl.LazyFrame:

    lf = performance_records.compute_industry_code(lf)
    lf = performance_records.compute_subsector_code(lf)
    lf = performance_records.compute_workforce_board_code(lf)
    lf = performance_records.compute_funding_stream(lf)

    return lf

@task
def task_performance_records_filter(lf: pl.LazyFrame) -> pl.LazyFrame:

    lf = performance_records.filter(lf)

    return lf

@task
def task_performance_records_write(lf: pl.LazyFrame) -> pl.DataFrame:

    df = lf.collect(engine="streaming")
    
    writers.write_parquet(df, f"{DATA_OUTPUT_PATH}performance_records.parquet", compression="zstd")

    return df

@task
def task_performance_records_write_sample(df: pl.DataFrame) -> pl.DataFrame:
    
    df_sample = performance_records.sample(df)

    writers.write_parquet(df_sample, f"{DATA_OUTPUT_PATH}performance_records_sample.parquet", compression="zstd")

    return df_sample


@flow
def main() -> None:
    dict_performance_records = task_performance_records_csv_read()
    lf_performance_records = task_performance_records_normalize(dict_performance_records)
    lf_performance_records = task_performance_records_compute_additional_columns(lf_performance_records)
    lf_performance_records = task_performance_records_filter(lf_performance_records)
    df = task_performance_records_write(lf_performance_records)
    df_sample = task_performance_records_write_sample(df)
    
    print("df: ", df.select(pl.len()))
    print("df_sample: ", df_sample.select(pl.len()))

    return 

if __name__ == "__main__":
    main.fn()