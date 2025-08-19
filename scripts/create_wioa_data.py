import polars as pl

FILE_PATH = "data/raw/Individual Records/WIOAPerformanceRecords_PY2024Q2_PUBLIC.csv"

data = pl.scan_csv(FILE_PATH)

result = (
    data.filter(

    # Do not consider rows which are served by the by a youth funding stream.
    (pl.col("CALC4003") != 1) &

    # Do not consider rows which are reportable individuals (as it seems
    # that detailed data is not collected on them).
    # Specifically, reportable individuals are anyone who has interacted with WIOA,
    # but has not necessarily participated in a funding stream.
    (pl.col("CALC4006") != 1))
    .collect(engine="streaming")
)


result.write_parquet("data/processed/wioa_data.parquet")