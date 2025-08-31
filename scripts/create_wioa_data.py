import polars as pl

FILE_PATH = "data/raw/Individual Records/WIOAPerformanceRecords_PY2024Q2_PUBLIC.csv"

data = pl.scan_csv(FILE_PATH)

print("Length before filtering: ", data.select(pl.col("PIRL100").len()).collect())

result = (
    data.filter(

    # Consider rows which are served by 
    # adult funding
    ((pl.col("CALC4001") == 1) |
    # dislocated worker funding
    (pl.col("CALC4002") == 1) |
    # dislocated worker grant
    (pl.col("CALC4004") == 1) |
    # the Wagner-Peyser Act
    (pl.col("CALC4005") == 1) |
    # Veterans' Programs
    (pl.col("PIRL914").is_in([1, 2]))) 

    # Do not consider rows which are served by the by a youth funding stream.
    & (pl.col("CALC4003") != 1)

    # Do not consider rows which are reportable individuals (as it seems
    # that detailed data is not collected on them).
    # Specifically, reportable individuals are anyone who has interacted with WIOA,
    # but has not necessarily participated in a funding stream.
    &(pl.col("CALC4006") != 1))
    
    .collect(engine="streaming")
)

print("Length after filtering: ", result.select(pl.col("PIRL100").len()))

result.write_parquet("data/processed/wioa_data.parquet")