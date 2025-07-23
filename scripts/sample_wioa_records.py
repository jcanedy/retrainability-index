import random
import polars as pl

FILE_PATH = "data/raw/Individual Records/WIOAPerformanceRecords_PY2024Q2_PUBLIC.csv"

with open(FILE_PATH) as f:
    lines = sum(1 for _ in f)

print(f"The file contains {lines} lines.")

SAMPLE_SIZE = int(lines * 0.1)

# Choose SAMPLE_SIZE random lines to keep (excluding header)
sample_indices = sorted(random.sample(range(1, lines), SAMPLE_SIZE))

# Skip all rows not in sample_indices
sample_data = (
    pl.scan_csv(FILE_PATH)  # Lazy reading
    .with_row_index("row_num")  # Add row numbers
    .filter(pl.col("row_num").is_in(sample_indices))
    .collect()  # Execute
    .to_pandas()  # Convert back if needed
)

#TODO(jcanedy@): Remove the `row_number` column as it is not needed.

sample_data.to_csv('processed/wioa_data_10_percent.csv', index=False)