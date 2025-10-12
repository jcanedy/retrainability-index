import pytest
import polars as pl
from pipeline.extract.readers import read_csv, read_parquet, read_excel

def test_read_csv(tmp_path):
    # Create small sample data
    df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    path = tmp_path / "data.csv"
    df.write_csv(path)

    # Read using wrapper
    out = read_csv(path)

    assert isinstance(out, pl.DataFrame)
    assert out.shape == (2, 2)
    assert out.to_dict(as_series=False) == df.to_dict(as_series=False)

def test_read_parquet(tmp_path):
    # Create small sample data
    df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    path = tmp_path / "data.parquet"
    df.write_parquet(path)

    # Read using wrapper
    out = read_parquet(path)

    assert isinstance(out, pl.DataFrame)
    assert out.shape == (2, 2)
    assert set(out.columns) == set(df.columns)
    assert out.to_dict(as_series=False) == df.to_dict(as_series=False)

def test_read_excel(tmp_path):
    # Create small sample data
    df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    path = tmp_path / "data.xlsx"
    df.write_excel(path)

    # Read using wrapper
    out = read_excel(path)

    assert isinstance(out, pl.DataFrame)
    assert out.shape == (2, 2)
    assert set(out.columns) == set(df.columns)
    assert out.to_dict(as_series=False) == df.to_dict(as_series=False)