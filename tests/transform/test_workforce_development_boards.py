import pytest
import polars as pl
from pipeline.transform.workforce_development_boards import (
    normalize, 
    filter, 
    join_with_datacommons_variables
)


def test_normalize():
    df = pl.DataFrame({
        "Program Year": [2024],
        "Region": ["East"],
        "State/Territory": ["NY"],
        "ETA Code": ["123"],
        "Local Board Name": ["Board A"],
        "Jurisdiction Name": ["City X"],
        "Created Timestamp": ["2024-01-01"],
        "Modified Timestamp": ["2024-02-01"],
        "Status": ["Approved"]
    })

    result = normalize(df)

    expected_cols = [
        "program_year", "region", "state", "workforce_board_code",
        "local_board", "jurisdiction", "created_timestamp",
        "modified_timestamp", "status"
    ]

    assert result.columns == expected_cols


def test_filter():
    df = pl.DataFrame({
        "program_year": [2024, 2023, 2024],
        "region": ["East", "East", "West"],
        "state": ["NY", "NY", "CA"],
        "created_timestamp": ["2024-01-01", "2023-01-01", "2024-01-02"],
        "modified_timestamp": ["2024-02-01", "2023-01-01", "2024-02-02"],
        "status": ["Approved", "Approved", "Rejected"],
        "jurisdiction": ["North County", "North County", "South County"],
        "workforce_board_code": ["123456", "123456", "78910"]
    })

    result = filter(df)

    # should only keep "Approved" status
    assert len(result) == 1
    assert result["state"][0] == "NY"
    assert result["jurisdiction_state"][0] == "North County, NY"

    # should drop the following columns
    assert "status" not in result.columns
    assert "created_timestamp" not in result.columns
    assert "modified_timestamp" not in result.columns


def test_join_with_datacommons_variables():
    df = pl.DataFrame({
        "jurisdiction_state": ["Medina, Ohio", "Cook County, Illinois", "Franklin County, Ohio"],
    })

    result = join_with_datacommons_variables(df)

    assert len(result) == len(df)
