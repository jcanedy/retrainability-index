import pytest
import polars as pl
from typing import Any
from pipeline.extract.fetchers import (
    fetch_datacommons_ids, 
    fetch_datacommons_land_areas, 
    fetch_datacommons_stats
)

def test_fetch_datacommons_ids():
    names = ["Medina, Ohio", "Cook County, Illinois", "Franklin County, Ohio"]

    result = fetch_datacommons_ids(names)

    assert len(result) == 3

def test_fetch_datacommons_land_areas():
    dcids = ["geoId/3948790", "geoId/17031", "geoId/39049"]

    result = fetch_datacommons_land_areas(dcids)

    assert len(result) == 3

def test_fetch_datacommons_stats():
    dcids = ["geoId/3948790", "geoId/17031", "geoId/39049"]

    result = fetch_datacommons_stats(dcids)

    assert len(result) == 3