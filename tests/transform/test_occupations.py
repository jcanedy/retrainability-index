import pytest
import polars as pl
from pipeline.transform.occupations import (
    normalize, 
    melt_occupation_levels
)