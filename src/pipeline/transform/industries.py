import polars as pl


def _expand_naics_code_range(code):
    """
    Expands a hyphenated BLS-style NAICS code range (e.g., '1131-2' or '31-330')
    into a list of 6-digit NAICS codes, using right-padding with zeros.
    """
    try:
  
      if "-" not in code:
          return [code]

      # Strip any trailing zero(s)
      code = code.rstrip("0")

      # Split the code into start and end parts
      code_start, code_end = code.split("-")

      shared_prefix_len = len(code_start) - len(code_end)
      prefix = code_start[:shared_prefix_len]

      # Get the numeric range
      start_suffix = int(code_start[shared_prefix_len:])
      end_suffix = int(code_end)

      full_codes = []
      for suffix in range(start_suffix, end_suffix + 1):
          full_code = prefix + str(suffix)
          padded_code = full_code.ljust(6, '0')
          full_codes.append(padded_code)

      return full_codes

    except Exception as e:
        return f"Error parsing code range: {e}"

def normalize(df: pl.DataFrame) -> pl.DataFrame:

    df_normalized = df.rename(
        {
            "Occupation title": "occupation_title",
            "Occupation code": "occupation_code",
            "Industry code": "industry_code",
            "Industry title": "industry_title",
            "Occupation type": "occupation_type",
            "Industry type": "industry_type",
            "2023 Employment": "2023_employment",
            "2033 Employment": "2033_employment",
            "2023 Percent of Industry": "2023_percent_of_industry",
            "2033 Percent of Industry": "2033_percent_of_industry"
        }
    ).with_columns(
        pl.col("occupation_code").str.replace("-", "")
    )

    # BLS encodes a small subset of industry codes as ranges, 
    # which are disambiguated by replicating the row
    # and using the individual industry code.

    df_normalized = (
        df_normalized.with_columns(
            pl.col("industry_code")
            .map_elements(_expand_naics_code_range, return_dtype=pl.List(pl.Utf8))
        )
        .explode("industry_code")
    )

    # Add sector, subsector, industry group, and NAICS Industry codes
    # data-source: https://www.census.gov/programs-surveys/economic-census/year/2017/economic-census-2017/guidance/understanding-naics.html#par_textimage_0
    df_normalized = df_normalized.with_columns(
        (pl.col("industry_code").str.slice(0, 2) + "0000").alias("sector_code"),
        (pl.col("industry_code").str.slice(0, 3) + "000").alias("subsector_code"),
        (pl.col("industry_code").str.slice(0, 4) + "00").alias("industry_group_code"),
        (pl.col("industry_code").str.slice(0, 5) + "0").alias("naics_industry_code"),
    ).select(
        pl.col("occupation_type"),
        pl.col("industry_type"),
        pl.col("industry_code"),
        pl.col("sector_code"),
        pl.col("subsector_code"),
        pl.col("industry_group_code"),
        pl.col("naics_industry_code"),
        pl.col("industry_title"),
        pl.col("occupation_code"),
        pl.col("2023_employment"),
        pl.col("2033_employment"),
        pl.col("2023_percent_of_industry"),
        pl.col("2033_percent_of_industry")
    )

    return df_normalized


def filter(df: pl.DataFrame) -> pl.DataFrame:

    df_filtered = (
        df
        # Filter to subsector codes (e.g., ends with 000)
        .filter(
            pl.col("occupation_type").eq(pl.lit("Line item")) &
            pl.col("industry_type").eq(pl.lit("Line item"))
        )
    )

    return df_filtered

def filter_to_sector(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to sector codes (i.e., ends with 0000)."""

    df_filtered = (
        df
        # Filter to subsector codes (e.g., ends with 000)
        .filter(
            pl.col("industry_code").str.ends_with(pl.lit("0000")) &
            ~pl.col("industry_code").str.ends_with(pl.lit("00000"))
        )
        .unique(subset=["industry_title", "industry_code"])
        .rename({
            "industry_title": "sector_title",
        })
        .select(
            pl.col("sector_code"),
            pl.col("sector_title")
        )
        .unique(subset="sector_code")
    )

    return df_filtered

def filter_to_subsector(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to subsector codes (i.e., ends with 000)."""

    df_filtered = (
        df
        # Filter to subsector codes (e.g., ends with 000)
        .filter(
            pl.col("industry_code").str.ends_with(pl.lit("000")) &
            ~pl.col("industry_code").str.ends_with(pl.lit("0000"))
        )
        .unique(subset=["industry_title", "industry_code"])
        .rename({
            "industry_title": "subsector_title",
        })
        .select(
            pl.col("subsector_code"),
            pl.col("subsector_title")
        )
        .unique(subset="subsector_code")
    )

    return df_filtered

def join_sector(df: pl.DataFrame, df_sector: pl.DataFrame) -> pl.DataFrame:

    df = (
        df.join(
            df_sector,
            on="sector_code",
            how="left"
        )
    )

    return df

def join_subsector(df: pl.DataFrame, df_subsector: pl.DataFrame) -> pl.DataFrame:

    df = (
        df.join(
            df_subsector,
            on="subsector_code",
            how="left"
        )
    )

    return df