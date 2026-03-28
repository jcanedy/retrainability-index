import polars as pl
from prefect import get_run_logger

from pipeline.extract import readers

def _read_industries_parquet() -> pl.DataFrame:
    return readers.read_parquet("gs://retrainability-index/processed/industries/industries.parquet")

def _read_occupations_parquet() -> pl.DataFrame:
    return readers.read_parquet("gs://retrainability-index/processed/occupations/occupations.parquet")

def normalize(df: pl.DataFrame) -> pl.DataFrame:

    df_normalized = (
        df
        .rename({
            "onetsoccode": "occupation_code"
        })
        .with_columns(
            # Cast to an Int64 to remove the decimal, then convert to a String.
            pl.col("occupation_code").cast(pl.Int64).cast(pl.String)
        )
        .select([
            "occupation_code",
            "r_cog",
            "r_man",
            "offshor"
        ])
        .unique(subset=["occupation_code"])
    )

    return df_normalized


def join_industries(df: pl.DataFrame) -> pl.DataFrame:
    logger = get_run_logger()

    df_industries = _read_industries_parquet()
    df_occupations = _read_occupations_parquet()
    
    columns = [
        "occupation_code", "occupation_title", "industry_code", "sector_code", "subsector_code", "industry_group_code", "naics_industry_code",
        "industry_title", "sector_title", "subsector_title",
        "2023_employment", "2033_employment",
        "2023_percent_of_industry", "2033_percent_of_industry",
        "r_cog", "r_man", "offshor"
    ]

    logger.info(f'Number of occupation codes (before join): {df["occupation_code"].n_unique()}')

    df = (
        df_industries.join(
            df,
            on="occupation_code",
            how="inner"
        )
        .join(
            df_occupations,
            on="occupation_code",
            how="inner"
        )
        .select(columns)
    )
    logger.info(f'Number of occupation codes (after join): {df["occupation_code"].n_unique()}')

    return df

def compute_industry(df: pl.DataFrame) -> pl.DataFrame:

    # Normalize the percent of occupation by industry code
    # So that within an industry, all remaining occupations sum to 1
    df = df.with_columns(
        (pl.col("2023_percent_of_industry") / 
        pl.col("2023_percent_of_industry").sum().over("industry_code"))
        .alias("2023_percent_of_industry_norm")
    )

    columns = [
        "industry_code", "sector_code", "subsector_code", "industry_group_code", "naics_industry_code",
        "industry_title", "sector_title", "subsector_title",
    ]

    # Calculate the industry_code level routine task intensity measures as a weighted sum
    df = (
        df
        .with_columns(
            (pl.col("2023_percent_of_industry_norm") * pl.col("r_cog")).alias("r_cog_industry"),
            (pl.col("2023_percent_of_industry_norm") * pl.col("r_man")).alias("r_man_industry"),
            (pl.col("2023_percent_of_industry_norm") * pl.col("offshor")).alias("offshor_industry")
        )
        .group_by(columns)
        .agg(
            pl.col("r_cog_industry").sum(),
            pl.col("r_man_industry").sum(),
            pl.col("offshor_industry").sum(),
        )
    )

    # For industry codes that are repeated, take the mean of the routine task intensity values
    df = df.group_by(columns).agg(
        pl.col("r_cog_industry").mean(),
        pl.col("r_man_industry").mean(),
        pl.col("offshor_industry").mean()
    )

    return df

def compute_subsector(df: pl.DataFrame, top_k: int = 10) -> pl.DataFrame:

    # Normalize the percent of occupation by subsector
    # So that within an subsector, all remaining occupations sum to 1
    df = df.with_columns(
        (pl.col("2023_employment") / 
        pl.col("2023_employment").sum().over("subsector_code"))
        .alias("2023_percent_of_subsector")
    )

    columns = [
        "sector_code", "subsector_code",
        "sector_title", "subsector_title",
    ]

    # Calculate the subsctor-level routine task intensity measures as a weighted sum
    df = (
        df
        .with_columns(
            (pl.col("2023_percent_of_subsector") * pl.col("r_cog")).alias("r_cog_subsector"),
            (pl.col("2023_percent_of_subsector") * pl.col("r_man")).alias("r_man_subsector"),
            (pl.col("2023_percent_of_subsector") * pl.col("offshor")).alias("offshor_subsector")
        )
        .group_by(columns)
        .agg(
            pl.col("r_cog_subsector").sum(),
            pl.col("r_man_subsector").sum(),
            pl.col("offshor_subsector").sum(),
              # Top-K occupations + their normalized share (weight)
              pl.struct([
                  pl.col("occupation_title"),
                  pl.col("2023_percent_of_subsector")
              ])
              .sort_by(pl.col("2023_percent_of_subsector"), descending=True)
              .head(top_k)
              .alias("subsector_top_occupation_titles")
        )
    )

    # For subsector codes that are repeated, take the mean of the routine task intensity values
    df = df.group_by(columns).agg(
        pl.col("r_cog_subsector").mean(),
        pl.col("r_man_subsector").mean(),
        pl.col("offshor_subsector").mean(),
        pl.col("subsector_top_occupation_titles").first()  # keep the list
    ).drop_nulls(columns)

    return df

def compute_sector(df: pl.DataFrame, top_k: int = 10) -> pl.DataFrame:

    # Normalize the percent of occupation by sector
    # So that within an sector, all remaining occupations sum to 1
    df = df.with_columns(
        (pl.col("2023_employment") / 
        pl.col("2023_employment").sum().over("sector_code"))
        .alias("2023_percent_of_sector")
    )

    columns = [
        "sector_code",
        "sector_title",
    ]

    # Calculate the sector-level routine task intensity measures as a weighted sum
    df = (
        df
        .with_columns(
            (pl.col("2023_percent_of_sector") * pl.col("r_cog")).alias("r_cog_sector"),
            (pl.col("2023_percent_of_sector") * pl.col("r_man")).alias("r_man_sector"),
            (pl.col("2023_percent_of_sector") * pl.col("offshor")).alias("offshor_sector")
        )
        .group_by(columns)
        .agg(
            pl.col("r_cog_sector").sum(),
            pl.col("r_man_sector").sum(),
            pl.col("offshor_sector").sum(),
              # Top-K occupations + their normalized share (weight)
              pl.struct([
                  pl.col("occupation_title"),
                  pl.col("2023_percent_of_sector")
              ])
              .sort_by(pl.col("2023_percent_of_sector"), descending=True)
              .head(top_k)
              .alias("top_occupation_titles")
        )
    )

    # For sector codes that are repeated, take the mean of the routine task intensity values
    df = df.group_by(columns).agg(
        pl.col("r_cog_sector").mean(),
        pl.col("r_man_sector").mean(),
        pl.col("offshor_sector").mean(),
        pl.col("top_occupation_titles").first()  # keep the list
    ).drop_nulls(columns)

    return df