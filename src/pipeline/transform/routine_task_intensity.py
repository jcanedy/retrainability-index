import polars as pl
from pipeline.extract import readers

def _read_industries_parquet() -> pl.DataFrame:
    df = readers.read_parquet("data/processed/industries/industries.parquet")
    return df


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

    df_industries = _read_industries_parquet()
    
    columns = [
        "occupation_code", "industry_code", "sector_code", "subsector_code", "industry_group_code", "naics_industry_code",
        "industry_title", "sector_title", "subsector_title",
        "2023_percent_of_industry", "2033_percent_of_industry",
        "r_cog", "r_man", "offshor"
    ]

    df = df.join(
        df_industries,
        on="occupation_code",
        how="left"
    ).select(columns)

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
            pl.col("offshor_industry").sum()
        )
    )

    # For industry codes that are repeated, take the mean of the routine task intensity values
    df = df.group_by(columns).agg(
        pl.col("r_cog_industry").mean(),
        pl.col("r_man_industry").mean(),
        pl.col("offshor_industry").mean()
    )

    return df

def compute_sector(df: pl.DataFrame) -> pl.DataFrame:
    """Computes the sector-level routine task intensity 
    by taking the mean of the industry-level routine task intensity."""

    columns = [
        "sector_code",
        "sector_title",
    ]

    # Computes the sector-level routine task intensity by taking the mean
    # of the industry-level routine task intensity. In the future, it may be more
    # appropriate to weight the industry-level routine task intensity by the
    # percentage of jobs the sector it accounts for.
    df = (
        df
        .group_by(columns)
        .agg(
            pl.col("r_cog_industry").mean().alias("r_cog_sector"),
            pl.col("r_man_industry").mean().alias("r_man_sector"),
            pl.col("offshor_industry").mean().alias("offshor_sector")
        )
    )

    return df


def compute_subsector(df: pl.DataFrame) -> pl.DataFrame:
    """Computes the subsector-level routine task intensity 
    by taking the mean of the industry-level routine task intensity."""

    columns = [
        "subsector_code", "sector_code",
        "subsector_title", "sector_title",
    ]

    # Computes the subsector-level routine task intensity by taking the mean
    # of the industry-level routine task intensity. In the future, it may be more
    # appropriate to weight the industry-level routine task intensity by the
    # percentage of jobs the subsector it accounts for.
    df = (
        df
        .group_by(columns)
        .agg(
            pl.col("r_cog_industry").mean().alias("r_cog_subsector"),
            pl.col("r_man_industry").mean().alias("r_man_subsector"),
            pl.col("offshor_industry").mean().alias("offshor_subsector")
        )
    )

    return df