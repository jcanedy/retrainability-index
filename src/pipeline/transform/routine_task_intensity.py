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
    )

    return df_normalized


def join_industries(df: pl.DataFrame) -> pl.DataFrame:

    df_industries = _read_industries_parquet()

    df = df.join(
        df_industries,
        on="occupation_code",
        how="left"
    )

    return df

def compute_industry(df: pl.DataFrame) -> pl.DataFrame:

    # Normalize the percent of occupation by industry code
    # So that within an industry, all remaining occupations sum to 1
    df = df.with_columns(
        (pl.col("2023_percent_of_industry") / 
        pl.col("2023_percent_of_industry").sum().over("industry_code"))
        .alias("2023_percent_of_industry_norm")
    )

    # Calculate the industry level routine task intensity measures as a weighted sum
    df = (
        df
        .with_columns(
            (pl.col("2023_percent_of_industry_norm") * pl.col("r_cog")).alias("r_cog_industry"),
            (pl.col("2023_percent_of_industry_norm") * pl.col("r_man")).alias("r_man_industry"),
            (pl.col("2023_percent_of_industry_norm") * pl.col("offshor")).alias("offshor_industry")
        )
        .group_by(["industry_code", "industry_title"])
        .agg(
            pl.col("r_cog_industry").sum(),
            pl.col("r_man_industry").sum(),
            pl.col("offshor_industry").sum()
        )
    )

    # For industry codes that are repeated, take the mean of the routine task intensity values
    df = df.group_by(["industry_code"]).agg(
        pl.col("r_cog_industry").mean(),
        pl.col("r_man_industry").mean(),
        pl.col("offshor_industry").mean()
    )

    return df