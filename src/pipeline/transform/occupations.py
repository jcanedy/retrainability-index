import polars as pl

def normalize(df: pl.DataFrame) -> pl.DataFrame:

    df_normalized = (
        df
        .rename({
            "Major Group": "major_group",
            "Minor Group": "minor_group",
            "Broad Group": "broad_group",
            "Detailed Occupation": "detailed_occupation",
            "": "occupation_title"
        })
        .with_columns(
            pl.col("detailed_occupation").str.replace("-", ""),
            pl.col("major_group").str.replace("-", ""),
            pl.col("minor_group").str.replace("-", ""),
            pl.col("broad_group").str.replace("-", ""),
        )
    )
    
    return df_normalized

def melt_occupation_levels(df: pl.DataFrame) -> pl.DataFrame:
    """Combines occupation levels (major_group, minor_group, etc.)
    into a column with the level and a column with the occupation_code
    and occupation_code_prefix.
    """
    
    df = (
        df.melt(
            id_vars="occupation_title",
            value_vars=[
                "major_group",
                "minor_group",
                "broad_group",
                "detailed_occupation"
            ],
            variable_name="occupation_level",
            value_name="occupation_code"
        )
        .drop_nulls(subset=["occupation_code"])
        .with_columns(
            pl.col("occupation_code").str.slice(0, 2).alias("occupation_code_prefix")
        )
    )

    return df
