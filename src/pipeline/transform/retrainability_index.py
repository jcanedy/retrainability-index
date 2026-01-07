import polars as pl

def join_routine_task_intensity(
    df: pl.DataFrame | pl.LazyFrame,
    df_rti_occupation: pl.DataFrame | pl.LazyFrame,
    df_rti_industry: pl.DataFrame | pl.LazyFrame,
    df_rti_subsector: pl.DataFrame | pl.LazyFrame 
) -> pl.DataFrame | pl.LazyFrame:

    df = (
        df
        .join(
            df_rti_occupation,
            left_on="occupational_code_pre",
            right_on="occupation_code",
            how="left"
        )
        .rename({
            "r_cog": "r_cog_pre",
            "r_man": "r_man_pre",
            "offshor": "offshor_pre"
        })
        .join(
            df_rti_occupation,
            left_on="occupational_code_post",
            right_on="occupation_code",
            how="left"
        )
        .rename({
            "r_cog": "r_cog_post",
            "r_man": "r_man_post",
            "offshor": "offshor_post"
        })
        .join(
            df_rti_industry,
            left_on="industry_code_pre",
            right_on="industry_code",
            how="left"
        )
        .rename({
            "r_cog_industry": "r_cog_industry_pre",
            "r_man_industry": "r_man_industry_pre",
            "offshor_industry": "offshor_industry_pre"
        })
        .drop([
            'sector_code', 
            'subsector_code', 
            'industry_group_code', 
            'naics_industry_code', 
            'industry_title', 
            'sector_title', 
            'subsector_title',
        ])
        .join(
            df_rti_industry,
            left_on="industry_code_post",
            right_on="industry_code",
            how="left"
        )
        .rename({
            "r_cog_industry": "r_cog_industry_post",
            "r_man_industry": "r_man_industry_post",
            "offshor_industry": "offshor_industry_post"
        })
        .drop([
            'sector_code', 
            'subsector_code', 
            'industry_group_code', 
            'naics_industry_code', 
            'industry_title', 
            'sector_title', 
            'subsector_title',
        ])
        .join(
            df_rti_subsector,
            left_on="subsector_code_pre",
            right_on="subsector_code",
            how="left"
        )
        .rename({
            "r_cog_subsector": "r_cog_subsector_pre",
            "r_man_subsector": "r_man_subsector_pre",
            "offshor_subsector": "offshor_subsector_pre",
            "subsector_title": "subsector_title_pre",
            "subsector_top_occupation_titles": "subsector_top_occupation_titles_pre"
        })
        .drop([
            "sector_code",
            "sector_title"
        ])
        .join(
            df_rti_subsector,
            left_on="subsector_code_post",
            right_on="subsector_code",
            how="left"
        )
        .rename({
            "r_cog_subsector": "r_cog_subsector_post",
            "r_man_subsector": "r_man_subsector_post",
            "offshor_subsector": "offshor_subsector_post",
            "subsector_title": "subsector_title_post",
            "subsector_top_occupation_titles": "subsector_top_occupation_titles_post"
        })
        .drop([
            "sector_code",
            "sector_title"
        ])
    )

    return df

def join_workforce_boards(
    df: pl.LazyFrame | pl.DataFrame,
    df_workforce_boards: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame | pl.DataFrame:

    df = (
        df
        .join(
            df_workforce_boards,
            left_on=["workforce_board_code", "entry_year"],
            right_on=["workforce_board_code", "program_year"],
            how="left"
        )
        .drop(["jurisdiction_count"])
        .rename({
            "population_per_sqkm": "workforce_board_population_per_sqkm",
            "population": "workforce_board_population",
            "median_age": "workforce_board_median_age",
            "median_income": "workforce_board_median_income",
            "unemployment_rate": "workforce_board_unemployment_rate",
            "diversity_index": "workforce_board_diversity_index",
            "household_debt_to_income_low": "workforce_board_household_debt_to_income_low",
            "household_debt_to_income_high": "workforce_board_household_debt_to_income_high",
            "mean_commuting_time_min": "workforce_board_mean_commuting_time_min",
            "rucc": "workforce_board_rucc",
            "jurisdictions": "workforce_board_jurisdictions"
        })
    )

    return df


def compute_routine_task_intensity_diff(
    df: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame | pl.DataFrame:

    # Calculate pre- and post-program difference in rti
    # A positive difference in pre- and post-program correspond to 
    # an occupation change that is has more routine cognitive/manual task 
    # or has more offshorable
    df = df.with_columns([
        (pl.col("r_cog_post") - pl.col("r_cog_pre")).alias("diff_r_cog"),
        (pl.col("r_man_post") - pl.col("r_man_pre")).alias("diff_r_man"),
        (pl.col("offshor_post") - pl.col("offshor_pre")).alias("diff_offshor"),
        (pl.col("r_cog_industry_post") - pl.col("r_cog_industry_pre")).alias("diff_r_cog_industry"),
        (pl.col("r_man_industry_post") - pl.col("r_man_industry_pre")).alias("diff_r_man_industry"),
        (pl.col("offshor_industry_post") - pl.col("offshor_industry_pre")).alias("diff_offshor_industry"),
        (pl.col("r_cog_subsector_post") - pl.col("r_cog_subsector_pre")).alias("diff_r_cog_subsector"),
        (pl.col("r_man_subsector_post") - pl.col("r_man_subsector_pre")).alias("diff_r_man_subsector"),
        (pl.col("offshor_subsector_post") - pl.col("offshor_subsector_pre")).alias("diff_offshor_subsector"),
    ])

    return df

def compute_index(
    df: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame | pl.DataFrame:

    cols = ["wages_mean_diff", "diff_r_cog_subsector", "diff_r_man_subsector"]

    # Create a variable that holds the winsorized column expression
    winsorized_cols = [pl.col(c).clip(
        lower_bound=pl.col(c).quantile(0.01),
        upper_bound=pl.col(c).quantile(0.99),
    ) for c in cols]

    df = (
        df
        .with_columns([
            # Apply Winsorization and Anchored Scaling
            pl.when(winsorized_col > 0)
                # Positive values (X > 0): Scale from 0.5 to 1.0
                .then(0.5 + 0.5 * (winsorized_col / winsorized_col.max()))

            .when(winsorized_col < 0)
                # Negative values (X < 0): Scale from 0.0 to 0.5
                .then(0.5 - 0.5 * (winsorized_col / winsorized_col.min()))

            # Zero values (X == 0): Anchor them exactly at 0.5
            .otherwise(0.5)
            .alias(f"{c}_normalized")
            for c, winsorized_col in zip(cols, winsorized_cols)
        ])
    )

    df = df.with_columns(
        (
        0.5 * pl.col("wages_mean_diff_normalized")
        + 0.25 * (1 - pl.col("diff_r_cog_subsector_normalized"))
        + 0.25 * (1 - pl.col("diff_r_man_subsector_normalized"))
        ).alias("index")
    )
    
    return df