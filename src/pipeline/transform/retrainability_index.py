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

def _winsorize_and_normalize_pair(
    df: pl.DataFrame,
    pre_col: str,
    post_col: str,
    out_pre: str,
    out_post: str,
) -> pl.DataFrame:
    """
    Winsorize and min-max normalize a pre/post column pair on a shared scale.
    Quantile bounds and min/max are computed across both columns combined.
    """
    combined = pl.concat([df[pre_col], df[post_col]])
    lower = combined.quantile(0.01)
    upper = combined.quantile(0.99)

    df = df.with_columns(
        pl.col(pre_col).clip(lower, upper).alias(f"{pre_col}_winsorized"),
        pl.col(post_col).clip(lower, upper).alias(f"{post_col}_winsorized"),
    )

    combined_min = df.select(
        pl.concat_list([f"{pre_col}_winsorized", f"{post_col}_winsorized"]).explode()
    ).min().item()
    combined_max = df.select(
        pl.concat_list([f"{pre_col}_winsorized", f"{post_col}_winsorized"]).explode()
    ).max().item()

    df = df.with_columns(
        ((pl.col(f"{pre_col}_winsorized") - combined_min) / (combined_max - combined_min)).alias(out_pre),
        ((pl.col(f"{post_col}_winsorized") - combined_min) / (combined_max - combined_min)).alias(out_post),
    )

    return df


def compute_index(
    df: pl.DataFrame
) -> pl.DataFrame:

    pre_cols  = ["wages_mean_pre_ihs", "r_cog_subsector_pre", "r_man_subsector_pre", "r_cog_pre", "r_man_pre"]
    post_cols = ["wages_mean_post_ihs", "r_cog_subsector_post", "r_man_subsector_post", "r_cog_post", "r_man_post"]
    norm_pre  = ["wages_mean_pre_ihs_normalized", "r_cog_subsector_pre_normalized", "r_man_subsector_pre_normalized", "r_cog_pre_normalized", "r_man_pre_normalized"]
    norm_post = ["wages_mean_post_ihs_normalized", "r_cog_subsector_post_normalized", "r_man_subsector_post_normalized", "r_cog_post_normalized", "r_man_post_normalized"]
    diff_cols = ["wages_mean_ihs_normalized_diff", "r_cog_subsector_normalized_diff", "r_man_subsector_normalized_diff", "r_cog_normalized_diff", "r_man_normalized_diff"]

    for pre_c, post_c, out_pre, out_post in zip(pre_cols, post_cols, norm_pre, norm_post):
        df = _winsorize_and_normalize_pair(df, pre_c, post_c, out_pre, out_post)

    df = df.with_columns([
        (pl.col(post_c) - pl.col(pre_c)).alias(diff_c)
        for pre_c, post_c, diff_c in zip(norm_pre, norm_post, diff_cols)
    ])

    df = df.with_columns(
        (
            0.5  * pl.col("wages_mean_ihs_normalized_diff")
            - 0.25 * pl.col("r_cog_normalized_diff")
            - 0.25 * pl.col("r_man_normalized_diff")
        ).alias("index"),
        (
            0.5  * pl.col("wages_mean_ihs_normalized_diff")
            - 0.25 * pl.col("r_cog_subsector_normalized_diff")
            - 0.25 * pl.col("r_man_subsector_normalized_diff")
        ).alias("index_subsector"),
    )

    return df