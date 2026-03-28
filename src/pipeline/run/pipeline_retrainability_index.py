from prefect import flow, task
import polars as pl
from pipeline.extract import readers
from pipeline.transform import retrainability_index
from pipeline.load import writers

PROJECT = "retraining-index"
DATASET = "staging"

_JOIN_QUERY = f"""
SELECT
    pr.*,
    rti_occ_pre.r_cog        AS r_cog_pre,
    rti_occ_pre.r_man        AS r_man_pre,
    rti_occ_pre.offshor      AS offshor_pre,
    rti_occ_post.r_cog       AS r_cog_post,
    rti_occ_post.r_man       AS r_man_post,
    rti_occ_post.offshor     AS offshor_post,
    rti_ind_pre.r_cog_industry    AS r_cog_industry_pre,
    rti_ind_pre.r_man_industry    AS r_man_industry_pre,
    rti_ind_pre.offshor_industry  AS offshor_industry_pre,
    rti_ind_post.r_cog_industry   AS r_cog_industry_post,
    rti_ind_post.r_man_industry   AS r_man_industry_post,
    rti_ind_post.offshor_industry AS offshor_industry_post,
    rti_sub_pre.r_cog_subsector   AS r_cog_subsector_pre,
    rti_sub_pre.r_man_subsector   AS r_man_subsector_pre,
    rti_sub_pre.offshor_subsector AS offshor_subsector_pre,
    rti_sub_pre.subsector_title   AS subsector_title_pre,
    rti_sub_post.r_cog_subsector   AS r_cog_subsector_post,
    rti_sub_post.r_man_subsector   AS r_man_subsector_post,
    rti_sub_post.offshor_subsector AS offshor_subsector_post,
    rti_sub_post.subsector_title   AS subsector_title_post,
    occ_pre.occupation_title        AS occupation_title_pre,
    occ_post.occupation_title       AS occupation_title_post,
    wb.population_per_sqkm          AS workforce_board_population_per_sqkm,
    wb.population                   AS workforce_board_population,
    wb.median_age                   AS workforce_board_median_age,
    wb.median_income                AS workforce_board_median_income,
    wb.unemployment_rate            AS workforce_board_unemployment_rate,
    wb.diversity_index              AS workforce_board_diversity_index,
    wb.household_debt_to_income_low  AS workforce_board_household_debt_to_income_low,
    wb.household_debt_to_income_high AS workforce_board_household_debt_to_income_high,
    wb.mean_commuting_time_min      AS workforce_board_mean_commuting_time_min,
    wb.rucc                         AS workforce_board_rucc
FROM `{PROJECT}.{DATASET}.wioa_performance_records` pr
LEFT JOIN `{PROJECT}.{DATASET}.routine_task_intensity_occupation` rti_occ_pre
    ON pr.occupation_code_pre = rti_occ_pre.occupation_code
LEFT JOIN `{PROJECT}.{DATASET}.routine_task_intensity_occupation` rti_occ_post
    ON pr.occupation_code_post = rti_occ_post.occupation_code
LEFT JOIN `{PROJECT}.{DATASET}.routine_task_intensity_industry` rti_ind_pre
    ON pr.industry_code_pre = rti_ind_pre.industry_code
LEFT JOIN `{PROJECT}.{DATASET}.routine_task_intensity_industry` rti_ind_post
    ON pr.industry_code_post = rti_ind_post.industry_code
LEFT JOIN `{PROJECT}.{DATASET}.routine_task_intensity_subsector` rti_sub_pre
    ON pr.subsector_code_pre = rti_sub_pre.subsector_code
LEFT JOIN `{PROJECT}.{DATASET}.routine_task_intensity_subsector` rti_sub_post
    ON pr.subsector_code_post = rti_sub_post.subsector_code
LEFT JOIN `{PROJECT}.{DATASET}.occupations` occ_pre
    ON pr.occupation_code_pre = occ_pre.occupation_code
LEFT JOIN `{PROJECT}.{DATASET}.occupations` occ_post
    ON pr.occupation_code_post = occ_post.occupation_code
LEFT JOIN `{PROJECT}.{DATASET}.workforce_boards_grouped` wb
    ON CAST(pr.workforce_board_code AS STRING) = CAST(wb.workforce_board_code AS STRING)
    AND pr.entry_year = wb.program_year
"""


@task
def task_retrainability_index_read() -> pl.DataFrame:
    return readers.read_bigquery(PROJECT, DATASET, table="", query=_JOIN_QUERY)


@task
def task_retrainability_index_compute_index(df: pl.DataFrame) -> pl.DataFrame:
    return retrainability_index.compute_index(df)


@task
def task_retrainability_index_write(df: pl.DataFrame) -> None:
    writers.write_bigquery(df, PROJECT, DATASET, "wioa_retrainability_index", if_exists="replace", sink=True)


@flow()
def retrainability_index_pipeline() -> None:
    df = task_retrainability_index_read()
    df = task_retrainability_index_compute_index(df)
    task_retrainability_index_write(df)

    return

if __name__ == "__main__":
    retrainability_index_pipeline()
