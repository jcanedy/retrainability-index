import polars as pl
import numpy as np

def _highest_educational_level_map(value):
    match value:
        case v if v <= 12:
            return 0
        case v if (v >=13) & (v <= 15):
            return 4
        case 16:
            return 7
        case 17:
            return 8
        case 87:
            return 1
        case 88:
            return 2
        case 89:
            return 5
        case 90:
            return 5
        case 91:
            return 6
        case 0:
            return 0
        case _:
            return None

NORMALIZATIONS = {
    "2024": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("CALC4020").alias("race"),
            pl.col("CALC4039").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Program Employment
            pl.col("PIRL403").alias("occupational_code_pre"),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Program Employment
            pl.col("PIRL1610").alias("occupational_code_post"),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
        
            # Program Information
            pl.col("PIRL108A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("CALC4000").alias("state"),
            (pl.col("CALC4001") == 1).alias("is_adult"),
            ((pl.col("CALC4002") == 1) | (pl.col("CALC4004") == 1)).alias("is_dislocated_worker"),
            (pl.col("CALC4003") == 1).alias("is_youth"),
            (pl.col("CALC4005") == 1).alias("is_wagner_peyser"),
            ((pl.col("CALC4006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter"),
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),

    "2023": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("CALC4020").alias("race"),
            pl.col("CALC4039").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Program Employment
            pl.col("PIRL403").alias("occupational_code_pre"),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Program Employment
            pl.col("PIRL1610").alias("occupational_code_post"),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL108A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("CALC4000").alias("state"),
            (pl.col("CALC4001") == 1).alias("is_adult"),
            ((pl.col("CALC4002") == 1) | (pl.col("CALC4004") == 1)).alias("is_dislocated_worker"),
            (pl.col("CALC4003") == 1).alias("is_youth"),
            (pl.col("CALC4005") == 1).alias("is_wagner_peyser"),
            ((pl.col("CALC4006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2022": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("CALC4020").alias("race"),
            pl.col("CALC4039").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Program Employment
            pl.col("PIRL403").alias("occupational_code_pre"),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Program Employment
            pl.col("PIRL1610").alias("occupational_code_post"),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL108A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("CALC4000").alias("state"),
            (pl.col("CALC4001") == 1).alias("is_adult"),
            ((pl.col("CALC4002") == 1) | (pl.col("CALC4004") == 1)).alias("is_dislocated_worker"),
            (pl.col("CALC4003") == 1).alias("is_youth"),
            (pl.col("CALC4005") == 1).alias("is_wagner_peyser"),
            ((pl.col("CALC4006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2021": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("PIRL4020").alias("race"),
            pl.col("PIRL4039").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("PIRL403").alias("occupational_code_pre"),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Separation Employment
            pl.col("PIRL1610").alias("occupational_code_post"),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL108A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("PIRL4000").alias("state"),
            (pl.col("PIRL4001") == 1).alias("is_adult"),
            ((pl.col("PIRL4002") == 1) | (pl.col("PIRL4004") == 1)).alias("is_dislocated_worker"),
            (pl.col("PIRL4003") == 1).alias("is_youth"),
            (pl.col("PIRL4005") == 1).alias("is_wagner_peyser"),
            ((pl.col("PIRL4006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2020": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("PIRL3023").alias("race"),
            pl.col("PIRL3042").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("PIRL403").alias("occupational_code_pre").cast(pl.Int64),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Separation Employment
            pl.col("PIRL1610").alias("occupational_code_post").cast(pl.Int64),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL108-A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108-B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108-C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("PIRL3000").alias("state"),
            (pl.col("PIRL3001") == 1).alias("is_adult"),
            ((pl.col("PIRL3002") == 1) | (pl.col("PIRL3004") == 1)).alias("is_dislocated_worker"),
            (pl.col("PIRL3003") == 1).alias("is_youth"),
            (pl.col("PIRL3005") == 1).alias("is_wagner_peyser"),
            ((pl.col("PIRL3006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2019": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("PIRL 3023").alias("race"),
            pl.col("PIRL 3042").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("PIRL403").alias("occupational_code_pre").cast(pl.Int64),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Separation Employment
            pl.col("PIRL1610").alias("occupational_code_post").cast(pl.Int64),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL108-A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108-B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108-C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("PIRL 3000").alias("state"),
            (pl.col("PIRL 3001") == 1).alias("is_adult"),
            ((pl.col("PIRL 3002") == 1) | (pl.col("PIRL 3004") == 1)).alias("is_dislocated_worker"),
            (pl.col("PIRL 3003") == 1).alias("is_youth"),
            (pl.col("PIRL 3005") == 1).alias("is_wagner_peyser"),
            ((pl.col("PIRL 3006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2018": lambda lf: (
        lf.select(
            pl.col("PIRL100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL201").alias("sex"),
            pl.col("PIRL 3023").alias("race"),
            pl.col("PIRL 3042").alias("age"),
            pl.col("PIRL408").alias("highest_educational_level"),
            pl.col("PIRL802").alias("low_income_status"),
            pl.col("PIRL400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("PIRL403").alias("occupational_code_pre").cast(pl.Int64),
            pl.col("PIRL404").alias("industry_code_q1_pre"),
            pl.col("PIRL404").alias("industry_code_q2_pre"),
            pl.col("PIRL406").alias("industry_code_q3_pre"),
            pl.col("PIRL1700").alias("wages_3q_pre"),
            pl.col("PIRL1701").alias("wages_2q_pre"),
            pl.col("PIRL1702").alias("wages_1q_pre"),

            # Post-Separation Employment
            pl.col("PIRL1610").alias("occupational_code_post").cast(pl.Int64),
            pl.col("PIRL1614").alias("industry_code_q1_post"),
            pl.col("PIRL1615").alias("industry_code_q2_post"),
            pl.col("PIRL1616").alias("industry_code_q3_post"),
            pl.col("PIRL1617").alias("industry_code_q4_post"),
            pl.col("PIRL1703").alias("wages_1q_post"),
            pl.col("PIRL1704").alias("wages_2q_post"),
            pl.col("PIRL1705").alias("wages_3q_post"),
            pl.col("PIRL1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL108-A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL108-B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL108-C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("PIRL 3000").alias("state"),
            (pl.col("PIRL 3001") == 1).alias("is_adult"),
            ((pl.col("PIRL 3002") == 1) | (pl.col("PIRL 3004") == 1)).alias("is_dislocated_worker"),
            (pl.col("PIRL 3003") == 1).alias("is_youth"),
            (pl.col("PIRL 3005") == 1).alias("is_wagner_peyser"),
            ((pl.col("PIRL 3006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL1300") == 1).alias("received_training"),
            pl.col("PIRL900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2017": lambda lf: (
        lf.select(
            pl.col("PIRL 100").alias("unique_id"),

            # Demographics Information
            pl.col("PIRL 201").alias("sex"),
            pl.col("PIRL 3023").alias("race"),
            pl.col("PIRL 3042").alias("age"),
            pl.col("PIRL 408").alias("highest_educational_level"),
            pl.col("PIRL 802").alias("low_income_status"),
            pl.col("PIRL 400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("PIRL 403").alias("occupational_code_pre").cast(pl.Int64),
            pl.col("PIRL 404").alias("industry_code_q1_pre"),
            pl.col("PIRL 404").alias("industry_code_q2_pre"),
            pl.col("PIRL 406").alias("industry_code_q3_pre"),
            pl.col("PIRL 1700").alias("wages_3q_pre"),
            pl.col("PIRL 1701").alias("wages_2q_pre"),
            pl.col("PIRL 1702").alias("wages_1q_pre"),

            # Post-Separation Employment
            pl.col("PIRL 1610").alias("occupational_code_post").cast(pl.Int64),
            pl.col("PIRL 1614").alias("industry_code_q1_post"),
            pl.col("PIRL 1615").alias("industry_code_q2_post"),
            pl.col("PIRL 1616").alias("industry_code_q3_post"),
            pl.col("PIRL 1617").alias("industry_code_q4_post"),
            pl.col("PIRL 1703").alias("wages_1q_post"),
            pl.col("PIRL 1704").alias("wages_2q_post"),
            pl.col("PIRL 1705").alias("wages_3q_post"),
            pl.col("PIRL 1706").alias("wages_4q_post"),
            
            # Program Information
            pl.col("PIRL 108-A").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.col("PIRL 108-B").alias("workforce_board_code_2").cast(pl.Utf8),
            pl.col("PIRL 180-C").alias("workforce_board_code_3").cast(pl.Utf8),
            pl.col("PIRL 3000").alias("state"),
            (pl.col("PIRL 3001") == 1).alias("is_adult"),
            ((pl.col("PIRL 3002") == 1) | (pl.col("PIRL 3004") == 1)).alias("is_dislocated_worker"),
            (pl.col("PIRL 3003") == 1).alias("is_youth"),
            (pl.col("PIRL 3005") == 1).alias("is_wagner_peyser"),
            ((pl.col("PIRL 3006") == 1)).alias("is_reportable_individual"),
            (pl.col("PIRL 1300") == 1).alias("received_training"),
            pl.col("PIRL 900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("entry_date"),
            pl.col("PIRL 901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2015": lambda lf: (
        lf.select(
            pl.col("Item_100").alias("unique_id"),

            # Demographics Information
            pl.col("Item_201").alias("sex"),
            pl.col("Item_3006").alias("race"),
            pl.col("Item_3004").alias("age"),
            pl.col("Item_410").map_elements(_highest_educational_level_map, return_dtype=pl.Int64).alias("highest_educational_level"),
            pl.col("Item_702").alias("low_income_status").cast(pl.Int64, strict=False),
            pl.col("Item_400").alias("employment_status"),


            # Pre-Separation Employment
            pl.col("Item_402").alias("occupational_code_pre").cast(pl.Int64),
            pl.col("Item_403").alias("industry_code_q1_pre").cast(pl.Int64),
            pl.col("Item_404").alias("industry_code_q2_pre").cast(pl.Int64),
            pl.col("Item_405").alias("industry_code_q3_pre").cast(pl.Int64),
            pl.col("Item_1600").alias("wages_3q_pre").cast(pl.Int64),
            pl.col("Item_1601").alias("wages_2q_pre").cast(pl.Int64),
            pl.col("Item_1602").alias("wages_1q_pre").cast(pl.Int64),

            # Post-Separation Employment
            pl.col("Item_1502").alias("occupational_code_post").cast(pl.Int64),
            pl.col("Item_1514").alias("industry_code_q1_post").cast(pl.Int64),
            pl.col("Item_1516").alias("industry_code_q2_post").cast(pl.Int64),
            pl.col("Item_1517").alias("industry_code_q3_post").cast(pl.Int64),
            pl.col("Item_1518").alias("industry_code_q4_post").cast(pl.Int64),
            pl.col("Item_1603").alias("wages_1q_post").cast(pl.Int64),
            pl.col("Item_1604").alias("wages_2q_post").cast(pl.Int64),
            pl.col("Item_1605").alias("wages_3q_post").cast(pl.Int64),
            pl.col("Item_1606").alias("wages_4q_post").cast(pl.Int64),

            # Program Information
            pl.col("Item_105").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.lit(None).alias("workforce_board_code_2").cast(pl.Utf8), # Secondary workforce board code was not reported in WIA
            pl.lit(None).alias("workforce_board_code_3").cast(pl.Utf8), # Tertiary workforce board code was not reported in WIA
            pl.col("Item_3002").alias("state"),
            (pl.col("Item_3007") == 1).alias("is_adult"),
            ((pl.col("Item_3008") == 1) | (pl.col("Item_3009") == 1) | (pl.col("Item_3010") == 1)).alias("is_dislocated_worker"),
            ((pl.col("Item_3011") == 1) | (pl.col("Item_3012") == 1)).alias("is_youth"),
            (pl.col("Item_951") == 1).alias("is_wagner_peyser"),
            (pl.col("Item_3013") == 0).alias("is_reportable_individual"),
            (pl.col("Item_3014") == 1).alias("received_training"),
            pl.col("Item_900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%m/%d/%Y")
                .alias("entry_date"),
            pl.col("Item_901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%m/%d/%Y")
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2014": lambda lf: (
        lf.select(
            pl.col("Item_100").alias("unique_id"),
            
            # Demographics Information
            pl.col("Item_201").alias("sex"),
            pl.col("Item_3006").alias("race").cast(pl.Int64, strict=False),
            pl.col("Item_3004").alias("age"),
            pl.col("Item_410").map_elements(_highest_educational_level_map, return_dtype=pl.Int64).alias("highest_educational_level"),
            pl.col("Item_702").alias("low_income_status").cast(pl.Int64, strict=False),
            pl.col("Item_400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("Item_402").alias("occupational_code_pre").cast(pl.Int64, strict=False),
            pl.col("Item_403").alias("industry_code_q1_pre").cast(pl.Int64, strict=False),
            pl.col("Item_404").alias("industry_code_q2_pre").cast(pl.Int64, strict=False),
            pl.col("Item_405").alias("industry_code_q3_pre").cast(pl.Int64, strict=False),
            pl.col("Item_1600").alias("wages_3q_pre").cast(pl.Int64, strict=False),
            pl.col("Item_1601").alias("wages_2q_pre").cast(pl.Int64, strict=False),
            pl.col("Item_1602").alias("wages_1q_pre").cast(pl.Int64, strict=False),

            # Post-Separation Employment
            pl.col("Item_1502").alias("occupational_code_post").cast(pl.Int64, strict=False),
            pl.col("Item_1514").alias("industry_code_q1_post").cast(pl.Int64, strict=False),
            pl.col("Item_1516").alias("industry_code_q2_post").cast(pl.Int64, strict=False),
            pl.col("Item_1517").alias("industry_code_q3_post").cast(pl.Int64, strict=False),
            pl.col("Item_1518").alias("industry_code_q4_post").cast(pl.Int64, strict=False),
            pl.col("Item_1603").alias("wages_1q_post").cast(pl.Int64, strict=False),
            pl.col("Item_1604").alias("wages_2q_post").cast(pl.Int64, strict=False),
            pl.col("Item_1605").alias("wages_3q_post").cast(pl.Int64, strict=False),
            pl.col("Item_1606").alias("wages_4q_post").cast(pl.Int64, strict=False),

            # Program Information
            pl.col("Item_105").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.lit(None).alias("workforce_board_code_2").cast(pl.Utf8), # Secondary workforce board code was not reported in WIA
            pl.lit(None).alias("workforce_board_code_3").cast(pl.Utf8), # Tertiary workforce board code was not reported in WIA
            pl.col("Item_3002").alias("state"),
            (pl.col("Item_3007") == 1).alias("is_adult"),
            ((pl.col("Item_3008") == 1) | (pl.col("Item_3009") == 1) | (pl.col("Item_3010") == 1)).alias("is_dislocated_worker"),
            ((pl.col("Item_3011") == 1) | (pl.col("Item_3012") == 1)).alias("is_youth"),
            (pl.col("Item_951") == 1).alias("is_wagner_peyser"),
            (pl.col("Item_3013") == "0").alias("is_reportable_individual"),
            (pl.col("Item_3014") == "1").alias("received_training"),
            pl.col("Item_900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%m/%d/%Y", strict=False)
                .alias("entry_date"),
            pl.col("Item_901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%m/%d/%Y", strict=False)
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    ),
    "2013": lambda lf: (
        lf.select(
            pl.col("Item_100").alias("unique_id"),
            
            # Demographics Information
            pl.col("Item_201").alias("sex"),
            pl.col("Item_3006").alias("race").cast(pl.Int64, strict=False),
            pl.col("Item_3004").alias("age"),
            pl.col("Item_410").map_elements(_highest_educational_level_map, return_dtype=pl.Int64).alias("highest_educational_level"),
            pl.col("Item_702").alias("low_income_status").cast(pl.Int64, strict=False),
            pl.col("Item_400").alias("employment_status"),

            # Pre-Separation Employment
            pl.col("Item_402").alias("occupational_code_pre").cast(pl.Int64, strict=False),
            pl.col("Item_403").alias("industry_code_q1_pre").cast(pl.Int64, strict=False),
            pl.col("Item_404").alias("industry_code_q2_pre").cast(pl.Int64, strict=False),
            pl.col("Item_405").alias("industry_code_q3_pre").cast(pl.Int64, strict=False),
            pl.col("Item_1600").alias("wages_3q_pre").cast(pl.Int64, strict=False),
            pl.col("Item_1601").alias("wages_2q_pre").cast(pl.Int64, strict=False),
            pl.col("Item_1602").alias("wages_1q_pre").cast(pl.Int64, strict=False),

            # Post-Separation Employment
            pl.col("Item_1502").alias("occupational_code_post").cast(pl.Int64, strict=False),
            pl.col("Item_1514").alias("industry_code_q1_post").cast(pl.Int64, strict=False),
            pl.col("Item_1516").alias("industry_code_q2_post").cast(pl.Int64, strict=False),
            pl.col("Item_1517").alias("industry_code_q3_post").cast(pl.Int64, strict=False),
            pl.col("Item_1518").alias("industry_code_q4_post").cast(pl.Int64, strict=False),
            pl.col("Item_1603").alias("wages_1q_post").cast(pl.Int64, strict=False),
            pl.col("Item_1604").alias("wages_2q_post").cast(pl.Int64, strict=False),
            pl.col("Item_1605").alias("wages_3q_post").cast(pl.Int64, strict=False),
            pl.col("Item_1606").alias("wages_4q_post").cast(pl.Int64, strict=False),

            # Program Information
            pl.col("Item_105").alias("workforce_board_code_1").cast(pl.Utf8),
            pl.lit(None).alias("workforce_board_code_2").cast(pl.Utf8), # Secondary workforce board code was not reported in WIA
            pl.lit(None).alias("workforce_board_code_3").cast(pl.Utf8), # Tertiary workforce board code was not reported in WIA
            pl.col("Item_3002").alias("state"),
            (pl.col("Item_3007") == 1).alias("is_adult"),
            ((pl.col("Item_3008") == 1) | (pl.col("Item_3009") == 1) | (pl.col("Item_3010") == 1)).alias("is_dislocated_worker"),
            ((pl.col("Item_3011") == 1) | (pl.col("Item_3012") == 1)).alias("is_youth"),
            (pl.col("Item_951") == 1).alias("is_wagner_peyser"),
            (pl.col("Item_3013") == "0").alias("is_reportable_individual"),
            (pl.col("Item_3014") == "1").alias("received_training"),
            pl.col("Item_900")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%m/%d/%Y", strict=False)
                .alias("entry_date"),
            pl.col("Item_901")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%m/%d/%Y", strict=False)
                .alias("exit_date"),
        )
        .with_columns(
            pl.col("entry_date").dt.year().alias("entry_year"),
            pl.col("entry_date").dt.quarter().alias("entry_quarter"),
            pl.col("exit_date").dt.year().alias("exit_year"),
            pl.col("exit_date").dt.quarter().alias("exit_quarter")
        )
        .filter(
            ~pl.col("is_reportable_individual"),
            pl.col("exit_date").is_not_null()
        )
    )
}

def normalize(lf: pl.LazyFrame, year: str) -> pl.LazyFrame:
    if year not in NORMALIZATIONS:
        raise ValueError(f"No transform defined for year {year}")

    return NORMALIZATIONS[year](lf)

def compute_industry_code(lf: pl.LazyFrame) -> pl.LazyFrame: 
    """Compute the pre- and post-program industry code by coalescing 
    the industry code closest to a participant's participation. 
    """

    lf = lf.with_columns(
        pl.coalesce(
            pl.col("industry_code_q1_pre"),
            pl.col("industry_code_q2_pre"),
            pl.col("industry_code_q3_pre"),
        ).alias("industry_code_pre"),
        pl.coalesce(
            pl.col("industry_code_q1_post"),
            pl.col("industry_code_q2_post"),
            pl.col("industry_code_q3_post"),
            pl.col("industry_code_q4_post")
        ).alias("industry_code_post")
    )
    
    return lf

def compute_workforce_board_code(lf: pl.LazyFrame) -> pl.LazyFrame:

    lf = lf.with_columns(
        pl.coalesce(
            pl.col("workforce_board_code_1"),
            pl.col("workforce_board_code_2"),
            pl.col("workforce_board_code_3"),
        ).alias("workforce_board_code"),
    )

    return lf

def compute_funding_stream(lf: pl.LazyFrame) -> pl.LazyFrame:
    """ Compute the funding stream based on whether participant received
    Adult, Dislocated Worker, Youth, etc. funding.
    """

    lf = lf.with_columns(
        pl.when( # Is Only Adult Funding (incl. Wagner Peyser)
            (pl.col("is_adult"))
            & (~pl.col("is_dislocated_worker"))
            & (~pl.col("is_youth"))
        ).then(pl.lit("Adult"))
        .when( # Is Only Dislocated Worker Funding  (incl. Wagner Peyser)
            (~pl.col("is_adult"))
            & (pl.col("is_dislocated_worker"))
            & (~pl.col("is_youth"))
        ).then(pl.lit("Dislocated Worker"))
        .when( # Is Only Youth Funding (incl. Wagner Peyser)
            (~pl.col("is_adult"))
            & (~pl.col("is_dislocated_worker"))
            & (pl.col("is_youth"))
        ).then(pl.lit("Youth"))
        .when( # Is Adult, Dislocated Worker, and/or Youth Funding  (incl. Wagner Peyser)
            ((pl.col("is_adult"))
            | (pl.col("is_dislocated_worker"))
            | (pl.col("is_youth")))
        ).then(pl.lit("Adult, Dislocated worker, or Youth"))
        .when( # Is Only Wagner Peyser Funding
            (~pl.col("is_adult"))
            & (~pl.col("is_dislocated_worker"))
            & (~pl.col("is_youth"))
            & (pl.col("is_wagner_peyser"))
        ).then(pl.lit("Wagner-Peyser"))
        .otherwise(pl.lit(None))
        .alias("funding_stream")
    )

    return lf

def _convert_industry_to_subsector_code(code):
    """
    Formats a numeric industry code by converting it to a 6-digit string 
    with the first 3 digits preserved and the last 3 set to '000'.

    If the input is NaN, it is returned unchanged.

    Parameters:
        code (float or int): The industry code to format.

    Returns:
        str or float: The formatted industry code as a string, or the original NaN.
    """
    if np.isnan(code):
        return code
    else:
        # Convert code to string
        code = str(code).split('.')[0]
        
        # Get first 3 digits and pad with zeros
        code = code[:3].ljust(6, "0")
        return code


def compute_subsector_code(lf: pl.LazyFrame) -> pl.LazyFrame:
    """ Compute pre- and post-subsector code based on industry code."""

    lf = lf.with_columns(
        pl.col("industry_code_pre")
        .map_elements(_convert_industry_to_subsector_code, return_dtype=pl.String)
        .alias("subsector_code_pre"), 
        pl.col("industry_code_post")
        .map_elements(_convert_industry_to_subsector_code, return_dtype=pl.String)
        .alias("subsector_code_post")
    )

    return lf

def filter(lf: pl.LazyFrame) -> pl.LazyFrame:

    lf_filtered = (
        lf.filter(
            ~pl.any_horizontal(
                pl.exclude([
                    "occupational_code_pre", 
                    "occupational_code_post", 
                    "industry_code_q1_pre", 
                    "industry_code_q2_pre", 
                    "industry_code_q3_pre", 
                    "industry_code_q1_post", 
                    "industry_code_q2_post", 
                    "industry_code_q3_post", 
                    "industry_code_q4_post",
                    "workforce_board_code_1", 
                    "workforce_board_code_2",
                    "workforce_board_code_3"
                ]).is_null()),
            pl.col("is_adult") | pl.col("is_dislocated_worker") | pl.col("is_youth") | pl.col("is_wagner_peyser"),
        )
    )

    return lf_filtered

def sample(df: pl.DataFrame) -> pl.DataFrame:
    return df.sample(fraction=0.01, with_replacement=False)