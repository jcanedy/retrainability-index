import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations
import polars as pl

print("Script starting ... ")

column_names = {
    'PIRL100': 'unique_individual_id_x',
    'PIRL108A': 'workforce_board_code_1', # IN 5
    'PIRL108B': 'workforce_board_code_2', # IN 5
    'PIRL108C': 'workforce_board_code_3', # IN 5
    'PIRL201': 'sex_x', # IN 1
    'PIRL400': 'employment_status_x', # IN 1
    'PIRL402': 'long_term_employment_status_x', # IN 1
    'PIRL403': 'occupation_code_x', # IN 6
    'PIRL404': 'industry_code_q1_x', # IN 6
    'PIRL405': 'industry_code_q2_x', # IN 6
    'PIRL406': 'industry_code_q3_x', # IN 6
    'PIRL407': 'highest_school_grade_level_x', # IN 2
    'PIRL408': 'highest_education_level_x', # IN 1
    'PIRL409': 'school_status_x', # IN 1
    'PIRL410': 'dislocation_date_x', # DT 8
    'PIRL412': 'tenure_with_employer_x', # IN 3,
    'PIRL413': 'farmworker_designation_x', # IN 1
    'PIRL800': 'homelessness_x', # IN 1
    'PIRL801': 'exoffender_x', # IN 1
    'PIRL802': 'low_income_x', # IN 1
    'PIRL803': 'english_language_learner_x', # IN 1
    'PIRL804': 'basic_skills_deficient_x', # IN 1
    'PIRL805': 'cultural_barriers_x', # IN 1
    'PIRL806': 'single_parent_x', # IN 1
    'PIRL807': 'displaced_homemaker_x', # IN 1
    'PIRL808': 'eligible_farmworker_x', # IN 1
    'PIRL900': 'program_entry_date_x', # DT 8
    'PIRL901': 'program_exit_date_y', # DT 8
    'PIRL1300': 'received_training_x', # IN 1
    'PIRL1301': 'training_provider_1_x', # AN 75
    'PIRL1303': 'training_service_type_1_x', # IN 2 
    'PIRL1600': 'employment_q1_y', # IN 1
    'PIRL1602': 'employment_q2_y', # IN 1
    'PIRL1604': 'employment_q3_y', # IN 1
    'PIRL1606': 'employment_q4_y', # IN 1
    'PIRL1610': 'occupation_code_y', # IN 6
    'PIRL1611': 'non_traditional_employment_y', # IN 1
    'PIRL1612': 'occupation_code_q2_y', # IN 6
    'PIRL1613': 'occupation_code_q4_y', # IN 6
    'PIRL1614': 'industry_code_q1_y', # IN 6
    'PIRL1615': 'industry_code_q2_y', # IN 6
    'PIRL1616': 'industry_code_q3_y', # IN 6
    'PIRL1617': 'industry_code_q4_y', # IN 6
    'PIRL1618': 'employer_retention_y', # IN 1
    'PIRL1700': 'wages_q3_x', # IN 6
    'PIRL1701': 'wages_q2_x', # IN 6
    'PIRL1702': 'wages_q1_x', # IN 6
    'PIRL1703': 'wages_q1_y', # IN 7
    'PIRL1704': 'wages_q2_y', # IN 7
    'PIRL1705': 'wages_q3_y', # IN 7
    'PIRL1706': 'wages_q4_y', # IN 7
    'CALC4000': 'state_x', # AN 2
    'CALC4001': 'adult_funding_x', # IN 1,
    'CALC4002': 'dislocated_funding_x', # IN 1
    'CALC4003': 'youth_funding_x', # IN 1
    'CALC4004': 'dislocated_grant_x', # IN 1
    'CALC4005': 'wagner_peyser_funding_x', # IN 1,
    'CALC4006': 'reportable_individual_x', # IN 1
    'CALC4007': 'measurable_skill_gains_y', # IN 1
    'CALC4013': 'employment_rate_q2_y', # IN 1
    'CALC4015': 'median_earnings_q2_y', # IN 7
    'CALC4017': 'employment_rate_q4_y', # IN 1
    'CALC4019': 'credential_rate_y', # IN 1
    'CALC4020': 'race_ethnicity_x', # IN 1
    'CALC4021': 'no_disability_x', # IN 1
    'CALC4030': 'industry_certificate_y', # IN 1
    'CALC4031': 'apprenticeship_certificate_y', # IN 1
    'CALC4032': 'government_certificate_y', # IN 1
    'CALC4039': 'age_x', # IN 2
    'CALC4040': 'program_year_exit_y', # IN 4
    'CALC4041': 'veterans_program_y', # IN 1
    'REPORT_QUARTER': 'report_quarter' # DT 8
 }

columns = column_names.keys()

# Load preprocessed data.
data = pl.read_parquet("data/processed/wioa_data.parquet")
occupations = pl.read_csv('data/processed/occupations.csv')
rti_by_occupation = pl.read_csv('data/processed/rti_by_occupation.csv')
rti_by_industry = pl.read_csv('data/processed/rti_by_industry.csv')
workforce_boards = pd.read_csv('data/processed/workforce_boards.csv')

#TODO(jcanedy27@gmail.com): Move to `compute_rti_by_industry.py`.
# Recast to string
rti_by_industry = rti_by_industry.with_columns(
    pl.col("industry_code", "industry_code_prefix").cast(pl.String)
)

# Rename WIOA columns to human readable column names.
data = data.select(columns).rename(column_names)
print(f"Data shape after initial load: {data.shape}")

# Remap numeric column values to interpretable values.
employment_status_x_map = {
    1: "Employed",
    2: "Employed, but Received Notice of \
    Termination of Employment or Military \
    Separation is pending",
    3: "Not in labor force",
    0: "Unemployed"
}

low_income_x_map = {
    1: "Yes",
    0: "No"
}

farmworker_designation_x_map = {
    1: "Seasonal Farmworker",
    2: "Migrant",
    0: "No"
}

received_training_x_map = {
  1: "Yes",
  0: "No"
}

race_ethnicity_x_map = {
    1: "Hispanic",
    2: "Asian (not Hispanic)",
    3: "Black (not Hispanic)",
    4: "Native Hawaiian or Pacific Islander (not Hispanic)",
    5: "American Indian or Alaska Native (not Hispanic)",
    6: "White (not Hispanic)",
    7: "Multiple Race (not Hispanic)",
}

def bin_age(age):
  if (age < 5):
    return "Under 5 years"
  elif (5 <= age < 17):
    return "5 to 17 years"
  elif (17 <= age < 25):
    return "17 to 24 years"
  elif (25 <= age < 35):
    return "25 to 34 years"
  elif (35 <= age < 45):
    return "35 to 44 years"
  elif (45 <= age < 55):
    return "45 to 54 years"
  elif (55 <= age < 65):
    return "55 to 64 years"
  elif (65 <= age < 85):
    return "65 to 84 years"
  elif (85 <= age < 100):
    return "85 to 99 years"
  elif (100 <= age):
    return "100 years and over"
  else:
    return "Unknown"

sex_x_map = {
    1: "Male",
    2: "Female",
    9: "Participant did not self-identify",
}

highest_education_level_x_map = {
  1: "Attained secondary school diploma",
  2: "Attained a secondary school equivalency",
  3: "The participant with a disability receives \
  a certificate of attendance/completion as a \
  result of successfully completing an \
  Individualized Education Program (IEP)",
  4: "Completed one of more years of postsecondary education",
  5: "Attained a postsecondary technical or vocational certificate (non-degree)",
  6: "Attained an Associate's degree",
  7: "Attained a Bachelor's degree",
  8: "Attained a degree beyond a Bachelor's degree",
  0: "No Educational Level Completed",
}

training_service_type_map = {
  1: "On the Job Training (non-WIOA Youth).",
  2: "Skill Upgrading",
  3: "Entrepreneurial Training (non-WIOA Youth)",
  4: "ABE or ESL (contextualized or other) in conjunction with Training",
  5: "Customized Training",
  6: "Occupational Skills Training (nonWIOA Youth)",
  7: "ABE or ESL (contextualized or other) \
  NOT in conjunction with training (funded \
  by Trade Adjustment Assistance only)",
  8: "Prerequisite Training",
  9: "Registered Apprenticeship",
  10: "Youth Occupational Skills Training",
  11: "Other Non-Occupational-Skills Training",
  12: "Job Readiness Training in conjunction \
  with other training",
  0: "No Training Service"
}

data = (
  data.lazy()
    .with_columns([
      pl.col("employment_status_x").replace_strict(employment_status_x_map, default=None).alias("employment_status_x"),
      pl.col("low_income_x").replace_strict(low_income_x_map, default=None).alias("low_income_x"),
      pl.col("farmworker_designation_x").replace_strict(farmworker_designation_x_map, default=None).alias("farmworker_designation_x"),
      pl.col("received_training_x").replace_strict(received_training_x_map, default=None).alias("received_training_x"),
      pl.col("race_ethnicity_x").replace_strict(race_ethnicity_x_map, default="Unknown").alias("race_ethnicity_x"),
      pl.col("age_x").map_elements(bin_age, return_dtype=pl.String).alias("age_x"),  # for function-based binning
      pl.col("sex_x").replace_strict(sex_x_map, default="Unknown").alias("sex_x"),
      pl.col("highest_education_level_x").replace_strict(highest_education_level_x_map, default="Unknown").alias("highest_education_level_x"),
      pl.col("training_service_type_1_x").replace_strict(training_service_type_map, default=None).alias("training_service_type_1_x"),
    ])
    .collect()
)

print(f"Data shape after remaping column values: {data.shape}")

# Add an individaual's pre-program occupation title based on the occupation code.
data = (
    data.join(
        occupations,
        left_on="occupation_code_x",
        right_on="occupation_code",
        how="left"
    )
    .drop(["soc_level", "occupation_code_prefix"])
    .rename({"occupation_title": "occupation_title_x"})
)

# Use an individual's pre-program occupation code to get rti.
data = (
    data.join(
        rti_by_occupation,     
        left_on="occupation_code_x",
        right_on="occupation_code",
        how="left"
    )
    .drop(["occ2000"])
    .rename({
      'nr_cog_anal': 'nr_cog_anal_x',
      'nr_cog_pers': 'nr_cog_pers_x',
      'r_cog': 'r_cog_x',
      'r_man': 'r_man_x',
      'nr_man_phys': 'nr_man_phys_x',
      'nr_man_pers': 'nr_man_pers_x',
      'offshor': 'offshor_x'
  })
)

# Add an individaual's post-program occupation title based on the occupation code.
data = (
  data.join(
      occupations, 
      left_on="occupation_code_y", 
      right_on="occupation_code",
      how="left"
  )
  .drop(["soc_level", "occupation_code_prefix"])
  .rename({"occupation_title": "occupation_title_y"})
)

# Use an individual's pre-program occupation code to get rti.
data = (
    data.join(
        rti_by_occupation,
        left_on="occupation_code_y",
        right_on="occupation_code",
        how="left"
    )
    .drop(["occ2000"])
    .rename({
      'nr_cog_anal': 'nr_cog_anal_y',
      'nr_cog_pers': 'nr_cog_pers_y',
      'r_cog': 'r_cog_y',
      'r_man': 'r_man_y',
      'nr_man_phys': 'nr_man_phys_y',
      'nr_man_pers': 'nr_man_pers_y',
      'offshor': 'offshor_y'
  })
)

print(f"Data shape after adding pre- and post-program occupation-level rti: {data.shape}")

# Use the latest industry code to represent the industry of individual's pre-program employment.
data = data.with_columns(
    pl.coalesce([
        pl.col("industry_code_q3_x"),
        pl.col("industry_code_q2_x"),
        pl.col("industry_code_q1_x"),
    ]).alias("industry_code_x")
)

# Use the lastest industry code to represent the industry of individual's post-program employment.
data = data.with_columns(
    pl.coalesce([
        pl.col("industry_code_q1_y"),
        pl.col("industry_code_q2_y"),
        pl.col("industry_code_q3_y"),
        pl.col("industry_code_q4_y"),
    ]).alias("industry_code_y")
)

def format_industry_code(code):
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
        code = code[:3] + '000'
        return code

# Format industry codes appropriately
data = data.with_columns(
    pl.col("industry_code_x", "industry_code_y").map_elements(format_industry_code, return_dtype=pl.String)
)
print(f"Data shape after adding pre- and post-program industry codes: {data.shape}")

# Join with industry-level rti for pre-program industry codes
data = (
   data.join(
      rti_by_industry,
      left_on="industry_code_x",
      right_on="industry_code",
      how="left"
   )
   .rename({
      "industry_title": "industry_title_x",
      "r_cog_industry": "r_cog_industry_x",
      "r_man_industry": "r_man_industry_x",
      "offshor_industry": "offshor_industry_x"
  })
)

# Join with industry-level rti for pre-program industry codes
data = (
   data.join(
      rti_by_industry,
      left_on="industry_code_y",
      right_on="industry_code",
      how="left"
   )
   .rename({
      "industry_title": "industry_title_y",
      "r_cog_industry": "r_cog_industry_y",
      "r_man_industry": "r_man_industry_y",
      "offshor_industry": "offshor_industry_y"
  })
)

print(f"Data shape after adding pre- and post-program industry-level rti: {data.shape}")

# Calculate pre- and post-program difference in wages
data = data.with_columns([
    pl.mean_horizontal("wages_q1_x", "wages_q2_x", "wages_q3_x").alias("wages_mean_x"),
    pl.mean_horizontal("wages_q1_y", "wages_q2_y", "wages_q3_y", "wages_q4_y").alias("wages_mean_y")
])

data = data.with_columns([
    (pl.col("wages_mean_y") - pl.col("wages_mean_x")).alias("diff_wages_mean_y"),
])


# Calculate pre- and post-program difference in rti
# A positive difference in pre- and post-program correspond to 
# an occupation change that is has more routine cognitive/manual task 
# or has more offshorable
data = data.with_columns([
   (pl.col("r_cog_y") - pl.col("r_cog_x")).alias("diff_r_cog_y"),
   (pl.col("r_man_y") - pl.col("r_man_x")).alias("diff_r_man_y"),
   (pl.col("offshor_y") - pl.col("offshor_x")).alias("diff_offshor_y"),
])

data = data.with_columns([
   (pl.col("r_cog_industry_y") - pl.col("r_cog_industry_x")).alias("diff_r_cog_industry_y"),
   (pl.col("r_man_industry_y") - pl.col("r_man_industry_x")).alias("diff_r_man_industry_y"),
   (pl.col("offshor_industry_y") - pl.col("offshor_industry_x")).alias("diff_offshor_industry_y"),
])

# Normalize output output columns
scaler = MinMaxScaler()
cols = ["diff_r_cog_y", 
        "diff_r_man_y",
        "diff_offshor_y", 
        "diff_r_cog_industry_y",
        "diff_r_man_industry_y",
        "diff_offshor_industry_y", 
        "diff_wages_mean_y",
      ]

# Convert to pandas, scale, convert back
cols_normalized = scaler.fit_transform(data.select(cols).to_pandas())
df_normalized = pl.DataFrame(cols_normalized, schema=[f"{col}_norm" for col in cols])

# Add the scaled columns back to original data
data = data.with_columns(df_normalized)

print(f"Data shape after constructing and normalizing response variables: {data.shape}")

data = data.with_columns([
    # 1 if routine cognitive tasks (strictly) increase
    (pl.col("diff_r_cog_industry_y") >= 0).cast(pl.Int64).alias("bin_r_cog_industry_y"), 
    # 1 if routine manual tasks (strictly) increase
    (pl.col("diff_r_man_industry_y") >= 0).cast(pl.Int64).alias("bin_r_man_industry_y"), 
    # 1 if offshorability (strictly) increase
    (pl.col("diff_offshor_industry_y") >= 0).cast(pl.Int64).alias("bin_offshor_industry_y"), 
    # 1 if routine cognitive tasks (strictly) increase
    (pl.col("diff_r_cog_y") >= 0).cast(pl.Int64).alias("bin_r_cog_y"), 
    # 1 if routine manual tasks (strictly) increase
    (pl.col("diff_r_man_y") >= 0).cast(pl.Int64).alias("bin_r_man_y"), 
    # 1 if offshorability (strictly) increase
    (pl.col("diff_offshor_y") >= 0).cast(pl.Int64).alias("bin_offshor_y"),
    # 1 if wages (strictly) increase
    (pl.col("diff_wages_mean_y") >= 0).cast(pl.Int64).alias("bin_wages_mean_y"), 
])

print(f"Data shape after binning response variables: {data.shape}")

data = data.with_columns(
   pl.when(
       # Tier 1: Has wages + all individual-level variables (most complete data)
       pl.col("bin_wages_mean_y").is_not_null() &
       pl.col("bin_r_cog_y").is_not_null() &
       pl.col("bin_r_man_y").is_not_null() &
       pl.col("bin_offshor_y").is_not_null()
   ).then(pl.lit("Tier 1"))
   .when(
       # Tier 2: Has wages + all industry-level variables (fallback data)
       pl.col("bin_wages_mean_y").is_not_null() &
       pl.col("bin_r_cog_industry_y").is_not_null() &
       pl.col("bin_r_man_industry_y").is_not_null() &
       pl.col("bin_offshor_industry_y").is_not_null()
   ).then(pl.lit("Tier 2"))
   .when(
       # Tier 3: Has wages only (minimal usable data)
       pl.col("bin_wages_mean_y").is_not_null()
   ).then(pl.lit("Tier 3"))
   # Excluded: Missing wages (unusable for analysis)
   .otherwise(pl.lit("Excluded"))
   .alias("outcome_tier")
)
print(f"Data shape after assigning outcome tier: {data.shape}")

# Define dimensions and metrics
dimensions = [
    'low_income_x', 'employment_status_x', 'received_training_x',
    'race_ethnicity_x', 'sex_x', 'age_x', 'highest_education_level_x',
    'training_service_type_1_x', 'industry_title_x'
]

metrics = [
    "bin_r_cog_industry_y mean", "bin_r_cog_industry_y count",
    "bin_r_man_industry_y mean", "bin_r_man_industry_y count", 
    "bin_offshor_industry_y mean", "bin_offshor_industry_y count",
    "bin_wages_mean_y mean", "bin_wages_mean_y count"
]

column_order = dimensions + metrics + ["__groupby__"]


# Clean data - drop rows with nulls in any dimension column
data = data.drop_nulls(subset=dimensions)
print(f"Data shape after dropping rows with NA in dimension columns: {data.shape}")

# Create all combinations of dimensions from 1 to N
grouping_sets = [list(c) for i in range(1, len(dimensions)+1) for c in combinations(dimensions, i)]

# Filter to Tier 2 data only
tier2_data = data.filter(pl.col("outcome_tier") == "Tier 2")

aggregates = []

# Process each grouping combination
for group in grouping_sets:
    # Aggregate by current group
    grouped = tier2_data.group_by(group).agg([
        pl.col("bin_r_cog_industry_y").mean().alias("bin_r_cog_industry_y mean"),
        pl.col("bin_r_cog_industry_y").count().alias("bin_r_cog_industry_y count"),
        pl.col("bin_r_man_industry_y").mean().alias("bin_r_man_industry_y mean"),
        pl.col("bin_r_man_industry_y").count().alias("bin_r_man_industry_y count"),
        pl.col("bin_offshor_industry_y").mean().alias("bin_offshor_industry_y mean"),
        pl.col("bin_offshor_industry_y").count().alias("bin_offshor_industry_y count"),
        pl.col("bin_wages_mean_y").mean().alias("bin_wages_mean_y mean"),
        pl.col("bin_wages_mean_y").count().alias("bin_wages_mean_y count"),
    ])
    
    # Add "All" for dimensions not in current group
    for dim in dimensions:
        if dim not in group:
            grouped = grouped.with_columns(pl.lit("All").alias(dim))
    
    # Add groupby identifier
    grouped = grouped.with_columns(pl.lit(','.join(group)).alias("__groupby__"))

    # Reorder columns to match final schema
    grouped = grouped.select(column_order)
    aggregates.append(grouped)

# Grand total row - aggregate all Tier 2 data
grand_total = tier2_data.select([
    pl.col("bin_r_cog_industry_y").mean().alias("bin_r_cog_industry_y mean"),
    pl.col("bin_r_cog_industry_y").count().alias("bin_r_cog_industry_y count"),
    pl.col("bin_r_man_industry_y").mean().alias("bin_r_man_industry_y mean"),
    pl.col("bin_r_man_industry_y").count().alias("bin_r_man_industry_y count"),
    pl.col("bin_offshor_industry_y").mean().alias("bin_offshor_industry_y mean"),
    pl.col("bin_offshor_industry_y").count().alias("bin_offshor_industry_y count"),
    pl.col("bin_wages_mean_y").mean().alias("bin_wages_mean_y mean"),
    pl.col("bin_wages_mean_y").count().alias("bin_wages_mean_y count"),
])

# Add all dimension columns as "All" and groupby marker
for dim in dimensions:
    grand_total = grand_total.with_columns(pl.lit("All").alias(dim))
grand_total = grand_total.with_columns(pl.lit("All").alias("__groupby__"))

# Reorder grand total columns to match schema
grand_total = grand_total.select(column_order)
aggregates.append(grand_total)

# Combine all aggregations
index_df = pl.concat(aggregates)

index_df.write_parquet("data/processed/index_tier2.parquet", compression="snappy")
print(f"Data shape after saving index Tier 2: {index_df.shape}")


# # Create Tier 1 Index
# metrics = ['bin_r_cog_y', 'bin_r_man_y', 'bin_offshor_y', 'bin_wages_mean_y']

# aggregates = []

# # Grouped aggregations with rollup handling
# isOutcomeTier1 = data['outcome_tier'] == "Tier 1"
# for group in grouping_sets:
#     grouped = data[isOutcomeTier1].groupby(group).agg({
#         'bin_r_cog_y': ['mean', 'count'],
#         'bin_r_man_y': ['mean', 'count'],
#         'bin_offshor_y': ['mean', 'count'],
#         'bin_wages_mean_y': ['mean', 'count'],
#     }).reset_index()

#     # Add 'All' for dimensions not in current group
#     for dim in dimensions:
#         if dim not in group:
#             grouped[dim] = "All"

#     grouped['__groupby__'] = ','.join(group)
#     grouped.columns = [' '.join(c).strip() for c in grouped.columns]
#     aggregates.append(grouped)

# # Grand total row (match schema exactly)
# agg = data[isOutcomeTier1].agg({
#     'bin_r_cog_y': ['mean', 'count'],
#     'bin_r_man_y': ['mean', 'count'],
#     'bin_offshor_y': ['mean', 'count'],
#     'bin_wages_mean_y': ['mean', 'count'],
# })

# agg_row = agg.T.stack().to_frame().T
# agg_row.columns =  [f"{col} {stat}" for col, stat in agg_row.columns]

# # Add all dimension columns and group marker
# for dim in dimensions:
#     agg_row[dim] = "All"
# agg_row['__groupby__'] = 'All'

# # Reorder columns to match other groupings
# # Put dimensions + __groupby__ first, then the metric columns
# # column_order = dimensions + ['__groupby__'] + list(agg_row.columns)
# # agg_row = agg_row[column_order]

# aggregates.append(agg_row)

# # Concatenate all
# index_df = pd.concat(aggregates, ignore_index=True)

# index_df.to_csv("data/processed/index_tier1.csv", index=False)
# print(f"Data shape after saving index Tier 1: {index_df.shape}")

print("Script finished!")


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# # Prepare data for modeling (encode categorical variables)
# tier2_data = data.filter(pl.col("outcome_tier") == "Tier 2").to_pandas()

# # Encode categorical variables
# encoders = {}
# X = tier2_data[dimensions].copy()
# for col in dimensions:
#     if X[col].dtype == 'object':
#         encoders[col] = LabelEncoder()
#         X[col] = encoders[col].fit_transform(X[col].astype(str))

# # For each outcome variable, get feature importance
# outcomes = ['bin_r_cog_industry_y', 'bin_r_man_industry_y', 'bin_offshor_industry_y', 'bin_wages_mean_y']

# importance_scores = {}
# for outcome in outcomes:
#     y = tier2_data[outcome].dropna()
#     X_clean = X.loc[y.index]
    
#     rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     rf.fit(X_clean, y)
    
#     importance_scores[outcome] = dict(zip(dimensions, rf.feature_importances_))

# # Average importance across all outcomes
# avg_importance = {dim: np.mean([importance_scores[outcome][dim] for outcome in outcomes]) 
#                   for dim in dimensions}

# # Sort by importance
# sorted_dims = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
# print("Dimension importance rankings:")
# for dim, score in sorted_dims:
#     print(f"{dim}: {score:.4f}")