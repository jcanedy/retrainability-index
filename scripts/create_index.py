# %%
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
    'PIRL1310': 'training_service_type_2_x', # IN 2
    'PIRL1315': 'training_service_type_3_x', # IN 2
    'PIRL1328': 'training_provided_online', # IN 1
    'PIRL1333': 'recieved_private_sector_training_x', # IN 1
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
# %%
# Load preprocessed data.
data = pl.read_parquet("data/processed/wioa_data.parquet")
occupations = pl.read_csv('data/processed/occupations.csv')
rti_by_subsector = pl.read_parquet('data/processed/rti_by_subsector.parquet')
rti_by_industry = pl.read_parquet('data/processed/rti_by_industry.parquet')
rti_by_occupation = pl.read_parquet('data/processed/rti_by_occupation.parquet')
workforce_boards = pd.read_csv('data/processed/workforce_boards.csv')

# Rename WIOA columns to human readable column names.
data = data.select(columns).rename(column_names)
print(f"Data shape after initial load: {data.shape}")
# %%

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
# %%
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
# %%
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

# %%

def convert_industry_to_subsector_code(code):
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

# Convert industry codes to subsector codes appropriately

data = data.with_columns(
    pl.col("industry_code_x", "industry_code_y").map_elements(convert_industry_to_subsector_code, return_dtype=pl.String)
).rename({
    "industry_code_x": "subsector_code_x",
    "industry_code_y": "subsector_code_y"
})

print(f"Data shape after adding pre- and post-program industry codes: {data.shape}")
# %%
# Join with industry-level rti for pre-program industry codes
data = (
   data.join(
      rti_by_subsector,
      left_on="subsector_code_x",
      right_on="subsector_code",
      how="left"
   )
   .rename({
      "subsector_title": "industry_title_x",
      "r_cog_industry": "r_cog_industry_x",
      "r_man_industry": "r_man_industry_x",
      "offshor_industry": "offshor_industry_x"
  })
)

# Join with industry-level rti for pre-program industry codes
data = (
   data.join(
      rti_by_subsector,
      left_on="subsector_code_y",
      right_on="subsector_code",
      how="left"
   )
   .rename({
      "subsector_title": "industry_title_y",
      "r_cog_industry": "r_cog_industry_y",
      "r_man_industry": "r_man_industry_y",
      "offshor_industry": "offshor_industry_y"
  })
)

print(f"Data shape after adding pre- and post-program industry-level rti: {data.shape}")

# %%
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

# %%
# Create indicator variables for pre- and post-program differences in 
# routine cognitive, routine manual, offshorability, and wages
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

# %%
# Define dimensions and metrics
dimensions = [
    'low_income_x', 'employment_status_x', 'received_training_x',
    'race_ethnicity_x', 'sex_x', 'age_x', 'highest_education_level_x',
    'industry_title_x', 'state_x'
]

#TODO(jcanedy27@): Consolidate count variables to a single column
metrics = [
    "bin_r_cog_industry_y mean",
    "bin_r_man_industry_y mean",
    "bin_offshor_industry_y mean",
    "bin_wages_mean_y mean",
    "diff_r_cog_industry_y median",
    "diff_r_cog_industry_y 25th",
    "diff_r_cog_industry_y 75th",
    "diff_r_man_industry_y median",
    "diff_r_man_industry_y 25th",
    "diff_r_man_industry_y 75th",
    "diff_offshor_industry_y median",
    "diff_offshor_industry_y 25th",
    "diff_offshor_industry_y 75th",
    "diff_wages_mean_y median",
    "diff_wages_mean_y 25th",
    "diff_wages_mean_y 75th",
    "diff_r_cog_industry_y mean",
    "diff_r_man_industry_y mean",
    "diff_offshor_industry_y mean",
    "diff_wages_mean_y mean",
    "count",
]

column_order = dimensions + metrics

# Clean data - drop rows with nulls in any dimension column
data = data.drop_nulls(subset=dimensions)
print(f"Data shape after dropping rows with NA in dimension columns: {data.shape}")

# Create all combinations of dimensions from 1 to N
grouping_sets = [list(c) for i in range(1, len(dimensions)+1) for c in combinations(dimensions, i)]

# Filter to Tier 2 data only
tier2_data = data.filter(pl.col("outcome_tier") == "Tier 2")

def consolidate_multiple_columns(data, columns, min_percentage=0.05):
    """Consolidate multiple columns at once, overwriting originals"""
    
    consolidated_data = data
    consolidated_columns = []
    non_consolidated_columns = []
    
    for column in columns:
        # Skip if column is not string/categorical
        if consolidated_data[column].dtype not in [pl.String, pl.Categorical]:
            non_consolidated_columns.append(column)
            print(f"{column}: skipped (not categorical)")
            continue
            
        total_count = consolidated_data.height
        value_counts = consolidated_data.select(pl.col(column).value_counts()).unnest(column)
        
        # Calculate percentages
        value_counts = value_counts.with_columns(
            (pl.col("count") / total_count).alias("percentage")
        )
        
        # Keep categories above threshold
        keep_categories = value_counts.filter(
            pl.col("percentage") >= min_percentage
        ).select(column).to_series().to_list()
        
        original_categories = value_counts.height
        kept_categories = len(keep_categories)
        
        # Only consolidate if we're actually reducing categories
        if kept_categories < original_categories:
            # Overwrite original column with consolidated version
            consolidated_data = consolidated_data.with_columns(
                pl.when(pl.col(column).is_in(keep_categories))
                .then(pl.col(column))
                .otherwise(pl.lit("Other"))
                .alias(column)  # Same name as original
            )
            consolidated_columns.append(column)
            print(f"{column}: consolidated to {kept_categories} categories (from {original_categories})")
        else:
            # No consolidation needed
            non_consolidated_columns.append(column)
            print(f"{column}: no consolidation needed ({original_categories} categories)")
    
    return consolidated_data, consolidated_columns, non_consolidated_columns

# Use it - much cleaner!
tier2_data, consolidated_cols, non_consolidated_cols = consolidate_multiple_columns(
    tier2_data, dimensions, min_percentage=0.02
)

print(f"Data shape after consolidating columns: {data.shape}")
# %%

# Save data for separate analysis
tier2_data.write_parquet("data/processed/wioa_data_tier2.parquet", compression="snappy")

# %%
aggregates = []

aggregations = [
    pl.col("bin_r_cog_industry_y").mean().alias("bin_r_cog_industry_y mean"),
    pl.col("bin_r_man_industry_y").mean().alias("bin_r_man_industry_y mean"),
    pl.col("bin_offshor_industry_y").mean().alias("bin_offshor_industry_y mean"),
    pl.col("bin_wages_mean_y").mean().alias("bin_wages_mean_y mean"),

    pl.col("diff_r_cog_industry_y").median().alias("diff_r_cog_industry_y median"),
    pl.col("diff_r_cog_industry_y").quantile(0.25, interpolation="linear").alias("diff_r_cog_industry_y 25th"),
    pl.col("diff_r_cog_industry_y").quantile(0.75, interpolation="linear").alias("diff_r_cog_industry_y 75th"),
    pl.col("diff_r_cog_industry_y").mean().alias("diff_r_cog_industry_y mean"),

    pl.col("diff_r_man_industry_y").median().alias("diff_r_man_industry_y median"),
    pl.col("diff_r_man_industry_y").quantile(0.25, interpolation="linear").alias("diff_r_man_industry_y 25th"),
    pl.col("diff_r_man_industry_y").quantile(0.75, interpolation="linear").alias("diff_r_man_industry_y 75th"),
    pl.col("diff_r_man_industry_y").mean().alias("diff_r_man_industry_y mean"),

    pl.col("diff_offshor_industry_y").median().alias("diff_offshor_industry_y median"),
    pl.col("diff_offshor_industry_y").quantile(0.25, interpolation="linear").alias("diff_offshor_industry_y 25th"),
    pl.col("diff_offshor_industry_y").quantile(0.75, interpolation="linear").alias("diff_offshor_industry_y 75th"),
     pl.col("diff_offshor_industry_y").mean().alias("diff_offshor_industry_y mean"),

    pl.col("diff_wages_mean_y").median().alias("diff_wages_mean_y median"),
    pl.col("diff_wages_mean_y").quantile(0.25, interpolation="linear").alias("diff_wages_mean_y 25th"),
    pl.col("diff_wages_mean_y").quantile(0.75, interpolation="linear").alias("diff_wages_mean_y 75th"),
    pl.col("diff_wages_mean_y").mean().alias("diff_wages_mean_y mean"),

    pl.len().alias("count"),
]

# Process each grouping combination
for group in grouping_sets:
    # Aggregate by current group
    grouped = tier2_data.group_by(group).agg(aggregations)
    
    # Add "All" for dimensions not in current group
    for dim in dimensions:
        if dim not in group:
            grouped = grouped.with_columns(pl.lit("All").alias(dim))

    # Reorder columns to match final schema
    grouped = grouped.select(column_order)
    aggregates.append(grouped)

# Grand total row - aggregate all Tier 2 data
grand_total = tier2_data.select(aggregations)

# Add all dimension columns as "All" and groupby marker
for dim in dimensions:
    grand_total = grand_total.with_columns(pl.lit("All").alias(dim))

# Reorder grand total columns to match schema
grand_total = grand_total.select(column_order)
aggregates.append(grand_total)

# Combine all aggregations
index_df = pl.concat(aggregates)

index_df = index_df.with_columns([
    pl.col(c).cast(pl.Categorical) for c in dimensions
])

# %%
index_df.head()

# %%
index_df.write_parquet("data/processed/index_tier2.parquet", compression="zstd")
print(f"Data shape after saving index Tier 2: {index_df.shape}")

print("Script finished!")
# %%
