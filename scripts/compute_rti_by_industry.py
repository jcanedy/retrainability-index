# %%
import pandas as pd

# %%
def expand_naics_code_range(code):
    """
    Expands a hyphenated BLS-style NAICS code range (e.g., '1131-2' or '31-330')
    into a list of 6-digit NAICS codes, using right-padding with zeros.
    """
    try:
  
      if "-" not in code:
          return [code]

      # Strip any trailing zero(s)
      code = code.rstrip("0")

      # Split the code into start and end parts
      code_start, code_end = code.split("-")

      shared_prefix_len = len(code_start) - len(code_end)
      prefix = code_start[:shared_prefix_len]

      # Get the numeric range
      start_suffix = int(code_start[shared_prefix_len:])
      end_suffix = int(code_end)

      full_codes = []
      for suffix in range(start_suffix, end_suffix + 1):
          full_code = prefix + str(suffix)
          padded_code = full_code.ljust(6, '0')
          full_codes.append(padded_code)

      return full_codes

    except Exception as e:
        return f"Error parsing code range: {e}"

# %%
print("Script starting ... ")

# data-source: https://www.bls.gov/emp/data/occupational-data.htm
matrix = pd.read_excel('data/raw/national_employment_matrix.xlsx')

# data-source: https://www.dropbox.com/scl/fi/86paxjkauqtg72o73gp2h/task-construct-1.zip?dl=0&e=1&rlkey=o21usoigscnrki3r77oo610pg
rti_by_occupation_code =  pd.read_stata('data/raw/rti_by_occupation_code.dta')

# %%
rti_by_occupation_code = rti_by_occupation_code.rename(
    columns={
        "onetsoccode": "occupation_code"
    }
)

# Convert occupation code from a float to an integer
rti_by_occupation_code["occupation_code"] = rti_by_occupation_code["occupation_code"].astype('Int64')

matrix = matrix.rename(
    columns={
        "Occupation title": "occupation_title",
        "Occupation code": "occupation_code",
        "Industry code": "industry_code",
        "Industry title": "industry_title",
        "Occupation type": "occupation_type",
        "Industry type": "industry_type",
        "2023 Percent of Industry": "2023_percent_of_industry",
    }
)

# %%
# Filter to occupation/industry codes which represent the detailed occupation/industry
isOccupationTypeLineItem = matrix["occupation_type"] == "Line item"
isIndustryTypeLineItem = matrix["industry_type"] == "Line item"
matrix = matrix.loc[isOccupationTypeLineItem & isIndustryTypeLineItem, [
    "industry_code",
    "industry_title",
    "occupation_code",
    "2023_percent_of_industry"
    ]]

# Format occupation code as integer 
matrix["occupation_code"] = matrix["occupation_code"].str.replace('-', '').astype('Int64')

# Join the routine task intensity measures by occupation code
matrix = matrix.merge(rti_by_occupation_code, how="left", on="occupation_code")

unique_occupations_before = matrix.occupation_code.nunique()
unique_industries_before = matrix.industry_code.nunique()

# Drop rows which do not have a routine task intensity measure
matrix = matrix.dropna(subset=["r_cog", "r_man", "offshor"])

unique_occupations_after = matrix.occupation_code.nunique()
unique_industries_after = matrix.industry_code.nunique()

print(f"{unique_occupations_after} of {unique_occupations_after} occupations and {unique_industries_after} of {unique_industries_before} industries were matched with routine task intensity metrics")

# %%
# Normalize the percent of occupation by industry code
# So that within an industry, all remaining occupations sum to 1
matrix['2023_percent_of_industry_norm'] = matrix.groupby('industry_code')['2023_percent_of_industry'].transform(
    lambda x: x / x.sum()
)

# Calculate the industry level routine task intensity measures as a weighted sum
matrix['r_cog_industry'] = matrix['2023_percent_of_industry_norm'] * matrix['r_cog']
matrix['r_man_industry'] = matrix['2023_percent_of_industry_norm'] * matrix['r_man']
matrix['offshor_industry'] = matrix['2023_percent_of_industry_norm'] * matrix['offshor']
rti_by_industry_code = matrix.groupby(['industry_code', 'industry_title'])[['r_cog_industry', 'r_man_industry', 'offshor_industry']].sum().reset_index()

# %%
# BLS encodes a small subset of industry codes as ranges, which are disambiguated by replicating the row
# and using the individual industry code.
rti_by_industry_code["industry_code"] = rti_by_industry_code["industry_code"].apply(expand_naics_code_range)
rti_by_industry_code = rti_by_industry_code.explode("industry_code", ignore_index=True)

# %%
# For industry codes that are repeated, take the mean of the rti values
num_repeated_codes = rti_by_industry_code.duplicated(subset="industry_code", keep=False).sum()

if num_repeated_codes > 0:
    print(f"Conslidating {num_repeated_codes} repeated industry codes")
    rti_by_industry_code = rti_by_industry_code.groupby(by=['industry_code'])[['r_cog_industry', 'r_man_industry', 'offshor_industry']].mean().reset_index()

# %%
# Add sector, subsector, industry group, and NAICS Industry codes
# data-source: https://www.census.gov/programs-surveys/economic-census/year/2017/economic-census-2017/guidance/understanding-naics.html#par_textimage_0
rti_by_industry_code['sector_code'] = rti_by_industry_code['industry_code'].apply(lambda x : x[:2] + '0000')
rti_by_industry_code['subsector_code'] = rti_by_industry_code['industry_code'].apply(lambda x : x[:3] + '000')
rti_by_industry_code['industry_group_code'] = rti_by_industry_code['industry_code'].apply(lambda x : x[:4] + '00')
rti_by_industry_code['naics_industry_code'] = rti_by_industry_code['industry_code'].apply(lambda x : x[:5] + '0')


# %%
# Save as csv files
rti_by_industry_code.to_csv("data/processed/rti_by_industry.csv", index=False)
rti_by_occupation_code.to_csv("data/processed/rti_by_occupation.csv", index=False)

print("Script finished!")
