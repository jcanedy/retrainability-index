import pandas as pd

#data-source: https://www.bls.gov/emp/data/occupational-data.htm
national_employment_matrix = pd.read_excel('data/raw/national_employment_matrix.xlsx')

#data-source: https://www.dropbox.com/scl/fi/86paxjkauqtg72o73gp2h/task-construct-1.zip?dl=0&e=1&rlkey=o21usoigscnrki3r77oo610pg
rti_by_occupation_code =  pd.read_stata('data/raw/rti_by_occupation_code.dta')

rti_by_occupation_code = rti_by_occupation_code.rename(
    columns={
        "onetsoccode": "occupation_code"
    }
)
rti_by_occupation_code["occupation_code"] = rti_by_occupation_code["occupation_code"].astype('Int64')

matrix = national_employment_matrix.copy()


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

isOccupationTypeLineItem = matrix["occupation_type"] == "Line item"
isIndustryTypeLineItem = matrix["industry_type"] == "Line item"

matrix = matrix.loc[isOccupationTypeLineItem, [
    "industry_code",
    "industry_title",
    "occupation_code",
    "2023_percent_of_industry"
    ]]

matrix['2023_percent_of_industry_norm'] = matrix.groupby('industry_code')['2023_percent_of_industry'].transform(
    lambda x: x / x.sum()
)

matrix["occupation_code"] = matrix["occupation_code"].str.replace('-', '').astype('Int64')
matrix = matrix.merge(rti_by_occupation_code, how="left", on="occupation_code")
matrix = matrix.dropna(subset=["r_cog", "r_man", "offshor"])

matrix['r_cog_industry'] = matrix['2023_percent_of_industry_norm'] * matrix['r_cog']
matrix['r_man_industry'] = matrix['2023_percent_of_industry_norm'] * matrix['r_man']
matrix['offshor_industry'] = matrix['2023_percent_of_industry_norm'] * matrix['offshor']

rti_by_industry_code = matrix.groupby(['industry_code', 'industry_title'])[['r_cog_industry', 'r_man_industry', 'offshor_industry']].sum().reset_index()

rti_by_industry_code['industry_code'] = rti_by_industry_code['industry_code'].apply(lambda x : x[:3] + '000')
rti_by_industry_code['industry_code_prefix'] = rti_by_industry_code['industry_code'].apply(lambda x : x[:2] + '0000')

rti_by_industry_code = rti_by_industry_code.groupby(by=['industry_code', 'industry_code_prefix'])[['r_cog_industry', 'r_man_industry', 'offshor_industry']].mean().reset_index()

rti_by_industry_code = (
    rti_by_industry_code.merge(
    matrix[['industry_code', 'industry_title']].drop_duplicates(), 
    left_on='industry_code_prefix', right_on='industry_code', 
    how='left')
    .dropna(subset=['industry_title'])
    .rename(columns={
        "industry_code_x": "industry_code"
    })
    .drop(columns=["industry_code_y"])
)

rti_by_industry_code.to_csv("data/processed/rti_by_industry.csv", index=False)
rti_by_occupation_code.to_csv("data/processed/rti_by_occupation.csv", index=False)