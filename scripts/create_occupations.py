import pandas as pd

soc_structure = pd.read_excel('data/raw/soc_codes_2018.xlsx', skiprows=7)

soc_structure = soc_structure.rename(columns={
    'Major Group': 'major_group',
    'Minor Group': 'minor_group',
    'Broad Group': 'broad_group',
    'Detailed Occupation': 'detailed_occupation',
    'Unnamed: 4': 'occupation_title'
})

soc_structure['detailed_occupation'] = soc_structure['detailed_occupation'].str.replace('-', '').astype('Int64')
soc_structure['major_group'] = soc_structure['major_group'].str.replace('-', '').astype('Int64')
soc_structure['minor_group'] = soc_structure['minor_group'].str.replace('-', '').astype('Int64')
soc_structure['broad_group'] = soc_structure['broad_group'].str.replace('-', '').astype('Int64')

soc_structure = soc_structure.melt(
    id_vars="occupation_title", 
    value_vars=[
        "major_group", 
        "minor_group", 
        "broad_group",
        "detailed_occupation"
    ],
    var_name="soc_level",
    value_name="occupation_code"
).dropna(subset="occupation_code").reset_index(drop=True)

soc_structure['occupation_code_prefix'] = soc_structure['occupation_code'].astype(str).str[:2]

soc_structure.to_csv("data/processed/occupations.csv")