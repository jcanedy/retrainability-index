import pandas as pd

df = pd.read_excel("data/raw/WCM_StateDashboard_2023All_Regions_All_States.xlsx", skiprows=8)

df['jurisdiction'] = df.apply(lambda x : x['Jurisdiction Name'] + ', ' + x['State/Territory'] , axis=1)
county_names = df['jurisdiction'].loc[:5].tolist()

# Fetch the corresponding DCIDs for the juristictions
counties = client.resolve.fetch_dcids_by_name(county_names).to_flat_dict()

# Extract just the DCIDs
county_dcids = [counties[county_name] for county_name in county_names]

df_county = client.observations_dataframe(entity_dcids=county_dcids, variable_dcids=['Count_Person', 'Median_Age_Person', 'Median_Income_Person'], date='latest')
df.to_csv("data/processed/workforce_boards.csv")

