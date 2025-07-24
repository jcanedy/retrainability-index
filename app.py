import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px

'''
# Retrainability Index

_A prototype._
'''


df = pd.read_csv("data/processed/index.csv", low_memory=False)

with st.expander("Configurations"):

    weights = np.array([1, 1, 1, 1]) / 4

    weight1 = st.slider("bin_r_cog_industry_y mean", 0.0, 1.0, value=weights[0])
    weight2 = st.slider("bin_r_man_industry_y mean", 0.0, 1.0, value=weights[1])
    weight3 = st.slider("bin_offshor_industry_y mean", 0.0, 1.0, value=weights[2])
    weight4 = st.slider("bin_wages_mean_y mean", 0.0, 1.0, value=weights[3])

    # Normalize weights so they sum to 1
    weight_array = np.array([weight1, weight2, weight3, weight4])
    weight_sum = weight_array.sum()

    # Avoid division by zero
    if weight_sum == 0:
        normalized_weights = np.array([1, 1, 1, 1]) / 4
    else:
        normalized_weights = weight_array / weight_sum

    # Use normalized weights
    df["index_y"] = (
        normalized_weights[0] * df["bin_r_cog_industry_y mean"] + 
        normalized_weights[1] * df["bin_r_man_industry_y mean"] + 
        normalized_weights[2] * df["bin_offshor_industry_y mean"] +
        normalized_weights[3] * df["bin_wages_mean_y mean"]
    )

    # df["index_y"] = df["index_y"] - df["index_y"].mean()

    st.write("Normalized weights:", normalized_weights)

    col1, col2, col3 = st.columns(3, vertical_alignment="center")

    with col1:
        race_ethnicities = df["race_ethnicity_x"].unique()
        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            race_ethnicities
        )

    with col2:
        sexes = df["sex_x"].unique()
        sex = st.selectbox(
            "Sex",
            sexes
        )

    with col3:
        received_trainings = list(df["received_training_x"].unique())
        received_training = st.selectbox(
            "Received Training",
            received_trainings
        )


    low_incomes = list(df["low_income_x"].unique())
    low_income = st.selectbox(
        "Low Income",
        low_incomes
    )

    employment_status_options = list(df["employment_status_x"].unique())
    employment_status = st.selectbox(
        "Employment Status",
        employment_status_options
    )

    age_options = list(df["age_x"].unique())
    age = st.selectbox(
        "Age",
        age_options
    )

    highest_education_level_options = list(df["highest_education_level_x"].unique())
    highest_education_level = st.selectbox(
        "Highest Education Level",
        highest_education_level_options
    )

    training_service_types = list(df["training_service_type_1_x"].unique())
    training_service_type = st.selectbox(
        "Training Service Type",
        training_service_types
    )

    # Group data together
    isRaceEthnicity = df.race_ethnicity_x.isin([race_ethnicity])
    isSex = df.sex_x.isin([sex])
    isReceivedTraining = df.received_training_x.isin([received_training])
    isLowIncome = df.low_income_x.isin([low_income])
    isEmploymentStatus = df.employment_status_x.isin([employment_status])
    isAge = df.age_x.isin([age])
    isHighestEducationLevel = df.highest_education_level_x.isin([highest_education_level])
    isTrainingServiceType = df.training_service_type_1_x.isin([training_service_type])
    

    # Include All as a comparison group
    isRaceEthnicityAll = df.race_ethnicity_x == "All"
    isSexAll = df.sex_x == "All"
    isReceivedTrainingAll = df.received_training_x == "All"
    isLowIncomeAll = df.low_income_x == "All"
    isEmploymentStatusAll = df.employment_status_x == "All"
    isAgeAll = df.age_x == "All"
    isHighestEducationLevelAll = df.highest_education_level_x == "All"
    isTrainingServiceTypeAll = df.training_service_type_1_x == "All"


    selected_df = df[(isRaceEthnicity 
                     & isSex 
                     & isReceivedTraining 
                     & isLowIncome
                     & isEmploymentStatus
                     & isAge
                     & isHighestEducationLevel
                     & isTrainingServiceType) |
                     (isRaceEthnicityAll 
                     & isSexAll
                     & isReceivedTrainingAll
                     & isLowIncomeAll
                     & isEmploymentStatusAll
                     & isAgeAll
                     & isHighestEducationLevelAll
                     & isTrainingServiceTypeAll)
                     ]
    
    selected_df["selection"] = ""
    
    selected_df.loc[(isRaceEthnicity 
                     & isSex 
                     & isReceivedTraining 
                     & isLowIncome
                     & isEmploymentStatus
                     & isAge
                     & isHighestEducationLevel
                     & isTrainingServiceType), "selection"] = "Selection"
        
    selected_df.loc[(isRaceEthnicityAll 
                     & isSexAll
                     & isReceivedTrainingAll
                     & isLowIncomeAll
                     & isEmploymentStatusAll
                     & isAgeAll
                     & isHighestEducationLevelAll
                     & isTrainingServiceTypeAll), "selection"] = "All"
    
    selected_rows = selected_df[selected_df["selection"] == "Selection"]["bin_wages_mean_y count"].sum()
    total_rows = selected_df[selected_df["selection"] == "All"]["bin_wages_mean_y count"].sum()
    
    
    selected_df = selected_df.melt(id_vars=["selection"],
                     value_vars=["bin_r_cog_industry_y mean", 
                                 "bin_r_man_industry_y mean", 
                                 "bin_offshor_industry_y mean",
                                 "bin_wages_mean_y mean",
                                 "index_y"])


st.write(f"Number of records: {selected_rows} of {total_rows} ({100 * selected_rows / total_rows:0.1f}%).")

fig = px.bar(selected_df, x="variable", y="value", color="selection", barmode="group")


# # Create distplot with custom bin_size
# fig = ff.create_distplot(
#     hist_data, 
#     group_labels=[race_ethnicity], 
#     bin_size=0.01, 
#     show_hist=False
# )


# # Add vertical line with annotation including median value
# fig.add_vline(
#     x=median_value,
#     line_dash="dash",
#     line_color="black",
#     annotation_text=f"Median: {median_value:.2f}",  # format to 2 decimal places
#     annotation_position="top right"
# )

# # Clamp x-axis
# fig.update_xaxes(range=[-4, 4])

# # Remove legend
# fig.update_layout(showlegend=False)


# Plot!
st.plotly_chart(fig)