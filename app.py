import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.figure_factory as ff

'''
# Retrainability Index

_A prototype._
'''


df = pd.read_csv("data.csv", low_memory=False)

total_rows = len(df)

with st.expander("Configurations"):

    weights = np.array([1, 1, 1]) / 3

    weight1 = st.slider("diff_r_cog_y_norm", 0.0, 1.0, value=weights[0])
    weight2 = st.slider("diff_offshor_y_norm", 0.0, 1.0, value=weights[1])
    weight3 = st.slider("wage_mean_ihs_diff_y_norm", 0.0, 1.0, value=weights[2])

    # Normalize weights so they sum to 1
    weight_array = np.array([weight1, weight2, weight3])
    weight_sum = weight_array.sum()

    # Avoid division by zero
    if weight_sum == 0:
        normalized_weights = np.array([1, 1, 1]) / 3
    else:
        normalized_weights = weight_array / weight_sum

    # Use normalized weights
    df["index_y"] = (
        normalized_weights[0] * df["diff_r_cog_industry_y_norm"] + 
        normalized_weights[1] * df["diff_offshor_industry_y_norm"] + 
        normalized_weights[2] * df["wage_mean_ihs_diff_y_norm"]
    )

    df["index_y"] = df["index_y"] - df["index_y"].mean()

    st.write("Normalized weights:", normalized_weights)

    col1, col2, col3 = st.columns(3, vertical_alignment="center")

    with col1:
        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            [
                "All",
                "American Indian or Alaska Native (not Hispanic)",
                "Asian (not Hispanic)",
                "Black (not Hispanic)",
                "Hispanic",
                "Multiple Race (not Hispanic)",
                "Native Hawaiian or Pacific Islander (not Hispanic)",
                "Unknown",
                "White (not Hispanic)"
            ]
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            [
                "All",
                "Male",
                "Female",
                "Participant did not self-identify",
                "Unknown"
            ]
        )

    with col3:
        states = list(df["state_x"].unique())
        state = st.selectbox(
            "State",
            ["All"] + states
        )

    education_levels = df["highest_education_level_x"].unique()
    education_levels.sort()
    education_levels = list(education_levels)
    education_level = st.selectbox(
        "Education Level",
        ["All"] + education_levels
    )

    occupation_titles = list(df["occupation_title_x"].unique())
    occupation_title = st.selectbox(
        "Occupation",
        ["All"] + occupation_titles
    )
    trainings = list(df["received_training_x"].unique())
    received_training = st.selectbox(
        "Training",
        ["All"] + trainings
    )
    training_service_types = list(df["training_service_type_1_x"].unique())
    training_service_type = st.selectbox(
        "Service Type",
        ["All"] + training_service_types
    )

    # Group data together
    if (race_ethnicity != "All"):
        isRaceEthnicity = df.race_ethnicity_x == race_ethnicity
        df = df.loc[isRaceEthnicity]

    if (sex != "All"):
        isSex = df.sex_x == sex
        df = df.loc[isSex]

    if (state != "All"):
        isState = df.state_x == state
        df = df.loc[isState]

    if (education_level != "All"):
        isEducationLevel = df.highest_education_level_x == education_level
        df = df.loc[isEducationLevel]

    if (occupation_title != "All"):
        isOccupationTitle = df.occupation_title_x == occupation_title
        df = df.loc[isOccupationTitle]

    if (received_training != "All"):
        isReceivedTraining = df.received_training_x == received_training
        df = df.loc[isReceivedTraining]

    if (training_service_type != "All"):
        isTrainingServiceType = df.training_service_type_1_x == training_service_type
        df = df.loc[isTrainingServiceType]

    hist_data = [df["index_y"].dropna().to_numpy()]

    selected_rows = len(hist_data[0])

    st.write(f"Number of records: {selected_rows} ({100 * selected_rows / total_rows:0.1f}%).")

# Create distplot with custom bin_size
fig = ff.create_distplot(
    hist_data, 
    group_labels=[race_ethnicity], 
    bin_size=0.01, 
    show_hist=False
)


# Calculate the median of the data
median_value = np.median(hist_data[0]) 

# Add vertical line with annotation including median value
fig.add_vline(
    x=median_value,
    line_dash="dash",
    line_color="black",
    annotation_text=f"Median: {median_value:.2f}",  # format to 2 decimal places
    annotation_position="top right"
)

# Clamp x-axis
fig.update_xaxes(range=[-4, 4])

# Remove legend
fig.update_layout(showlegend=False)


# Plot!
st.plotly_chart(fig)