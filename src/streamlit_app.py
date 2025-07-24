import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

'''
# Retrainability Index
v0.0.2

The Retainability Index is composite metric designed to evaluate how effectively programs are helping workers access retraining, gain future-ready skills, and secure quality employment. 
Built from WIOA performance data, the index highlights demographic differences in program outcomes to identify where retraining efforts are leading to positive retraining.
'''

methodology_expander = st.expander("Methodology")

with methodology_expander:
    '''

    These metrics capture changes in job characteristics at the industry level, comparing pre- and post-program outcomes. Each indicator is computed by aggregating data across occupations within an industry. 
    Specifically, we use a composite measures of routine task intensity and offshorability derived from the occupational mix of each industry. 
    The values are weighted by occupational employment shares to reflect the dominant task content and characteristics of each sector. 
    All metrics are transformed so that higher values consistently indicate more desirable or positive changes post-program.

    - **`bin_r_cog_industry_y mean`**: Represents the average change in cognitive routine task intensity, where higher values indicate a shift toward *less cognitively routine* (i.e., more cognitively complex) work post-program.

    - **`bin_r_man_industry_y mean`**: Measures the average change in manual routine task intensity. Higher values reflect a movement away from *manual routine* work toward more varied or skilled tasks after the program.

    - **`bin_offshor_industry_y mean`**: Captures the average change in offshorability. Higher values imply a shift toward *less offshorable* work — suggesting increased job embeddedness or resilience post-program.

    - **`bin_wages_mean_y mean`**: Indicates the average change in wage levels. Higher values represent *increased wages* following the program.

    - **`index_y`**: A composite indicator summarizing positive changes across routine intensity, offshorability, and wages. Higher values denote an *overall positive impact* of the program on job quality and task characteristics.

    '''


df = pd.read_csv("data/processed/index.csv", low_memory=False)

with st.expander("Index Configurations"):

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

    # Unpack weights
    w1, w2, w3, w4 = normalized_weights

    # Construct the LaTeX formula with actual weights
    latex_formula = rf'''
    \text{{Index}}_y = {w1:.2f} \cdot \overline{{r_{{\text{{cog}},y}}}} + 
                    {w2:.2f} \cdot \overline{{r_{{\text{{man}},y}}}} + 
                    {w3:.2f} \cdot \overline{{r_{{\text{{offshor}},y}}}} + 
                    {w4:.2f} \cdot \overline{{w_{{\text{{mean}},y}}}}
    '''

    # Display the formula
    st.latex(latex_formula)

container = st.container(border=True)

with container:

    '''
    ### Participant Demographics
    '''

    row1 = st.columns(3, vertical_alignment="center")

    with row1[0]:
        race_ethnicities = df["race_ethnicity_x"].unique()
        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            race_ethnicities
        )

    with row1[1]:
        sexes = df["sex_x"].unique()
        sex = st.selectbox(
            "Sex",
            sexes
        )

    with row1[2]:
        age_options = list(df["age_x"].unique())
        age = st.selectbox(
            "Age",
            age_options
        )

    row2 = st.columns(3, vertical_alignment="center")

    with row2[0]:
        highest_education_level_options = list(df["highest_education_level_x"].unique())
        highest_education_level = st.selectbox(
            "Highest Education Level",
            highest_education_level_options
        )

    with row2[1]:
        low_incomes =  sorted(list(df["low_income_x"].unique()), key=lambda x: (x != "All", x))
        low_income = st.selectbox(
            "Low Income",
            low_incomes
        )
    with row2[2]:
        employment_status_options = list(df["employment_status_x"].unique())
        employment_status = st.selectbox(
            "Employment Status",
            employment_status_options
        )


    '''
    ### Program Information
    '''
    row3 = st.columns(3, vertical_alignment="center")

    with row3[0]:
        received_trainings = list(df["received_training_x"].unique())
        received_training = st.selectbox(
            "Received Training",
            received_trainings
        )

    with row3[1]:
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
                    & isTrainingServiceType)
                    ]

all_df = df[(isRaceEthnicityAll 
                    & isSexAll
                    & isReceivedTrainingAll
                    & isLowIncomeAll
                    & isEmploymentStatusAll
                    & isAgeAll
                    & isHighestEducationLevelAll
                    & isTrainingServiceTypeAll)
                    ]


selected_df["Group"] = "Selected Participants"
    
all_df["Group"] = "All Participants"

selected_rows = selected_df["bin_wages_mean_y count"].sum()
total_rows = all_df["bin_wages_mean_y count"].sum()

plot_df = pd.concat([selected_df, all_df], ignore_index=True)


plot_df = plot_df.melt(id_vars=["Group"],
                    value_vars=["bin_r_cog_industry_y mean", 
                                "bin_r_man_industry_y mean", 
                                "bin_offshor_industry_y mean",
                                "bin_wages_mean_y mean",
                                "index_y"],
                    var_name="Statistic",
                    value_name="Value")


'''
Based on your selections, the chart below compares these metrics to all participants in the 10% sample dataset. 
The sample is a subset and may not reflect all regional or program-specific nuances of the full WIOA dataset.

'''

left, center, right = st.columns(3)

left.metric(label="Percent of Participants", value=f"{100 * selected_rows / total_rows:0.1f}%")
center.metric(label="Selected Participants (Sample)", value=f"{selected_rows:0.0f}")
right.metric(label="All Participants (Sample)", value=f"{total_rows:0.0f}")


# fig = px.bar(plot_df, x="Statistic", y="Value", color="Group", barmode="group")


# # Clamp x-axis
# fig.update_xaxes(range=[-4, 4])

# # Plot
# st.plotly_chart(fig)

all_plot_df = plot_df[plot_df["Group"] == "All Participants"]
selected_plot_df = plot_df[plot_df["Group"] == "Selected Participants"]

# Merge the two dataframes on the Statistic column
dumbbell_df = all_plot_df.merge(
    selected_plot_df,
    on="Statistic",
    suffixes=("_all", "_selected")
)

fig = go.Figure()

# Add the dumbbell lines (connecting All to Selected)
for _, row in dumbbell_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["Value_all"], row["Value_selected"]],
        y=[row["Statistic"], row["Statistic"]],
        mode="lines",
        line=dict(color="grey"),
        showlegend=False,
    ))

# Add the Selected Participants dots
fig.add_trace(go.Scatter(
    x=dumbbell_df["Value_selected"],
    y=dumbbell_df["Statistic"],
    mode="markers",
    name="Selected Participants",
    marker=dict(color="green", size=10)
))

# Add the All Participants dots
fig.add_trace(go.Scatter(
    x=dumbbell_df["Value_all"],
    y=dumbbell_df["Statistic"],
    mode="markers",
    name="All Participants",
    marker=dict(color="grey", size=10)
))


fig.update_layout(
    title="All vs Selected Participants",
    xaxis_title="Percent of Participants",
    yaxis_title="Statistic",
    template="simple_white"
)

# Plot
st.plotly_chart(fig)