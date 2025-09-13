from google.cloud import bigquery
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st

from functools import reduce
import operator


'''
# Retrainability Index
v0.0.7 _(research prototpye)_

_Author(s): Julian Jacobs ([LinkedIn](https://www.linkedin.com/in/julian-jacobs-a729b87a/), [Website](https://www.juliandjacobs.com/)), Jordan Canedy-Specht ([LinkedIn](https://www.linkedin.com/in/jordancanedy/), [Github](https://github.com/jcanedy))_

_Code Repository: <https://github.com/jcanedy/retrainability-index>_

This AI ‘Retrainability’ Index is designed to visualize and quantify how well United States public retraining programs help workers reskill in the face of technological automation. 

### Why study retrainability?

In this project, we look at outcomes for the NUMBER of Americans from YEAR RANGE who participated in the retraining offered through the [Workforce Innovation and Opportunity Act (WIOA)](https://www.dol.gov/agencies/eta/performance). 

Policymakers - both historically and today - often regard worker retraining (or reskilling) as an effective policy response to support workers who have been displaced by technological change. And yet, there is currently limited research studying the efficacy of these retraining programs in supporting technology labor market adjustment.

This project’s core contribution is the development of a ‘retrainability index’, which calculates at the individual level a person’s likelihood of successfully retraining into work that is both better compensated and more resilient in the face of automation than their prior employment. 

We proxy for automation exposure through routine task intensity (RTI) - a measure of how ‘routine’ the cognitive and manual work a particular job is, which has been [used](https://shapingwork.mit.edu/research/skills-tasks-and-technologies-implications-for-employment-and-earnings/) by social scientists for decades. Generally, more routine cognitive and manual tasks have been more at risk of automation.

It is worth noting here that, since we are looking at WIOA performance data from YEAR RANGE, this tool is necessarily backwards looking. We are looking at how well US public retraining has supported worker adjustment to digitalization shocks, for instance, not the potential forthcoming AI labor market shock. This backward focus is intentional - AI economic effects are only beginning to emerge, and existing variables of AI exposure tend to primarily capture AI complementarity, as opposed to displacement, risks. 

Our Retrainability Index is a composite of how much retraining impacts wages and the routinization of work that retainers perform. 
'''

with st.expander("How to use this tool"):
    '''
    Like automation risk, ‘retrainability’ is not spread evenly across American demographics and geography. 

    Automation risk tells us how likely it is a person will lose their job or experience wage stagnation due to technological change.

    On the other hand, the retrainability index tells us how likely it is that a person will become more upwardly mobile in the face of automation, through US public retraining programs. 

    For the most part, US public retraining participants serve vulnerable lower income Americans, many of whom have been long - term unemployed. As a result, the population represented in this research is not representative of the broader United States. However, it nonetheless allows us to paint a picture of how some of America's most vulnerable workers have fared through public retraining, pointing to where future investment and programmatic improvements might be necessary. 
    '''

TABLE = 'index_tier2_v0_0_8'

@st.cache_data(ttl=None, show_spinner="Loading")
def get_unique_values_for_columns(columns: list[str]) -> dict:
    # Build SELECT clause dynamically
    select_clauses = [
        f"ARRAY_AGG(DISTINCT {col} ORDER BY {col}) AS {col}" for col in columns
    ]
    select_sql = ",\n  ".join(select_clauses)

    query = f"""
        SELECT
          {select_sql}
        FROM `retraining-index.processed.{TABLE}`
    """

    client = bigquery.Client()
    df = client.query(query).to_dataframe()

    # The result will be a single-row DataFrame with arrays as columns
    result = {col: df[col][0] for col in columns}
    return result


@st.cache_data(ttl=None, show_spinner="Loading")
def get_single_row(filters: dict):
    where_clauses = []
    parameters = []

    for key, value in filters.items():
        if value:
            where_clauses.append(f"{key} = @{key}")
            parameters.append(bigquery.ScalarQueryParameter(key, "STRING", value))

    where_sql = " AND ".join(where_clauses) or "TRUE"

    query = f"""
        SELECT *
        FROM `retraining-index.processed.{TABLE}`
        WHERE {where_sql}
        LIMIT 1
    """

    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(query_parameters=parameters)
    return client.query(query, job_config=job_config).to_dataframe()

# Define mapping from input column names to output variable names with '_options' suffix
columns = {
    "race_ethnicity_x": "race_ethnicity_options",
    "sex_x": "sex_options",
    "age_x": "age_options",
    "highest_education_level_x": "highest_education_level_options",
    "low_income_x": "low_income_options",
    "employment_status_x": "employment_status_options",
    "state_x": "state_options",
    "industry_title_x": "industry_title_options",
    "industry_title_y": "exit_industry_title_options",
    "funding_stream_x": "funding_stream_options", 
}

column_options = get_unique_values_for_columns([
    "race_ethnicity_x",
    "sex_x",
    "age_x",
    "highest_education_level_x",
    "low_income_x",
    "employment_status_x",
    "state_x",
    "industry_title_x",
    "industry_title_y",
    "funding_stream_x",
])

def sort_with_all_other(options, custom_order=None):
    options_set = set(options)
    
    # Separate out "All" and "Other" if they exist
    all_present = "All" in options_set
    other_present = "Other" in options_set

    # Remove them for sorting
    core = [opt for opt in options if opt not in {"All", "Other"}]

    # Sort core either by custom order or alphabetically
    if custom_order:
        order_map = {val: idx for idx, val in enumerate(custom_order)}
        core.sort(key=lambda x: order_map.get(x, len(order_map)))
    else:
        core.sort()

    return (["All"] if all_present else []) + core + (["Other"] if other_present else [])

# Use custom sort for age
age_custom_order = [
    "17 to 24 years",
    "25 to 34 years",
    "35 to 44 years",
    "45 to 54 years",
    "55 to 64 years",
    "65 to 84 years"
]
age_options = sort_with_all_other(column_options["age_x"], age_custom_order)

# Use custom sort for education
education_custom_order = [
    "No Educational Level Completed",
    "Attained secondary school diploma",
    "Attained a secondary school equivalency",
    "Attained a postsecondary technical or vocational certificate (non-degree)",
    "Completed one of more years of postsecondary education",
    "Attained an Associate's degree",
    "Attained a Bachelor's degree",
    "Attained a degree beyond a Bachelor's degree"
]
highest_education_level_options = sort_with_all_other(column_options["highest_education_level_x"], education_custom_order)

# Apply default sort (alphabetical + All/Other positioning) to all others
race_ethnicity_options = sort_with_all_other(column_options["race_ethnicity_x"])
sex_options = sort_with_all_other(column_options["sex_x"])
low_income_options = sort_with_all_other(column_options["low_income_x"])
employment_status_options = sort_with_all_other(column_options["employment_status_x"])
state_options = sort_with_all_other(column_options["state_x"])
industry_title_options = sort_with_all_other(column_options["industry_title_x"])
exit_industry_title_options = sort_with_all_other(column_options["industry_title_y"])
funding_stream_options = sort_with_all_other(column_options["funding_stream_x"])


# Config: field name -> label for display
sidebar_fields = [
    ("race_ethnicity_x", "Race / Ethnicity"),
    ("sex_x", "Sex"),
    ("age_x", "Age"),
    ("highest_education_level_x", "Highest Education Level"),
    ("low_income_x", "Low Income"),
    ("employment_status_x", "Employment Status"),
    ("state_x", "State"),
    ("industry_title_x", "Entry Industry Code"),
    ("industry_title_y", "Exit Industry Code"),
    ("funding_stream_x", "Funding Stream"),
]

# Use the *_options variables already created
options_lookup = {
    "race_ethnicity_x": race_ethnicity_options,
    "sex_x": sex_options,
    "age_x": age_options,
    "highest_education_level_x": highest_education_level_options,
    "low_income_x": low_income_options,
    "employment_status_x": employment_status_options,
    "state_x": state_options,
    "industry_title_x": industry_title_options, 
    "industry_title_y": exit_industry_title_options,
    "funding_stream_x": funding_stream_options,
}

# Store selected values here
selections = {}

with st.sidebar:
    st.markdown("### Participant Demographics")

    # Handle first 6 fields (3 rows of 2 columns)
    
    for i in range(0, 6, 2):
        col1, col2 = st.columns(2)
        field1, label1 = sidebar_fields[i]
        field2, label2 = sidebar_fields[i + 1]

        with col1:
            selections[field1] = st.selectbox(label1, options_lookup[field1])
        with col2:
            selections[field2] = st.selectbox(label2, options_lookup[field2])

    st.markdown("### Program Information")

    col1, col2 = st.columns(2)

    with col1:

        field, label = sidebar_fields[6]
        selections[field] = st.selectbox(label, options_lookup[field])

    with col2:

        field, label = sidebar_fields[9]
        selections[field] = st.selectbox(label, options_lookup[field])   

    st.markdown("### Prior Employment")

    col1, col2 = st.columns(2)

    field1, label1 = sidebar_fields[7]
    field2, label2 = sidebar_fields[8]

    with col1:
        selections[field1] = st.selectbox(
           label1, options_lookup[field1]
        )

    with col2:
        selections[field2] = st.selectbox(
           label2, options_lookup[field2]
        )

all = {
    "race_ethnicity_x":"All",
    "sex_x":"All","age_x":"All",
    "highest_education_level_x":"All",
    "low_income_x":"All",
    "employment_status_x":"All",
    "state_x":"All",
    "industry_title_x":"All",
    "industry_title_y":"All",
    "funding_stream_x":"All",
}

results_all = get_single_row(all)
results_all["Group"] = "All Participants"

results_selections = get_single_row(selections)
results_selections["Group"] = "Selected Participants"

results = pd.concat([results_all, results_selections], ignore_index=True)

#TODO(@jcanedy27): Move weight calculation to `create_index.py`

weights = np.array([0.25, 0.25, 0.5])

# Unpack weights
w1, w2, w3 = weights

results["r_cog_adj"] = (-2 * results["bin_r_cog_industry_y mean"] - 0.5)
results["r_man_adj"] = (-2 * results["bin_r_man_industry_y mean"] - 0.5)
results["wages_mean_adj"] = (2 * results["bin_wages_mean_y mean"] - 0.5)

results["Index"] = (
    w1 * results["r_cog_adj"]
    + w2 * results["r_man_adj"]
    + w3 * results["wages_mean_adj"]
)

isAllParticipants = results.Group == "All Participants"

# Extract the index
index_all = results.loc[isAllParticipants, "Index"].item()
index_selections = results.loc[~isAllParticipants, "Index"].item()

# Extract the count sums
count_all = results_all["count"].item()
count_selections = results_selections["count"].item()

results = (
    results
    .rename(columns={
        "bin_r_cog_industry_y mean": "Higher Routine Cognitive Exposure",
        "bin_r_man_industry_y mean": "Higher Routine Manual Exposure ", 
        "bin_wages_mean_y mean": "Wage Gain",

        "diff_r_cog_industry_y median": "Median Routine Cognitive Exposure",
        "diff_r_cog_industry_y 25th": "25th Routine Cognitive Exposure",
        "diff_r_cog_industry_y 75th": "75th Routine Cognitive Exposure",
        "diff_r_cog_industry_y mean": "Mean Routine Cognitive Exposure",

        "diff_r_man_industry_y median": "Median Routine Manual Exposure ",
        "diff_r_man_industry_y 25th": "25th Routine Manual Exposure ",
        "diff_r_man_industry_y 75th": "75th Routine Manual Exposure ",
        "diff_r_man_industry_y mean": "Mean Routine Manual Exposure ",

        "diff_wages_mean_y median": "Median Wage Gain",
        "diff_wages_mean_y 25th": "25th Wage Gain",
        "diff_wages_mean_y 75th": "75th Wage Gain",
        "diff_wages_mean_y mean": "Mean Wage Gain",

        "count": "Count",
    })
    .melt(
        id_vars=["Group"],
        value_vars=["Higher Routine Cognitive Exposure", "Higher Routine Manual Exposure ", "Wage Gain",
            "Median Routine Cognitive Exposure", "Median Routine Manual Exposure " , "Median Wage Gain",
            "25th Routine Cognitive Exposure", "25th Routine Manual Exposure " , "25th Wage Gain",
            "75th Routine Cognitive Exposure", "75th Routine Manual Exposure " , "75th Wage Gain",
            "Mean Routine Cognitive Exposure", "Mean Routine Manual Exposure ", "Mean Wage Gain"],
        var_name="Statistic",
        value_name="Value"
    )
)

# Convert to pandas if needed for plotting
plot_df = results

tab1, tab2, tab3 = st.tabs(["Index", "Methodology", "Version History"])

with tab2:
    st.markdown(
        """
        These statistics assess the **post-program job quality** of WIOA participants based on their employment outcomes by industry. Each metric is expressed as a **proportion between 0 and 1**, representing the share of participants meeting a specific condition. These values are directly interpretable as percentages (e.g., 0.65 = 65%).

        ### Statistics

        - **Routine Cognitive Exposure** (`rce`):  
        Proportion of participants employed in industries with **high cognitive routine** task intensity (e.g., clerical, administrative support).  
        Higher values imply more exposure to automation-prone cognitive tasks.

        - **Routine Manual Exposure** (`rme`):  
        Proportion of participants employed in industries with **high manual routine** task intensity (e.g., machine operation, basic manufacturing).  
        Higher values indicate greater exposure to physically repetitive tasks vulnerable to automation.

        - **Wage Gain** (`wg`):  
        Proportion of participants who experienced an **increase in wages** post-program compared to pre-program.  
        A higher value signals stronger economic mobility.

        These raw values are **not transformed** for reporting purposes, and are intended to be easily interpretable by practitioners.

        ### Transformation

        To combine the statistics into a single composite score, each is **transformed to range from –1 to 1**, centered at 0. This allows for consistent directionality, weighting, and comparison across components.

        The transformation formula is:

        \n
        $$
        x' = 2(x - 0.5)
        $$

        - This maps 0.5 (neutral) to 0  
        - Values above 0.5 become positive (better than neutral)  
        - Values below 0.5 become negative (worse than neutral)

        For **Routine Exposure metrics** (`rce`, `rme`), the transformed values are **inverted (multiplied by –1)** so that higher index scores always represent *better* outcomes (i.e., less routine exposure).

        ### Composite Index

        The **Index Score** (`index`) is a weighted average of the transformed metrics. It captures overall post-program job quality, with higher scores reflecting:

        - Less routine task exposure  
        - Greater wage gains  

        The index ranges from –1 (least favorable) to +1 (most favorable).

        """
    )

    st.latex(
        rf"""
        \text{{Index}} = {w1:.2f} \cdot \text{{rce}}' \ +\ 
                        {w2:.2f} \cdot \text{{rme}}' \ +\ 
                        {w3:.2f} \cdot \text{{wg}}'
        """
    )


with tab1:
    # Format user selections
    selected_filters = {
        label: selections[field]
        for field, label in sidebar_fields
        if selections[field] != "All"
    }

    if selected_filters:
        selection_description = "; ".join(f"**{label}**: {value}" for label, value in selected_filters.items())
    else:
        selection_description = "*No subgroup filters applied. Displaying all participants.*"

    # Markdown paragraph with academic tone + selection summary
    st.markdown(
        f"""
        The chart below compares retraining-related outcomes for the selected subgroup of participants to those of all participants 
        in the WIOA dataset.

        **Selected Filters:** {selection_description}
        """
    )

    # --- Index Display ---
    is_index = plot_df['Statistic'] == "index_y"
    index_df = plot_df[is_index][["Group", "Value"]].set_index("Group")

    is_quantile = plot_df['Statistic'].str.startswith(("25th", "Median", "75th"))
    is_mean = plot_df['Statistic'].str.startswith(("Mean"))

    is_all_selected = all == selections

    if is_all_selected:
        index_selections = index_all
        diff = None
    else:
        diff = index_selections - index_all if index_selections is not None and index_all is not None else None

    if index_selections is not None:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 2.8em; font-weight: bold; margin-top: 1em; margin-bottom: 1em">
                Index: {index_selections:.2f}
                {f'<div style="font-size: 0.4em; font-weight: lighter; color: {"green" if diff > 0 else "red"};">({diff:+.2f} compared to all participants)</div>' if diff is not None else ''}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style='text-align: center; font-size: 2.8em; font-weight: bold; margin-top: 1em;'>
                Index: N/A
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Summary Metrics ---
    col1, col2, col3 = st.columns(3)

    if not is_all_selected and index_selections is not None:
        col1.metric("Percent of Participants", f"{100 * count_selections / count_all:0.1f}%")
        col2.metric("Selected Participants", f"{count_selections:,.0f}")
    else:
        col1.metric("Percent of Participants", "100.0%")
        col2.metric("Selected Participants", f"{count_all:,.0f}")

    col3.metric("All Participants", f"{count_all:,.0f}")

    # --- Dumbbell Chart ---
    fig = go.Figure()
    non_index_df = plot_df[~is_index & ~is_quantile & ~is_mean]
    pivoted_df = non_index_df.pivot(index="Statistic", columns="Group", values="Value")

    for i, stat in enumerate(pivoted_df.index):
        val_all = pivoted_df.loc[stat].get("All Participants")
        val_sel = pivoted_df.loc[stat].get("Selected Participants")

        if pd.notna(val_all) and pd.notna(val_sel):
            fig.add_trace(go.Scatter(
                y=[val_all, val_sel], x=[stat, stat],
                mode="lines", line=dict(color="lightgrey"), showlegend=False
            ))

        if pd.notna(val_all):
            fig.add_trace(go.Scatter(
                y=[val_all], x=[stat], mode="markers",
                name="All Participants" if i == 0 else None,
                marker=dict(color="grey", size=10),
                showlegend=(i == 0)
            ))

        if pd.notna(val_sel):
            fig.add_trace(go.Scatter(
                y=[val_sel], x=[stat], mode="markers",
                name="Selected Participants" if i == 0 else None,
                marker=dict(color="green", size=10),
                showlegend=(i == 0)
            ))

    fig.update_layout(
        title=dict(
            text="Share of Participants with Increases in Wages and Routine Task Intensity by Subgroup",
            x=0.5,                  # 0.5 = center
            xanchor="center",
            font=dict(size=18)
        ),
        yaxis_title="Percent of Participants",
        xaxis_title="Statistic",
        template="simple_white",
        legend=dict(
            orientation="h",              # horizontal layout
            yanchor="bottom",
            y=-0.3,                       # move below the chart
            xanchor="center",
            x=0.5,
            title=None,
            traceorder="normal",
            valign="middle",
            font=dict(size=12),
            itemsizing="constant"
        ),
        legend_tracegroupgap=0
    )

    st.plotly_chart(fig)

    # --- Mean Metrics ---

    # --- Box Chart ---
    # .pivot_table(index="Group", columns=["Statistic"], values="Value")
    quantiles_df = plot_df[is_quantile].set_index("Group").pivot_table(index="Statistic", columns="Group", values="Value")

    is_median = quantiles_df.index.str.startswith("Median")
    is_q1 = quantiles_df.index.str.startswith("25th")
    is_q3 = quantiles_df.index.str.startswith("75th")

    median = quantiles_df[is_median]
    q1 = quantiles_df[is_q1]
    q3 = quantiles_df[is_q3]

    # Define subplot titles (same order as columns in median/q1/q3)
    subplot_titles = ['Routine Cognitive Exposure', 'Routine Manual Exposure', 'Wage Gain']

    # Use group names from columns
    groups = median.columns.tolist()

    # Define colors for consistency
    if len(groups) == 2:
        colors = {
            groups[0]: "gray",
            groups[1]: "green"
        }
    else:
        colors = {
            groups[0]: "gray",
        }

    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)

    # Loop over each subplot (indexed 0 to 2)
    for i, title in enumerate(subplot_titles):
        for group in groups:
            fig.add_trace(
                go.Box(
                    q1=[q1.loc[q1.index.str.contains(title), group].values[0]],
                    median=[median.loc[median.index.str.contains(title), group].values[0]],
                    q3=[q3.loc[q3.index.str.contains(title), group].values[0]],
                    name=group,
                    marker_color=colors[group],
                    boxpoints=False,
                    showlegend=(i == 0),  # Show legend only in first column
                ),
                row=1, col=i+1
            )

    fig.update_xaxes(showticklabels=False)

    # Format layout
    fig.update_layout(
        title=dict(
            text="Outcome Differences for Selected Participants: Wages and Routine Task Intensity",
            x=0.5,                  # 0.5 = center
            xanchor="center",
            font=dict(size=18)
        ),
        boxmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            traceorder="normal",
            font=dict(size=12),
            itemsizing="constant"
        )
    )

    st.plotly_chart(fig)

    st.write("Below are the wage gain and routine task intensity differences for selected participants compared to all participants.")

    mean_df = plot_df[is_mean]
    mean_pivoted_df = mean_df.pivot(index="Statistic", columns="Group", values="Value")

    stat_cols = st.columns(len(mean_pivoted_df.index))

    for i, stat in enumerate(mean_pivoted_df.index):
        group_a_val = mean_pivoted_df.loc[stat, 'Selected Participants']
        group_b_val = mean_pivoted_df.loc[stat, 'All Participants']

        # Show Group A as main value, Group B as fake delta
        stat_cols[i].metric(
            label=stat,
            value=f"{group_a_val:.2f}",
            delta=f"{group_a_val - group_b_val:.2f}"  # Just showing Group B as "delta"
        )


    


with tab3:
    """
        **v0.0.8 (31.08.2025)**
        - Added `Funding Stream` filter.

        **v0.0.7 (19.08.2025)**
        - Added exit industry filter.
        - Removed `Received Training` filter.

        **v0.0.6 (06.08.2025)**
        - Added means of pre- and post-program changes in Wage Gain, Routine Cognitive Exposure, and Routine Manual Exposure.

        **v0.0.5 (02.08.2025)**
        - Added box plot figure to capture magnitude of pre- and post-program changes in Wage Gain, Routine Cognitive Exposure, and Routine Manual Exposure.
        
        **v0.0.4 (31.07.2025)**
        - Modified index and subindex calculation (see Methodology for additional details).
        - Added consolidation to demographic selectors, grouping attributes with low participant counts into 'Other'.
        - Added `State` selector.
        - Removed `Training Service Type` selector.  
        - Removed `Tier 1` index.
    """

    
