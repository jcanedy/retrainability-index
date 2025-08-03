import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from functools import reduce
import operator


'''
# Retrainability Index
v0.0.5 _(research prototpye)_

_Author(s): Jordan Canedy-Specht [LinkedIn](https://www.linkedin.com/in/jordancanedy/), [Github](https://github.com/jcanedy)_

_Code Repository: <https://github.com/jcanedy/retrainability-index>_

The Retainability Index is a composite metric designed to evaluate how effectively workforce programs help participants access retraining, develop future-ready skills, and secure quality employment. This research prototype is built using [data from the Workforce Innovation and Opportunity Act (WIOA) program](https://www.dol.gov/agencies/eta/performance)—the U.S. Department of Labor’s flagship workforce development system. The WIOA dataset includes individual-level records for millions of participants in adult, dislocated worker, and youth programs, capturing demographics, services received, and employment and wage outcomes before and after program exit.

The index incorporates measures of routine task intensity (RTI) based on the task framework developed by [Daron Acemoglu and David Autor (2011)](https://shapingwork.mit.edu/research/skills-tasks-and-technologies-implications-for-employment-and-earnings/). RTI scores are calculated at the industry level, using the composition of occupations typically employed within each industry in the broader economy. While RTI has previously been used to study labor market polarization and automation risk, this project applies it in a new context: as a component of a composite metric evaluating retraining program outcomes. Combined with participant-level wage progression metrics, the index provides a forward-looking signal of how well public training programs are positioning individuals for resilient, automation-resistant employment.

As a proof of concept, the index also highlights demographic differences in outcomes. Going forward, we aim to expand the Retainability Index deeply, by incorporating additional outcome variables such as job tenure, benefits, and occupational mobility; and broadly, by adapting the methodology for use in other countries as comparable labor and training data become available.
'''

df_lazy = pl.scan_parquet("cloud/storage/processed/index_tier2.parquet")


#TODO(@jcanedy27): Move weight calculation to `create_index.py`

weights = np.array([0.5, 0.25, 0.25])

# Unpack weights
w1, w2, w3 = weights

# Use normalized weights
df_lazy = df_lazy.with_columns([
    # Center and rescale to [-1, 1], then flip sign for routine vars
    (-2 * (pl.col("bin_r_cog_industry_y mean") - 0.5)).alias("r_cog_adj"),
    (-2 * (pl.col("bin_r_man_industry_y mean") - 0.5)).alias("r_man_adj"),
    ( 2 * (pl.col("bin_wages_mean_y mean")     - 0.5)).alias("wages_adj")
]).with_columns([
    # Compute final index
    (w1 * pl.col("r_cog_adj") +
     w2 * pl.col("r_man_adj") +
     w3 * pl.col("wages_adj")).alias("index_y")
])

# Define mapping from input column names to output variable names with '_options' suffix
columns = {
    "race_ethnicity_x": "race_ethnicity_options",
    "sex_x": "sex_options",
    "age_x": "age_options",
    "highest_education_level_x": "highest_education_level_options",
    "low_income_x": "low_income_options",
    "employment_status_x": "employment_status_options",
    "received_training_x": "received_training_options",
    "state_x": "state_options",
    "industry_title_x": "industry_title_options",
}

# Lazily compute unique values per column
unique_lazyframes = {
    out_name: df_lazy.select(pl.col(col)).unique()
    for col, out_name in columns.items()
}

# Materialize and extract unique values into lists
unique_values = {
    out_name: lf.collect().get_column(col).to_list()
    for (col, out_name), lf in zip(columns.items(), unique_lazyframes.values())
}

race_ethnicity_options = unique_values["race_ethnicity_options"]
sex_options = unique_values["sex_options"]
age_options = unique_values["age_options"]
highest_education_level_options = unique_values["highest_education_level_options"]
low_income_options = unique_values["low_income_options"]
employment_status_options = unique_values["employment_status_options"]
received_training_options = unique_values["received_training_options"]
state_options = unique_values["state_options"]
industry_title_options = unique_values["industry_title_options"]


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
age_options = sort_with_all_other(age_options, age_custom_order)

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
highest_education_level_options = sort_with_all_other(highest_education_level_options, education_custom_order)

# Apply default sort (alphabetical + All/Other positioning) to all others
race_ethnicity_options = sort_with_all_other(race_ethnicity_options)
sex_options = sort_with_all_other(sex_options)
low_income_options = sort_with_all_other(low_income_options)
employment_status_options = sort_with_all_other(employment_status_options)
received_training_options = sort_with_all_other(received_training_options)
state_options = sort_with_all_other(state_options)
industry_title_options = sort_with_all_other(industry_title_options)


# Config: field name -> label for display
sidebar_fields = [
    ("race_ethnicity_x", "Race / Ethnicity"),
    ("sex_x", "Sex"),
    ("age_x", "Age"),
    ("highest_education_level_x", "Highest Education Level"),
    ("low_income_x", "Low Income"),
    ("employment_status_x", "Employment Status"),
    ("state_x", "State"),
    ("received_training_x", "Received Training"),
    ("industry_title_x", "Industry Code"),
]

# Use the *_options variables already created
options_lookup = {
    "race_ethnicity_x": race_ethnicity_options,
    "sex_x": sex_options,
    "age_x": age_options,
    "highest_education_level_x": highest_education_level_options,
    "low_income_x": low_income_options,
    "employment_status_x": employment_status_options,
    "received_training_x": received_training_options,
    "state_x": state_options,
    "industry_title_x": industry_title_options,  
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

    # Handle next 2 fields
    col1, col2 = st.columns(2)
    field1, label1 = sidebar_fields[6]
    field2, label2 = sidebar_fields[7]

    with col1:
        selections[field1] = st.selectbox(label1, options_lookup[field1])
    with col2:
        selections[field2] = st.selectbox(label2, options_lookup[field2])

    st.markdown("### Prior Employment")
    selections["industry_title_x"] = st.selectbox(
        "Industry Code", options_lookup["industry_title_x"]
    )


# Build combined expression using &
selected_filter = reduce(
    operator.and_,
    [pl.col(field).is_in([value]) for field, value in selections.items()]
)

# Same for the 'all' filter
all_filter = reduce(
    operator.and_,
    [pl.col(field) == "All" for field in selections.keys()]
)

# Perform all operations in one optimized query
result = (
    df_lazy
    .with_columns([
        # Add filter flags
        selected_filter.alias("is_selected"),
        all_filter.alias("is_all")
    ])
    .with_columns([
        # Calculate sums for counts
        pl.when(pl.col("is_selected"))
        .then(pl.col("count"))
        .otherwise(0)
        .sum()
        .over(pl.lit(1))  # Window over entire dataset
        .alias("selected_count_sum"),
        
        pl.when(pl.col("is_all"))
        .then(pl.col("count"))
        .otherwise(0)
        .sum()
        .over(pl.lit(1))  # Window over entire dataset
        .alias("all_count_sum")
    ])
    .filter(pl.col("is_selected") | pl.col("is_all"))
    .with_columns([
        # Add Group column
        pl.when(pl.col("is_all"))
        .then(pl.lit("All Participants"))
        .otherwise(pl.lit("Selected Participants"))
        .alias("Group")
    ])
    .drop("bin_offshor_industry_y mean")
    .rename({
        "bin_r_cog_industry_y mean": "Higher Routine Cognitive Tasks",
        "bin_r_man_industry_y mean": "Higher Routine Manual Tasks", 
        "bin_wages_mean_y mean": "Higher Mean Wages",

        "diff_r_cog_industry_y median": "Median Routine Cognitive Tasks",
        "diff_r_cog_industry_y 25th": "25th Routine Cognitive Tasks",
        "diff_r_cog_industry_y 75th": "75th Routine Cognitive Tasks",

        "diff_r_man_industry_y median": "Median Routine Manual Tasks",
        "diff_r_man_industry_y 25th": "25th Routine Manual Tasks",
        "diff_r_man_industry_y 75th": "75th Routine Manual Tasks",

        "diff_wages_mean_y median": "Median Mean Wages",
        "diff_wages_mean_y 25th": "25th Mean Wages",
        "diff_wages_mean_y 75th": "75th Mean Wages",
    })
    .unpivot(
        on=["Higher Routine Cognitive Tasks", "Higher Routine Manual Tasks", "Higher Mean Wages",
            "Median Routine Cognitive Tasks", "Median Routine Manual Tasks" , "Median Mean Wages",
            "25th Routine Cognitive Tasks", "25th Routine Manual Tasks" , "25th Mean Wages",
            "75th Routine Cognitive Tasks", "75th Routine Manual Tasks" , "75th Mean Wages",
            "index_y"],
        index=["Group", "selected_count_sum", "all_count_sum"],
        variable_name="Statistic",
        value_name="Value"
    )
    .collect()
)

# Extract the count sums
selected_rows = result["selected_count_sum"][0]
total_rows = result["all_count_sum"][0]

# Clean up the final dataframe
plot_df_pl = result.select(["Group", "Statistic", "Value"])

# Convert to pandas if needed for plotting
plot_df = plot_df_pl.to_pandas()

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

        - **Wage Gain Share** (`wgs`):  
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
                        {w3:.2f} \cdot \text{{wgs}}'
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

    is_all_selected = all(value == "All" for value in selections.values())
    selected_index = index_df["Value"].get("Selected Participants")
    all_index = index_df["Value"].get("All Participants")

    if is_all_selected:
        selected_index = all_index
        diff = None
    else:
        diff = selected_index - all_index if selected_index is not None and all_index is not None else None

    if selected_index is not None:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 2.8em; font-weight: bold; margin-top: 1em; margin-bottom: 1em">
                Index: {selected_index:.2f}
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

    if not is_all_selected and selected_index is not None:
        col1.metric("Percent of Participants", f"{100 * selected_rows / total_rows:0.1f}%")
        col2.metric("Selected Participants", f"{selected_rows:,.0f}")
    else:
        col1.metric("Percent of Participants", "100.0%")
        col2.metric("Selected Participants", f"{total_rows:,.0f}")

    col3.metric("All Participants", f"{total_rows:,.0f}")

    # --- Dumbbell Chart ---
    fig = go.Figure()
    non_index_df = plot_df[~is_index & ~is_quantile]
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
    subplot_titles = ['Mean Wages', 'Routine Cognitive Tasks', 'Routine Manual Tasks']

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


with tab3:
    """
        **v0.0.5 (02.08.2025)**
        - Added box plot figure to capture magnitude of pre- and post-program changes in wages, routine cognitive tasks, and routine manual tasks.
        
        **v0.0.4 (31.07.2025)**
        - Modified index and subindex calculation (see Methodology for additional details).
        - Added consolidation to demographic selectors, grouping attributes with low participant counts into 'Other'.
        - Added `State` selector.
        - Removed `Training Service Type` selector.  
        - Removed `Tier 1` index.
    """

    
