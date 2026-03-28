# Retrainability Index

A composite metric designed to evaluate how effectively workforce programs help participants access retraining, develop future-ready skills, and secure quality employment. This research prototype analyzes data from the Workforce Innovation and Opportunity Act (WIOA) program — the U.S. Department of Labor's flagship workforce development system.

## Overview

The Retrainability Index combines measures of routine task intensity (RTI) based on the task framework developed by [Daron Acemoglu and David Autor (2011)](https://shapingwork.mit.edu/research/skills-tasks-and-technologies-implications-for-employment-and-earnings/) with participant-level wage progression metrics. While RTI has previously been used to study labor market polarization and automation risk, this project applies it in a new context: as a component of a composite metric evaluating retraining program outcomes.

## Data Sources

- **WIOA Performance Records**: Individual-level records for participants in adult, dislocated worker, and youth programs, program years 2017–2023 (U.S. Department of Labor)
- **National Employment Matrix**: Bureau of Labor Statistics occupational employment by industry
- **Routine Task Intensity Measures**: Task-based occupation classifications (Acemoglu & Autor framework)
- **SOC Occupation Codes**: Standard Occupational Classification system mappings
- **Consumer Price Index**: BLS CPI data used for inflation-adjusting wages
- **Workforce Development Board Codes**: Local board jurisdiction mappings with geographic and demographic variables sourced from Data Commons (population, median income, unemployment rate, median age, commute time, RUCC, household debt-to-income, diversity index)

## Methodology

### Index Calculation

The index is computed at two levels of industry aggregation — **occupation** and **subsector** — using the same approach:

**Step 1 — Shared-scale winsorization**: For each pre/post column pair, values from both columns are pooled to compute shared 1st/99th percentile bounds. Both columns are clipped to those bounds.

**Step 2 — Shared min-max normalization**: A combined min/max is computed across both winsorized columns and applied to scale each to [0, 1].

**Step 3 — Normalized diff**: `post_normalized - pre_normalized` captures the direction and magnitude of change on a common scale.

**Step 4 — Index formula**: Applied to three column pairs (wages, routine cognitive RTI, routine manual RTI):

```
index          = 0.5 × wages_diff - 0.25 × r_cog_diff          - 0.25 × r_man_diff
index_subsector = 0.5 × wages_diff - 0.25 × r_cog_subsector_diff - 0.25 × r_man_subsector_diff
```

Wages are IHS-transformed (`wages_mean_pre_ihs`, `wages_mean_post_ihs`) prior to winsorization. A positive index score indicates wage growth into less automatable work; a negative score indicates the reverse.

## Architecture

Raw data is stored in **GCS** (`gs://retrainability-index/`) and processed into **BigQuery** (`retraining-index.staging.*`). Pipelines are orchestrated with **Prefect**.

### Pipeline DAG

```
pipeline_consumer_price_index  ──┐
pipeline_industries            ──┤
pipeline_occupations           ──┤──► pipeline_routine_task_intensity
pipeline_performance_records   ──┤
                                 │
pipeline_occupations           ──┐
pipeline_routine_task_intensity ─┤──► pipeline_retrainability_index ──► staging.wioa_retrainability_index
pipeline_performance_records   ──┤
pipeline_workforce_development_boards ──┘
```

### BigQuery Output Tables (`retraining-index.staging`)

| Table | Description |
|---|---|
| `wioa_performance_records` | Normalized, filtered WIOA participant records with inflation-adjusted wages |
| `occupations` | SOC occupation code/title mappings |
| `industries` / `sectors` / `subsectors` | BLS industry hierarchy |
| `routine_task_intensity_occupation` | RTI scores at occupation level |
| `routine_task_intensity_industry` | RTI scores at industry level |
| `routine_task_intensity_subsector` | RTI scores at subsector level |
| `consumer_price_index` | CPI by year for inflation adjustment |
| `workforce_boards` / `workforce_boards_grouped` / `workforce_boards_all` | Workforce board jurisdictions with geographic/demographic variables |
| `wioa_retrainability_index` | Final index output joined with all upstream data |

## Installation

**Prerequisites**: Python 3.12+, [uv](https://github.com/astral-sh/uv), GCP credentials with access to `retraining-index` project.

```bash
uv sync
```

## Running Pipelines

Each pipeline can be run independently. Run upstream pipelines before downstream ones.

```bash
# Upstream (can run in parallel)
python src/pipeline/run/pipeline_consumer_price_index.py
python src/pipeline/run/pipeline_industries.py
python src/pipeline/run/pipeline_occupations.py
python src/pipeline/run/pipeline_performance_records.py
python src/pipeline/run/pipeline_routine_task_intensity.py
python src/pipeline/run/pipeline_workforce_development_boards.py

# Downstream (requires all upstream tables)
python src/pipeline/run/pipeline_retrainability_index.py
```
