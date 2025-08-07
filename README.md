# Retrainability Index

A composite metric designed to evaluate how effectively workforce programs help participants access retraining, develop future-ready skills, and secure quality employment. This research prototype analyzes data from the Workforce Innovation and Opportunity Act (WIOA) program—the U.S. Department of Labor's flagship workforce development system.

## Overview

The Retrainability Index combines measures of routine task intensity (RTI) based on the task framework developed by [Daron Acemoglu and David Autor (2011)](https://shapingwork.mit.edu/research/skills-tasks-and-technologies-implications-for-employment-and-earnings/) with participant-level wage progression metrics. While RTI has previously been used to study labor market polarization and automation risk, this project applies it in a new context: as a component of a composite metric evaluating retraining program outcomes.

**Key Features:**
- Interactive Streamlit dashboard for exploring participant outcomes by demographic subgroups
- Industry-level routine task intensity calculations based on occupational composition
- Data processing pipeline for WIOA performance records
- Scoring methodology combining wage gains with automation exposure metrics to create composite index

## Data Sources

- **WIOA Performance Records**: Individual-level records for millions of participants in adult, dislocated worker, and youth programs (U.S. Department of Labor)
- **National Employment Matrix**: Bureau of Labor Statistics occupational employment by industry data
- **Routine Task Intensity Measures**: Task-based occupation classifications (Acemoglu & Autor framework)
- **SOC Occupation Codes**: Standard Occupational Classification system mappings

## Methodology

### Metrics

The index incorporates three key statistics (expressed as proportions 0-1):

1. **Routine Cognitive Exposure (RCE)**: Proportion of participants employed in industries with high cognitive routine task intensity (e.g., clerical, administrative support)
2. **Routine Manual Exposure (RME)**: Proportion employed in industries with high manual routine task intensity (e.g., machine operation, basic manufacturing)  
3. **Wage Gain (WG)**: Proportion experiencing wage increases post-program compared to pre-program

### Transformation & Scoring

Each metric is transformed to range from -1 to 1, centered at 0:
```
x' = 2(x - 0.5)
```

For routine exposure metrics, values are inverted so higher index scores represent better outcomes (less routine exposure). The composite index is calculated as:

```
Index = 0.25 × RCE' + 0.25 × RME' + 0.50 × WG'
```

Higher scores reflect less routine task exposure and greater wage gains, indicating better job quality outcomes.

## Installation & Usage

### Local Development

1. **Prerequisites**: Python 3.12+, uv package manager

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run data processing pipeline**:
   ```bash
   # Process routine task intensity by industry
   python scripts/compute_rti_by_industry.py
   
   # Create composite index
   python scripts/create_index.py
   ```

4. **Launch dashboard**:
   ```bash
   streamlit run app/streamlit_app.py
   ```



