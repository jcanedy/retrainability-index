import streamlit as st

'''
# Retrainability Index
v0.0.3

The Retainability Index is composite metric designed to evaluate how effectively programs are helping workers access retraining, gain future-ready skills, and secure quality employment. 
Built from WIOA performance data, the index highlights demographic differences in program outcomes to identify where retraining efforts are leading to positive retraining.
'''

'''
## Methodology
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