# Minority Language Dynamics: Basque, Welsh and Irish

This repo contains the data, scripts and outputs for my thesis on minority language dynamics. It fits a spatially extended Abrams–Strogatz ODE model to census data for three languages.

---

## Scripts

### `src/basque/basque_age.py`

Run this before `basque.py`. It does a descriptive age cohort analysis on the EUSTAT data. It reads from `cohort_lang.xlsx` and produces 5 figures.

- fig1: fluency by age group comparing 1981 and 2021
- fig2: fluency over time for the youngest age groups with a marker at the 1982 statute
- fig3: pseudo-cohort tracking where each line follows a birth cohort aging across census waves
- fig4: heatmap of fluency in young age groups by comarca for 1981 vs 2021
- fig5: scatter of 1981 vs 2021 fluency for 5–9 year olds by comarca

### `src/basque/basque.py`

Main fitting script for Basque. 20 comarcas, 9 census waves from 1981 to 2021. Fits three nested models (M0ns, M0, MR) and compares them with AIC/BIC. Reads the shapefile to build the adjacency graph and runs RK4 integration. Produces 5 figures.

- fig1: AIC/BIC/AICc comparison bar chart
- fig2: spatial resistance vs initial speaker fraction
- fig3: per-comarca resistance bar chart
- fig4: per-comarca trajectory plots comparing M0ns and MR
- fig5: prestige grounding scatter and basin classification

### `src/basque/synthetic_val.py`

Validation script. Generates synthetic data from known parameters (`a=1.31`, `D=0.05`) with added noise, fits the model to it, and checks whether the parameters are recovered. Then fits the same model to the real census data as a comparison. Produces one 4-panel figure..

### `src/irish/irish_with_natural_asdynamics.py`

Same structure as `basque.py` but for Irish. 26 counties, 4 census waves from 2006 to 2022. The dependent variable here is daily speakers outside the education system as a fraction of total county population, not self-reported ability. Also includes pure A–S variants (`AS_shared`, `AS_indiv`, `AS_free`) as a comparison. Produces 6 figures.

- fig1: AIC/BIC/AICc comparison bar chart
- fig2: resistance vs inital Irish fraction scatter plot
- fig3: per-county resistance bar chart 
- fig4: per-county trajectories
- fig5: prestige grounding with reference lines for Basque and Welsh
- fig6: pure A–S analysis, per-county volatility and prestige grounding

### `src/welsh/welsh.py`

Same structure again but for Welsh. 22 local authorities, 4 census waves from 1991 to 2021. M0 is excluded here because the 2001 census changed both the question wording and the eligible age group which makes fitting a shock term pointless. The adjacency list is manually specified rather than computed from the shapefile because Anglesey is an island connected by the Menai Bridge so there is no shared land border with Gwynedd. Produces 5 figures.

- fig1: AIC/BIC/AICc comparison bar chart
- fig2: resistance vs initial Welsh fraction scatter
- fig3: per-district resistance
- fig4: trajectories with the 2001 discontinuity marked
- fig5: prestige grounding and basin classification

---

## Running the scripts

Update the file paths at the top of each script before running. Each script has a config block with paths to the input data and the output directory. Then just run them. For Basque, run `basque_age.py` first.

Note: Running these scripts takes a few minutes per language on a normal laptop, especially MR which has a lot of parameters.

```
pip install -r requirements.txt
```
