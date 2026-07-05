# Process_Mining_2026


# Deliverables

Title, names & Abstract
• Main content, including:
o 1. Introduction & Problem Context (including research objective)
o 2. Related Work
o 3. Methodology (e.g., PM2 structure*, tool usage, data cleaning, mining techniques)
o 4. Case Description (e.g., data)
o 5. Analysis & Findings (e.g., visualizations, KPIs, bottlenecks, deviations)
o [can be omitted] 6. Improvement Plan (including e.g., feasibility, benefits, and
challenges)
o 7. Recommendations (e.g., for case owners, dataset improvements]
o 8. Discussion & Limitations
o 9. Conclusion & Future Work
• Declaration of Generative AI and AI-assisted technologies**
• Reproducibility & Artifact Availability***
• References
• Appendices (if needed)


## Important Note on Reproduceability and availablity

## Future Work: Predictive Process Monitoring (PPM) Pipeline

To bridge the gap between historical diagnostic profiling and active shop-floor intervention, this repository includes an initial, production-ready machine learning framework in the script `predictive.py`. This script acts as the structural foundation for transitioning the project into a predictive and prescriptive monitoring system.

### Pipeline Execution Framework

The predictive module is designed around three process-aware phases:

1. **Prefix Slicing & Feature Engineering:** Continuous event logs are parsed into chronological step-length snapshots to simulate incomplete, active processes on the floor:
   Each slice aggregates dynamic temporal attributes (`elapsed_time`, `prefix_length`) with decoded static environmental configurations (`num_agvs`, `dispatching_rule`, `layout_direction`).

2. **Supervised Gradient Boosting:** Engineered features are exposed to an **XGBoost Regressor** to model non-linear agent interactions and accurately forecast the remaining cycle time down to the terminal `Drain` station.

3. **Prescriptive "What-If" Analysis:** The framework provides the baseline mathematical engine to evaluate 27 synthetic permutations of floor layouts against a live running case, sorting configurations by efficiency to actively prevent Service Level Agreement (SLA) breaches.

---
