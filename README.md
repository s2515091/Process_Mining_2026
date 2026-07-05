# Resource-Centric Process Mining Approach to Bottleneck Analysis in Agent-Based Manufacturing

This repository contains the complete Python data science pipeline used to analyze post-execution event logs from an agent-based manufacturing system. Following the **Process Mining Project Methodology (PMPM)** framework, this project bypasses traditional control-flow modeling to isolate systemic constraints using a performance-oriented resource lens. The documentation and analysis can be thoroughly followed in the `resource_analysis.ipynb` notebook.

### Installation

Ensure Python 3.10+ is installed. Clone this repository and configure your virtual environment

### Execution

Launch the Jupyter instance to run the scripts step-by-step: 
Bash
```
jupyter notebook resource_analysis.ipynb
```

## Future Work: Predictive Process Monitoring (PPM) Pipeline

To bridge the gap between historical diagnostic profiling and active shop-floor intervention, this repository includes an initial, production-ready machine learning framework in the script `predictive.py`. This script acts as the structural foundation for transitioning the project into a predictive and prescriptive monitoring system.

### Pipeline Execution Framework

The predictive module is designed around three process-aware phases:

1. **Prefix Slicing & Feature Engineering:** Continuous event logs are parsed into chronological step-length snapshots to simulate incomplete, active processes on the floor:
   Each slice aggregates dynamic temporal attributes (`elapsed_time`, `prefix_length`) with decoded static environmental configurations (`num_agvs`, `dispatching_rule`, `layout_direction`).

2. **Supervised Gradient Boosting:** Engineered features are exposed to an **XGBoost Regressor** to model non-linear agent interactions and accurately forecast the remaining cycle time down to the terminal `Drain` station.

3. **Prescriptive "What-If" Analysis:** The framework provides the baseline mathematical engine to evaluate 27 synthetic permutations of floor layouts against a live running case, sorting configurations by efficiency to actively prevent Service Level Agreement (SLA) breaches.

### Prerequisites

Ensure you have Python 3.8+ installed along with the required analytical libraries:
Bash

```
pip install numpy pandas xgboost scikit-learn pm4py
```
### Running the predictive baseline

To execute the prefix-slicing and train the baseline XGBoost model on the pre-filtered simulation data:
Bash
```
python predictive.py
```
### Result: Data Schema & Feature Structure

The `predictive.py` script transforms the sequential `.xes` event stream into a structured, tabular CSV matrix optimized for supervised machine learning. Each row in the resulting table represents a unique historical case execution snapshot at a specific **Prefix Length ($L$)**.

The engineered CSV file follows this strict schema:

| Column Name | Data Type | Feature Class | Description |
| :--- | :--- | :--- | :--- |
| `case_id` | String | Identifier | Unique trace index from the Bemthuis et al. simulation logs. |
| `prefix_length` | Integer | Dynamic Process | The sequence step count (e.g., $L=3$ implies the state after the first 3 events). |
| `current_activity` | Categorical | Dynamic Process | The specific station activity milestone at step $L$ (e.g., `Drilling`, `Painting`). |
| `elapsed_time` | Float | Dynamic Process | Total minutes accumulated since the job entered the floor via the `Arrival` station. |
| `num_agvs` | Integer | Static Control | Resource density configuration of the floor fleet ($X \in \{4, 5, 6\}$). |
| `dispatching_rule` | Categorical | Static Control | The active task allocation logic ($Y \in \{\text{Random}, \text{LWT}, \text{Nearest}\}$). |
| `layout_direction` | Categorical | Static Control | Transport lane pathing constraints ($Z \in \{\text{Forward}, \text{Backward}, \text{Bidirectional}\}$). |
| `product_type` | Categorical | Static Control | The product category profile (*Console*, *Helicopter*, *Robot*). |
| `remaining_time` | Float | **Target ($Y$)** | **Prediction Goal:** The exact time delta remaining until the case records a terminal `Drain` event. |

### Intermediate Artifact Generation
When running the future work script, this table is automatically generated and split into training matrices:
* **Features ($X$):** One-hot encoded representations of the dynamic and static attributes.
* **Labels ($Y$):** Continuous target column (`remaining_time`).
---
