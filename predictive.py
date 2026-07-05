import os
import re
import pandas as pd
import pm4py
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def build_pmpm_prefixes(df):
    prefix_data = []

    # Group by unique case run
    for case_id, group in df.groupby('case:concept:name'):
        group = group.reset_index(drop=True)
        total_events = len(group)

        # Total time taken for this product run (in minutes)
        total_duration = (group['time:timestamp'].iloc[-1] - group['time:timestamp'].iloc[0]).total_seconds() / 60.0

        for prefix_len in range(1, total_events):
            current_event = group.iloc[prefix_len - 1]

            elapsed_time = (current_event['time:timestamp'] - group['time:timestamp'].iloc[0]).total_seconds() / 60.0
            remaining_time = total_duration - elapsed_time

            prefix_data.append({
                'prefix_length': prefix_len,
                'current_activity': current_event['concept:name'],
                'elapsed_time': elapsed_time,
                'num_agvs': current_event['attr_num_agvs'],
                'layout_direction': current_event['attr_direction'],
                'dispatching_rule': current_event['attr_dispatching_rule'],
                'remaining_cycle_time': remaining_time
            })

    return pd.DataFrame(prefix_data)

def parse_pmpm_metadata(filename):
    """
    PMPM Data Enrichment: Extract exact factors from the authors'
    naming convention to use as machine learning features.
    """
    # Look for a 3-digit pattern
    match = re.search(r'(\d)(\d)(\d)\.xes', filename)
    if match:
        num_agvs = int(match.group(1))

        direction_map = {'1': 'Forward', '2': 'Backward', '3': 'Bidirectional'}
        direction = direction_map.get(match.group(2), 'Unknown')

        rule_map = {'1': 'Random', '2': 'Longest_Waiting_Time', '3': 'Nearest_Vehicle'}
        rule = rule_map.get(match.group(3), 'Unknown')

        return num_agvs, direction, rule
    return 4, 'Unknown', 'Unknown'


def aggregate_filtered_logs(directory_path):
    all_cases = []

    for file in os.listdir(directory_path):
        if file.endswith('.xes'):
            num_agvs, direction, rule = parse_pmpm_metadata(file)

            # Directly read the filtered XES file
            log = pm4py.read_xes(os.path.join(directory_path, file))
            df = pm4py.convert_to_dataframe(log)

            # Ensure chronological order per case
            df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

            # Inject context features
            df['attr_num_agvs'] = num_agvs
            df['attr_direction'] = direction
            df['attr_dispatching_rule'] = rule

            all_cases.append(df)

    return pd.concat(all_cases, ignore_index=True)

def generate_what_if_matrix(current_activity, elapsed_time, prefix_length, critical_sla=80.0):
    """
    PMPM Operational Matrix: Simulates changes to the multi-agent system setup
    to prevent an active job from violating its target deadline.
    """
    rules = ['Random', 'Longest_Waiting_Time', 'Nearest_Vehicle']
    agv_counts = [4, 5, 6]
    directions = ['Forward', 'Backward', 'Bidirectional']

    matrix_scenarios = []

    for agv in agv_counts:
        for rule in rules:
            for direction in directions:
                hypothetical_input = pd.DataFrame([{
                    'current_activity': current_activity,
                    'layout_direction': direction,
                    'dispatching_rule': rule,
                    'prefix_length': prefix_length,
                    'elapsed_time': elapsed_time,
                    'num_agvs': agv
                }])

                pred_remaining = predictive_framework.predict(hypothetical_input)[0]
                expected_total = elapsed_time + pred_remaining
                status = "SLA BREACH" if expected_total > critical_sla else "COMPLIANT"

                matrix_scenarios.append({
                    'AGV Count': agv, 'Rule': rule, 'Direction': direction,
                    'Predicted Remaining (min)': round(pred_remaining, 2),
                    'Expected Total (min)': round(expected_total, 2),
                    'Status': status
                })

    return pd.DataFrame(matrix_scenarios).sort_values(by='Expected Total (min)')

df_clean = aggregate_filtered_logs("./data/FilteredFiles/")
ml_df = build_pmpm_prefixes(df_clean)

# Train-Test Split & Modeling Pipeline
categorical_features = ['current_activity', 'layout_direction', 'dispatching_rule']
numerical_features = ['prefix_length', 'elapsed_time', 'num_agvs']

X = ml_df[categorical_features + numerical_features]
y = ml_df['remaining_cycle_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder_pipeline = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

predictive_framework = Pipeline(steps=[
    ('preprocessor', encoder_pipeline),
    ('regressor', XGBRegressor(n_estimators=180, learning_rate=0.06, max_depth=6, random_state=42))
])

predictive_framework.fit(X_train, y_train)

# Example: A 'Helicopter' product is currently undergoing Painting,
# has been in the system for 10 minutes, and is at step length 5.
decision_sheet = generate_what_if_matrix('Painting', 10.0, 5, critical_sla=17.0)
decision_sheet.to_csv('prediction.csv')