#import libraries
import os
import sys
import json
import pandas as pd
import graphviz as graphviz

sys.path.insert(0, "..")

import constants as cst


def alerts_graph(batch_name: str, batch_id: str):
    '''TODO'''
    alerts = high_level_alert(batch_name, batch_id)
    alert_drift = max(alerts[2:])
    alerts.insert(3, alert_drift)
    names = ["Data quality", "Outlier detection", "Drift",
            "Concept drift", "Numerical drift", "Categorical drift",
            "Binary drift"]
    colors = ["green" if alerts[i]==0 else "red" for i in range(len(alerts))]
    names_emo = [names[i]+" ✔️" if alerts[i]==0 else names[i]+" ❗" for i in range(len(alerts))] 
    graph = graphviz.Digraph()
    graph.attr('node', style='filled')
    for i, alert in enumerate(alerts):
        graph.node(str(i+1), names_emo[i], color=colors[i])
    graph.attr(rankdir='LR')
    graph.edges(['12', '23', '34', '35', '36', '37'])
    return graph


def high_level_alert(batch_name: str, batch_id: str) -> list:
    '''TODO'''
    with open(cst.MONITORING_METRICS_FILE_PATH, "r") as read_file:
        dict_metrics = json.load(read_file)
    data_quality = max(dict_metrics[batch_id][batch_name]['data_quality'].values())
    outliers = dict_metrics[batch_id][batch_name]['outliers']['alert']
    concept_drift = dict_metrics[batch_id][batch_name]['metrics']['PSI']['alert']
    numerical_drift = pd.DataFrame(
        dict_metrics[batch_id][batch_name]['metrics']['covariate_drift_metrics']['numerical_metrics'])\
        .applymap(lambda x: dict(x)['alert']).transpose().max().max()
    categorical_drift = pd.DataFrame(
        dict_metrics[batch_id][batch_name]['metrics']['covariate_drift_metrics']['categorical_metrics'])\
        .applymap(lambda x: dict(x)['alert']).transpose().max().max()
    binary_drift = pd.DataFrame(
        dict_metrics[batch_id][batch_name]['metrics']['covariate_drift_metrics']['binary_metrics'])\
        .applymap(lambda x: dict(x)['alert']).transpose().max().max()
    return [data_quality, outliers, concept_drift, numerical_drift, categorical_drift, binary_drift]