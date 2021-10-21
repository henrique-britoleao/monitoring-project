# import libraries
import os
import sys
import json
import pandas as pd
import graphviz as graphviz
import plotly.express as px

sys.path.insert(0, "..")

from Dashboard import read_alerts
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
    data_quality = read_alerts.read_data_quality_alerts(dict_metrics, batch_name, batch_id)
    outliers = read_alerts.read_outliers_alert(dict_metrics, batch_name, batch_id)
    concept_drift = read_alerts.read_psi_alert(dict_metrics, batch_name, batch_id)
    numerical_drift, categorical_drift, binary_drift = read_alerts.read_covariate_metrics_alerts(dict_metrics, batch_name, batch_id)
    return [max(data_quality.values()), outliers, concept_drift, numerical_drift.max().max(), categorical_drift.max().max(), binary_drift.max().max()]


def alerts_matrix(batch_name: str, batch_id: str):
    '''TODO'''
    with open(cst.MONITORING_METRICS_FILE_PATH, "r") as read_file:
        dict_metrics = json.load(read_file)
    numerical_drift, categorical_drift, binary_drift = read_alerts.read_covariate_metrics_alerts(dict_metrics, batch_name, batch_id)
    covariance_drift = numerical_drift.append(categorical_drift).append(binary_drift).fillna(0)
    alert_heatmap = px.imshow(covariance_drift.transpose())
    return alert_heatmap

        

