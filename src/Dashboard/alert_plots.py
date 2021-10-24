# import libraries
import sys
import json
import pandas as pd
import graphviz as graphviz
import plotly.express as px

sys.path.insert(0, "..")

from dashboard import read_alerts
import constants as cst


def alerts_graph(batch_name: str, batch_id: str) -> graphviz.Digraph:
    """
    Plots a graph summarising, for each type of alert, if an alert has been
    raised or not

    Args:
        batch_name: name of the new batch of data
        batch_id: numerical id associated to the batch, the batch id "i" means 
        that it is the ith uploaded batch on the dashboard.

    Returns:
        graph: graph with one node for each type of alerts. Node is red if an
        alert has been raised, green otherwise  
    """
    alerts = high_level_alert(batch_name, batch_id)
    alert_drift = max(alerts[2:])
    alerts.insert(3, alert_drift)
    names = [
        "Data quality",
        "Outlier detection",
        "Drift",
        "Concept drift",
        "Numerical drift",
        "Categorical drift",
        "Binary drift",
    ]  # possible types of alerts
    colors = [
        "green" if alerts[i] == 0 else "red" for i in range(len(alerts))
    ]  # associate color indicating if an alert has been raised or not
    names_emo = [
        names[i] + " ✔️" if alerts[i] == 0 else names[i] + " ❗"
        for i in range(len(alerts))
    ]
    graph = graphviz.Digraph()
    graph.attr("node", style="filled")
    for i, alert in enumerate(alerts):
        graph.node(str(i + 1), names_emo[i], color=colors[i])
    graph.attr(rankdir="LR")
    graph.edges(["12", "23", "34", "35", "36", "37"])
    return graph


def high_level_alert(batch_name: str, batch_id: str) -> list:
    """
    For each type of alerts, indicates if an alert has been raised

    Args:
        batch_name: name of the new batch of data
        batch_id: numerical id associated to the batch, the batch id "i" means 
        that it is the ith uploaded batch on the dashboard.

    Returns:
        alert_list: for each type of alerts has value 1 if an alert has been
        raised, 0 otherwise
    """
    with open(cst.MONITORING_METRICS_FILE_PATH, "r") as read_file:
        dict_metrics = json.load(read_file)
    data_quality = read_alerts.read_data_quality_alerts(
        dict_metrics, batch_name, batch_id
    )
    outliers = read_alerts.read_outliers_alert(dict_metrics, batch_name, batch_id)
    concept_drift = read_alerts.read_psi_alert(dict_metrics, batch_name, batch_id)
    (
        numerical_drift,
        categorical_drift,
        binary_drift,
    ) = read_alerts.read_covariate_metrics_alerts(dict_metrics, batch_name, batch_id)
    alerts_list = [
        max(data_quality.values()),
        outliers,
        concept_drift,
        numerical_drift.max().max(),
        categorical_drift.max().max(),
        binary_drift.max().max(),
    ]
    return alerts_list


def alerts_matrix(batch_name: str, batch_id: int) -> px.imshow:
    """
    Plots a heatmap to see which columns are raising covariate drift alerts
    and the statistical test involved

    Args:
        batch_name: name of the new batch of data
        batch_id: numerical id associated to the batch, the batch id "i" means 
        that it is the ith uploaded batch on the dashboard.

    Returns:
        alert_heatmap (px.imshow): heatmap of statistical tests vs columns. The 
        color is indicative of the presence of an alert 
    """
    with open(cst.MONITORING_METRICS_FILE_PATH, "r") as read_file:
        dict_metrics = json.load(read_file)
    (
        numerical_drift,
        categorical_drift,
        binary_drift,
    ) = read_alerts.read_covariate_metrics_alerts(
        dict_metrics, batch_name, batch_id
    )  # read the alerts associated to covariate drift
    covariates_drift = (
        numerical_drift.append(categorical_drift).append(binary_drift).fillna(0)
    )
    alert_heatmap = px.imshow(covariates_drift.transpose())
    return alert_heatmap
