import json
import constants as cst
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_performance(
    batch_perf_path: str = cst.PERFORMANCE_METRICS_FILE_PATH,
    train_perf_path: str = cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH,
    batch_name: str = "batch1.csv",
):
    """Plot model performance on the training set (cross-eval) vs on the batch dataset

    Args:
        batch_perf_path (str, optional): json file containing the latest performance metrics
            computed on the batch data (incl. true labels). Defaults to cst.PERFORMANCE_METRICS_FILE_PATH.
        train_perf_path (str, optional): json file containing the training metrics. 
            Defaults to cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH.
        batch_name (str, optional): batch name. Defaults to "batch1".

    Returns:
        Figure: plotly figure with performance metrics on the train vs batch sets
    """
    # TRAINING DATA
    # GET VALUES OF TRAIN
    with open(train_perf_path) as json_file:
        train_perf = json.load(json_file)
    # saving performance keys
    train_metrics = list(train_perf.keys())
    # delete confusion matrix
    train_metrics.pop(4)
    train_values = list(train_perf.values())
    # delete confusion matrix
    train_values.pop(4)
    # BATCH DATA
    # Opening JSON file
    
    with open(batch_perf_path) as json_file:
        performance_metrics = json.load(json_file)
    
    batch_perf = _get_batch_metrics(batch_name, performance_metrics)    
    
    # saving performance keys of chosen batch
    perf_metrics = list(batch_perf[batch_name].keys())
    # delete confusion matrix
    perf_metrics.pop(4)
    # saving performance values of chosen batch
    perf_values = list(batch_perf[batch_name].values())
    # delete confusion matrix
    perf_values.pop(4)
    # define bar colors
    colors = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Training Performance",
                x=train_metrics,
                y=train_values,
                text=[f"{np.round(v*100)}%" for v in train_values],
                textposition="outside",
                marker_color=colors[0],
            ),
            go.Bar(
                name="Batch Performance ",
                x=perf_metrics,
                y=perf_values,
                text=[f"{np.round(v*100)}%" for v in perf_values],
                textposition="outside",
                marker_color=colors[1],
            ),
        ]
    )
    # Change the bar mode
    fig.update_layout(
        barmode="group",
        title_text=f"Model performance of training sample vs batch",
    )
    return fig

def _get_batch_metrics(batch_name: str, performance_metrics: list[dict]) -> dict:
    print(f'{batch_name=}')
    for batch_data in performance_metrics[::-1]: # recover performances from last run to first
        recovered_name = list(batch_data.keys())[0]
        print(f'{recovered_name=}')
        if recovered_name == batch_name:
            return batch_data
        else:
            pass