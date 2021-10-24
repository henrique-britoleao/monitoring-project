import json
import constants as cst
import plotly.graph_objects as go
import pandas as pd


def plot_performance(
    batch_perf_path: str = cst.PERFORMANCE_METRICS_FILE_PATH,
    batch_id:int = 0,
    train_perf_path: str = cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH,
    batch_name: str = "batch1",
):
    """Creates barplot to compare training performance vs batch performance
    Args:
        json_path (str, optional): path to json file with performance metrics of models.
                                   Defaults to constants.PERFORMANCE_METRICS_FILE_PATH.
        batch_name (str, optional): Which batch is being analysed. Defaults to "batch1".
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
        batch_perf = json.load(json_file)

    # saving performance keys of chosen batch
    print(batch_id, batch_name)
    perf_metrics = list(batch_perf[batch_id][batch_name].keys())
    # delete confusion matrix
    perf_metrics.pop(4)
    # saving performance values of chosen batch
    perf_values = list(batch_perf[batch_id][batch_name].values())
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
                marker_color=colors[0],
            ),
            go.Bar(
                name="Batch Performance ",
                x=perf_metrics,
                y=perf_values,
                marker_color=colors[1],
            ),
        ]
    )
    # Change the bar mode
    fig.update_layout(
        barmode="group",
        title_text=f"Model performance of training sample vs {batch_name}",
    )
    return fig

def plot_perf_evolution(
    metric: str,
    batch_perf_path: str = cst.PERFORMANCE_METRICS_FILE_PATH,
    train_perf_path: str = cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH,
):
    '''TODO'''
    with open(train_perf_path) as json_file:
        train_perf = json.load(json_file)

    train_value = train_perf[metric]
    with open(batch_perf_path) as json_file:
        batch_perf = json.load(json_file)

    batch_values = []
    for i in range(len(batch_perf)):
        batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=i+1)[:-4]
        batch_values.append(batch_perf[i][batch_name][metric])

    x = [f'Batch{i+1}' for i in range(len(batch_values))]
    train_values = [train_value]*len(batch_values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=batch_values, name="Batches",
                        hoverinfo='text',
                        line_shape='spline'))

    fig.add_trace(go.Scatter(x=x, y=train_values, name="Training",
                        line = dict(color='red', width=4, dash='dash')))

    fig.update_traces(hoverinfo='name', mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))
    fig.show()

    return fig


