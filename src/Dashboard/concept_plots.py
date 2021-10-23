# -*- coding: utf-8 -*-

#####  Imports  #####
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

import constants as cst


def graph_target_prob_dist(
    sample_df_pred_path: str,
    batch_df_pred_path: str = cst.PREDICTED_TRAIN_FILE_PATH,
    colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"],
):
    """Plots the distribution of the prediction probabilities.
    Args:
        sample_df_pred (pd.DataFrame): original sample with prediction probabilities
        batch_df_pred (pd.DataFrame): new sample with prediction probabilities
        colors (list, optional): List of colors to use. Defaults to
        ["rgb(0, 0, 100)", "rgb(0, 200, 200)"].
    Returns:
        ff.Figure: Histogram with probability density curve

    """
    # load dataframe
    sample_df_pred = pd.read_csv(sample_df_pred_path, sep=";")
    batch_df_pred = pd.read_csv(batch_df_pred_path, sep=";")

    # Histogram configuration
    hist_data = [
        sample_df_pred[cst.y_pred_proba_main_class],
        batch_df_pred[cst.y_pred_proba_main_class],
    ]
    group_labels = ["Sample Prediction Probabilities", "New Prediction Probabilities"]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
        hist_data=hist_data,
        group_labels=group_labels,
        bin_size=0.01,
        colors=colors,
        curve_type="kde",
        show_rug=False,
        histnorm="probability density",
    )
    # Add title
    fig.update(
        layout_title_text=f"Probability Distribution of target {cst.y_pred_proba}"
    )
    return fig


def graph_target_labels(
    sample_df_pred: pd.DataFrame,
    batch_df_pred: pd.DataFrame,
    colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"],
):
    """Plots distribution of (predicted) labels for sample and new batch
    Args:
        sample_df_pred (pd.DataFrame): initial sample with predictions
        batch_df_pred (pd.DataFrame): new sample with predictions
        colors (list, optional): Bar colors. Defaults to ["rgb(0, 0, 100)", "rgb(0, 200, 200)"].
    """

    # compute target distribution ratio
    class_0_sample = np.round(
        sample_df_pred[cst.y_pred].value_counts()[0] / sample_df_pred.shape[0], 2
    )
    class_1_sample = np.round(
        sample_df_pred[cst.y_pred].value_counts()[1] / sample_df_pred.shape[0], 2
    )

    class_0_new = np.round(
        batch_df_pred[cst.y_pred].value_counts()[0] / batch_df_pred.shape[0], 2
    )
    class_1_new = np.round(
        batch_df_pred[cst.y_pred].value_counts()[1] / batch_df_pred.shape[0], 2
    )

    fig = go.Figure(
        data=[
            go.Bar(
                name="Sample Predictions",
                x=cst.y_class_labels,
                y=[class_0_sample, class_1_sample],
                text=[f"{int(class_0_sample*100)}%", f"{int(class_1_sample*100)}%"],
                marker_color=colors[0],
            ),
            go.Bar(
                name="Batch Predictions",
                x=cst.y_class_labels,
                y=[class_0_new, class_1_new],
                text=[f"{int(class_0_new*100)}%", f"{int(class_1_new*100)}%"],
                marker_color=colors[1],
            ),
        ]
    )
    # Change the bar mode
    fig.update_layout(barmode="group", title_text="Label Prediction Distribution in %")
    return fig
