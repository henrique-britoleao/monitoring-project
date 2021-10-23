# -*- coding: utf-8 -*-

#####  Imports  #####
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard import plot_utils as pu

### CATEGORICAL VARIABLES ###
def graph_categorical_dist(    
    sample_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    categorical_col: str,
    colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"],
):
    """Graph to plot the distribution of sample and batch data for a given column

    Args:
        sample_df (pd.DataFrame): training data
        batch_df (pd.DataFrame): batch data
        categorical_col (str): categorical column to plot distribution 
        colors (list, optional): Defaults to ["rgb(0, 0, 100)", "rgb(0, 200, 200)"].

    Returns:
        Figure: figure to display in the dashboard
    """
    sample_col_distrib = sample_df[categorical_col].value_counts(normalize=True)*100
    batch_col_distrib = batch_df[categorical_col].value_counts(normalize=True)*100

    fig = go.Figure(
            data=[
            go.Bar(
                x=sample_col_distrib.index, 
                y=sample_col_distrib.values,
                text=[f"{np.round(v, 1)}%" for v in sample_col_distrib.values],
                marker_color=colors[0],
                name=f"Sample Distrib. of {categorical_col}",
            ),
            go.Bar(
                x=batch_col_distrib.index, 
                y=batch_col_distrib.values,
                text=[f"{np.round(v, 1)}%" for v in batch_col_distrib.values],
                marker_color=colors[1],
                name=f"Batch Distrib. of {categorical_col}",
            ),
        ]
    )

    pu.update_fig_centered_title(fig, f'Sample vs Batch Class Distribution in column {categorical_col}')

    fig.update_layout(legend=dict(
        yanchor="top",
        xanchor="right",
    ))

    return fig

def graph_categorical_dist_diff(    
    sample_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    categorical_col: str,
    colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"],
):
    """Graph to plot the difference in distribution between the sample and batch data for a given column

    Args:
        sample_df (pd.DataFrame): training data
        batch_df (pd.DataFrame): batch data
        categorical_col (str): categorical column to plot distribution 
        colors (list, optional): Defaults to ["rgb(0, 0, 100)", "rgb(0, 200, 200)"].

    Returns:
        Figure: figure to display in the dashboard
    """
    sample_col_distrib = sample_df[categorical_col].value_counts(normalize=True)*100
    batch_col_distrib = batch_df[categorical_col].value_counts(normalize=True)*100
    sample_batch_distrib_diff = (batch_col_distrib - sample_col_distrib).fillna(0)

    fig = go.Figure(
            data=[
            go.Bar(
                x=sample_batch_distrib_diff.index, 
                y=sample_batch_distrib_diff.values,
                text=[f"{np.round(v, 1)}%" for v in sample_batch_distrib_diff.values],
                marker_color=colors[0],
            ),
        ]
    )

    pu.update_fig_centered_title(fig, f'Sample vs Batch Class Distribution Difference in column {categorical_col}')
    fig.update_layout(legend=dict(
        yanchor="top",
        xanchor="right",
    ))

    return fig

