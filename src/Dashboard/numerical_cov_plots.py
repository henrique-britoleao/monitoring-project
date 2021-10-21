# -*- coding: utf-8 -*-

#####  Imports  #####
import ast
import pandas as pd
from scipy import stats
import numpy as np
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

import constants as cst

from Dashboard import plot_utils as pu

### NUMERICAL VARIABLES ###
def plot_scaled_means(sample_df, batch_df, colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"]):
    scaler = MinMaxScaler()
    scaled_sample_df, scaled_batch_df = sample_df.loc[:, cst.numerical_columns], batch_df.loc[:, cst.numerical_columns]

    scaled_sample_df.loc[:, cst.numerical_columns] = scaler.fit_transform(scaled_sample_df.loc[:, cst.numerical_columns])
    scaled_batch_df.loc[:, cst.numerical_columns] = scaler.transform(scaled_batch_df.loc[:, cst.numerical_columns])

    sample_means = pd.DataFrame(
        data={
            "Source": "Sample",
            "Numerical Column": scaled_sample_df.mean().index, 
            "Scaled Mean": scaled_sample_df.mean().values
        }
    )

    batch_means = pd.DataFrame(
        data={
            "Source": "Batch",
            "Numerical Column": scaled_batch_df.mean().index, 
            "Scaled Mean": scaled_batch_df.mean().values
        }
    )
    data = pd.concat([sample_means, batch_means])

    fig = px.bar(
        data, 
        x='Numerical Column', 
        y='Scaled Mean', 
        text=[f"{np.round(m, 2)}" for m in data['Scaled Mean']],
        color='Source', 
        barmode='group', 
        color_discrete_sequence=colors
    )

    pu.update_fig_centered_title(fig, "Sample vs Batch Scaled Means for numerical features")
    
    return fig

def plot_quartiles_numerical_variables(sample_df, batch_df, numerical_col, colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"]):
    sample_data = pd.DataFrame(data={'source': 'Sample', numerical_col: sample_df[numerical_col]})
    batch_data = pd.DataFrame(data={'source': 'Batch', numerical_col: batch_df[numerical_col]})
    data = pd.concat([sample_data, batch_data])

    fig = px.box(data, x="source", y=numerical_col, color="source", color_discrete_sequence=colors)
    fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
    fig.update_layout(
        xaxis_title="Dataset", 
        yaxis_title=numerical_col
        )

    pu.update_fig_centered_title(fig, f"Sample vs Batch Distribution in column {numerical_col}")
 
    return fig

def plot_distributions_numerical_variables(sample_df, batch_df, numerical_col, colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"]):
    optim_n_bins = find_optimal_n_bins(sample_df[numerical_col])
    distrib_data = get_binned_data(sample_df[numerical_col], batch_df[numerical_col], optim_n_bins)

    unstacked_distrib_data = (
        distrib_data
            .set_index('bin_upper_bound')
            .unstack()
            .reset_index()
            .rename(columns={0: 'proportion', 'level_0': 'source'})
    )

    fig = px.bar(
        unstacked_distrib_data,
        x="bin_upper_bound", 
        y="proportion", 
        color="source", 
        barmode="overlay",
        color_discrete_sequence=colors
    )

    pu.update_fig_centered_title(fig, f"Sample vs Batch Distribution in column {numerical_col}")

    fig.update_layout(
            legend=dict(
                title="",
                yanchor="top",
                xanchor="right",
            ), 
            xaxis_title='Bin upper bound',
            yaxis_title='Distribution (%)'
    )
    return fig

def get_binned_data(sample_data: pd.Series, batch_data: pd.Series, n_bins: int):

    # Bin sample and batch data 
    binned_sample_data, bins_sample_data = pd.cut(sample_data, bins=n_bins, retbins=True)
    binned_batch_data = pd.cut(batch_data, bins=bins_sample_data)

    # Compute distribution in each bin
    distrib_sample_data = binned_sample_data.astype(str).value_counts(normalize=True).sort_index()
    distrib_batch_data = binned_batch_data.astype(str).value_counts(normalize=True).sort_index()
    
    # Combine binned sample and batch data
    distrib_data = pd.DataFrame(index=distrib_sample_data.index)
    distrib_data['sample_distribution'] = distrib_sample_data
    distrib_data['batch_distribution'] = distrib_batch_data
    distrib_data['batch_distribution'].fillna(0, inplace=True)

    # Get upper bound in interval
    distrib_data['bin_upper_bound'] = [get_upper_bound_interval(interval) for interval in distrib_data.index]
    return distrib_data.sort_values(by='bin_upper_bound', ascending=True)
 
def get_upper_bound_interval(interval):
    upper_bound = ast.literal_eval(interval.replace('(', '['))[1]
    return int(upper_bound)

def find_optimal_n_bins(sample_data):
    sample_iqr = stats.iqr(sample_data)
    sample_size = len(sample_data)
    optim_binwidth = np.round(2*sample_iqr/np.cbrt(sample_size))
    optim_n_bins = int((sample_data.max() - sample_data.min())/optim_binwidth)
    return optim_n_bins