# -*- coding: utf-8 -*-

#####  Imports  #####
import ast
import pandas as pd
from scipy import stats
import numpy as np
import plotly.express as px

import plot_utils as pu

### NUMERICAL VARIABLES ###
def plot_distributions_numerical_variables(numerical_col, sample_df, batch_df):
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
        barmode="overlay"
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