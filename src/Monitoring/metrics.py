# -*- coding: utf-8 -*-

#####  Imports  #####
from typing import Callable, Tuple
import logging

from scipy.stats import chi2_contingency, fisher_exact, kruskal, ks_2samp

import pandas as pd
import numpy as np

#####  Set logger  #####
logger = logging.getLogger("main_logger")

#####  Covariate drift metrics  #####
StatisticComputer = Callable[[pd.Series, pd.Series], dict[str, dict[str, float]]]

def compute_covariate_drift_metrics(
    numerical_cols: list, categorical_cols: list, binary_cols: list,
    sample_df: pd.DataFrame, batch_df: pd.DataFrame, 
    metrics_dict: dict[str, dict[str, StatisticComputer]]) -> dict:
    """Computes the covariate drift metrics comparing the distributions of the
    covariates in the original dataset and the new batch.  

    Args:
        numerical_cols (list): column names containg numerical variables
        categorical_cols (list): column names containg categorical variables
        binary_cols (list): column names containg binary variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        metrics_dict (dict): contains the functions to be used to calculate the
        statistics on each column. Should follow the structure:
        {
            "numerical": {
                "stats_name_1": StatisticComputer_1,
                ...
            },
            "categorical": {
                "stats_name_2": StatisticComputer_2,
                ...
            },
            "binary": {
                "stats_name_3": StatisticComputer_3,
                ...
            },
        }

    Returns:
        dict: covariate drift metrics as specified in metrics_example.json
    """
    # initialize output dict
    metrics = {}
    
    dfs = [sample_df, batch_df]
    # compute metrics
    numerical_metrics = compute_statistics(numerical_cols, *dfs,
                                           metrics_dict['numerical'])
    categorical_metrics = compute_statistics(categorical_cols, *dfs, 
                                             metrics_dict['categorical'])
    binary_metrics = compute_statistics(binary_cols, *dfs, 
                                        metrics_dict['binary'])

    # update output dict
    metrics.update(
        {
            "covariate_drift_metrics": {
                "numerical_metrics": numerical_metrics,
                "categorical_metrics": categorical_metrics,
                "binary_metrics": binary_metrics,
            }
        }
    )

    return metrics

def compute_statistics(selected_cols: list, sample_df: pd.DataFrame,
                       batch_df: pd.DataFrame, 
                       stats_dict: dict[str, StatisticComputer]) -> dict:
    """Computes required test statistics and the associated p-values for a set
    of columns present in both the original data and the batch data. 
    
    Args:
        selected_cols (list): column names to be used
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        stats_dict (dict):  dictionary containing stats name and corresponding 
        StatisticComputer. Must be of the shape:
        {
            "stats_1": StatisticComoputer1
            ...
        }
        
    Returns:
        dict: contains test statistics and p-values following the schema set in 
        metrics_example.json
    """
    # initilaize output dictionary
    metrics_dict = dict()
    # populate metrics dictionary
    for col in selected_cols:
        metrics_dict[col] = {}
        # calculate metrics
        for stats_name, stats_func in stats_dict.items():
            stats_results = stats_func(sample_df[col], batch_df[col])
            # store metrics in dict
            metrics_dict[col].update(
                {stats_name: stats_results}
            )

    return metrics_dict


#####  Statistics wrappers  #####
def compute_chi_sq_stats(sample_data: pd.Series, batch_data: pd.Series, 
                         **kwargs):
    """Wrapper of scipy.stats.chi2_contingency function."""
    contingency_table = build_contingency_table(sample_data, batch_data)
    test_val, p_val, _, _ = chi2_contingency(contingency_table, *kwargs)

    return {"test_val": test_val, "p_val": p_val}

def compute_fisher_stats(sample_data: pd.Series, batch_data: pd.Series, 
                         **kwargs):
    """Wrapper of scipy.stats.fisher_exact function."""
    contingency_table = build_contingency_table(sample_data, batch_data)
    test_val, p_val = fisher_exact(contingency_table, *kwargs)

    return {"test_val": test_val, "p_val": p_val}

def compute_kruskal_wallis_test(sample_data: pd.Series, batch_data: pd.Series, 
                                **kwargs):
    """Wrapper of scipy.stats.kruskal function."""
    test_val, p_val = kruskal(sample_data, batch_data, **kwargs)
    
    return {"test_val": test_val, "p_val": p_val}

def compute_kolmogorov_smirnov_test(sample_data: pd.Series, 
                                    batch_data: pd.Series, **kwargs):
    """Wrapper of scipy.stats.ks_2samp function."""
    test_val, p_val = ks_2samp(sample_data, batch_data, **kwargs)
    
    return {"test_val": test_val, "p_val": p_val}

def build_contingency_table(sample_data: pd.Series, 
                            batch_data: pd.Series) -> pd.DataFrame:
    """Builds a contingency table for categorical and binary data contained in 
    the original and batch data.

    Args:
        sample_data (pd.Series): column containing the data from the original 
        sample
        batch_data (pd.Series): column containing the data from the batch sample

    Returns:
        pd.DataFrame: contingency table
    """
    categorical_values = pd.concat([sample_data, batch_data])
    data_origins = np.array(
        ["sample"] * len(sample_data) + ["batch"] * len(batch_data)
    )

    return pd.crosstab(index=categorical_values, columns=data_origins)