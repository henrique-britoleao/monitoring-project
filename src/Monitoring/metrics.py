# -*- coding: utf-8 -*-

#####  Imports  #####
from typing import Callable, Tuple
import logging

from scipy.stats import chi2_contingency, fisher_exact
import pandas as pd
import numpy as np

#####  Set logger  #####
logger = logging.getLogger("main_logger")

#####  Covariate drift metrics  #####
def compute_covariate_drift_metrics(categorical_cols: list, binary_cols: list,
                                    sample_df: pd.DataFrame, 
                                    batch_df: pd.DataFrame) -> dict:
    """Computes the covariate drift metrics comparing the distributions of the
    covariates in the original dataset and the new batch.  

    Args:
        categorical_cols (list): column names containg categorical variables
        binary_cols (list): column names containg binary variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch

    Returns:
        dict: covariate drift metrics as specified in metrics_example.json
    """
    # initialize output dict
    metrics = {}
    # compute metrics
    categorical_metrics = compute_statistic(
        categorical_cols,
        sample_df,
        batch_df,
        stats_name="chi_squared",
        stats_func=compute_chi_sq_stats,
    )
    binary_metrics = compute_statistic(
        binary_cols,
        sample_df,
        batch_df,
        stats_name="fisher_test",
        stats_func=fisher_exact,
    )

    metrics.update(
        {
            "covariate_drift_metrics": {
                "categorical_metrics": categorical_metrics,
                "binary_metrics": binary_metrics,
            }
        }
    )

    return metrics


StatisticComputer = Callable[[pd.DataFrame], Tuple[float, float]]


def compute_chi_sq_stats(
    contingency_table: pd.DataFrame, *kwargs
) -> Tuple[float, float]:
    """Wrapper of scipy.stats chi2_contingency that makes it a StatisticComputer."""
    test_val, p_val, _, _ = chi2_contingency(contingency_table, *kwargs)

    return test_val, p_val


def compute_statistic(selected_cols: list, sample_df: pd.DataFrame,
                      batch_df: pd.DataFrame, stats_name: str,
                      stats_func: StatisticComputer) -> dict:
    """Computes a test statistic and the associated p-value for a set of columns
    present in both the original data and the batch data. 

    Args:
        selected_cols (list): column names to be used
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        stats_name (str): name of the statistic to be assigned in the output 
        dictionary. Must follow the naming principles set in metrics_example.json
        stats_func (StatisticComputer): function that computes the metric

    Returns:
        dict: contains test statistics and p-values following the schema set in 
        metrics_example.json
    """
    # initilaize output dictionary
    metrics_dict = dict()
    # populate metrics dictionary
    for col in selected_cols:
        contingency_table = build_contingency_table(
            sample_data=sample_df.loc[:, col], batch_data=batch_df.loc[:, col]
        )
        # calculate metrics
        test_val, p_val = stats_func(contingency_table)
        # store metrics in dict
        metrics_dict.update(
            {col: {stats_name: {"test_val": test_val, "p_val": p_val}}}
        )

    return metrics_dict


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
