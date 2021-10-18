# -*- coding: utf-8 -*-

#####  Imports  #####
import logging

from scipy import stats
import pandas as pd

#####  Set logger  #####
logger = logging.getLogger("main_logger")

numerical_metrics_dict = {
    "kruskal_wallis": stats.kruskal,
    "kolmogorov_smirnov": stats.ks_2samp
}

#####  Covariate drift metrics  #####
def compute_numerical_drift_metrics(numerical_cols: list,
                                    sample_df: pd.DataFrame, 
                                    batch_df: pd.DataFrame,
                                    numerical_metrics_dict: dict) -> dict:
    """Computes the numerical drift metrics comparing the distributions of the
    numerical features in the original dataset and the new batch.  
    Args:
        numerical_cols (list): column names containg numerical variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        numerical_metrics_dict (dict): dictionary containing stats name and corresponding scipy function for numerical value tests
    Returns:
        dict: numerical drift metrics as specified in metrics_example.json
    """
    # initialize output dict
    metrics = {}

    # compute numerical metrics
    numerical_metrics = compute_statistic(
        numerical_cols,
        sample_df,
        batch_df,
        numerical_metrics_dict
    )

    metrics.update(
        {
            "numerical_drift_metrics": {
                "numerical_metrics": numerical_metrics,
            }
        }
    )

    return metrics

def compute_statistic(selected_cols: list, sample_df: pd.DataFrame,
                      batch_df: pd.DataFrame, stats_dict: dict) -> dict:
    """Computes required test statistics and the associated p-values for a set of columns
    present in both the original data and the batch data. 
    Args:
        selected_cols (list): column names to be used
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        stats_dict (dict):  dictionary containing stats name and corresponding scipy 
        function for numerical value tests. Must follow the naming principles set in metrics_example.json
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
            test_val, p_val = stats_func(sample_df[col], batch_df[col])
            # store metrics in dict
            metrics_dict[col].update(
                {stats_name: {"test_val": test_val, "p_val": p_val}}
            )

    return metrics_dict