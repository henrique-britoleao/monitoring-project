# -*- coding: utf-8 -*-

#####  Imports  #####
from typing import Callable
import logging

from scipy.stats import chi2_contingency, fisher_exact, kruskal, ks_2samp

from skmultiflow.drift_detection import ADWIN, HDDM_A

import pandas as pd
import numpy as np

import constants as cst
import monitoring.detect_alert as detect_alert
from operator import le, lt, ge, gt

#####  Set logger  #####
logger = logging.getLogger("main_logger")

#####  Covariate drift metrics  #####
StatisticComputer = Callable[[pd.Series, pd.Series], dict[str, float]]

def compute_covariate_drift_metrics(
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
        E.g.: 
        
        {
            "covariate_drift_metrics": {
                "numerical_metrics": {
                    "column_1": {
                        "test_1": {
                            "test_value": "double"
                        },
                        ...
                    }
                },
                "categorical_metrics": {
                    "column_1": {
                        "test_1": {
                            "test_value": "double"
                        },
                    }
                },
                "binary_metrics": {
                    "column_1": {
                        "test_1": {
                            "test_value": "double"
                        },
                    }
                }
        }
    """
    # initialize output dict
    metrics = dict()
    
    dfs = [sample_df, batch_df]
    # compute metrics
    numerical_metrics = compute_statistics(cst.numerical_columns, *dfs,
                                           metrics_dict['numerical'])
    categorical_metrics = compute_statistics(cst.categorical_columns, *dfs, 
                                             metrics_dict['categorical'])
    binary_metrics = compute_statistics(cst.binary_columns, *dfs, 
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
        for stats_name, config_dict in stats_dict.items():
            stats_func = globals()[config_dict['function']]
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
    alert = detect_alert.alert(p_val, "chi_squared", "categorical", "covariate_drift", le)

    return {"test_val": test_val, "p_val": p_val, "alert": alert}

def compute_fisher_stats(sample_data: pd.Series, batch_data: pd.Series, 
                         **kwargs):
    """Wrapper of scipy.stats.fisher_exact function."""
    contingency_table = build_contingency_table(sample_data, batch_data)
    test_val, p_val = fisher_exact(contingency_table, *kwargs)
    alert = detect_alert.alert(p_val, "fisher_test", "binary", "covariate_drift", le)

    return {"test_val": test_val, "p_val": p_val, "alert": alert}

def compute_kruskal_wallis_test(sample_data: pd.Series, batch_data: pd.Series, 
                                **kwargs):
    """Wrapper of scipy.stats.kruskal function."""
    test_val, p_val = kruskal(sample_data, batch_data, **kwargs)
    alert = detect_alert.alert(p_val, "kruskal_wallis", "numerical", "covariate_drift", le)
    
    return {"test_val": test_val, "p_val": p_val, "alert": alert}

def compute_kolmogorov_smirnov_test(sample_data: pd.Series, 
                                    batch_data: pd.Series, **kwargs):
    """Wrapper of scipy.stats.ks_2samp function."""
    test_val, p_val = ks_2samp(sample_data, batch_data, **kwargs)
    alert = detect_alert.alert(p_val, "kolmogorov_smirnov", "numerical", "covariate_drift", le)
    
    return {"test_val": test_val, "p_val": p_val, "alert": alert}


#####  Compute CSI metrics  #####
def compute_csi_numerical(
    sample_data: pd.Series, batch_data: pd.Series, buckets: int=10
) -> float:
    """Computes covariate drift between sample and batch data of specific column
       for numerical columns

    Args:
        sample_data (pd.Series): column containing the data from the original
        sample
        batch_data (pd.Series): column containing the data from the batch sample
        n_buckets (int, optional): number of buckets to split values in. Defaults to 10.

    Returns:
        float: CSI value
    """
    # define probability breakpoints for each bucket
    raw_breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(raw_breakpoints, float(np.min(sample_data)), float(np.max(sample_data)))

    # value count of probabilities for each bucket
    sample_counts = np.histogram(sample_data, breakpoints)[0]
    batch_counts = np.histogram(batch_data, breakpoints)[0]

    # create dataframe with buckets for each breakpoint value, counts for sample and batch values
    csi_df = pd.DataFrame(
        {
            "Bucket": np.arange(1, buckets + 1),
            "Breakpoint Value": breakpoints[1:],
            "Initial Count": sample_counts,
            "New Count": batch_counts,
        }
    )
    csi_df["Initial Percent"] = csi_df["Initial Count"] / len(sample_data)
    csi_df["New Percent"] = csi_df["New Count"] / len(batch_data)

    # add psi value to each bucket
    csi_df["CSI"] = (csi_df["New Percent"] - csi_df["Initial Percent"]) * np.log(
        csi_df["New Percent"] / csi_df["Initial Percent"]
    )
    csi_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    csi_df.fillna(0, inplace=True)

    # get overall PSI
    csi_total = np.round(sum(csi_df["CSI"]), 3)
    alert = detect_alert.alert(csi_total , "CSI", "numerical", "covariate_drift", ge)
    
    return {"csi_value": csi_total, "alert": alert}


def compute_csi_categorical(
    sample_data: pd.Series, batch_data: pd.Series, buckets: int=10
) -> float:
    """Computes covariate drift between sample and batch data of specific column
       for categorical columns

    Args:
        sample_data (pd.Series): column containing the data from the original
        sample
        batch_data (pd.Series): column containing the data from the batch sample
        n_buckets (int, optional): number of buckets to split values in. Defaults to 10.

    Returns:
        float: CSI value
    """

    # encode category and save in array
    sample_dist = sample_data.astype("category").cat.codes
    batch_dist = batch_data.astype("category").cat.codes

    # define number of buckets as number of cats
    buckets = sample_dist.nunique()
    breakpoints = np.append(
        sample_dist.unique(), max(sample_dist.unique()) + 1
    ).tolist()
    breakpoints.sort()

    # value count of probabilities for each bucket
    sample_counts = np.histogram(sample_dist, breakpoints)[0]
    batch_counts = np.histogram(batch_dist, breakpoints)[0]

    # create dataframe with buckets for each breakpoint value, counts for dev and new probabilities
    csi_df = pd.DataFrame(
        {
            "Bucket": np.arange(1, buckets + 1),
            "Breakpoint Value": breakpoints[1:],
            "Initial Count": sample_counts,
            "New Count": batch_counts,
        }
    )
    csi_df["Initial Percent"] = csi_df["Initial Count"] / len(sample_dist)
    csi_df["New Percent"] = csi_df["New Count"] / len(batch_dist)

    # add psi value to each bucket
    csi_df["CSI"] = (csi_df["New Percent"] - csi_df["Initial Percent"]) * np.log(
        csi_df["New Percent"] / csi_df["Initial Percent"]
    )
    csi_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    csi_df.fillna(0, inplace=True)

    # get overall PSI
    csi_total = np.round(sum(csi_df["CSI"]), 3)
    alert = detect_alert.alert(csi_total , "CSI", "categorical", "covariate_drift", ge)

    return {"csi_value": csi_total, "alert": alert}


#####  Compute data drift  #####
def detect_drift_adwin_method(sample_data: pd.Series, batch_data: pd.Series) -> dict[str, int]:
    """
    TODO: redo this
    Detects numerical drift by adding stream elements from the new batch to ADWIN
    and verifying if drift occurred (in the batch data only).
    
    Args:
        numerical_cols (list): column names containg numerical variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
    Return:
        None
    """
    adwin = ADWIN()
    data_stream = np.concatenate((sample_data, batch_data))
    col_name = sample_data.name
    
    # initialize alert
    alert = 0
    
    # Adding stream elements to ADWIN and verifying if drift occurred
    for i in range(len(data_stream)):
        adwin.add_element(data_stream[i])
        if adwin.detected_change():
            alert = 1
            logger.warning('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i) + 'for column:' + col_name)
            
    return {"alert": alert}
    
#####  Utils functions  #####
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

def scale_range(input, min, max) -> np.array:
    """Scales the raw breakpoints from 0 to 1.

    Args:
        input (array): raw breakpoints
        min (float): minimum score of sample batch
        max (float): maximum score of sample batch

    Returns:
        array: scaled breakpoints
    """
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input