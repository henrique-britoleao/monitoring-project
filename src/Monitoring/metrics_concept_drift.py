# -*- coding: utf-8 -*-

#####  Imports  #####
import pandas as pd
import numpy as np

from typing import Callable
import logging

import detect_alert
from operator import le, lt, ge, gt

#####  Set logger  #####
logger = logging.getLogger(__name__)

#####  Concept drift metrics  #####
StatisticComputer = Callable[[pd.Series, pd.Series], dict[str, float]]

def compute_concept_drift_metrics(
    sample_target: pd.Series, batch_target: pd.Series,  
    metrics_dict: dict[str, StatisticComputer]
) -> dict:
    metrics = dict()
    
    for metric_name, config_dict in metrics_dict.items():
        # recover function from name
        statistic_computer = globals()[config_dict['function']]
        # compute statistic
        statistic_value = statistic_computer(sample_target, batch_target)
        # update metrics dict
        metrics.update({metric_name: statistic_value})
    
    return metrics

def compute_psi(
    sample_data: pd.Series, batch_data: pd.Series, buckets: int = 10
) -> float:
    """Computes covariate drift between sample and batch data of specific column
       for numerical columns

    Args:
        sample_data (pd.Series): column containing the prediction probabilities
        from the original sample
        batch_data (pd.Series): column containing the prediction probabilities
        from the batch sample
        n_buckets (int, optional): number of buckets to split values in. Defaults to 10.

    Returns:
        float: PSI value
    """
    # define probability breakpoints for each bucket
    raw_breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(raw_breakpoints, np.min(sample_data), np.max(sample_data))

    # value count of probabilities for each bucket
    sample_counts = np.histogram(sample_data, breakpoints)[0]
    batch_counts = np.histogram(batch_data, breakpoints)[0]

    # create dataframe with buckets for each breakpoint value, counts for sample and batch values
    psi_df = pd.DataFrame(
        {
            "Bucket": np.arange(1, buckets + 1),
            "Breakpoint Value": breakpoints[1:],
            "Initial Count": sample_counts,
            "New Count": batch_counts,
        }
    )
    psi_df["Initial Percent"] = psi_df["Initial Count"] / len(sample_data)
    psi_df["New Percent"] = psi_df["New Count"] / len(batch_data)

    # add psi value to each bucket
    psi_df["PSI"] = (psi_df["New Percent"] - psi_df["Initial Percent"]) * np.log(
        psi_df["New Percent"] / psi_df["Initial Percent"]
    )
    psi_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    psi_df.fillna(0, inplace=True)

    # get overall PSI
    psi_total = np.round(sum(psi_df["PSI"]), 3)

    alert = detect_alert.alert(psi_total, "PSI", "", "concept_drift", ge)

    return {"psi_value": psi_total, "alert": alert}

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
