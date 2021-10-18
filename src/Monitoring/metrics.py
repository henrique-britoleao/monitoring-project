# -*- coding: utf-8 -*-

#####  Imports  #####
import pandas as pd
import seaborn as sns
import numpy as np

import logging

#####  Set logger  #####
logger = logging.getLogger("main_logger")

#####  Concept drift metrics  #####
def compute_concept_drift_metrics(
    sample_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    n_buckets: int,
    pred_col: str,
) -> dict:
    """Computes the concept drift metrics comparing the distributions of the
    targets in the original dataset and the new batch.

    Args:
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        n_buckets (int): number of buckets to split targets in
        pred_col (str): column name of prediction, for which PSI is computed

    Returns:
        dict: concept drift metrics as specified in metrics_example.json
    """
    # initialize output dict
    psi_metric = {}
    psi_df = compute_prob_table(
        target_col=pred_col,
        sample_df=sample_df,
        batch_df=batch_df,
        n_buckets=n_buckets,
    )
    psi = compute_sum_metric(metric_df=psi_df)
    psi_metric.update({"concept_drift_metrics": {"PSI": psi}})

    return psi_metric


#####  Concept drift metrics  #####
def compute_covariate_drift_metrics(
    sample_df: pd.DataFrame, batch_df: pd.DataFrame, n_buckets: int
) -> dict:
    """Computes the covariate drift metrics comparing the distributions of the
    covariates in the original dataset and the new batch.

    Args:
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        n_buckets (int): number of buckets to split targets in

    Returns:
        dict: dict: covariate drift metrics as specified in metrics_example.json
    """

    # initialize output dict
    csi_metrics = {}
    for col in sample_df.columns:
        csi_df = compute_prob_table(
            target_col=col,
            sample_df=sample_df,
            batch_df=batch_df,
            n_buckets=n_buckets,
        )
        # compute metrics
        csi = compute_sum_metric(metric_df=csi_df)
        csi_metrics.update({col: {"CSI": csi}})

    return {"covariate_drift_metrics": csi_metrics}


### functions for PSI and CSI ###
def compute_sum_metric(metric_df: pd.DataFrame) -> float:
    """Computes over PSI value

    Args:
        psi_df (pd.DataFrame): breakpoint values, counts and percentage count
       for the sample batch and new batch distribution of target values per bucket

    Returns:
        float: sum of PSI values
    """

    # get overall PSI
    metric_total = np.round(sum(metric_df["PSI"]), 3)
    return metric_total


def compute_prob_table(
    target_col: str, sample_df: pd.DataFrame, batch_df: pd.DataFrame, n_buckets: int
) -> pd.DataFrame:
    """Generates dataframe with breakpoint values, counts and percentage count
       for the sample batch and new batch.

    Args:
        target_col (str): name of column to analyse (y for PSI, variables for CSI)
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
        n_buckets (int): number of buckets to split targets in

    Returns:
        pd.DataFrame: distribution of target values per bucket
    """
    # save list of categorical values
    cat_features = [i for i in sample_df.columns if sample_df.dtypes[i] == "object"]

    if target_col in cat_features:
        # encode category and save in array
        sample_dist = sample_df[target_col].astype("category").cat.codes
        batch_dist = batch_df[target_col].astype("category").cat.codes
        # define number of buckets as number of cats
        buckets = sample_dist.nunique()
        breakpoints = np.append(
            sample_dist.unique(), max(sample_dist.unique()) + 1
        ).tolist()
        breakpoints.sort()

    else:
        # save array
        sample_dist = sample_df[target_col]
        batch_dist = batch_df[target_col]
        buckets = n_buckets
        # define probability breakpoints for each bucket
        raw_breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = scale_range(
            raw_breakpoints, np.min(sample_dist), np.max(sample_dist)
        )

    # value count of probabilities for each bucket
    sample_counts = np.histogram(sample_dist, breakpoints)[0]
    batch_counts = np.histogram(batch_dist, breakpoints)[0]

    # create dataframe with buckets for each breakpoint value, counts for dev and new probabilities
    psi_df = pd.DataFrame(
        {
            "Bucket": np.arange(1, buckets + 1),
            "Breakpoint Value": breakpoints[1:],
            "Initial Count": sample_counts,
            "New Count": batch_counts,
        }
    )
    psi_df["Initial Percent"] = psi_df["Initial Count"] / len(sample_dist)
    psi_df["New Percent"] = psi_df["New Count"] / len(batch_dist)

    # add psi value to each bucket
    psi_df["PSI"] = (psi_df["New Percent"] - psi_df["Initial Percent"]) * np.log(
        psi_df["New Percent"] / psi_df["Initial Percent"]
    )
    psi_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    psi_df.fillna(0, inplace=True)

    return psi_df


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
