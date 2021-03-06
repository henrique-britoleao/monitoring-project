# -*- coding: utf-8 -*-

#####  Imports  #####
from monitoring.metrics_cov_drift import compute_covariate_drift_metrics
from monitoring.metrics_concept_drift import compute_concept_drift_metrics
import pandas as pd
import json
import constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger

logger = MainLogger.getLogger(__name__)

#####  Compute metrics  #####


def compute_metrics(
    sample_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    sample_pred: pd.Series,
    batch_pred: pd.Series,
    metrics_dict: dict,
) -> dict:
    """Computes concept and convariate drifts metrics and tests 
    on batch and sample data.

    Args:
        sample_df (pd.DataFrame): sample data
        batch_df (pd.DataFrame): batch data
        sample_pred (pd.Series): sample data predictions
        batch_pred (pd.Series): batch data predictions
        metrics_dict (dictionary): metrics to compute

    Returns:
        dictionary: computed drift metrics
    """
    # initialize output dict
    metrics = dict()
    # recover metrics config
    covariate_drift_metrics_dict = metrics_dict["covariate_drift"]
    concept_drift_metrics_dict = metrics_dict["concept_drift"]

    # calculate metrics
    covariate_drift_metrics = compute_covariate_drift_metrics(
        sample_df, batch_df, metrics_dict=covariate_drift_metrics_dict
    )
    concept_drift_metrics = compute_concept_drift_metrics(
        sample_pred, batch_pred, metrics_dict=concept_drift_metrics_dict
    )

    metrics.update(covariate_drift_metrics)
    metrics.update(concept_drift_metrics)

    return metrics
