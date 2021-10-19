# -*- coding: utf-8 -*-

#####  Imports  #####
from metrics_cov_drift import compute_covariate_drift_metrics
from metrics_concept_drift import compute_concept_drift_metrics
import pandas as pd
import json

#####  Set logger  #####
import logging
logger = logging.getLogger(__name__)

#####  Compute metrics  #####

def compute_metrics(numerical_cols: list, categorical_cols: list, binary_cols: list,
    sample_df: pd.DataFrame, batch_df: pd.DataFrame, target_col: str, metrics_dict):
    # initialize output dict
    metrics = dict()
    # recover metrics config
    covariate_drift_metrics_dict = metrics_dict['covariate_drift']
    concept_drift_metrics_dict = metrics_dict['concept_drift']
    # calculate metrics
    covariate_drift_metrics = compute_covariate_drift_metrics(numerical_cols=numerical_cols, categorical_cols=categorical_cols, binary_cols=binary_cols, sample_df=sample_df, batch_df=batch_df, metrics_dict=covariate_drift_metrics_dict)
    concept_drift_metrics = compute_concept_drift_metrics(sample_df=sample_df, batch_df=batch_df, target_col=target_col, metrics_dict=concept_drift_metrics_dict)
    
    metrics.update(covariate_drift_metrics)
    metrics.update(concept_drift_metrics)
    
    return metrics 