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

def compute_metrics(sample_df: pd.DataFrame, batch_df: pd.DataFrame, model, metrics_dict) -> dict:
    # initialize output dict
    metrics = dict()
    # recover metrics config
    covariate_drift_metrics_dict = metrics_dict['covariate_drift']
    concept_drift_metrics_dict = metrics_dict['concept_drift']
    
    sample_pred, batch_pred = model.predict_proba(sample_df)[:, -1], model.predict_proba(batch_df)[:, -1]
    
    # calculate metrics
    covariate_drift_metrics = compute_covariate_drift_metrics(sample_df, batch_df, metrics_dict=covariate_drift_metrics_dict)
    concept_drift_metrics = compute_concept_drift_metrics(sample_pred, batch_pred, metrics_dict=concept_drift_metrics_dict)
    
    metrics.update(covariate_drift_metrics)
    metrics.update(concept_drift_metrics)
    
    return metrics 