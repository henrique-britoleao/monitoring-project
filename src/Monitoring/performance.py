# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import recall_score, precision_score, f1_score, 
from sklearn.metrics import accuracy_score, roc_auc_score, jaccard_score
from sklearn.metrics import confusion_matrix
import numpy as np
import json

def main_model_evaluation(clf: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """
    Main Evaluation Function: computes metrics and save them into a json file
    Args:
        clf: trained model used for the metrics
        X_test (pd.DataFrame): X_test
        y_test (pd.DataFrame): y_test
        conf (dict):  Configuration file stored as a json object

    Returns: Dict of classification performance metrics
    """
    y_test_pred = np.array([clf.predict(X_test) >= 0.5], dtype=np.float32)[0]

    dict_metrics = {}
    dict_metrics['f1_score'] = f1_score(y_test, y_test_pred)
    dict_metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
    dict_metrics['recall'] = recall_score(y_test, y_test_pred)
    dict_metrics['precision'] = precision_score(y_test, y_test_pred)
    dict_metrics['confusion_matrix'] = metric_confusion_matrix(y_test, y_test_pred)
    dict_metrics['auc'] = roc_auc_score(y_test, y_test_pred)
    dict_metrics['jaccard_score'] = jaccard_score(y_test, y_test_pred)

    return dict_metrics

def metric_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tn, fp, fn, tp =  confusion_matrix(y_true, y_pred).ravel()
    dict_confusion_matrix = {'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    return dict_confusion_matrix

def save_training_performance_metrics(dict_metrics: dict, conf: dict) -> None:
    """
    Saves the dictionary containing model performance metrics to a json file

    Args:
        dict_metrics (dict): Dict of classification performance metrics
        conf (dict): Configuration file stored as a json object
    """
    with open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'training_metrics_'
            + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as outfile:
        json.dump(str(dict_metrics), outfile)

def save_batch_performance_metrics(dict_batch_metrics: dict, batch_id: str, conf: dict) -> None:
    with open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'monitoring_metrics_'
            + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as metrics_json_path:
            dict_metrics = json.load(metrics_json_path)

    dict_metrics[batch_id]["metrics"].update(
        {
            "model_performance_metrics": dict_batch_metrics
        }
    )

    with open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'monitoring_metrics_'
            + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as metrics_json_path:
        json.dump(dict_metrics,metrics_json_path)