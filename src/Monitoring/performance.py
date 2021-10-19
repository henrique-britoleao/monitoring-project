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

    metrics = {}
    metrics['f1_score'] = f1_score(y_test, y_test_pred)
    metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
    metrics['recall'] = recall_score(y_test, y_test_pred)
    metrics['precision'] = precision_score(y_test, y_test_pred)
    metrics['confusion_matrix'] = metric_confusion_matrix(y_test, y_test_pred)
    metrics['auc'] = roc_auc_score(y_test, y_test_pred)
    metrics['jaccard_score'] = jaccard_score(y_test, y_test_pred)

    return metrics

def metric_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tn, fp, fn, tp =  confusion_matrix(y_true, y_pred).ravel()
    dict_confusion_matrix = {'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    return dict_confusion_matrix