# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from sklearn.metrics import recall_score, precision_score
import numpy as np
import json

#This script can be optimized depending on your needs

def main_evaluation(clf,X_train,y_train, X_test, y_test, conf):
    """
    Main Evaluation Function: computes metrics and save them into a json file
    Args:
        clf: model used for the metrics
        X_train:  X_train
        y_train: y_train
        X_test: X_test
        y_test: y_test
        conf:  Conf File

    Returns: Dict of metrics and saves the metrics into a json file

    """


    y_test_pred = np.array([clf.predict(X_test) >= 0.5], dtype=np.float32)[0]
    y_train_pred = np.array([clf.predict(X_train) >= 0.5], dtype=np.float32)[0]

    dict_metrics = {}
    dict_metrics['f1_score'] = metric_f1_score(y_test, y_test_pred)
    dict_metrics['accuracy'] = metric_accuracy(y_test, y_test_pred)
    dict_metrics['recall'] = metric_recall(y_test, y_test_pred)
    dict_metrics['precision'] = metric_precision(y_test, y_test_pred)
    dict_metrics['confusion_matrix'] = metric_confusion_matrix(y_test, y_test_pred)

    with open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'metrics_'
              + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as outfile:
        json.dump(str(dict_metrics), outfile)
    return dict_metrics

def metric_f1_score( y_true, y_pred, average = 'binary'):
     return f1_score(y_true, y_pred, average = average)

def metric_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def metric_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def metric_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def metric_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp =  confusion_matrix(y_true, y_pred).ravel()
    dict_confusion_matrix = {'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    return dict_confusion_matrix