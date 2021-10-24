# -*- coding: utf-8 -*-

#####  Imports  #####
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    jaccard_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from sklearn.base import ClassifierMixin, clone

from preprocessing import preprocessing
import constants as cst
import numpy as np

#####  Set Logger  #####
from src.utils.loggers import MainLogger

logger = MainLogger.getLogger(__name__)


def cross_evaluate_model_performance(
    clf: ClassifierMixin, X: pd.DataFrame, y: pd.DataFrame
) -> dict:
    """
    Main Cross Evaluation Function: computes metrics based on a basic train-test split
    and save them into a json file
    Args:
        clf: trained model used for the metrics
        X (pd.DataFrame): features
        y (pd.DataFrame): target
        conf (dict):  Configuration file stored as a json object

    Returns: Dict of classification performance metrics
    """
    df = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = preprocessing.basic_split(df, cst.y_name)

    clf2 = clone(clf)
    clf2.fit(X_train, y_train)

    cv_metrics = evaluate_model_performance_on_test(clf2, X_test, y_test)
    return cv_metrics


def evaluate_model_performance_on_test(
    clf: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> dict:
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
    metrics["f1_score"] = f1_score(y_test, y_test_pred)
    metrics["accuracy"] = accuracy_score(y_test, y_test_pred)
    metrics["recall"] = recall_score(y_test, y_test_pred)
    metrics["precision"] = precision_score(y_test, y_test_pred)
    metrics["confusion_matrix"] = metric_confusion_matrix(y_test, y_test_pred)
    metrics["auc"] = roc_auc_score(y_test, y_test_pred)
    metrics["jaccard_score"] = jaccard_score(y_test, y_test_pred)

    return metrics


def metric_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    dict_confusion_matrix = {"tn": str(tn), "fp": str(fp), "fn": str(fn), "tp": str(tp)}
    return dict_confusion_matrix
