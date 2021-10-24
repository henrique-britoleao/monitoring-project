# -*- coding: utf-8 -*-
import pandas as pd

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from modeling.pipeline_builder import build_prediction_pipeline

#####  Set Logger  #####
from src.utils.loggers import MainLogger

logger = MainLogger.getLogger(__name__)

####### LIGHTGBM  ########
def get_GS_params_lightgbm():
    """
    Gives params and models to use for the grid_search using LightGBM Classifier
    Returns:Estimator and params for the grid_search

    """
    params_grid = {
        "estimator__boosting_type": ["gbdt"],
        "estimator__objective": ["binary"],
        "estimator__num_boost_round": [200],
        "estimator__learning_rate": [0.01, 0.1],
        "estimator__max_depth": [6, 100],
        "estimator__reg_alpha": [0, 0.1],
        "estimator__min_data_in_leaf": [5, 10],
        "estimator__learning_rate": [0.01],
        "estimator__scale_pos_weight": [0.2, 1, 3, 10],
        "estimator__verbose": [-1],
    }

    estimator = lgb.LGBMClassifier()
    return estimator, params_grid


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    """
    Trains a LightGBM model
    Args:
        X_train: X_train
        y_train: y_train
        params: params to use for the fitting

    Returns: trained lightgbm model

    """
    # Extract estimator parameters ("estimator__{value}")
    estimator_params = extract_estimator_params_from_grid(params)
    estimator = lgb.LGBMClassifier(**estimator_params)
    pipeline = build_prediction_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    return pipeline


####### RANDOM FOREST  ########
def get_GS_params_RFClassifier():
    """
    Gives params and models to use for the grid_search using Random Forest Classifier
    Returns: Estimator and params for the grid_search
    """
    params_grid = {
        "estimator__bootstrap": [True],
        "estimator__criterion": ["entropy"],
        "estimator__max_depth": [3, 6],
        "estimator__max_features": [3, 10],
        "estimator__min_samples_leaf": [4],
        "estimator__min_samples_split": [3],
    }

    estimator = RandomForestClassifier()

    return estimator, params_grid


def train_RFClassifier(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    """
    Trains a random forest Classifier
    Args:
        X_train:
        y_train:
        params: params to use for the fitting

    Returns: trained random forest model
    """

    # Extract estimator parameters ("estimator__{value}")
    estimator_params = extract_estimator_params_from_grid(params)

    estimator = RandomForestClassifier(**estimator_params)
    pipeline = build_prediction_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    return pipeline


### UTILS ###
def extract_estimator_params_from_grid(params_grid):
    estimator_params = {}
    for param, param_value in params_grid.items():
        estimator_params[param.replace("estimator__", "")] = param_value

    return estimator_params
