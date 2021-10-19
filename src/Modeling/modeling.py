# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

from typing import Tuple
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def main_modeling_from_name (X_train, y_train, conf, categorical_columns, numerical_columns):
    """
    Main modeling function to fit a model on the training set using the best parameters for the selected model

    Args:
        X_train (pd.DataFrame): X_train
        y_train (pd.Series): y_train
        conf (dict): configuration file containing the selected model
        categorical_columns (list): list of categorical columns
        numerical_columns (list): list of numerical columns

    Returns:
        dict: best parameters 
        float: best score
    """

    dict_function_GS_params = {'random_forest': 'get_GS_params_RFClassifier',
                                'lightgbm': 'get_GS_params_lightgbm'}
    dict_function_train_model = {'random_forest': 'train_RFClassifier',
                                'lightgbm': 'train_lightgbm'}

    selected_model = conf['selected_model']
    function_get_GS_params = globals()[dict_function_GS_params[selected_model]]
    estimator, params_grid = function_get_GS_params()

    logger.info('Beginning of Grid Search using ' + selected_model)
    best_params, best_score = main_GS_from_estimator_and_params(X_train, y_train, estimator, params_grid, categorical_columns, numerical_columns)

    function_train = globals()[dict_function_train_model[selected_model]]
    model = function_train(X_train, y_train, best_params, categorical_columns, numerical_columns)
    logger.info('End of Grid Search using ' + selected_model)
    logger.info('Best parameteres are :')
    logger.info(best_params)
    logger.info('best score' + str(best_score))

    return model, best_params

def main_GS_from_estimator_and_params(X_train: pd.DataFrame, y_train: pd.Series, estimator, params_grid: dict, categorical_columns: list, numerical_columns: list) -> Tuple[dict, float]:
    """
    Main function to run a grid search

    Args:
        X_train (pd.DataFrame): X_train
        y_train (pd.Series): y_train
        estimator (model): prediction model to search parameters for 
        params_grid (dict): parameters grid to test in a grid search
        categorical_columns (list): list of categorical columns
        numerical_columns (list): list of numerical columns

    Returns:
        dict: best parameters 
        float: best score
    """

    gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=y_train)
    pipeline = build_prediction_pipeline(estimator, categorical_columns, numerical_columns)
    
    gsearch = GridSearchCV(estimator=pipeline, param_grid=params_grid, cv=gkf,
                           scoring=make_scorer(f1_score), verbose=1, n_jobs = -1)
    best_model = gsearch.fit(X=X_train, y=y_train)

    return best_model.best_params_, best_model.best_score_


####### LIGHTGBM  ########
def get_GS_params_lightgbm():
    """
    Gives params and models to use for the grid_search using LightGBM Classifier
    Returns:Estimator and params for the grid_search

    """
    params_grid = {
        'estimator__boosting_type': ['gbdt'],
        'estimator__objective': ['binary'],
        'estimator__num_boost_round': [200], 
        'estimator__learning_rate': [0.01, 0.1],
        'estimator__max_depth': [6, 100],
        'estimator__reg_alpha': [0, 0.1],
        'estimator__min_data_in_leaf': [5, 10],
        'estimator__learning_rate': [0.01],
        'estimator__scale_pos_weight': [0.2, 1, 3, 10],
        'estimator__verbose': [-1]
    }

    estimator = lgb.LGBMClassifier()
    return estimator, params_grid

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict, categorical_columns: list, numerical_columns: list):
    """
    Trains a LightGBM model
    Args:
        X_train: X_train
        y_train: y_train
        params: params to use for the fitting

    Returns: trained lightgbm model

    """
    estimator = lgb.LGBMClassifier(**params)
    pipeline = build_prediction_pipeline(estimator, categorical_columns, numerical_columns)
    pipeline.fit(X_train, y_train)
    return pipeline


####### RANDOM FOREST  ########
def get_GS_params_RFClassifier():
    """
    Gives params and models to use for the grid_search using Random Forest Classifier
    Returns: Estimator and params for the grid_search
    """
    params_grid = {'estimator__bootstrap': [True],
              'estimator__criterion': ['entropy'],
              'estimator__max_depth': [3,6],
              'estimator__max_features': [3,10],
              'estimator__min_samples_leaf': [4],
              'estimator__min_samples_split': [3]}

    estimator = RandomForestClassifier()

    return estimator, params_grid

def train_RFClassifier(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict, categorical_columns: list, numerical_columns: list):
    """
    Trains a random forest Classifier
    Args:
        X_train: 
        y_train: 
        params: params to use for the fitting

    Returns: trained random forest model
    """
    estimator = RandomForestClassifier(**params).fit(X_train,y_train)
    pipeline = build_prediction_pipeline(estimator, categorical_columns, numerical_columns)
    return pipeline


####### PIPELINE  ########
def build_prediction_pipeline(estimator, categorical_columns: list, numerical_columns: list):
    """
    Builds a pipeline to combine the column encoder and the estimator

    Args:
        estimator: prediction model
    """
    encoder = create_column_encoder(categorical_columns, numerical_columns)
    pipeline = Pipeline(([('encoder', encoder), ('estimator', estimator)]))
    return pipeline

####### PREPROCESSING  ########
def create_column_encoder(categorical_columns: list, numerical_columns: list):
    """
    Encodes categorical columns and scales numerical columns 

    Args:
        categorical_columns (list): list of categorical columns 
    """
    encoder = make_column_transformer(
        (OneHotEncoder(), categorical_columns), 
        remainder="passthrough"
    )
    return encoder