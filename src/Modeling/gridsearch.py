# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

from typing import Tuple
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from Modeling.pipeline_builder import build_prediction_pipeline

###### GRIDSEARCH ######
def main_GS_from_estimator_and_params(X_train: pd.DataFrame, y_train: pd.Series, estimator, params_grid: dict) -> Tuple[dict, float]:
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
    pipeline = build_prediction_pipeline(estimator)
    
    gsearch = GridSearchCV(estimator=pipeline, param_grid=params_grid, cv=gkf,
                           scoring=make_scorer(f1_score), verbose=-1, n_jobs=-1)
    best_model = gsearch.fit(X=X_train, y=y_train)

    return best_model.best_params_, best_model.best_score_