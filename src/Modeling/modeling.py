# -*- coding: utf-8 -*-
from modeling.gridsearch import main_GS_from_estimator_and_params
from modeling.models import *

import constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

def main_modeling_from_name (X_train, y_train):
    """
    Main modeling function to fit a model on the training set using the best parameters for the selected model

    Args:
        X_train (pd.DataFrame): X_train
        y_train (pd.Series): y_train
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

    function_get_GS_params = globals()[dict_function_GS_params[cst.selected_model]]
    estimator, params_grid = function_get_GS_params()

    logger.info('Beginning of Grid Search using ' + cst.selected_model)
    best_params, best_score = main_GS_from_estimator_and_params(X_train, y_train, estimator, params_grid)

    function_train = globals()[dict_function_train_model[cst.selected_model]]
    model = function_train(X_train, y_train, best_params)
    logger.info('End of Grid Search using ' + cst.selected_model)
    logger.info('Best parameters are :')
    logger.info(best_params)
    logger.info('best score :' + str(best_score))

    return model, best_params