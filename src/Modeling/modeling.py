# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, fbeta_score

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


# What can be done in this script: (not mandatory)
# Implementing a cross validation (function) to check the robustness of a parameter set
# Implementing on it a grid search to select the best parameter set
# Implementing other models



def main_modeling_from_name (X_train,y_train, conf):
    """
    Main modeling function: it launches a grid search using the correct model according to the conf file
    Args:
        X_train: X_train
        y_train: y_train
        conf: configuration file

    Returns: model fitted on the train set and its best params

    """

    dict_function_GS_params = {'random_forest': 'get_GS_params_RFClassifier',
                                'lightgbm': 'get_GS_params_lightgbm'}
    dict_function_train_model = {'random_forest': 'train_RFClassifier',
                                'lightgbm': 'train_lightgbm'}

    selected_model = conf['selected_model']
    function_get_GS_params = globals()[dict_function_GS_params[selected_model]]
    estimator, params_grid = function_get_GS_params()

    logger.info('Beginning of Grid Search using ' + selected_model)
    best_params, best_score = main_GS_from_estimator_and_params(X_train, y_train, estimator, params_grid, conf)

    function_train = globals()[dict_function_train_model[selected_model]]
    model = function_train(X_train,y_train, best_params)
    logger.info('Enfd of Grid Search using ' + selected_model)
    logger.info('Best parameteres are :')
    logger.info(best_params)
    logger.info('best score' + str(best_score))

    return model, best_params


def main_GS_from_estimator_and_params(X_train,y_train, estimator, params_grid, conf):
    """
    Main function to run a grid search
    Args:
        X_train: X_train
        y_train:  y_train
        estimator: unfit model to use
        params_grid: grid search to run
        conf: conf file

    Returns: best params and score achieved in the GS

    """
    gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=y_train)


    gsearch = GridSearchCV(estimator=estimator, param_grid=params_grid, cv=gkf,
                           scoring=make_scorer(f1_score), verbose=1, n_jobs = -1)
    best_model = gsearch.fit(X=X_train, y=y_train)

    return best_model.best_params_, best_model.best_score_


####### LIGHTGBM  ########

def get_GS_params_lightgbm():
    """
    Gives params and models to use for the grid_search using LightGBM Classifier
    Returns:Estimator and params for the grid_search

    """
    params_grid = {'objective': ['binary'],
    'max_depth': [6,100],
    'reg_alpha': [0, 0.1],
    'min_data_in_leaf': [5, 10],
    'learning_rate': [0.01],
    'scale_pos_weight': [0.2, 1, 3, 10]
    }

    estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=200, learning_rate=0.01)

    return estimator, params_grid

def train_lightgbm(X_train,y_train,params):
    """
    Training function for a Lightgbm
    Args:
        X_train: X_train
        y_train: y_train
        params: params to use for the fitting

    Returns: trained lightgbm model

    """
    dftrainLGB = lgb.Dataset(data=X_train, label=y_train, feature_name=list(X_train))
    model = lgb.train(params,dftrainLGB)
    return model


####### RANDOM FOREST  ########
def get_GS_params_RFClassifier():
    """
    Gives params and models to use for the grid_search using Random Forest Classifier
    Returns:Estimator and params for the grid_search
    """
    params_grid = {'bootstrap': [True],
              'criterion': ['entropy'],
              'max_depth': [3,6],
              'max_features': [3,10],
              'min_samples_leaf': [4],
              'min_samples_split': [3]}
    estimator = RandomForestClassifier()

    return estimator, params_grid

def train_RFClassifier(X_train,y_train,params):
    """
    Training function for a random forest Classifier
    Args:
        X_train: 
        y_train: 
        params: params to use for the fitting

    Returns: trained random forest model

    """
    model = RandomForestClassifier(**params).fit(X_train,y_train)
    return model
