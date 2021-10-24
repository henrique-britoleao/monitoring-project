# -*- coding: utf-8 -*-

#####  Imports  #####
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

import constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger

logger = MainLogger.getLogger(__name__)

####### PIPELINE  ########
def build_prediction_pipeline(estimator):
    """
    Builds a pipeline to combine the column encoder and the estimator

    Args:
        estimator: prediction model
    """
    encoder = create_column_encoder(cst.categorical_columns, cst.numerical_columns)
    pipeline = Pipeline(([("encoder", encoder), ("estimator", estimator)]))
    return pipeline


def create_column_encoder(categorical_columns: list, numerical_columns: list):
    """
    Encodes categorical columns and scales numerical columns

    Args:
        categorical_columns (list): list of categorical columns
    """
    encoder = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        remainder="passthrough",
    )
    return encoder
