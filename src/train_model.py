# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import pandas as pd

from Loading import loading
from Preprocessing import preprocessing
from Modeling import modeling
from Evaluation import evaluation
from Utils import utils as u

import constants as cst

def main_model_training_pipeline():
    """Function to load, preprocess training data
    Train and save a model with optimal hyperparameters 
    Compute training performance metrics, make predictions on the training set and save results
    """
    sample_df = load_training_data()

    # Preprocess and save training data 
    print('Preprocess')
    sample_df_preprocessed = preprocess_training_data(sample_df)
    save_preprocessed_training_data(sample_df_preprocessed)

    # Train and save model
    print('Train model')
    model, _ = modeling.main_modeling_from_name(
        sample_df_preprocessed.drop(columns=[cst.y_name]),
        sample_df_preprocessed[cst.y_name]
        )
    u.save_model(model)

    # Compute and save model performance metrics
    training_performance_metrics = evaluation.cross_evaluate_model_performance(
        model,
        sample_df_preprocessed.drop(columns=[cst.y_name]),
        sample_df_preprocessed[cst.y_name]
    )
    u.save_training_performance_metrics(training_performance_metrics)

    # Make predictions on the training set and save results
    sample_df_preprocessed_pred = make_predictions_on_training_data(model, sample_df_preprocessed)
    save_predicted_training_data(sample_df_preprocessed_pred)

def load_training_data() -> pd.DataFrame:
    """Load training data

    Returns:
        pd.DataFrame: raw training data
    """
    logger.info("Load training data")
    sample_df = loading.read_csv_from_path(cst.RAW_TRAIN_FILE_PATH)
    return sample_df 

def preprocess_training_data(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess training data

    Args:
        sample_df (pd.DataFrame): raw training data

    Returns:
        pd.DataFrame: preprocessed training data
    """
    logger.info("Preprocess training data")
    preprocessor = preprocessing.MarketingPreprocessor()
    df_preprocessed = preprocessor(sample_df, cst.column_types)
    return df_preprocessed

def save_preprocessed_training_data(sample_df_preprocessed) -> None:
    loading.write_csv_from_path(sample_df_preprocessed, cst.PREPROCESSED_TRAIN_FILE_PATH)

def make_predictions_on_training_data(model, sample_df_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """Add predictions to the preprocessed training dataset

    Args:
        model (Pipeline): trained model
        sample_df_preprocessed (pd.DataFrame): preprocessed training data

    Returns:
        pd.DataFrame: preprocessed training data with predicted labels and probabilities
    """
    sample_df_preprocessed_pred = sample_df_preprocessed.copy()
    
    # drop target column
    if cst.y_name in sample_df_preprocessed_pred.columns:
        sample_df_preprocessed_pred = sample_df_preprocessed.drop(cst.y_name, axis=1)
        
    sample_df_preprocessed_pred[cst.y_pred] = model.predict(sample_df_preprocessed)
    sample_df_preprocessed_pred[
        [f"{cst.y_pred_proba}_{cst.y_class_labels[0]}", 
         f"{cst.y_pred_proba}_{cst.y_class_labels[1]}"]
    ] = model.predict_proba(sample_df_preprocessed.drop(cst.y_name, axis=1))

    return sample_df_preprocessed_pred

def save_predicted_training_data(sample_df_preprocessed_pred: pd.DataFrame) -> None:
    """Save training data with predicted labels and corresponding probabilities

    Args:
        sample_df_preprocessed_pred (pd.DataFrame): preprocessed train data with predicted labels and probabilities
    """
    loading.write_csv_from_path(sample_df_preprocessed_pred, cst.PREDICTED_TRAIN_FILE_PATH)