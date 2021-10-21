# -*- coding: utf-8 -*-

#####  Imports  #####
from Loading import loading
from Modeling import modeling
from Evaluation import evaluation
from Utils import utils as u
import constants as cst

import pandas as pd

import logging
logger = logging.getLogger(__name__)

#####  Processors  #####
def main_model_training_pipeline() -> None:
    """
    Loads and preprocesses the training data, trains and saves a model with 
    optimal hyperparameters, computes training performance metrics, make 
    predictions on the training set and saves results.
    """
    sample_df = loading.load_training_data()

    # Preprocess and save training data 
    sample_df_preprocessed = cst.PREPROCESSOR(sample_df, cst.column_types)
    save_preprocessed_training_data(sample_df_preprocessed)

    # Train and save model
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


def make_predictions_on_training_data(model, sample_df_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """Add predictions to the preprocessed training dataset

    Args:
        model (Pipeline): trained model
        sample_df_preprocessed (pd.DataFrame): preprocessed training data

    Returns:
        pd.DataFrame: preprocessed training data with predicted labels and probabilities
    """
    sample_df_preprocessed_pred = sample_df_preprocessed.copy()
    
    sample_df_preprocessed_pred[cst.y_pred] = model.predict(sample_df_preprocessed_pred.drop(cst.y_name, axis=1))
    sample_df_preprocessed_pred[
        [f"{cst.y_pred_proba}_{cst.y_class_labels[0]}", 
         f"{cst.y_pred_proba}_{cst.y_class_labels[1]}"]
    ] = model.predict_proba(sample_df_preprocessed_pred.drop([cst.y_name, cst.y_pred], axis=1))

    return sample_df_preprocessed_pred


def save_predicted_training_data(sample_df_preprocessed_pred: pd.DataFrame) -> None:
    """Save training data with predicted labels and corresponding probabilities

    Args:
        sample_df_preprocessed_pred (pd.DataFrame): preprocessed train data with predicted labels and probabilities
    """
    loading.write_csv_from_path(sample_df_preprocessed_pred, cst.PREDICTED_TRAIN_FILE_PATH)
    
    
def save_preprocessed_training_data(sample_df_preprocessed: pd.DataFrame) -> None:
    """Save preprocessed training data."""
    loading.write_csv_from_path(sample_df_preprocessed, cst.PREPROCESSED_TRAIN_FILE_PATH)
    
    
if __name__ == "__main__":
    # create and set up logger
    import logging
    logging.basicConfig(filename = cst.TRAINING_LOG_FILE_PATH,
                        filemode = "w",
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    
    main_model_training_pipeline()