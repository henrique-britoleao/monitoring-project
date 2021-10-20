# -*- coding: utf-8 -*-

################## Importing libraries ####################
import sys

# from src.Utils.utils import load_batch
sys.path.insert(0,"Loading/")
sys.path.insert(0,"Preprocessing/")
sys.path.insert(0,"Modeling/")
sys.path.insert(0,"Evaluation/")
sys.path.insert(0,"Interpretability/")
sys.path.insert(0,"Monitoring/")
sys.path.insert(0,"Utils/")

import loading
import preprocessing
import modeling
import utils as u
import monitoring
import data_quality  
import train_model

import constants as cst
import logging, json, pickle
from datetime import datetime as dt
import os

import pandas as pd

logging.basicConfig(filename = cst.LOG_FILE_PATH,
                    filemode = "w",
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def main(batch_id):
    batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=batch_id)
    batch_df = loading.read_csv_from_path(os.path.join(cst.BATCHES_PATH, batch_name))
    
    sample_df = train_model.load_training_data()
    sample_preprocesssed = loading.read_csv_from_path(cst.PREPROCESSED_TRAIN_FILE_PATH)
    
    data_quality_alerts = data_quality.check_data_quality(batch_df, sample_df)
    batch_preprocessed = batch_preprocess(batch_df, cst.column_types, preprocessing.MarketingPreprocessor())
    
    model = u.load_model()
    predicted_batch = train_model.make_predictions_on_training_data(model, batch_preprocessed)
    save_predicted_batch(predicted_batch)
    
    predicted_sample = loading.read_csv_from_path(cst.PREDICTED_TRAIN_FILE_PATH)
    
    monitoring_metrics = monitoring.compute_metrics( 
        sample_preprocesssed, 
        batch_preprocessed, 
        predicted_batch.loc[:, f'{cst.y_pred_proba}_{cst.y_class_labels[1]}'],
        predicted_sample.loc[:, f'{cst.y_pred_proba}_{cst.y_class_labels[1]}'],
        cst.selected_dataset_information['metrics_setup']
    )
    
    # initialize output dict
    records = {batch_name: dict()}
    # populate records
    records[batch_name]['date'] = str(dt.now())
    records[batch_name].update({'data_quality': data_quality_alerts})
    records[batch_name].update({'metrics': monitoring_metrics})
    
    
    # with open(cst.MONITORING_METRICS_FILE_PATH, 'r') as monitoring_file:a
    #     metrics = json.load(monitoring_file)

    with open(cst.MONITORING_METRICS_FILE_PATH, 'w') as monitoring_file:
        json.dump(records, monitoring_file)
        

def batch_preprocess(batch_df: pd.DataFrame, column_types: dict[str, list[str]], preprocessor: preprocessing.Preprocessor):
    return preprocessor(batch_df, column_types)


def save_predicted_batch(sample_df_preprocessed_pred: pd.DataFrame) -> None:
    """Save training data with predicted labels and corresponding probabilities

    Args:
        sample_df_preprocessed_pred (pd.DataFrame): preprocessed train data with predictions
    """
    loading.write_csv_from_path(sample_df_preprocessed_pred, cst.PREDICTED_BATCH_FILE_PATH)


if __name__ == "__main__":
    main(1)