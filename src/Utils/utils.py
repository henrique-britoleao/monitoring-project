# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import json
import os

import constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

#####  Utility functions  #####
def save_model(clf, name: str="") -> None:
    """Pickles a model object to a file in the output models path. 

    Args:
        clf: trained model object
        name (str, optional): name of the file to which the model will be 
        pickled. Defaults to "".
    """
    if len(name)==0:
        name = f"{cst.selected_dataset}_{cst.selected_model}"
    filename = cst.MODELS_PATH + name + ".sav"
    
    with open(filename, 'wb') as model_file:
        pickle.dump(clf, model_file)

    logger.info('Saved model: ' + filename)
    

def load_model(name: str=""):
    """Loads a pickled model object. 

    Args:
        name (str, optional): name of the file to which the model was pickled. 
        Defaults to "".

    Returns:
        model: the trained model object
    """
    if len(name)==0:
        name = f"{cst.selected_dataset}_{cst.selected_model}"
    filename = cst.MODELS_PATH + name + ".sav"
    
    with open(filename, 'rb') as model_file:
        clf = pickle.load(model_file)
        logger.info('Loaded model: ' + filename)
    
    return clf


def save_training_performance_metrics(metrics: dict) -> None: 
    """Saves a ditionary of trainig metrics to a json file.

    Args:
        metrics (dict): dictionary with metrics following the structure set in
        metrics_example.json
    """
    file_path = cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH
    with open(file_path, 'w') as f:
        json.dump(metrics, f)
        
    logger.info(f"Wrote performance metrics to: {file_path}")
    
        
def load_batch(batch_id: str) -> pd.DataFrame:
    """Loads the data contained in a batch. 
    
    Args:
        batch_id (str): ID of batch

    Returns:
        pd.DataFrame: df containg the batch data
    """
    batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=batch_id)
    batch_path = os.path.join(cst.BATCHES_PATH, batch_name)
    
    return pd.read_csv(batch_path)


def append_to_json(data: dict, file_path: str) -> None:
    """Appends data to a json file. Creates the file if it has not been created.

    Args:
        data (dict): data to be appended to json  file
        file_path (str): path to json file.
    """
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        content = []
    
    content.append(data)
    
    with open(file_path, 'w') as f:
        content = json.dump(content, f)
        
    logger.info(f"Appended data to {file_path}")