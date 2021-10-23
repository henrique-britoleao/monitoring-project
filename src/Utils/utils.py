# -*- coding: utf-8 -*-
import pandas as pd
import logging
import pickle
import json
import os
import io

import constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

def my_get_logger(path_log, log_level, my_name =""):
    """
    Instanciation du logger et paramétrisation
    :param path_log: chemin du fichier de log
    :param log_level: Niveau du log
    :return: Fichier de log
    """
    
    log_level_dict = {"CRITICAL": logging.CRITICAL,
                        "ERROR": logging.ERROR,
                        "WARNING": logging.WARNING,
                        "INFO": logging.INFO,
                        "DEBUG": logging.DEBUG}
    
    LOG_LEVEL = log_level_dict[log_level]

    if my_name != "":
        logger = logging.getLogger(my_name)
        logger.setLevel(LOG_LEVEL)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(LOG_LEVEL)
    
    # create a file handler
    handler = logging.FileHandler(path_log)
    handler.setLevel(LOG_LEVEL)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)-8s: %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def save_model(clf, name=""):
    if len(name)==0:
        name = f"{cst.selected_dataset}_{cst.selected_model}"
    filename = cst.MODELS_PATH + name + ".sav"
    
    with open(filename, 'wb') as model_file:
        pickle.dump(clf, model_file)

    logger.info('Saved model: ' + filename)
    

def load_model(name=""):
    if len(name)==0:
        name = f"{cst.selected_dataset}_{cst.selected_model}"
    filename = cst.MODELS_PATH + name + ".sav"
    
    with open(filename, 'rb') as model_file:
        clf = pickle.load(model_file)
        logger.info('Loaded model: ' + filename)
    
    return clf


def get_y_column_from_conf():
    return cst.y_name

def save_training_performance_metrics(metrics: dict): #conf: dict) -> None:
    """TODO"""
    file_path = cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH
    with open(file_path, 'w') as f:
        content = json.dump(metrics, f)
        
    logger.info(f"Wrote performance metrics to: {file_path}")
        
def load_batch(batch_id: str) -> pd.DataFrame:
    batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=batch_id)
    batch_path = os.path.join(cst.BATCHES_PATH, batch_name)
    
    return pd.read_csv(batch_path)

def append_to_json(data: dict, file_path: str) -> None:
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        content = []
    
    content.append(data)
    
    with open(file_path, 'w') as f:
        content = json.dump(content, f)
        
    logger.info(f"Appended data to {file_path}")
