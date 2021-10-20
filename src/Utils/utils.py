# -*- coding: utf-8 -*-
import logging
import pickle
import json
logger = logging.getLogger('main_logger')

import constants as cst

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
    pickle.dump(clf, open(filename, 'wb'))
    logger.info('Saved model: ' + filename)

def load_model(name=""):
    if len(name)==0:
        name = f"{cst.selected_dataset}_{cst.selected_model}"
    filename = cst.MODELS_PATH + name + ".sav"
    clf = pickle.load(open(filename, 'rb'))
    logger.info('Loaded model: ' + filename)
    return clf

def get_y_column_from_conf():
    return cst.y_name

def save_training_performance_metrics(metrics: dict): #conf: dict) -> None:
    """
    Saves the dictionary containing model performance metrics to a json file

    Args:
        metrics (dict): Dict of classification performance metrics
        conf (dict): Configuration file stored as a json object
    """
    #with open(cst.MONITORING_PATH + '_' + cst.selected_model + ".txt", w) as outfile:
    #open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'training_metrics_'
    #        + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as outfile:
    #    json.dump(str(metrics), outfile)
    pass