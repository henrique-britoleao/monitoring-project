# -*- coding: utf-8 -*-
import pandas as pd
import logging
import pickle
import json
import os

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


def save_model(clf, conf, name =""):
    if len(name)==0:
        name = conf['selected_dataset']+'_'+conf['selected_model']
    filename = conf["paths"]["Outputs_path"]+conf["paths"]["folder_models"] + name+'.sav'
    pickle.dump(clf, open(filename, 'wb'))
    logger.info('Modele sauvergarde: ' + filename)
    return 'OK'

def load_model(conf,name=""):
    if len(name)==0:
        name = conf['selected_dataset']+'_'+conf['selected_model']
    filename = conf["paths"]["Outputs_path"]+conf["paths"]["folder_models"] + name+'.sav'
    print(filename)
    clf = pickle.load( open(filename, 'rb'))
    logger.info('Modele charge: ' + filename)
    return clf

def get_y_column_from_conf(conf):
    return conf["dict_info_files"][conf['selected_dataset']]["y_name"]

def save_training_performance_metrics(metrics: dict, conf: dict) -> None:
    """
    Saves the dictionary containing model performance metrics to a json file

    Args:
        metrics (dict): Dict of classification performance metrics
        conf (dict): Configuration file stored as a json object
    """
    with open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'training_metrics_'
            + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as outfile:
        json.dump(str(metrics), outfile)
        
def load_batch(batch_id: str) -> pd.DataFrame:
    batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=batch_id)
    batch_path = os.path.join(cst.BATCHES_PATH, batch_name)
    
    return pd.read_csv(batch_path)