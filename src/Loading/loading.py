# -*- coding: utf-8 -*-
import pandas as pd
import logging
logger = logging.getLogger('main_logger')

#Global:


def read_csv_from_name(conf):
    """
    Calls the read_csv_from_path function giving it a path read in the conf file
    Args:
        conf: conf file

    Returns: dataframe read from csv

    """
    selected_dataset = conf['selected_dataset']
    path = conf["paths"]["Inputs_path"]+ conf["dict_info_files"][selected_dataset]["path_file"]

    return read_csv_from_path(path)

def read_csv_from_path(path):
    """
    Reads a csv from a path
    Args:
        path: path of the file to read

    Returns: dataframe

    """
    #This function CAN be sophisticated: detection of format, delimiter etc.

    df = pd.read_csv(path,sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    logger.debug('file read : '+ path)

    return df

def write_preprocessed_csv_from_name(df,conf):
    """
    Writes the preprocessed dataframe
    Args:
        df: preprocessed dataframe
        conf: conf file

    Returns: "ok"

    """
    selected_dataset = conf['selected_dataset']
    path = conf["paths"]["Outputs_path"]+conf["paths"]["folder_preprocessed"]+conf["dict_info_files"][selected_dataset]["path_file_preprocessed"]
    df.to_csv(path,sep=";", index =False)
    logger.debug('file wrote : '+ path)

    return "OK"

def load_preprocessed_csv_from_name(conf):
    """
    Loads the preprocessed dataframe
    Args:
        conf:  conf file

    Returns: dataframe read

    """
    selected_dataset = conf['selected_dataset']
    path = conf["paths"]["Outputs_path"]+conf["paths"]["folder_preprocessed"]+conf["dict_info_files"][selected_dataset]["path_file_preprocessed"]
    df = read_csv_from_path(path)
    logger.debug('file read : '+ path)

    return df