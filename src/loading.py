# -*- coding: utf-8 -*-
import constants as cst

from pathlib import Path
import pandas as pd

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

def read_csv_from_path(path: Path):
    """
    Reads a csv from a path
    Args:
        path: path of the file to read

    Returns: dataframe

    """
    df = pd.read_csv(path,sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    logger.debug('Read file: '+ path)

    return df

def write_csv_from_path(df: pd.DataFrame, path: Path):
    """
    Writes the dataframe to the correct file path
    Args:
        df: dataframe to write
        path: local file path from constants 
    """
    df.to_csv(path, sep=";", index=False)
    logger.debug('Wrote file in: '+ path)
    
def load_training_data() -> pd.DataFrame:
    """Load training data

    Returns:
        pd.DataFrame: raw training data
    """
    logger.info("Load training data")
    sample_df = read_csv_from_path(cst.RAW_TRAIN_FILE_PATH)
    return sample_df 