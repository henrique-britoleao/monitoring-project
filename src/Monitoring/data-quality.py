# -*- coding: utf-8 -*-

#####  Imports  #####
from typing import Callable, Tuple
import logging

import pandas as pd
import numpy as np

#####  Set logger  #####
logger = logging.getLogger("main_logger")


def check_colnames(sample_df: pd.DataFrame, batch_df: pd.DataFrame) -> bool:
    """Checks if the names of the columns of the new batch of data are identical
    to the ones of the training data

    Args:
        sample_data (pd.DataFrame): original dataframe used for training
        batch_data (pd.DataFrame): new dataframe used to make predictions

    Returns:
        bool: if the names of the columns are identical
    """
    same_colnames = True

    if sample_df.columns.equals(batch_df.columns):
        logger.info(
            f"Columns names are identical in the train set and the new batch ✔️"
        )
    else:
        same_colnames = False
        for i, name in enumerate(sample_df.columns.tolist()):
            batch_name = batch_df.columns.tolist()[i]
            if batch_name != name:
                logger.critical(f'Column "{name}" became "{batch_name}"❌')

    return same_colnames


def check_dtypes(sample_df: pd.DataFrame, batch_df: pd.DataFrame) -> bool:
    """Checks if the data types of the new batch of data are identical
    to the ones of the training data when their columns both have the
    same names

    Args:
        sample_data (pd.DataFrame): original dataframe used for training
        batch_data (pd.DataFrame): new dataframe used to make predictions

    Returns:
        bool: if the data types of the columns are identical
    """
    same_colnames = check_colnames(sample_df, batch_df)
    same_dtypes = True

    if same_colnames == True:
        if sample_df.dtypes.equals(batch_df.dtypes):
            logger.info(
                f"Data types are identical in the train set and the new batch ✔️"
            )
        else:
            same_dtypes = False
            diff_df = (
                pd.concat([df1.dtypes, df3.dtypes])
                .drop_duplicates(keep=False)
                .reset_index()
            )
            diff_df.columns = ["colname", "dtype"]
            for name in diff_df.colname.unique():
                colname = name
                dtype_train = diff_df.loc[diff_df.colname == name].iloc[0, 1]
                dtype_batch = diff_df.loc[diff_df.colname == name].iloc[1, 1]
                logger.critical(
                    f'Column "{colname}" has dtype "{dtype_train}" in the train set whereas it has dtype "{dtype_batch}" in the new batch.❌'
                )
        return same_dtypes

    else:
        return None


def checkNANs(batch_df: pd.DataFrame) -> bool:
    """Checks for the presence of NANs values 
    in a new batch of data

    Args:
        batch_data (pd.DataFrame): new dataframe used to make predictions
    """
    for col in batch_df:
        num_NaNs = batch_df[col].isnull().sum()
        if num_NaNs != 0:
            logger(f"Column: {col} has {num_NaNs} NAN values ❌")
            break
        else:
            logger.info(f"There is no NAN values in the new set ✔️")
        break


def check_data_quality(sample_df: pd.DataFrame, batch_df: pd.DataFrame):
    """Checks overall data quality of a new batch of data comapred to 
    the original training data

    Args:
        batch_data (pd.DataFrame): new dataframe used to make predictions
    """
    check_dtypes(sample_df, batch_df)
    checkNANs(batch_df)
