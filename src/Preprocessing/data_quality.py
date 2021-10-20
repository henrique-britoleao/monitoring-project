# -*- coding: utf-8 -*-

#####  Imports  #####
import logging
import pandas as pd

#####  Set logger  #####
logger = logging.getLogger(__name__)

def check_data_quality(sample_df: pd.DataFrame, batch_df: pd.DataFrame) -> dict[str, bool]:
    """Checks overall data quality of a new batch of data comapred to 
    the original training data

    Args:
        batch_data (pd.DataFrame): new dataframe used to make predictions
    """
    try:
        alert_schema = check_schema(sample_df, batch_df)
        alert_nans = checkNANs(batch_df)
        
        return {"alert_schema": alert_schema,
                "alert_nans": alert_nans}
    
    except TypeError:
        alert_nans = checkNANs(batch_df)
        return {"alert_schema": 1,
                "alert_nans": alert_nans}


def check_schema(sample_df: pd.DataFrame, batch_df: pd.DataFrame) -> bool:
    """TODO"""
    return (check_dtypes(sample_df, batch_df) or 
            check_colnames(sample_df, batch_df))

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
    alert_dtypes = False

    if sample_df.dtypes.equals(batch_df.dtypes):
        logger.info(
            f"Data types are identical in the train set and the new batch ✔️"
        )
    else:
        alert_dtypes = True
        diff_df = (
            pd.concat([sample_df.dtypes, batch_df.dtypes])
            .drop_duplicates(keep=False)
            .reset_index()  # find differences in dtypes
        )
        diff_df.columns = ["colname", "dtype"]
        for name in diff_df.colname.unique():
            colname = name
            dtype_train = diff_df.loc[diff_df.colname == name].iloc[0, 1]
            dtype_batch = diff_df.loc[diff_df.colname == name].iloc[1, 1]
            logger.error(
                f'Column "{colname}" has dtype "{dtype_train}" in the train set '
                f'whereas it has dtype "{dtype_batch}" in the new batch.❌'
            )
    return alert_dtypes


def check_colnames(sample_df: pd.DataFrame, batch_df: pd.DataFrame) -> bool:
    """Checks if the names of the columns of the new batch of data are identical
    to the ones of the training data

    Args:
        sample_data (pd.DataFrame): original dataframe used for training
        batch_data (pd.DataFrame): new dataframe used to make predictions

    Returns:
        bool: if the names of the columns are identical
    """
    alert_colnames = False

    if sample_df.columns.equals(batch_df.columns):
        logger.info(
            f"Columns names are identical in the train set and the new batch ✔️"
        )
    else:
        alert_colnames = True
        for i, name in enumerate(sample_df.columns.tolist()):
            batch_name = batch_df.columns.tolist()[i]
            if batch_name != name:
                message = f'Column "{name}" became "{batch_name}"❌'
                logger.error(message)

    return alert_colnames


def checkNANs(batch_df: pd.DataFrame) -> bool:
    """Checks for the presence of NANs values 
    in a new batch of data

    Args:
        batch_data (pd.DataFrame): new dataframe used to make predictions
    """
    alert_nans = False
    for col in batch_df:
        num_nans = batch_df[col].isnull().sum()
        if num_nans != 0:
            logger.warning(f"Column: {col} has {num_nans} NAN values ❌. "
                           "This may cause the model not to fit.")
            alert_nans = True
    if alert_nans == False:
        logger.info(f"There are no NAN values in the new set ✔️")
    return alert_nans