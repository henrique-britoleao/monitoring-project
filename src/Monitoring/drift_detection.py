# -*- coding: utf-8 -*-

#####  Imports  #####
import logging

from skmultiflow.drift_detection import ADWIN, HDDM_A
import pandas as pd
import numpy as np

#####  Set logger  #####
logger = logging.getLogger("main_logger")

def detect_drift_in_streaming_data(numerical_cols: list,
                                        sample_df: pd.DataFrame, 
                                        batch_df: pd.DataFrame,
                                        ) -> None:
    """Detects numerical drift by adding stream elements from the new batch to a window
    and verifying if drift occurred (in the batch data only).
    
    Args:
        numerical_cols (list): column names containg numerical variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
    Return:
        None
    """

    detect_drift_adwin_method(numerical_cols, sample_df, batch_df)
    detect_drift_hddm_a_method(numerical_cols, sample_df, batch_df)

def detect_drift_adwin_method(numerical_cols: list,
                                    sample_df: pd.DataFrame, 
                                    batch_df: pd.DataFrame) -> None:
    """Detects numerical drift by adding stream elements from the new batch to ADWIN
    and verifying if drift occurred (in the batch data only).
    
    Args:
        numerical_cols (list): column names containg numerical variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
    Return:
        None
    """
    for col in numerical_cols:
        adwin = ADWIN()
        data_stream = np.concatenate((sample_df[col], batch_df[col]))

        # Adding stream elements to ADWIN and verifying if drift occurred
        for i in range(len(data_stream)):
            adwin.add_element(data_stream[i])
            if adwin.detected_change():
                logger.info('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i) + 'for column:' + col)

def detect_drift_hddm_a_method(numerical_cols: list,
                                    sample_df: pd.DataFrame, 
                                    batch_df: pd.DataFrame) -> None:
    """Detects numerical drift by adding stream elements from the new batch to HDDM_A
    and verifying if drift occurred (in the batch data only).
    
    Args:
        numerical_cols (list): column names containg numerical variables
        sample_df (pd.DataFrame): data used to train original model
        batch_df (pd.DataFrame): data received from latest batch
    Return:
        None
    """
    for col in numerical_cols:
        hddm_a = HDDM_A()
        sample_size = sample_df.shape[0]
        data_stream = np.concatenate((sample_df[col], batch_df[col]))

        # Adding stream elements to ADWIN and verifying if drift occurred
        for i in range(len(data_stream)):
            hddm_a.add_element(data_stream[i])
            if i < sample_size:
                pass
            else:
                if hddm_a.detected_warning_zone():
                    logger.info(f"Warning zone detected in data (HDDM_A): {str(data_stream[i])} - at index: {str(i)} for column {col}")
                if hddm_a.detected_change():
                    logger.info(f"Change detected in data (HDDM_A): {str(data_stream[i])} - at index: {str(i)} for column {col}")