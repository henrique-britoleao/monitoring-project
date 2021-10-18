# -*- coding: utf-8 -*-

#####  Imports  #####
import logging

from skmultiflow.drift_detection import ADWIN
import pandas as pd
import numpy as np

#####  Set logger  #####
logger = logging.getLogger("main_logger")

def detect_drift_adaptive_windowing(numerical_cols: list,
                                    sample_df: pd.DataFrame, 
                                    batch_df: pd.DataFrame) -> None:
    """Detects numerical drift by adding stream elements from the new batch to ADWIN
    and verifying if drift occurred. 
    
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