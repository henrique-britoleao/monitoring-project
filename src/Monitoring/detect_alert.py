# -*- coding: utf-8 -*-

#####  Imports  #####
import constants as cst
from typing import Callable
from numbers import Number
from operator import ge

#####  Set Logger  #####
from src.utils.loggers import AlertLogger

logger = AlertLogger.getLogger("Alert Logger")

#####  Alert  #####
SignOperator = Callable[[Number, Number], bool]


def alert(
    value: float,
    metric: str,
    data_type: str,
    drift_type: str,
    alert_msg: str,
    sign: SignOperator = ge,
) -> bool:
    """outputs alert 0 or 1 depending on treshhold rule
    a < b -> lt(a, b)
    a <= b -> le(a, b)
    a >= b -> ge(a, b)
    a > b -> gt(a, b)
    
    Args:
        value (int) : value of metric
        metric (str): metric we want to evalue
        sign ([type], optional): bigger than, smaller than. Defaults to ge.
        alert_msg (str): message to be logged when alert is triggered
    
    Reeturns:
        bool: flag indicating if there is an alert or not
    """

    if drift_type == "concept_drift":
        threshold = cst.selected_dataset_information["metrics_setup"][drift_type][
            metric
        ]["threshold"]
    else:
        threshold = cst.selected_dataset_information["metrics_setup"][drift_type][
            data_type
        ][metric]["threshold"]

    if sign(value, threshold):
        alert = 1
        logger.warning(alert_msg)
    else:
        alert = 0
    return alert
