# import libraries
import pandas as pd


def read_covariate_metrics_alerts(dict_metrics: dict, batch_name: str, batch_id: str):
    """Function to read covariate metrics alerts from the json metrics file 

    Args:
        dict_metrics (dict): json metrics file for data drift and concept drift
        batch_name (str): batch name
        batch_id (str): batch id 

    Returns:
        list: numerical, categorical and binary drift metrics in dataframes
    """
    numerical_drift = pd.DataFrame( 
        dict_metrics[batch_id][batch_name]['metrics']['covariate_drift_metrics']['numerical_metrics'])\
        .applymap(lambda x: dict(x)['alert']).transpose()
    categorical_drift = pd.DataFrame(
        dict_metrics[batch_id][batch_name]['metrics']['covariate_drift_metrics']['categorical_metrics'])\
        .applymap(lambda x: dict(x)['alert']).transpose()
    binary_drift = pd.DataFrame(
        dict_metrics[batch_id][batch_name]['metrics']['covariate_drift_metrics']['binary_metrics'])\
        .applymap(lambda x: dict(x)['alert']).transpose()
    return numerical_drift, categorical_drift, binary_drift


def read_data_quality_alerts(dict_metrics: dict, batch_name: str, batch_id: str):
    """Read data quality alerts from the json metrics file

    Args:
        dict_metrics (dict): json metrics file for data drift and concept drift
        batch_name (str): batch name
        batch_id (str): batch id 

    Returns:
        int: 1 if alert else 0
    """
    return dict_metrics[batch_id][batch_name]['data_quality']


def read_outliers_alert(dict_metrics: dict, batch_name: str, batch_id: str):
    """Read outliers alerts from the json metrics file

    Args:
        dict_metrics (dict): json metrics file for data drift and concept drift
        batch_name (str): batch name
        batch_id (str): batch id 

    Returns:
        int: 1 if alert else 0
    """
    return dict_metrics[batch_id][batch_name]['outliers']['alert']


def read_psi_alert(dict_metrics: dict, batch_name: str, batch_id: str):
    """Read PSI alerts from the json metrics file

    Args:
        dict_metrics (dict): json metrics file for data drift and concept drift
        batch_name (str): batch name
        batch_id (str): batch id 

    Returns:
        int: 1 if alert else 0
    """
    return dict_metrics[batch_id][batch_name]['metrics']['PSI']['alert']
