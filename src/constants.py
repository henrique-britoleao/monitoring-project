# -*- coding: utf-8 -*-
"""Reads the configurations from params/conf/conf.json and stores values in 
constants to be used across the project."""

#####  Imports  #####
# from preprocessing.preprocessing import MarketingPreprocessor

import json, os
from string import Template

#####  Constants definitions  #####
# load configurations
PATH_CONF = "params/conf/conf.json"
conf = json.load(open(PATH_CONF, "r"))

selected_dataset = conf["selected_dataset"]
selected_model = conf["selected_model"]
selected_dataset_information = conf["dict_info_files"][selected_dataset]
selected_dataset_main_target_class = selected_dataset_information["y_labels"][0]

y_name = selected_dataset_information["y_name"]
y_pred = f"{y_name}_pred"
y_pred_proba = f"{y_name}_pred_proba"
y_pred_proba_main_class = f"{y_pred_proba}_{selected_dataset_main_target_class}"
y_class_labels = selected_dataset_information["y_labels"]

columns_nature = selected_dataset_information["column_nature"]
categorical_columns = columns_nature["categorical_columns"]
binary_columns = columns_nature["binary_columns"]
numerical_columns = columns_nature["numerical_columns"]

column_types = selected_dataset_information["column_types"]


# load paths
INPUTS_PATH = conf["paths"]["Inputs_path"]
BATCHES_PATH = os.path.join(INPUTS_PATH, conf["paths"]["folder_batches"])
OUTPUTS_PATH = conf["paths"]["Outputs_path"]
PREPROCESSED_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_preprocessed"])
PREDICTED_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_predicted"])
MODELS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_models"])
METRICS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_metrics"])
MONITORING_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_monitoring"])

# build file paths
RAW_TRAIN_FILE_NAME = selected_dataset_information["file_name"]
RAW_TRAIN_FILE_PATH = os.path.join(INPUTS_PATH, RAW_TRAIN_FILE_NAME)

PREPROCESSED_TRAIN_FILE_NAME = "sample_df_preprocessed.csv"
PREPROCESSED_TRAIN_FILE_PATH = os.path.join(
    PREPROCESSED_PATH, PREPROCESSED_TRAIN_FILE_NAME
)

PREDICTED_TRAIN_FILE_NAME = "sample_df_predicted.csv"
PREDICTED_TRAIN_FILE_PATH = os.path.join(PREDICTED_PATH, PREDICTED_TRAIN_FILE_NAME)

PREPROCESSED_BATCH_FILE_NAME = "batch_df_preprocessed.csv"
PREPROCESSED_BATCH_FILE_PATH = os.path.join(
    PREPROCESSED_PATH, PREPROCESSED_BATCH_FILE_NAME
)

PREDICTED_BATCH_FILE_NAME = "batch_df_predicted.csv"
PREDICTED_BATCH_FILE_PATH = os.path.join(PREDICTED_PATH, PREDICTED_BATCH_FILE_NAME)

TRAINING_METRICS_FILE_NAME = "training_metrics"
TRAINING_METRICS_FILE_PATH = os.path.join(METRICS_PATH, TRAINING_METRICS_FILE_NAME)

MONITORING_METRICS_FILE_NAME = "metrics.json"
MONITORING_METRICS_FILE_PATH = os.path.join(
    MONITORING_PATH, MONITORING_METRICS_FILE_NAME
)

PERFORMANCE_METRICS_FILE_NAME = "performance.json"
PERFORMANCE_METRICS_FILE_PATH = os.path.join(
    MONITORING_PATH, PERFORMANCE_METRICS_FILE_NAME
)

TRAIN_PERFORMANCE_METRICS_FILE_NAME = (
    f"{selected_dataset}_{selected_model}_train_performance.json"
)
TRAIN_PERFORMANCE_METRICS_FILE_PATH = os.path.join(
    METRICS_PATH, TRAIN_PERFORMANCE_METRICS_FILE_NAME
)

# Log files
MAIN_LOG_FILE_PATH = conf["paths"]["main_logger"]
DEBUG_LOG_FILE_PATH = conf["paths"]["debug_logger"]
TRAINING_LOG_FILE_PATH = conf["paths"]["training_logger"]
ALERT_LOG_FILE_PATH = conf["paths"]["alert_logger"]

# Naming templates
BATCH_NAME_TEMPLATE = Template("batch${id}.csv")
