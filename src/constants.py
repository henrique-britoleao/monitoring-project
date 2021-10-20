# -*- coding: utf-8 -*-

#####  Imports  #####
import json, os
from string import Template

# load configurations
#PATH_CONF = "../params/conf/conf.json"
PATH_CONF = "params/conf/conf.json"
conf = json.load(open(PATH_CONF, 'r'))

# load paths
INPUTS_PATH = conf["paths"]["Inputs_path"]
BATCHES_PATH = os.path.join(INPUTS_PATH, conf["paths"]["folder_batches"])
OUTPUTS_PATH = conf["paths"]["Outputs_path"]
PREPROCESSED_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_preprocessed"])
PREDICTED_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_predicted"]) 
MODELS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_models"])
METRICS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_metrics"])
MONITORING_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_monitoring"])
PREDICTIONS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_predictons"])


# build file paths
RAW_TRAIN_FILE_NAME = "sample_df.csv"
RAW_TRAIN_FILE_PATH = os.path.join(INPUTS_PATH, RAW_TRAIN_FILE_NAME)

RAW_BATCH_FILE_NAME = "batch_df.csv"
RAW_BATCH_FILE_PATH = os.path.join(INPUTS_PATH, RAW_BATCH_FILE_NAME)

PREPROCESSED_TRAIN_FILE_NAME = "sample_df_preprocessed.csv"
PREPROCESSED_TRAIN_FILE_PATH = os.path.join(PREPROCESSED_PATH, PREPROCESSED_TRAIN_FILE_NAME)

PREDICTED_TRAIN_FILE_NAME = "sample_df_predicted.csv"
PREDICTED_TRAIN_FILE_PATH = os.path.join(PREPROCESSED_PATH, PREPROCESSED_TRAIN_FILE_NAME)

PREPROCESSED_BATCH_FILE_NAME = "batch_df_preprocessed.csv"
PREPROCESSED_BATCH_FILE_PATH = os.path.join(PREPROCESSED_PATH, PREPROCESSED_BATCH_FILE_NAME)

PREDICTED_BATCH_FILE_NAME = "batch_df_predicted.csv"
PREDICTED_BATCH_FILE_PATH = os.path.join(PREDICTED_PATH, PREDICTED_BATCH_FILE_NAME)

TRAINING_METRICS_FILE_NAME = "training_metrics"
TRAINING_METRICS_FILE_PATH = os.path.join(METRICS_PATH, TRAINING_METRICS_FILE_NAME)

MONITORING_METRICS_FILE_NAME = "metrics.json"
MONITORING_METRICS_FILE_PATH = os.path.join(MONITORING_PATH, MONITORING_METRICS_FILE_NAME)

PROBA_PREDICTIONS_FILE_NAME = "batch_proba_predictions.pickle"
PROBA_PREDICTIONS_FILE_PATH = os.path.join(PREDICTIONS_PATH, PROBA_PREDICTIONS_FILE_NAME)

PREDICTIONS_FILE_NAME = "batch_predictions.pickle"
PREDICTIONS_FILE_PATH = os.path.join(PREDICTIONS_PATH, PREDICTIONS_FILE_NAME)


selected_dataset = conf["selected_dataset"]
selected_model = conf["selected_model"]
selected_dataset_information = conf["dict_info_files"][selected_dataset]

y_name = selected_dataset_information["y_name"]
y_pred = f"{y_name}_pred"
y_pred_proba = f"{y_name}_pred_proba"
y_class_labels = selected_dataset_information["y_labels"]

columns_nature = selected_dataset_information["column_nature"]
categorical_columns = columns_nature["categorical_columns"]
binary_columns = columns_nature["binary_columns"]
numerical_columns = columns_nature["numerical_columns"]

column_types = selected_dataset_information["column_types"]

BATCH_NAME_TEMPLATE = Template('batch${id}.csv')
