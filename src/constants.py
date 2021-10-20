import json, os

# load configurations
PATH_CONF = "../params/conf/conf.json"
conf = json.load(open(PATH_CONF, 'r'))

# load paths
INPUTS_PATH = conf["paths"]["Inputs_path"]
OUTPUTS_PATH = conf["paths"]["Outputs_path"]
PREPROCESSED_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_preprocessed"])
MODELS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_models"])
METRICS_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_metrics"])
MONITORING_PATH = os.path.join(OUTPUTS_PATH, conf["paths"]["folder_monitoring"])

MONITORING_METRICS_FILE_NAME = "metrics.json"
MONITORING_METRICS_FILE_PATH = os.path.join(MONITORING_PATH, MONITORING_METRICS_FILE_NAME)


selected_dataset = conf["selected_dataset"]
selected_dataset_information = conf["dict_info_files"][selected_dataset]

y_name = selected_dataset_information["y_name"]

columns_nature = selected_dataset_information["column_nature"]
categorical_columns = columns_nature["categorical_columns"]
binary_columns = columns_nature["binary_columns"]
numerical_columns = columns_nature["numerical_columns"]

column_types = selected_dataset_information["column_types"]