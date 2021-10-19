import json

path_conf = "../params/conf/conf.json"
conf = json.load(open(path_conf, 'r'))

selected_dataset = conf["selected_dataset"]
selected_dataset_information = conf["dict_info_files"][selected_dataset]

y_name = selected_dataset_information["y_name"]

columns_nature = selected_dataset_information["column_nature"]
categorical_columns = columns_nature["categorical_columns"]
binary_columns = columns_nature["binary_columns"]
numerical_columns = columns_nature["numerical_columns"]

column_types = selected_dataset_information["column_types"]