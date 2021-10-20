# -*- coding: utf-8 -*-

################## Importing libraries ####################
import sys
sys.path.insert(0,"Loading/")
sys.path.insert(0,"Preprocessing/")
sys.path.insert(0,"Modeling/")
sys.path.insert(0,"Evaluation/")
sys.path.insert(0,"Interpretability/")
sys.path.insert(0,"Monitoring/")
sys.path.insert(0,"Utils/")

import loading
import preprocessing
import modeling
import utils as u
import monitoring  

import constants as cst
import logging, json

import pandas as pd

logger = logging.getLogger(__name__)

def main():
    conf = cst.conf
    
    clf = u.load_model(conf)
    preprocessor = preprocessing.MarketingPreprocessor()
    
    batch_df = pd.read_csv('../Inputs/Batches/batch1.csv')
    preprocessed_batch = preprocessor(batch_df, column_types=cst.selected_dataset_information['column_types'])
    
    data = loading.read_csv_from_name(conf)
    preprocessed_df = preprocessor(data, column_types=cst.selected_dataset_information['column_types'])
    
    monitoring_metrics = monitoring.compute_metrics(preprocessed_df, preprocessed_batch, clf, cst.selected_dataset_information['metrics_setup'])
    
    # with open(cst.MONITORING_METRICS_FILE_PATH, 'r') as monitoring_file:
    #     metrics = json.load(monitoring_file)
    metrics = list()
    metrics.append(monitoring_metrics)
    with open(cst.MONITORING_METRICS_FILE_PATH, 'w') as monitoring_file:
        json.dump(metrics, monitoring_file)

if __name__ == "__main__":
    main()