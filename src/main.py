# -*- coding: utf-8 -*-

################## Importing libraries ####################
import os
print(os.getcwd())

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
import evaluation
import interpretability
import monitoring
import utils as u

import argparse
import json
from time import time

import logging

## The parser allow to get arguments from the command line in order to launch only selected steps
parser = argparse.ArgumentParser(description='1st Industrialised ML Project',
epilog="This has been developped by Quinten")

parser.add_argument('--step', help='integer that tells which step to run', default=-1)
parser.add_argument('--step_from',
help='integer that tells from which step to run main. It then run all the steps from step_from',
default=0)
parser.add_argument('--step_list', help='list of integer that tells which steps to run', default=[])

parser.add_argument("--pathconf", help="path to conf file", default="../params/conf/conf.json")

args = parser.parse_args()
step = int(args.step)
step_from = int(args.step_from)
step_list = args.step_list
path_conf = args.pathconf

# path_conf ='../conf/conf.json'
conf = json.load(open(path_conf, 'r'))

path_log = conf['path_log'] # "../log/my_log_file.txt"
log_level = conf['log_level'] # "DEBUG"

# instanciation of the logger
logger = u.my_get_logger(path_log, log_level, my_name="main_logger")


def main(logger, step_list, NB_STEP_TOT, path_conf = '../conf/conf.json'):
    """
    Main function launching step by step the ML Pipeline
    Args:
        logger: Logger file
        step_list: List of steps to be executes
        NB_STEP_TOT: By default = number of total step to laucnh them all if no specific steps are given
        path_conf: path of the conf file
    """
    START = time()

    #Computation of the steps to complete
    if len(step_list) > 0:
        step_list = eval(step_list)
    
    if (step == -1) and (len(step_list) == 0):
        step_list = list(range(step_from, NB_STEP_TOT + 1))
    
    print(step_list)
    logger.debug('Steps to execute :' + ', '.join(map(str,step_list)))
    
    #Reading conf file
    conf = json.load(open(path_conf, 'r'))
    seed = 42

    #Launch of each selected step
    if (step == 1) or (1 in step_list):
        logger.debug("Beginning of step 1 - Loading and Preprocessing")
        # Reading of the dataset selected in the conf file
        df = loading.read_csv_from_name(conf)

        # Preprocessing of the selected dataset
        df_preprocessed, X_columns, y_column = preprocessing.main_preprocessing_from_name(df, conf)

        # Writting of the preprocessed dataset
        loading.write_preprocessed_csv_from_name(df_preprocessed, conf)

        logger.debug("End of step 1 ")

    if (step == 2) or (2 in step_list):
        
        logger.debug(" Beginning of step 2 - Loading Preprocessed ")
        # Loading of the preprocessed dataset
        df_preprocessed = loading.load_preprocessed_csv_from_name(conf)
        # Basic Splitting between train and test
        y_column = u.get_y_column_from_conf(conf)
        X_columns = [x for x in df_preprocessed.columns if x != y_column]
        X_train, X_test, y_train, y_test = preprocessing.basic_split(df_preprocessed, 0.25, X_columns, y_column , seed=seed)

        logger.debug(" End of step 2 ")

    if (step == 3) or (3 in step_list):
        if 2 not in step_list: #Step 2 must be launched with step 3
            df_preprocessed = loading.load_preprocessed_csv_from_name(conf)
            # Basic Splitting between train and test
            y_column = u.get_y_column_from_conf(conf)
            X_columns = [x for x in df_preprocessed.columns if x != y_column]
            X_train, X_test, y_train, y_test = preprocessing.basic_split(df_preprocessed, 0.25, X_columns, y_column, seed=seed)

        logger.debug(" Beginning of step 3 - Modeling ")
        
        # Modelisation using the model selected in the conf file
        clf, best_params = modeling.main_modeling_from_name(X_train, y_train, conf)
        # Saving the model
        u.save_model(clf, conf)
        logger.debug(" End of step 3 ")


    if (step == 4) or (4 in step_list):
        if 2 not in step_list: #Step 2 must be launched with step 4
            df_preprocessed = loading.load_preprocessed_csv_from_name(conf)
            # Basic Splitting between train and test
            y_column = u.get_y_column_from_conf(conf)
            X_columns = [x for x in df_preprocessed.columns if x != y_column]
            X_train, X_test, y_train, y_test = preprocessing.basic_split(df_preprocessed, 0.25, X_columns, y_column, seed=seed)

        logger.debug(" Beginning of step 4 - Evaluation")

        # Loading of the model
        clf = u.load_model(conf)

        # Computing metrics
        dict_metrics = evaluation.main_evaluation(clf, X_train, y_train, X_test, y_test, conf)


        logger.debug("End of step 4 ")


    logger.debug("Time for total execution :" + str(time() - START))
        

if __name__ == '__main__':
    try:
        main(logger, step_list, NB_STEP_TOT = 4, path_conf=path_conf)
    
    except Exception as e:
        logger.error("Error during execution", exc_info=True)
