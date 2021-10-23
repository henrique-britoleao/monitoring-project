#####  Imports  #####
import logging
import sys
import os

#####  Get logger  ######
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

#####  Set globals  #####
required_folders = [
    './Outputs/Metrics',
    './Outputs/Models',
    './Outputs/Monitoring',
    './Outputs/Predicted',
    './Outputs/Preprocessed',
    './Inputs/Batches',
    './params/logs'
]

required_files = [
    './Outputs/Monitoring/metrics.json',
    './Outputs/Monitoring/performance.json',
    './logs/main_logs.log',
    './logs/debug_logs.log',
	'./logs/training_logs.log'
]

#####  Set up folder structure  #####
if __name__=="__main__":
    logger.info('Started setup')

    for folder in required_folders:
        try:
            os.makedirs(folder)
            logger.info(f'Created folder: {folder}')
        except FileExistsError:
            logger.info(f'Already found folder: {folder}')
            

    for file in required_files:
        with open(file, mode='w'): 
            logger.info(f'Ensured file is created: {file}')