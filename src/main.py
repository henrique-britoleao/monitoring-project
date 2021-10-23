# -*- coding: utf-8 -*-

#####  Imports  #####
from monitoring import monitor_runner
import constants as cst

import argparse
import logging

#####  Set Logger  #####
# TODO: handle loggers better
logging.basicConfig(filename = cst.MAIN_LOG_FILE_PATH,
                    filemode = "w",
                    level = logging.INFO)

logger = logging.getLogger(__name__)    

#####  Main script  #####
# Set the argument parser
argument_parser = argparse.ArgumentParser(
    description='Process to monitor a batch.'
)
argument_parser.add_argument('-m', '--mode', default='process', nargs=1, choices=['process', 'evaluate'], help='Defines the type of monitoring to perform')
argument_parser.add_argument('--batch-id', help='Id of batch to be parsed', nargs=1, required=True)

# Define main script
def main(batch_id, mode="process"):
    # initialize runner
    runner = monitor_runner.MonitorRunner(batch_id)
    
    if mode == "process":
        runner.process_batch(preprocessor=cst.PREPROCESSOR)
    if mode == "evaluate":
        runner.evaluate_batch(preprocessor=cst.PREPROCESSOR)
        

def usage():
    # TODO
    pass
        
if __name__ == "__main__":
    args = argument_parser.parse_args()
    main(*args.batch_id, *args.mode)