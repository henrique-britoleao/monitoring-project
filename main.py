# -*- coding: utf-8 -*-

#####  Imports  #####
from src.monitoring import monitor_runner
from src.preprocessing.preprocessing import get_preprocessor
import src.constants as cst

import argparse
import logging

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

#####  Main script  #####
# Set the argument parser
argument_parser = argparse.ArgumentParser(
    description='Process to monitor a batch.'
)
argument_parser.add_argument(
    '-m', '--mode', 
    default='process', 
    nargs=1, 
    choices=['process', 'evaluate'], 
    help='Defines the type of monitoring to perform'
)
argument_parser.add_argument(
    '--batch-id', 
    help='Id of batch to be parsed', 
    nargs=1, 
    required=True)

# Define main script
def main(batch_id, mode="process"):
    """Runs monitoring on batch."""
    # initialize runner
    runner = monitor_runner.MonitorRunner(batch_id)
    preprocessor = get_preprocessor()
    
    if mode == "process":
        runner.process_batch(preprocessor=preprocessor)
    if mode == "evaluate":
        runner.evaluate_batch(preprocessor=preprocessor)

        
if __name__ == "__main__":
    args = argument_parser.parse_args()
    main(*args.batch_id, *args.mode)