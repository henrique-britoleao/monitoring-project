# -*- coding: utf-8 -*-

#####  Imports  #####
import constants as cst

from functools import wraps
import logging

#####  Loggers  #####
logs_format = '[%(levelname)s - %(asctime)s] %(name)s: %(message)s'

class MainLogger(logging.Logger):
    """TODO"""
    @classmethod
    def getLogger(cls, name=None):
        """TODO"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        
        file_handler = logging.FileHandler(cst.MAIN_LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        debug_file_handler = logging.FileHandler(cst.DEBUG_LOG_FILE_PATH)
        debug_file_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_file_handler)
        
        logger.addFormatter(logging.Formatter(logs_format))