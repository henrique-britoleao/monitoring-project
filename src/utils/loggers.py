# -*- coding: utf-8 -*-

#####  Imports  #####
import constants as cst

import logging

#####  Loggers  #####
logs_format = '[%(levelname)s - %(asctime)s] %(name)s: %(message)s'

class MainLogger:
    """TODO"""
    formatter = logging.Formatter(logs_format)
    @classmethod
    def getLogger(cls, name=None) -> logging.Logger:
        """TODO"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        if not (len(logger.handlers)):
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(cls.formatter)
            logger.addHandler(stream_handler)
            
            file_handler = logging.FileHandler(cst.MAIN_LOG_FILE_PATH)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(cls.formatter)
            logger.addHandler(file_handler)
            
            debug_file_handler = logging.FileHandler(cst.DEBUG_LOG_FILE_PATH)
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_handler.setFormatter(cls.formatter)
            logger.addHandler(debug_file_handler)
        
        return logger
    
class AlertLogger:
    formatter = logging.Formatter(logs_format)
    @classmethod
    def getLogger(cls, name=None) -> logging.Logger:
        """TODO"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        
        if not len(logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.WARNING)
            stream_handler.setFormatter(cls.formatter)
            logger.addHandler(stream_handler)
            
            file_handler = logging.FileHandler(cst.ALERT_LOG_FILE_PATH)
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(cls.formatter)
            logger.addHandler(file_handler)
        
        return logger