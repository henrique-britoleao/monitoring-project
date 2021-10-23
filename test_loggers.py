from src.utils.loggers import MainLogger

if __name__ == "__main__":
    logger = MainLogger.getLogger()     
    logger.debug('Test')
    logger.info('Test')
    logger.error('Test')