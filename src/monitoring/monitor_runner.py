# -*- coding: utf-8 -*-

#####  Imports  #####
from preprocessing import preprocessing, data_quality
from monitoring import monitoring, outlier_detection
from evaluation import evaluation
import constants as cst
from utils import utils
import train_model
import loading

from datetime import datetime as dt
import pandas as pd
import os

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

#####  Monitor runner  #####
class MonitorRunner:
    def __init__(self, batch_id: int) -> None:
        self.batch_id = batch_id
        self.batch_file_name = cst.BATCH_NAME_TEMPLATE.substitute(id=batch_id)
    
    def process_batch(self, preprocessor: preprocessing.Preprocessor):
        batch_data = loading.read_csv_from_path(os.path.join(cst.BATCHES_PATH, self.batch_file_name))
        # check quality of batch
        data_quality_alerts = self._check_batch_quality(batch_data)
        # preprocess batch data
        batch_preprocessed = self._preprocess_batch(batch_data, preprocessor)
        # load preprocessed sample data
        predicted_batch = self._make_batch_predictions(batch_preprocessed, "") # TODO: model name
        
        # load sample data
        sample_preprocesssed = loading.read_csv_from_path(cst.PREPROCESSED_TRAIN_FILE_PATH)
        predicted_sample = loading.read_csv_from_path(cst.PREDICTED_TRAIN_FILE_PATH)
        # compute drift metrics
        monitoring_metrics = monitoring.compute_metrics( 
            sample_preprocesssed, 
            batch_preprocessed, 
            predicted_batch.loc[:, f'{cst.y_pred_proba}_{cst.y_class_labels[1]}'],
            predicted_sample.loc[:, f'{cst.y_pred_proba}_{cst.y_class_labels[1]}'],
            cst.selected_dataset_information['metrics_setup']
        )
        # detect outliers
        outlier_alerts = outlier_detection.build_outlier_dict(batch_preprocessed)
        
        # initialize output dict
        records = {self.batch_file_name: dict()}
        # populate records
        records[self.batch_file_name]['date'] = str(dt.now())
        records[self.batch_file_name].update({'data_quality': data_quality_alerts})
        records[self.batch_file_name].update({'metrics': monitoring_metrics})
        records[self.batch_file_name].update({'outliers': outlier_alerts})
        
        logger.info(f'Done creating records for batch {self.batch_id}')

        utils.append_to_json(records, cst.MONITORING_METRICS_FILE_PATH)
        
    def evaluate_batch(self, preprocessor: preprocessing.Preprocessor): 
        batch_data = loading.read_csv_from_path(os.path.join(cst.BATCHES_PATH, self.batch_file_name))
        batch_preprocessed = self._preprocess_batch(batch_data, preprocessor) # TODO: laod saved based on batch id and raise error if file not present
        # load model
        model = utils.load_model()
        
        X_batch = batch_preprocessed.drop(columns=cst.y_name)
        y_batch = batch_preprocessed[cst.y_name]

        performance_metrics = evaluation.evaluate_model_performance_on_test(model, X_batch, y_batch)
            
        utils.append_to_json({self.batch_file_name: performance_metrics}, cst.PERFORMANCE_METRICS_FILE_PATH)
        
    def _check_batch_quality(self, batch_data: pd.DataFrame):
        sample_data = loading.load_training_data()
        data_quality_alerts = data_quality.check_data_quality(batch_data, sample_data)
        
        return data_quality_alerts
    
    def _preprocess_batch(self, batch_data:pd.DataFrame, preprocessor: preprocessing.Preprocessor) -> pd.DataFrame:
        preprocessed_batch_data = preprocessor(batch_data, cst.column_types)
        loading.write_csv_from_path(preprocessed_batch_data, cst.PREPROCESSED_BATCH_FILE_PATH)
        logger.info('Saved batch predicitons') # TODO: save based on batch id
        
        return preprocessed_batch_data
    
    def _make_batch_predictions(self, preprocessed_batch_data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        model = utils.load_model(model_name) # load pre-trained-model
        predicted_batch = train_model.make_predictions(model, preprocessed_batch_data)
        
        return predicted_batch
