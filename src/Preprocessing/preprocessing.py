# -*- coding: utf-8 -*-

#####  Imports  #####
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd

#####  Processors  #####
class Preprocessor(ABC):
    """Abstract class to preprocess supported datasets."""
    def __call__(self, data: pd.DataFrame, 
                 column_types: dict[str, list[str]]) -> pd.DataFrame:
        """Runs a pipeline with the three steps defined in the Preprocessor 
        class.

        Args:
            data (pd.DataFrame): data to be preprocessed.

        Returns:
            pd.DataFrame: preprocessed data
        """
        clean_data = self.clean_data(data)
        enforced_data = self.enforce_types(clean_data, column_types)
        preprocessed_data = self.treat_nans(enforced_data)
        
        return preprocessed_data
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the columns of the data."""
    
    @abstractmethod
    def enforce_types(self, data: pd.DataFrame, 
                      column_types: dict[str, list[str]]) -> pd.DataFrame:
        """Ensures all columns are typed according to configuration file."""
        
    @abstractmethod
    def treat_nans(self, data:pd.DataFrame) -> pd.DataFrame:
        """Deals with missing data in the data."""
    
class MarketingPreprocessor(Preprocessor):
    """Preprocessor to treat the marketing dataset."""
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info('Cleaning marketing data.')
        
        logger.debug('Cleaning Income column')
        #Cleaning Income
        data[' Income '] = data[' Income '].str.replace('$', '')
        data[' Income '] = data[' Income '].str.replace(',', '')
        data[' Income '] = data[' Income '].str.replace(' ', '')
        data[' Income '] =  data[' Income '].fillna(0)
        #Renaming
        data = data.rename(columns={" Income ": "Income"})
        
        logger.debug('Cleaning dates columns.')
        #Cleaning Dates:
        data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
        data['Dt_year'] = data['Dt_Customer'].dt.year.astype(int)
        data['Dt_month'] = data['Dt_Customer'].dt.month.astype(int)
        
        logger.debug('Dropping unique columns')
        # Drop id:
        data = data.drop(['ID','Dt_Customer'], axis=1)
        
        return data
        
    
    def enforce_types(self, data: pd.DataFrame, 
                      column_types: dict[str, list[str]]) -> pd.DataFrame:
        logger.info('Enforcing types')
        for dtype, cols in column_types.items():
            for col in cols:
                try:
                    # enforce type
                    data.loc[:, col] = data.loc[:, col].astype(dtype)
                except KeyError:
                    logger.debug(f'Could not find the column {col} in dataset')
                except Exception as e:
                    logger.error(e)
                    raise e
           
        return data

    def treat_nans(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.isna().sum().sum() > 0:
            raise ValueError('Found NAN values in the data. Please remove miss'
                             'ing values from the data before passing it to th'
                             'e pipeline.')
        
        return data

def basic_split(df: pd.DataFrame, target_column: str, train_size: float = 0.7, seed=42):
    """Splits the dataframe in train, test sets

    Args:
        df (pd.DataFrame): dataframe to split
        target_column (str): name of the target variable column
        train_size (float, optional): proportion of train set as a ratio. Defaults to 0.7
        seed (int, optional): Random state. Defaults to 42

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: train features, test features, 
        train labels, test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]), 
        df[target_column], 
        train_size=train_size, 
        random_state=seed
        )
    return X_train, X_test, y_train, y_test

