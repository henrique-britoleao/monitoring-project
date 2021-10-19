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
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs a pipeline with the three steps defined in the Preprocessor 
        class.

        Args:
            data (pd.DataFrame): data to be preprocessed.

        Returns:
            pd.DataFrame: preprocessed data
        """
        clean_data = self.clean_data(data)
        enforced_data = self.enforce_types(clean_data)
        preprocessed_data = self.treat_nans(enforced_data)
        
        return preprocessed_data
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the columns of the data."""
    
    @abstractmethod
    def enforce_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensures all columns are correctly typed."""
        
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
        
    
    def enforce_types(self, data: pd.DataFrame) -> pd.DataFrame:
        column_types = {
            int: ['Year_Birth', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 
                  'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                  'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                  'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 
                  'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response', 
                  'Complain'],
            str: ['Education', 'Marital_Status', 'Income', 'Country'],
        }
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

#####  Splitting  #####
def basic_split(df, size_of_test, target_column, seed=42):
    """
    Split the dataframe in train, test sets
    Args:
        df: Dataframe to Split
        size_of_test: proportion of test dataset
        X_columns: Columns for the variables
        y_column: Column for the output
        seed: Random state/seed

    Returns: Train and test datasets for variables and output.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=target_column), 
        df[target_column], 
        test_size=size_of_test, 
        random_state =seed,
    )
    return X_train, X_test, y_train, y_test