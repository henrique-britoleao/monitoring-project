# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

from sklearn.model_selection import train_test_split
import utils as u
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

#TODO: complete this script, or create sub_scripts such as preprocessing_smote etc..

# Description of this script:
# main function calling the sub preprocessing function for the dataset selected
# subfunctions applying preprocessing (ex: one hot encoding, dropping etc..)


def main_preprocessing_from_name (df, conf):
    """
    Main Preprocessing function: it launches the correct function in order to preprocess the selected dataset
    Args:
        df: Dataframe
        conf: Conf file

    Returns: Preprocessed Dataframe

    """

    dict_function_preprocess = {'drift': 'preprocessing_for_drift_dataset',
                                'fraude': 'preprocessing_for_fraud_dataset',
                                'stroke': 'preprocessing_for_stroke_dataset',
                                'banking':'preprocessing_for_banking_dataset',
                                'diabetic':'preprocessing_for_diabetic_dataset',
                                'wine':'preprocessing_for_wine_dataset',
                                'marketing':'preprocessing_for_marketing_dataset'}

    selected_dataset = conf['selected_dataset']
    function_preprocess = globals()[dict_function_preprocess[selected_dataset]]
    logger.info('Beginning of preprocessing function: ' + dict_function_preprocess[selected_dataset] )
    df_preprocessed = function_preprocess(df, conf)
    logger.info('End of preprocessing function: ' + dict_function_preprocess[selected_dataset] )

    return df_preprocessed


def preprocessing_for_drift_dataset(df,conf):
    """
    Preprocessing for the DRIFT dataset
    Args:
        df: Drift dataset
        conf:  conf file

    Returns: Preprocessed Drift Dataset

    """
    #Steps:
    # Clean the output (as 0 or 1)
    # one hot elevel, car, zipcode
    # drop id
    logger.debug('Cleaning output')
    # Cleaning Output:
    df['class'] = df['class'].map({'groupA': 1, 'groupB': 0})

    logger.debug('One Hot Encoding')
    # one hot encoding
    cols = ['elevel','car','zipcode',]
    df = one_hot_encoder(df, cols)

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df.drop(['id'], axis=1)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column ]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.debug('preprocessing drift ok')

    return df_preprocessed, X_columns, y_column

def preprocessing_for_stroke_dataset(df,conf):
    """
    Preprocessing for the STROKE dataset
    Args:
        df: Stroke dataset
        conf:  conf file

    Returns: Preprocessed Stroke Dataset

    """
    #Steps:
    # one hot elevel, car, zipcode
    # Yes/No in 1/0
    # transform NA values
    # drop id

    logger.debug('One Hot Encoding')

    # one hot encoding
    cols = ['gender', 'work_type', 'Residence_type', 'smoking_status']
    df = one_hot_encoder(df, cols)

    logger.debug('mapping values')
    # transform Yes/No in 1/0
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})

    logger.debug('FillNA')
    # transform NA values
    df['bmi'] = df['bmi'].fillna(-1)

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df.drop(['id'], axis=1)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column ]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.info('preprocessing stroke ok')

    return df_preprocessed, X_columns, y_column


def preprocessing_for_banking_dataset(df, conf):
    """
    Preprocessing for the BANKING dataset
    Args:
        df: Banking dataset
        conf:  conf file

    Returns: Preprocessed Banking Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # one hot elevel, car, zipcode
    # drop id

    logger.debug('Cleaning Output')

    # Cleaning Output:
    df['deposit_subscription'] = df['deposit_subscription'].map({'yes': 1, 'no': 0})

    logger.debug('Mapping Values')
    #Cleaning Other fields:
    for col in  ['loan','housing','default']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df["month"] = df["month"].map({'jan': 1, 'feb': 2,'mar': 3, 'apr': 4,'may': 5, 'jun': 6,'jul': 7, 'aug': 8,'sep': 9, 'oct': 10,'nov': 11, 'dec': 12})

    logger.debug('One Hot Encoding')
    # one hot encoding
    cols = ['poutcome', 'contact', 'education','marital','job' ]
    df_preprocessed = one_hot_encoder(df, cols)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.info('preprocessing banking ok')

    return df_preprocessed, X_columns, y_column

def preprocessing_for_diabetic_dataset(df, conf):
    """
    Preprocessing for the DIABETIC dataset
    Args:
        df: Diabetic dataset
        conf:  conf file

    Returns: Preprocessed Diabetic Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # one hot elevel, car, zipcode
    # drop id

    logger.debug('Cleaning Output')
    # Cleaning Output:
    df['readmitted'] = df['readmitted'].map({'>30': 1, '<30':1, 'NO': 0})

    logger.debug('mapping Values')
    #Cleaning Other fields:
    for col in  ['race','gender','weight','payer_code','medical_specialty']:
        df[col] = df[col].str.replace('?', 'unknown')


    df['age'] = df['age'].map({'unknown':0, '[0-10)':5, '[10-20)':15,'[20-30)':25,'[30-40)':35,'[40-50)':45,
                           '[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95})

    df['weight'] = df['weight'].map({'unknown':0, '[0-25)':15, '[25-50)':40,'[50-75)':65,'[75-100)':90,'[100-125)':115,
                           '[125-150)':140,'[150-175)':165,'[175-200)':190,'>200':215})

    logger.debug('One hot Encoding')
    # one hot encoding
    cols = ['gender','race','payer_code', 'medical_specialty', 'max_glu_serum','A1Cresult','metformin',
            'repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide',
            'glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
            'examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed' ]
    df = one_hot_encoder(df, cols)

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df.drop(['encounter_id','patient_nbr'], axis=1)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column]

    logger.debug('Verification of float and na values')

    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.info('preprocessing Diabetic ok')

    return df_preprocessed, X_columns, y_column

def preprocessing_for_wine_dataset(df, conf):

    """
    Preprocessing for the WINE dataset
    Args:
        df: Wine dataset
        conf:  conf file

    Returns: Preprocessed Wine Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # one hot elevel, car, zipcode
    # drop id

    logger.debug('Cleaning Output')
    # Cleaning Output:
    df['quality'] = np.where(df['quality']>5,1,0)

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df.drop(['Time'], axis=1)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.info('preprocessing Wine ok')

    return df_preprocessed, X_columns, y_column

def preprocessing_for_marketing_dataset(df, conf):
    """
    Preprocessing for the WINE dataset
    Args:
        df: Wine dataset
        conf:  conf file

    Returns: Preprocessed Wine Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # one hot
    # drop id

    logger.debug('Cleaning ')
    #Cleaning Income
    df[' Income '] = df[' Income '].str.replace('$', '')
    df[' Income '] = df[' Income '].str.replace(',', '')
    df[' Income '] = df[' Income '].str.replace(' ', '')
    df[' Income '] =  df[' Income '].fillna(0)
    #Renaming
    df = df.rename(columns={" Income ": "Income"})

    #Cleaning Dates:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Dt_year'] = df['Dt_Customer'].dt.year
    df['Dt_month'] = df['Dt_Customer'].dt.month

    logger.debug('One hot Encoding')
    # one hot encoding
    cols = ['Education','Marital_Status','Country' ]
    df = one_hot_encoder(df, cols)

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df.drop(['ID','Dt_Customer'], axis=1)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.info('preprocessing Marketing ok')

    return df_preprocessed, X_columns, y_column


def preprocessing_for_fraud_dataset(df, conf):
    """
    Preprocessing for the FRAUD dataset
    Args:
        df: Fraud dataset
        conf:  conf file

    Returns: Preprocessed Fraud Dataset

    """
    logger.debug('Scaling Data')

    rob_scaler = RobustScaler()
    std_scaler = StandardScaler()

    df_preprocessed = df.copy()

    df_preprocessed['Amount'] = std_scaler.fit_transform(df_preprocessed['Amount'].values.reshape(-1, 1))
    df_preprocessed['Time'] = rob_scaler.fit_transform(df_preprocessed['Time'].values.reshape(-1, 1))

    logger.debug('Selection of X and Y')
    y_column = u.get_y_column_from_conf(conf)
    columns_V = [col for col in df_preprocessed.columns if col not in ["Time", "Amount"] if col not in y_column]
    df_preprocessed[columns_V] = std_scaler.fit_transform(df_preprocessed[columns_V])

    X_columns = [col for col in df_preprocessed.columns if col not in y_column]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA présent dans "+ col)
    logger.info('preprocessing Fraud ok')

    return df_preprocessed, X_columns, y_column


############################## Preprocessing Utils Functions ###########
def one_hot_encoder(df, cols):
    """
    One hot encoder, while dropping the encoded columns
    Args:
        df: dataframe
        cols: cols to encode

    Returns:dataframe with one hot encoded columns

    """
    #transform categorical features in OHE
    df_added = pd.get_dummies(df[cols], prefix=cols)
    df = pd.concat([df.drop(cols,axis=1), df_added],axis=1)
    return df


# This part can be optimized as a selection of function from the conf file (as the preprocessing is)
def basic_split(df, size_of_test, X_columns, y_column, seed = 42):
    """
    Split the dataframe in train, test sets
    Args:
        df: Dataframe to Split
        size_of_test: proportion of test dataset
        X_columns: Columns for the variables
        y_column: Column for the output
        seed: Random state/seed

    Returns: Train and test datasets for variables and output

    """
    X_train, X_test, y_train, y_test = train_test_split(df[X_columns], df[y_column], test_size = size_of_test , random_state =seed )
    return X_train, X_test, y_train, y_test


