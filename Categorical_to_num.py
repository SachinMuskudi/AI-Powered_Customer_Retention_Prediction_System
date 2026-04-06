import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging
from logging_code import setup_logging
logger  = setup_logging('Categorical_to_num')
import sys
import os
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

def c_t_n(X_train_cat,X_test_cat):
    try:
        #Nominal Encoding
        logger.info(f'Before Nominal X_train_cat column: {X_train_cat.shape}: \n{X_train_cat.columns}')
        logger.info(f'Before Nominal X_test_cat column: {X_test_cat.shape}: \n{X_test_cat.columns}')
        oh = OneHotEncoder(drop='first')
        oh.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner']])
        train_values = oh.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner']]).toarray()
        test_values = oh.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner']]).toarray()
        t1 = pd.DataFrame(train_values)
        t2 = pd.DataFrame(test_values)
        t1.columns = oh.get_feature_names_out()
        t2.columns = oh.get_feature_names_out()
        X_train_cat.reset_index(drop=True,inplace=True)
        X_test_cat.reset_index(drop=True,inplace=True)
        t1.reset_index(drop=True,inplace=True)
        t2.reset_index(drop=True,inplace=True)
        X_train_cat = pd.concat([X_train_cat,t1],axis=1)
        X_test_cat = pd.concat([X_test_cat,t2],axis=1)
        X_train_cat = X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner'],axis=1)
        X_test_cat = X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner'],axis=1)
        logger.info(f'After Nominal X_train_cat column: {X_train_cat.shape}: \n{X_train_cat.columns}')
        logger.info(f'After Nominal X_test_cat column: {X_test_cat.shape}: \n{X_test_cat.columns}')
        
        
        #Ordinal Encoding
        logger.info(f'Before Ordinal X_train_cat column: {X_train_cat.shape}: \n{X_train_cat.columns}')
        logger.info(f'Before Ordinal X_test_cat column: {X_test_cat.shape}: \n{X_test_cat.columns}')

        od = OrdinalEncoder()
        od.fit(X_train_cat[['Contract']])
        train_res = od.transform(X_train_cat[['Contract']])
        test_res = od.transform(X_test_cat[['Contract']])
        p1 = pd.DataFrame(train_res)
        p2 = pd.DataFrame(test_res)
        p1.columns = od.get_feature_names_out()+'_od'
        p2.columns = od.get_feature_names_out()+'_od'
        p1.reset_index(drop=True,inplace=True)
        p2.reset_index(drop=True,inplace=True)
        X_train_cat = pd.concat([X_train_cat,p1],axis=1)
        X_test_cat = pd.concat([X_test_cat,p2],axis=1)
        X_train_cat = X_train_cat.drop(['Contract'], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)

        logger.info(f'After Ordinal X_train_cat column: {X_train_cat.shape}: \n{X_train_cat.columns}')
        logger.info(f'After Ordinal X_test_cat column: {X_test_cat.shape}: \n{X_test_cat.columns}')

        logger.info(f'After Ordinal Train Null Values: \n{X_train_cat.isnull().sum()}')
        logger.info(f'After Ordinal Test Null Values: \n{X_test_cat.isnull().sum()}')

        return X_train_cat,X_test_cat
        
    except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")