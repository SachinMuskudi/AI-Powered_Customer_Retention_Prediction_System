import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
from logging_code import setup_logging
logger = setup_logging('handle_missing_values')


def handling_missing_values(X_train,X_test): #Using Mode Method
    try:
        logger.info(f'Before X_train Handling Null Values : {X_train.shape} \n{X_train.columns} \n{X_train.isnull().sum()}')
        logger.info(f'Before X_test Handling Null Values : {X_test.shape} \n{X_test.columns} \n{X_test.isnull().sum()}')
        
        for i in X_train.columns:
            
            if X_train[i].isnull().sum() > 0:
                X_train[i+'_mode'] = X_train[i].fillna(X_train[i].mode()[0])
                X_test[i+'_mode'] = X_test[i].fillna(X_test[i].mode()[0])
                X_train = X_train.drop([i],axis=1)
                X_test = X_test.drop([i],axis=1)
                
        logger.info(f'After X_train Handling Null Values : {X_train.shape} \n{X_train.columns} \n{X_train.isnull().sum()}')
        logger.info(f'After X_test Handling Null Values : {X_test.shape} \n{X_test.columns} \n{X_test.isnull().sum()}')
                
        
        return X_train,X_test
        
    except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")