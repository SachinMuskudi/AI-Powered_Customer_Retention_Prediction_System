import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
from logging_code import setup_logging
logger = setup_logging('var_out')
from scipy.stats import yeojohnson


def vt_outliers(X_train_num,X_test_num):
    try:
        logger.info(f'X_train_num before transformation: {X_train_num.shape} \n{X_train_num.columns} \n{X_train_num.isnull().sum()}')
        logger.info(f'X_test_num before transformation: {X_test_num.shape} \n{X_test_num.columns} \n{X_test_num.isnull().sum()}')
        
        for i in X_train_num.columns:
            if i != 'SeniorCitizen':
                #Variable Transformation
                X_train_num[i+'_yeo'],lam_val = yeojohnson(X_train_num[i])
                X_test_num[i+'_yeo'],lam_val = yeojohnson(X_test_num[i])
                
                X_train_num = X_train_num.drop([i],axis=1)
                X_test_num = X_test_num.drop([i],axis=1)
                
                #Outliers Detection
                iqr = X_train_num[i+'_yeo'].quantile(0.75) - X_train_num[i+'_yeo'].quantile(0.25)
                lower_limit = X_train_num[i+'_yeo'].quantile(0.25) - (1.5 * iqr)
                upper_limit = X_train_num[i+'_yeo'].quantile(0.75) + (1.5 * iqr)
                
                X_train_num[i+'_trim'] = np.where(X_train_num[i+'_yeo'] > upper_limit,upper_limit,
                                            np.where(X_train_num[i+'_yeo'] < lower_limit,lower_limit,X_train_num[i+'_yeo']))
                
                X_test_num[i+'_trim'] = np.where(X_test_num[i+'_yeo'] > upper_limit,upper_limit,
                                            np.where(X_test_num[i+'_yeo'] < lower_limit,lower_limit,X_test_num[i+'_yeo']))
                
                X_train_num = X_train_num.drop([i+'_yeo'],axis=1)
                X_test_num = X_test_num.drop([i+'_yeo'],axis=1)
                
        logger.info(f'X_train_num After transformation: {X_train_num.shape} \n{X_train_num.columns} \n{X_train_num.isnull().sum()}')
        logger.info(f'X_test_num After transformation: {X_test_num.shape} \n{X_test_num.columns} \n{X_test_num.isnull().sum()}')
        
        return X_train_num,X_test_num     
        
    except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")