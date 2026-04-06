'''
In this file I am going to call all related functions for data cleaning and model
development
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
from logging_code import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.model_selection import train_test_split
from handle_missing_values import handling_missing_values
from var_out import vt_outliers
from filter_methods import fm
from Categorical_to_num import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs

class Churn:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            
            logger.info(f'Total Data Size: {self.df.shape}')
            logger.info(f'Before New column: \n{self.df.columns}')
            
            #Adding Sim Column as Telecom Partner
            rng = np.random.default_rng(seed=42)
            partners = ['Airtel', 'VI-!dea', 'BSNL', 'Jio']
            self.df['Telecom_Partner'] = rng.choice(partners, size=len(self.df))

            # Verify distribution is roughly equal across all 4 partners
            logger.info(f'Telecom_Partner distribution:\n{self.df["Telecom_Partner"].value_counts()}')
            
            logger.info(f'After New Column: {self.df.shape} \n{self.df.columns}')
            logger.info(f'Null Values are: \n{self.df.isnull().sum()}')
            
            for i in self.df.columns:
                logger.info(f'Before Replacement {i} : {self.df[i].isnull().sum()}')
            
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ',np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
            
            logger.info('========================================================')
            logger.info(f'Replaced String type to Numeric type')
            
            for i in self.df.columns:
                logger.info(f'After Replacement {i} : {self.df[i].isnull().sum()}')
            
            
            self.X = self.df.drop(['Churn'],axis=1)
            self.y = self.df['Churn']
            
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            
            self.y_train = self.y_train.map({'Yes':0,'No':1}).astype(int)
            self.y_test = self.y_test.map({'Yes':0,'No':1}).astype(int)
            
            logger.info(f'Train Data Size: {len(self.X_train)} : {len(self.y_train)} \n{self.X_train.shape}')
            logger.info(f'Test Data Size: {len(self.X_test)} : {len(self.y_test)} \n{self.X_test.shape}')
            
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
            
    def missing_values(self):
        try:
            logger.info(f'Before Train Null Values :{self.X_train.shape} \n{self.X_train.isnull().sum()} \n{self.X_train.columns}')
            logger.info(f'Before Test Null Values : {self.X_test.shape} \n{self.X_test.isnull().sum()} \n{self.X_test.columns}')
            
            self.X_train,self.X_test = handling_missing_values(self.X_train,self.X_test)
            
            logger.info(f'After Train Null Values : {self.X_train.shape} \n{self.X_train.isnull().sum()} \n{self.X_train.columns} ')
            logger.info(f'After Test Null Values : {self.X_test.shape} \n{self.X_test.isnull().sum()} \n{self.X_test.columns}')
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
            
            
    def data_separation(self):
        try:
            logger.info(f'Splitting Data from X_train and X_test')
            
            #creating numberic columns from X_train and X_test
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')
            
            #creating categorical columns from X_train and X_test
            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')
            
            logger.info(f'X_train_num_cols: {self.X_train_num_cols.shape} \n{self.X_train_num_cols.columns}')
            logger.info(f'X_test_num_cols: {self.X_test_num_cols.shape} \n{self.X_test_num_cols.columns}')
            logger.info('================================================================================')
            logger.info(f'X_train_cat_cols: {self.X_train_cat_cols.shape} \n{self.X_train_cat_cols.columns}')
            logger.info(f'X_test_cat_cols: {self.X_test_cat_cols.shape} \n{self.X_test_cat_cols.columns}')
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
            
    def variable_transformation(self):
        try:
            logger.info(f'X_train number values Before Vt and Outliers: {self.X_train_num_cols.shape} \n{self.X_train_num_cols.columns}')
            logger.info(f'X_test number values Before Vt and Outliers : {self.X_test_num_cols.shape} \n{self.X_test_num_cols.columns}')
            
            
            self.X_train_num_cols,self.X_test_num_cols = vt_outliers(self.X_train_num_cols,self.X_test_num_cols)
            
            logger.info('================================================================================')
            logger.info(f'X_train number values After Vt and Outliers: {self.X_train_num_cols.shape} \n{self.X_train_num_cols.columns}')
            logger.info(f'X_test number values After Vt and Outliers : {self.X_test_num_cols.shape} \n{self.X_test_num_cols.columns}') 
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
            
    
    def feature_selection(self):
        try:
            self.X_train_num_cols,self.X_test_num_cols = fm(self.X_train_num_cols,self.X_test_num_cols,self.y_train,self.y_test)
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
            
    
    def cat_to_num(self):
        try:
            self.X_train_cat_cols = self.X_train_cat_cols.drop(['customerID'],axis=1)
            self.X_test_cat_cols = self.X_test_cat_cols.drop(['customerID'],axis=1)
            
            self.X_train_cat_cols,self.X_test_cat_cols = c_t_n(self.X_train_cat_cols,self.X_test_cat_cols)

            #combine data
            self.X_train_cat_cols.reset_index(drop=True,inplace=True)
            self.X_train_num_cols.reset_index(drop=True,inplace=True)
            self.X_test_cat_cols.reset_index(drop=True,inplace=True)
            self.X_test_num_cols.reset_index(drop=True,inplace=True)
            
            self.training_data = pd.concat([self.X_train_num_cols,self.X_train_cat_cols],axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols,self.X_test_cat_cols],axis=1)

            logger.info('====================================================')
            logger.info(f'Final Training Data : {self.training_data.shape} \n{self.training_data.columns}')
            logger.info(f'Checking Null Values in Training Data: \n{self.training_data.isnull().sum()}')

            logger.info(f'Final Testing Data : {self.testing_data.shape} \n{self.testing_data.columns}')
            logger.info(f'Checking Null Values in Testing Data: \n{self.testing_data.isnull().sum()}')
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
            
    
    def data_balancing(self):
        try:
            logger.info(f'Number of Rows for Good Columns {1}: {sum(self.y_train == 1)}')
            logger.info(f'Number of Rows for Bad Columns {0}: {sum(self.y_train == 0)}')
            logger.info(f'Training data size: {self.training_data.shape}')

            sm = SMOTE(random_state=42)
            self.training_data_bal,self.y_train_bal = sm.fit_resample(self.training_data,self.y_train)

            logger.info(f"Number of Rows for Good Customer {1} : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad Customer {0} : {sum(self.y_train_bal == 0)}")
            logger.info(f"Training data size : {self.training_data_bal.shape}")

            fs(self.training_data_bal,self.y_train_bal,self.testing_data,self.y_test)
            
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")

if __name__ == '__main__':
    obj = Churn('Telco-Customer-Churn.csv')
    obj.missing_values()
    obj.data_separation()
    obj.variable_transformation()
    obj.feature_selection()
    obj.cat_to_num()
    obj.data_balancing()
    