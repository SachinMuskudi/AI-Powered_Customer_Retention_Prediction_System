import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging
from logging_code import setup_logging
logger = setup_logging('feature_scaling')
import sys
import os
from sklearn.preprocessing import Normalizer #Z_Score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from All_Models import common
import pickle

def fs(X_train,y_train,X_test,y_test):
    try:
        logger.info(f"Training data independent size : {X_train.shape}")
        logger.info(f"Training data dependent size : {y_train.shape}")
        logger.info(f"Testing data independent size : {X_test.shape}")
        logger.info(f"Testing data dependent size : {y_test.shape}")
        logger.info(f"before : {X_train.head(1)}")

        Norm = Normalizer()
        Norm.fit(X_train)
        X_train_Norm = Norm.transform(X_train)
        X_test_Norm = Norm.transform(X_test)

        with open('Normalizer.pkl','wb') as f:
            pickle.dump(Norm,f)

        logger.info(f'{X_train_Norm}')
        common(X_train_Norm,y_train,X_test_Norm,y_test)
        reg = LogisticRegression(C= 1, class_weight= 'balanced', max_iter= 100, penalty= 'l1', solver= 'liblinear')
        reg.fit(X_train_Norm,y_train) #training completed
        logger.info(f"Test Accuracy : {accuracy_score(y_test, reg.predict(X_test_Norm))}")
        logger.info(f"Test Confusion Matrix : \n{confusion_matrix(y_test, reg.predict(X_test_Norm))}")
        logger.info(f"Classification report : \n{classification_report(y_test, reg.predict(X_test_Norm))}")

        with open('Model.pkl','wb') as f:
            pickle.dump(reg,f)


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")