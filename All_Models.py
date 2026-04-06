import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
from logging_code import setup_logging
logger = setup_logging("all_models")
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV,cross_validate

def knn(X_train,y_train,X_test,y_test):
  global knn_reg
  knn_reg = KNeighborsClassifier(n_neighbors=5)
  knn_reg.fit(X_train,y_train)
  predictions = knn_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global knn_predictions
  knn_predictions = knn_reg.predict(X_test)

def nb(X_train,y_train,X_test,y_test):
  global nb_reg
  nb_reg = GaussianNB()
  nb_reg.fit(X_train,y_train)
  predictions = nb_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global nb_predictions
  nb_predictions = nb_reg.predict(X_test)

def lr(X_train,y_train,X_test,y_test):
  global lr_reg
  lr_reg = LogisticRegression()
  lr_reg.fit(X_train,y_train)
  predictions = lr_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global lr_predictions
  lr_predictions = lr_reg.predict(X_test)


def dt(X_train,y_train,X_test,y_test):
  global dt_reg
  dt_reg = DecisionTreeClassifier(criterion='entropy')
  dt_reg.fit(X_train,y_train)
  predictions = dt_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global dt_predictions
  dt_predictions = dt_reg.predict(X_test)

def rf(X_train,y_train,X_test,y_test):
  global rf_reg
  rf_reg = RandomForestClassifier(criterion='entropy',n_estimators=5)
  rf_reg.fit(X_train,y_train)
  predictions = rf_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global rf_predictions
  rf_predictions = rf_reg.predict(X_test)

def adab(X_train,y_train,X_test,y_test):
  global ada_reg
  lr = LogisticRegression()
  ada_reg = AdaBoostClassifier(estimator=lr , n_estimators=5)
  ada_reg.fit(X_train,y_train)
  predictions = ada_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global ada_predictions
  ada_predictions = ada_reg.predict(X_test)

def gb(X_train,y_train,X_test,y_test):
  global gr_reg
  gr_reg = GradientBoostingClassifier(n_estimators=5)
  gr_reg.fit(X_train,y_train)
  predictions = gr_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global gb_predictions
  gb_predictions = gr_reg.predict(X_test)

def xgb(X_train,y_train,X_test,y_test):
  global xgb_reg
  xgb_reg = XGBClassifier(n_estimators = 5)
  xgb_reg.fit(X_train,y_train)
  predictions = xgb_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global xgb_predictions
  xgb_predictions = xgb_reg.predict(X_test)

def svm(X_train,y_train,X_test,y_test):
  global sv_reg
  sv_reg = SVC(kernel = 'rbf')
  sv_reg.fit(X_train,y_train)
  predictions = sv_reg.predict(X_test)
  logger.info(f'\n{confusion_matrix(y_test,predictions)}')
  logger.info(f'\n{accuracy_score(y_test,predictions)}')
  logger.info(f'\n{classification_report(y_test,predictions)}')
  global svm_predictions
  svm_predictions = sv_reg.predict(X_test)

def auc_roc_tech(X_train,y_train,X_test,y_test):
    knn_fpr, knn_tpr, knn_th = roc_curve(y_test, knn_predictions)
    nb_frp, nb_tpr, nb_th = roc_curve(y_test, nb_predictions)
    lr_fpr, lr_tpr, lr_th = roc_curve(y_test, lr_predictions)
    dt_fpr, dt_tpr, dt_th = roc_curve(y_test, dt_predictions)

    rf_fpr, rf_tpr, rf_th = roc_curve(y_test, rf_predictions)
    ada_fpr, ada_tpr, ada_th = roc_curve(y_test, ada_predictions)
    gb_fpr, gb_tpr, gb_th = roc_curve(y_test, gb_predictions)
    xgb_fpr, xgb_tpr, xgb_th = roc_curve(y_test, xgb_predictions)

    # svm_fpr, svm_tpr, svm_th = roc_curve(y_test, svm_predictions)

    plt.figure(figsize=(5, 3))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ALL Models AUC Curve")

    plt.plot(knn_fpr, knn_tpr, label='KNN')
    plt.plot(lr_fpr, lr_tpr, label='LR')
    plt.plot(nb_frp, nb_tpr, label='NB')
    plt.plot(dt_fpr, dt_tpr, label='dt')
    plt.plot(rf_fpr, rf_tpr, label='RF')
    plt.plot(ada_fpr, ada_tpr, label='ada')
    plt.plot(gb_fpr, gb_tpr, label='GB')
    plt.plot(xgb_fpr, xgb_tpr, label='XGB')
   # plt.plot(svm_fpr, svm_tpr, label='SVM')

    plt.legend(loc=0)
    plt.show()


def HyperParameter(X_train,y_train,X_test,y_test):
  try:
      parameters_list = [
                  # ----- L2 -----
                  {
                      'penalty': ['l2'],
                      'solver': ['lbfgs', 'liblinear'],
                      'C': [0.1, 1, 10],
                      'max_iter': [100, 200, 500],
                      'class_weight': [None, 'balanced'],
                  },

                  # ----- L1 -----
                  {
                      'penalty': ['l1'],
                      'solver': ['liblinear', 'saga'],
                      'C': [0.01, 0.1, 1],
                      'max_iter': [100, 200, 500],
                      'class_weight': [None, 'balanced'],
                  }
              ]

      grid_reg = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters_list, scoring='f1', cv=5)
      grid_result = grid_reg.fit(X_train, y_train)
      logger.info(f'The grid_result: \n{grid_result}')
      logger.info(f'The grid best parameter are: \n{grid_result.best_params_}')
      logger.info(f'The grid Accuracy Score: \n{grid_result.best_score_}')
  except Exception as e:
      er_type, er_msg, er_line = sys.exc_info()
      logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")          
      
def common(X_train,y_train,X_test,y_test):
    try:
        logger.info('-----------knn----------------')
        knn(X_train, y_train, X_test, y_test)
        logger.info('-----------Naive Bayes----------------')
        nb(X_train, y_train, X_test, y_test)
        logger.info('-----------LR----------------')
        lr(X_train, y_train, X_test, y_test)
        logger.info('-----------dt----------------')
        dt(X_train, y_train, X_test, y_test)
        logger.info('-----------rf----------------')
        rf(X_train, y_train, X_test, y_test)
        logger.info('-----------adaboost----------------')
        adab(X_train, y_train, X_test, y_test)
        logger.info('-----------gradient boosting----------------')
        gb(X_train, y_train, X_test, y_test)
        logger.info('-----------xtreme GB----------------')
        xgb(X_train, y_train, X_test, y_test)
        # logger.info('-----------SVM----------------')
        # svm(X_train, y_train, X_test, y_test)
        logger.info('------------AUC and ROC-----------------')
        auc_roc_tech(X_train,y_train,X_test,y_test)
        logger.info(f'AUC - ROC Image Successfully Created')
        # logger.info('----------HyperParameter Tuning--------------')
        HyperParameter(X_train,y_train,X_test,y_test)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"The type of error {er_type} in line no : {er_line.tb_lineno} due to : {er_msg}")
