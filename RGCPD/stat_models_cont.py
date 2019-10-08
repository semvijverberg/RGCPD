#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:03:31 2019

@author: semvijverberg
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import make_scorer


def GBR(RV, df_norm, keys=None, kwrgs_GBR=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
    
    if keys is None:
        no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
        keys = df_norm.columns
        keys = [k for k in keys if k not in no_data_col]
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':3,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_ts = RV.RV_ts_fit
    # Get training years
    TrainIsTrue = df_norm['TrainIsTrue'] 
    # Get mask to make only prediction for RV_mask dates
    pred_mask   = df_norm['RV_mask']
  
    X_train = X[TrainIsTrue]
    y_train = RV_ts[TrainIsTrue.values] 
    
    # add sample weight mannually
#    y_train[y_train > y_train.mean()] = 10 * y_train[y_train > y_train.mean()]
    
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    regressor = GradientBoostingRegressor(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
#        scoring   = 'r2'
        scoring   = 'neg_mean_squared_error'
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=5, refit=scoring, 
                  return_train_score=False)
        regressor.fit(X_train, y_train.values.ravel())
        results = regressor.cv_results_
        scores = results['mean_test_score'] 
        improv = int(100* (min(scores)-max(scores)) / max(scores))
        print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
              "best params {}".format(
                improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    

    prediction = pd.DataFrame(regressor.predict(X[pred_mask]),
                              index=X[pred_mask].index, columns=[0])

    #%%
    return prediction, regressor