#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:03:31 2019

@author: semvijverberg
"""

import pandas as pd
import numpy as np
# import statsmodels.formula.api as sm
# from statsmodels.api import add_constant
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import make_scorer

from stat_models import get_cv_accounting_for_years, get_masks
from stat_models import feature_selection

def ridgeCV(y_ts, df_norm, keys=None, kwrgs_model=None):

    
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries
    '''
    #%%
    if keys is None:
            no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
            keys = df_norm.columns
            keys = [k for k in keys if k not in no_data_col]
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings("ignore", category=FutureWarning)

    if kwrgs_model == None:
        # use Bram settings
        kwrgs_model = { 'fit_intercept':True,
                        'alphas':(.01, .1, 1.0, 10.0)}


    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_model.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_model.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    if 'feat_sel' in kwrgs:
        feat_sel = kwrgs.pop('feat_sel')
    else:
        feat_sel = None
        
    # Get training years
    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = get_masks(df_norm)

    X = df_norm[keys]
    # X = add_constant(X)
    X_train = X[x_fit_mask.values]
    X_pred  = X[x_pred_mask.values]

    RV_bin_fit = y_ts['ts']
    # y_ts dates no longer align with x_fit  y_fit masks
    y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values
    y_train = RV_bin_fit[y_fit_mask].squeeze()

    # if y_pred_mask is not None:
    #     y_dates = RV_bin_fit[y_pred_mask.values].index
    # else:
    y_dates = RV_bin_fit.index
    
    X = X_train

    # # Create stratified random shuffle which keeps together years as blocks.
    # kwrgs_cv = ['kfold', 'seed']
    # kwrgs_cv = {k:i for k, i in kwrgs.items() if k in kwrgs_cv}
    # [kwrgs.pop(k) for k in kwrgs_cv.keys()]
   
    # cv = get_cv_accounting_for_years(y_train, **kwrgs_cv)
    cv = None
    model = RidgeCV(cv=cv,
                    store_cv_values=True,
                    **kwrgs)
                    
    if feat_sel is not None:
        if feat_sel['model'] is None:
            feat_sel['model'] = model
        model, new_features, rfecv = feature_selection(X_train, y_train.values, **feat_sel)
        X_pred = X_pred[new_features]
    else:
        model.fit(X_train, y_train)


    y_pred = model.predict(X_pred)

    prediction = pd.DataFrame(y_pred, index=y_dates, columns=[0])
    model.X_pred = X_pred
    #%%
    return prediction, model