#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:05:28 2020

@author: semvijverberg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV
import multiprocessing
max_cpu = multiprocessing.cpu_count()
import itertools
flatten = lambda l: list(itertools.chain.from_iterable(l))

def get_cv_accounting_for_years(y_train=pd.DataFrame, kfold: int=5, 
                                seed: int=1):
    '''
    Train-test split that gives priority to keep data of same year as blocks, 
    datapoints of same year are very much not i.i.d. and should be seperated. 
       

    Parameters
    ----------
    total_size : int
        total length of dataset.
    kfold : int
        prefered number of folds, however, if folds do not fit the number of 
        years, kfold is incremented untill it does. 
    seed : int, optional
        random seed. The default is 1.

    Returns
    -------
    cv : sk-learn cross-validation generator

    '''
    
    freq = y_train.groupby(y_train.index.year).sum()
    freq = (freq > freq.mean()).astype(int)
    
    all_years = np.unique(freq.index)
    while all_years.size % kfold != 0:
        kfold += 1
    
    
    cv_strat = StratifiedKFold(n_splits=kfold, shuffle=True,
                               random_state=seed)
    test_yrs = []
    for i, j in cv_strat.split(X=freq.index, y=freq.values):
        test_yrs.append(freq.index[j].values)
    
    label_test = np.zeros( y_train.size , dtype=int)
    for i, test_fold in enumerate(test_yrs):
        for j, yr in enumerate(y_train.index.year):
            if yr in list(test_fold):
                label_test[j] = i

    cv = PredefinedSplit(label_test)
    return cv

def feature_selection(X_train, y_train, model='logitCV', scoring='brier_score_loss', folds=5,
                      verbosity=0):
    
    cv = get_cv_accounting_for_years(len(y_train), folds, seed=1)
    
    if model == 'logitCV':
        kwrgs_logit = { 'class_weight':{ 0:1, 1:1},
                        'scoring':'brier_score_loss',
                        'penalty':'l2',
                        'solver':'lbfgs'}

        
        model = LogisticRegressionCV(Cs=10, fit_intercept=True,
                                     cv=cv,
                                     n_jobs=1,
                                     **kwrgs_logit)

    rfecv = RFECV(estimator=model, step=1, cv=cv,
              scoring='brier_score_loss')
    rfecv.fit(X_train, y_train)
    new_features = X_train.columns[rfecv.ranking_==1]
    new_model = rfecv.estimator_
#    rfecv.fit(X_train, y_train)
    if verbosity != 0:
        print("Optimal number of features : %d" % rfecv.n_features_)
    return new_model, new_features, rfecv

def get_masks(df_norm):
    '''
    x_fit and y_fit can be encompass more data then x_pred, therefore they
    are split from x_pred and y_pred.
    y_pred is needed in the special occasion no past (lagged) data for X avaible
    if these are not given, then model x_fit, y_fit, x_pred & y_pred are
    fitted according to TrainIsTrue at lag=0.
    '''
    TrainIsTrue = df_norm['TrainIsTrue']
    if 'x_fit' in df_norm.columns:
        x_fit_mask = np.logical_and(TrainIsTrue, df_norm['x_fit'])
    else:
        x_fit_mask = TrainIsTrue
    if 'y_fit' in df_norm.columns:
        y_dates = df_norm['y_fit'][df_norm['y_fit']].index
        TrainIsTrue_yfit = TrainIsTrue.loc[y_dates]
        y_fit_mask = np.logical_and(TrainIsTrue_yfit, df_norm['y_fit'].loc[y_dates])
        y_fit_mask = y_fit_mask
    else:
        y_fit_mask = TrainIsTrue
    if 'x_pred' in df_norm.columns:
        x_pred_mask = df_norm['x_pred']
    else:
        x_pred_mask = pd.Series(np.repeat(True, x_fit_mask.size),
                                index=x_fit_mask.index)
    if 'y_pred' in df_norm.columns:
        y_pred_dates = df_norm['y_pred'][df_norm['y_pred']].index
        y_pred_mask = df_norm['y_pred'].loc[y_pred_dates]
    else:
        y_pred_mask = None
    return x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask


def standardize_on_train(c, TrainIsTrue):
                return (c - c[TrainIsTrue.values].mean()) \
                        / c[TrainIsTrue.values].std()

