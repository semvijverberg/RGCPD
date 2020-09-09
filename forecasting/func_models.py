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
import functions_pp

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
                  scoring=scoring)
    rfecv.fit(X_train, y_train)
    new_features = X_train.columns[rfecv.ranking_==1]
    new_model = rfecv.estimator_
#    rfecv.fit(X_train, y_train)
    if verbosity != 0:
        print("Optimal number of features : %d" % rfecv.n_features_)
    return new_model, new_features, rfecv

def _check_y_fitmask(fit_masks, lag_i, base_lag):
    ''' If lag_i is uneven, taking the mean over the RV period may result in
    a shorter y_fit (RV_mask) then the original RV_mask (where the time mean
    bins were done on its own time axis. Hence y_fit is redefined by adding
    lag_i+base_lag to x_fit mask.

    Note: y_fit_mask and y_pred_mask the same now
    '''
    fit_masks_n = fit_masks.copy()
    y_fit = fit_masks['y_fit'] ; x_fit = fit_masks['x_fit']
    y_dates_RV = x_fit[x_fit].index + pd.Timedelta(lag_i+base_lag, 'd')
    y_dates_pr = y_fit[y_fit].index
    mismatch = (functions_pp.get_oneyr(y_dates_pr)[0]- \
                functions_pp.get_oneyr(y_dates_RV)[0] ).days
    y_fit_corr = y_dates_RV + pd.Timedelta(mismatch, 'd')
    y_fit_mask = [True if d in y_fit_corr else False for d in x_fit.index]
    fit_masks_n.loc[:,'y_fit'] = np.array(y_fit_mask)

    y_pred = fit_masks['y_pred'] ; x_pred = fit_masks['x_pred']
    y_dates_RV = x_pred[x_pred].index + pd.Timedelta(lag_i+base_lag, 'd')
    y_dates_pr = y_pred[y_pred].index
    mismatch = (functions_pp.get_oneyr(y_dates_pr)[0]- \
                functions_pp.get_oneyr(y_dates_RV)[0] ).days
    y_pred_corr = y_dates_RV + pd.Timedelta(mismatch, 'd')
    y_pred_mask = [True if d in y_pred_corr else False for d in x_pred.index]
    fit_masks_n.loc[:,'y_pred'] = np.array(y_pred_mask)
    size_y_fit = fit_masks_n['y_fit'][fit_masks_n['y_fit']].dropna().size
    assert  size_y_fit == y_dates_RV.size, ('y_fit mask will not match RV '
                ' dates length')
    return fit_masks_n

def apply_shift_lag(fit_masks, lag_i):
    '''
    only shifting the boolean masks, Traintest split info is contained
    in the TrainIsTrue mask.
    '''
    if 'fit_model_mask' not in fit_masks.columns:
        fit_masks['fit_model_mask'] = fit_masks['RV_mask'].copy()

    RV_mask = fit_masks['RV_mask'].copy()
    x_pred = RV_mask.shift(periods=-int(lag_i))
    x_pred[~x_pred.notna()] = False

    # y_fit, accounting for if test data of X is used to predict train data y
    # due to lag

    y_fit = fit_masks['fit_model_mask'].copy()
    if lag_i > 0:
        # y_date cannot be predicted elsewise mixing
        # test test test train train train, cannot use dates test -> train
        # dates left boundary
        left_boundary = fit_masks['TrainIsTrue'].shift(periods=-lag_i,
                                            fill_value=fit_masks['TrainIsTrue'][-1])
        # train train test test test train train, cannot use dates train -> test
        # dates right boundary
        right_boundary = fit_masks['TrainIsTrue'].shift(periods=lag_i,
                                            fill_value=fit_masks['TrainIsTrue'][-1])
        diff_left = left_boundary.astype(int) - fit_masks['TrainIsTrue'].astype(int)
        diff_right = right_boundary.astype(int) - fit_masks['TrainIsTrue'].astype(int)
        diff_traintest = np.logical_or(diff_left, diff_right)
        dates_boundary_due_to_lag = diff_traintest[diff_traintest.astype(bool)].index
        y_fit.loc[dates_boundary_due_to_lag] = False


    x_fit = y_fit.shift(periods=-int(lag_i))
    n_nans = x_fit[~x_fit.notna()].size
    # set last x_fit date to False if x_fit caused nan
    if n_nans > 0:
        # take into account that last x_fit_train should be False to have
        # equal length y_train & x_fit
        x_fit[~x_fit.notna()] = False

    # cannot predict first values of y because there is no X
    dates_no_X_info = RV_mask.index[:int(lag_i)]
    y_pred = RV_mask.copy()
    y_pred.loc[dates_no_X_info] = False
    y_fit.loc[dates_no_X_info] = False


    fit_masks['x_fit'] = x_fit
    fit_masks['y_fit'] = y_fit
    fit_masks['x_pred'] = x_pred
    fit_masks['y_pred'] = y_pred
    fit_masks = fit_masks.drop(['RV_mask'], axis=1)
    fit_masks = fit_masks.drop(['fit_model_mask'], axis=1)
    return fit_masks.astype(bool)

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

def robustscaling_on_train(c, TrainIsTrue):
    return (c - c[TrainIsTrue.values].quantile(q=.25)) \
            / (c[TrainIsTrue.values].quantile(q=.75) - c[TrainIsTrue.values].quantile(q=.25))

def minmaxscaler_on_train(c, TrainIsTrue):
    return (c - c[TrainIsTrue.values].min()) \
            / (c[TrainIsTrue.values].max() - c[TrainIsTrue.values].min())