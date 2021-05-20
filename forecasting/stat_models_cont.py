#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:03:31 2019

@author: semvijverberg
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV

import func_models as utils


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
    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = utils.get_masks(df_norm)

    X = df_norm[keys]
    X = X.dropna(axis='columns') # drop only nan columns
    # X = add_constant(X)
    X_train = X[x_fit_mask.values]
    X_pred  = X[x_pred_mask.values]

    RV_fit = y_ts['ts'].loc[y_fit_mask.index] # y_fit may be shortened
    # because X_test was used to predict y_train due to lag, hence train-test
    # leakage.

    # y_ts dates may no longer align with x_fit  y_fit masks
    y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values
    y_train = RV_fit[y_fit_mask].squeeze()

    # if y_pred_mask is not None:
    #     y_dates = RV_fit[y_pred_mask.values].index
    # else:
    # y_dates = RV_fit.index

    X = X_train

    # # Create stratified random shuffle which keeps together years as blocks.
    kwrgs_cv = ['kfold', 'seed']
    kwrgs_cv = {k:i for k, i in kwrgs.items() if k in kwrgs_cv}
    [kwrgs.pop(k) for k in kwrgs_cv.keys()]
    if len(kwrgs_cv) >= 1:
        cv = utils.get_cv_accounting_for_years(y_train, **kwrgs_cv)
        kwrgs['store_cv_values'] = False
    else:
        cv = None
        kwrgs['store_cv_values'] = True
    model = RidgeCV(cv=cv,
                    **kwrgs)

    if feat_sel is not None:
        if feat_sel['model'] is None:
            feat_sel['model'] = model
        model, new_features, rfecv = utils.feature_selection(X_train, y_train.values, **feat_sel)
        X_pred = X_pred[new_features]
    else:
        model.fit(X_train, y_train)


    y_pred = model.predict(X_pred)

    prediction = pd.DataFrame(y_pred, index=y_pred_mask.index, columns=[0])
    model.X_pred = X_pred
    model.name = 'Ridge Regression'
    #%%
    return prediction, model


class ScikitModel:

    def __init__(self, scikitmodel=None, verbosity=1):
        if scikitmodel is None:
            scikitmodel = RidgeCV
        self.scikitmodel = scikitmodel
        self.verbosity = verbosity

    def fit_wrapper(self, y_ts, df_norm, keys=None, kwrgs_model=None):
        '''
        X contains all precursor data, incl train and test
        X_train, y_train are split up by TrainIsTrue
        Preciction is made for whole timeseries
        '''
        #%%

        scikitmodel = self.scikitmodel

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
        if 'scoringCV' in kwrgs.keys():
            scoring = kwrgs.pop('scoringCV')
        else:
            scoring = None
        # Get training years
        x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = utils.get_masks(df_norm)

        X = df_norm[keys]
        X = X.dropna(axis='columns') # drop only nan columns
        # X = add_constant(X)
        X_train = X[x_fit_mask.values]
        X_pred  = X[x_pred_mask.values]

        RV_fit = y_ts['ts'].loc[y_fit_mask.index] # y_fit may be shortened
        # because X_test was used to predict y_train due to lag, hence train-test
        # leakage.

        # y_ts dates may no longer align with x_fit  y_fit masks
        y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values
        y_train = RV_fit[y_fit_mask == True].squeeze()

        # if y_pred_mask is not None:
        #     y_dates = RV_fit[y_pred_mask.values].index
        # else:
        # y_dates = RV_fit.index

        X = X_train

        # # Create stratified random shuffle which keeps together years as blocks.
        kwrgs_cv = ['kfold', 'seed']
        kwrgs_cv = {k:i for k, i in kwrgs.items() if k in kwrgs_cv}
        [kwrgs.pop(k) for k in kwrgs_cv.keys()]
        if len(kwrgs_cv) >= 1:
            cv = utils.get_cv_accounting_for_years(y_train, **kwrgs_cv)
        else:
            cv = None
        try:
            model = scikitmodel(cv=cv, **kwrgs)
        except:
            model = scikitmodel(**kwrgs)

        if len(kwrgs_gridsearch) != 0:
            # get cross-validation splitter
            # if 'kfold' in kwrgs.keys():
            #     kfold = kwrgs.pop('kfold')
            # else:
            #     kfold = 5
            # cv = utils.get_cv_accounting_for_years(y_train, kfold, seed=1)

            model = GridSearchCV(model,
                      param_grid=kwrgs_gridsearch,
                      scoring=scoring, cv=cv, refit=True,
                      return_train_score=True, verbose=self.verbosity,
                      n_jobs=3)
            model.fit(X_train, y_train.values.ravel())
            model.best_estimator_.X_pred = X_pred # add X_pred to model
            # if self.verbosity == 1:
            #     results = model.cv_results_
            #     scores = results['mean_test_score']
            #     greaterisbetter = model.scorer_._sign
            #     improv = int(100* greaterisbetter*(max(scores)- min(scores)) / max(scores))
            #     print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
            #           "best params {}".format(
            #             improv, model.best_score_, model.best_params_))
        else:
            model.fit(X_train, y_train.values.ravel())
            model.X_pred = X_pred # add X_pred to model

        if np.unique(y_train).size < 5:
            y_pred = model.predict_proba(X_pred)[:,1] # prob. event prediction
        else:
            y_pred = model.predict(X_pred)

        prediction = pd.DataFrame(y_pred, index=y_pred_mask.index, columns=[0])
        #%%
        return prediction, model
