#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:03:31 2019

@author: semvijverberg
"""

import numpy as np
import pandas as pd
import sklearn.model_selection as msel
import sklearn.linear_model as lm
from typing import Union

from . import func_models as utils


def fit_df_data_sklearn(df_data: pd.DataFrame=None,
                        keys: Union[list, np.ndarray]=None,
                        target: Union[str,pd.DataFrame]=None,
                        tau_min: int=1,
                        tau_max: int=1,
                        match_lag_region_to_lag_fc=False,
                        transformer=None,
                        fcmodel=None,
                        kwrgs_model: dict={'scoring':'neg_mean_squared_error'}):
    '''
    Perform cross-validated Ridge regression to predict the target.

    Parameters
    ----------
    keys : Union[list, np.ndarray]
        list of nparray of predictors that you want to merge into one.
    target : str, optional
        target timeseries to predict. The default target is the RV.
    tau_max : int, optional
        prediction is repeated at lags 0 to tau_max. It might be that the
        signal stronger at a certain lag. Relationship is established for
        all lags in range(tau_min, tau_max+1) and the strongest is
        relationship is kept.
    fcmodel : function of stat_models, None
        Give function that satisfies stat_models format of I/O, default
        Ridge regression
    kwrgs_model : dict, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    predict (DataFrame), weights (DataFrame), models_lags (dict).

    '''
    # df_data=None;keys=None;target=None;tau_min=1;tau_max=1;transformer=None
    # kwrgs_model={'scoring':'neg_mean_squared_error'};match_lag_region_to_lag_fc=False

    lags = range(tau_min, tau_max+1)
    splits = df_data.index.levels[0]

    if 'TrainIsTrue' not in df_data.columns:
        TrainIsTrue = pd.DataFrame(np.ones( (df_data.index.size), dtype=bool),
                                   index=df_data.index, columns=['TrainIsTrue'])
        df_data = df_data.merge(TrainIsTrue, left_index=True, right_index=True)

    if 'RV_mask' not in df_data.columns:
        RV_mask = pd.DataFrame(np.ones( (df_data.index.size), dtype=bool),
                               index=df_data.index, columns=['RV_mask'])
        df_data = df_data.merge(RV_mask, left_index=True, right_index=True)

    RV_mask = df_data.loc[0]['RV_mask'] # not changing
    if target is None: # not changing
        target_ts = df_data.loc[0].iloc[:,[0]][RV_mask]

    if keys is None:
        keys = [k for k in df_data.columns if k not in ['TrainIsTrue', 'RV_mask']]
        # remove col with same name as target_ts
        keys = [k for k in keys if k != target_ts.columns[0]]


    models_lags = dict()
    for il, lag in enumerate(lags):
        preds = np.zeros( (splits.size), dtype=object)
        wghts = np.zeros( (splits.size) , dtype=object)
        models_splits_lags = dict()
        for isp, s in enumerate(splits):
            fit_masks = df_data.loc[s][['RV_mask', 'TrainIsTrue']]
            TrainIsTrue = df_data.loc[s]['TrainIsTrue']

            df_s = df_data.loc[s]
            if type(keys) is dict:
                _ks = keys[s]
                _ks = [k for k in _ks if k in df_s.columns] # keys split
            else:
                _ks = [k for k in keys if k in df_s.columns] # keys split

            if match_lag_region_to_lag_fc:
                ks = [k for k in _ks if k.split('..')[0] == str(lag)]
                l = lag ; valid = len(ks) !=0 and ~df_s[ks].isna().values.all()
                while valid == False:
                    ks = [k for k in _ks if k.split('..')[0] == str(l)]
                    if len(ks) !=0 and ~df_s[ks].isna().values.all():
                        valid = True
                    else:
                        l -= 1
                    print(f"\rNot found lag {lag}, using lag {l}", end="")
                    assert l > 0, 'ts @ lag not found or nans'
            else:
                ks = _ks

            if transformer is not None and transformer != False:
                df_trans = df_s[ks].apply(transformer,
                                        args=[TrainIsTrue])
            elif transformer == False:
                df_trans = df_s[ks] # no transformation
            else: # transform to standard normal
                df_trans = df_s[ks].apply(utils.standardize_on_train_and_RV,
                                          args=[fit_masks, lag])
                                        # result_type='broadcast')

            if type(target) is str:
                target_ts = df_data.loc[s][[target]][RV_mask]
            elif type(target) is pd.DataFrame:
                target_ts = target.copy()
                if hasattr(target.index, 'levels'):
                    target_ts = target.loc[s]

            shift_lag = utils.apply_shift_lag
            df_norm = df_trans.merge(shift_lag(fit_masks.copy(), lag),
                                     left_index=True,
                                     right_index=True)


            if fcmodel is None:
                fcmodel = ScikitModel()
            pred, model = fcmodel.fit_wrapper({'ts':target_ts},
                                              df_norm, ks, kwrgs_model)

            # if len(lags) > 1:
            models_splits_lags[f'split_{s}'] = model


            if il == 0:#  and isp == 0:
            # add truth
                prediction = target_ts.copy()
                prediction = prediction.merge(pred.rename(columns={0:lag}),
                                              left_index=True,
                                              right_index=True)
            else:
                prediction = pred.rename(columns={0:lag})

            coeff = utils.SciKitModel_coeff(model, lag)

            preds[isp] = prediction
            wghts[isp] = coeff
        if il == 0:
            predict = pd.concat(list(preds), keys=splits)
            weights = pd.concat(list(wghts), keys=splits)
        else:
            predict = predict.merge(pd.concat(list(preds), keys=splits),
                          left_index=True, right_index=True)
            weights = weights.merge(pd.concat(list(wghts), keys=splits),
                          left_index=True, right_index=True)
        models_lags[f'lag_{lag}'] = models_splits_lags

    return predict, weights, models_lags

class ScikitModel:

    def __init__(self, scikitmodel=None, verbosity=1):
        if scikitmodel is None:
            scikitmodel = lm.RidgeCV
        self.scikitmodel = scikitmodel
        self.verbosity = verbosity

    def fit_wrapper(self, y_ts, df_norm, keys=None, kwrgs_model={}):
        '''
        X contains all precursor data, incl train and test
        X_train, y_train are split up by TrainIsTrue
        Preciction is made for whole timeseries
        if 'search_method' is given in kwrgs_model, it will specifify the
        model selection method from scikit-learn. Default is GridSearchCV.
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

        # find parameters for gridsearch optimization
        kwrgs_gridsearch = {k:i for k, i in kwrgs_model.items() if type(i) == list}
        # only the constant parameters are kept
        kwrgs = kwrgs_model.copy()
        [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]

        # extract some specified parameters if given
        if 'search_method' in kwrgs.keys():
            search_method = kwrgs.pop('search_method')
        else:
            search_method = 'GridSearchCV'
        if 'scoringCV' in kwrgs.keys():
            scoring = kwrgs.pop('scoringCV')
        else:
            scoring = None
        if 'n_jobs' in kwrgs.keys():
            n_jobs = kwrgs.pop('n_jobs')
        else:
            n_jobs = None

        # Get training years
        x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = utils.get_masks(df_norm)

        X = df_norm[keys]
        X = X.dropna(axis='columns') # drop only nan columns
        X_train = X[x_fit_mask.values]
        X_pred  = X[x_pred_mask.values]

        RV_fit = y_ts['ts'].loc[y_fit_mask.index] # y_fit may be shortened
        # because X_test was used to predict y_train due to lag, hence train-test
        # leakage.

        # y_ts dates may no longer align with x_fit  y_fit masks
        y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values==1
        y_train = RV_fit[y_fit_mask].squeeze()
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
            gridsearch = msel.__dict__[search_method]
            model = gridsearch(model,
                      param_grid=kwrgs_gridsearch,
                      scoring=scoring, cv=cv, refit=True,
                      return_train_score=True, verbose=self.verbosity,
                      n_jobs=n_jobs)
            model.fit(X_train, y_train.values.ravel())
            model.best_estimator_.X_pred = X_pred # add X_pred to model
            model.best_estimator_.df_norm = df_norm # add df_norm to model for easy reproduction
            model.best_estimator_.target = RV_fit
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
            model.df_norm = df_norm # add df_norm to model for easy reproduction
            model.target = RV_fit

        if np.unique(y_train).size < 5:
            y_pred = model.predict_proba(X_pred)[:,1] # prob. event prediction
        else:
            y_pred = model.predict(X_pred)

        # changed on 27-01-2022
        # prediction = pd.DataFrame(y_pred, index=x_pred_mask[x_pred_mask].index,
                                  # columns=[0])
        prediction = pd.DataFrame(y_pred, index=RV_fit.index,
                                  columns=[0])
        #%%
        return prediction, model



# def ridgeCV(y_ts, df_norm, keys=None, kwrgs_model=None):
#     '''
#     X contains all precursor data, incl train and test
#     X_train, y_train are split up by TrainIsTrue
#     Preciction is made for whole timeseries
#     '''
#     #%%
#     if keys is None:
#             no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
#             keys = df_norm.columns
#             keys = [k for k in keys if k not in no_data_col]
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     # warnings.filterwarnings("ignore", category=FutureWarning)

#     if kwrgs_model == None:
#         # use Bram settings
#         kwrgs_model = { 'fit_intercept':True,
#                         'alphas':(.01, .1, 1.0, 10.0)}


#     # find parameters for gridsearch optimization
#     kwrgs_gridsearch = {k:i for k, i in kwrgs_model.items() if type(i) == list}
#     # only the constant parameters are kept
#     kwrgs = kwrgs_model.copy()
#     [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
#     if 'feat_sel' in kwrgs:
#         feat_sel = kwrgs.pop('feat_sel')
#     else:
#         feat_sel = None

#     # Get training years
#     x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = utils.get_masks(df_norm)

#     X = df_norm[keys]
#     X = X.dropna(axis='columns') # drop only nan columns
#     # X = add_constant(X)
#     X_train = X[x_fit_mask.values]
#     X_pred  = X[x_pred_mask.values]

#     RV_fit = y_ts['ts'].loc[y_fit_mask.index] # y_fit may be shortened
#     # because X_test was used to predict y_train due to lag, hence train-test
#     # leakage.

#     # y_ts dates may no longer align with x_fit  y_fit masks
#     y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values==1
#     y_train = RV_fit[y_fit_mask].squeeze()

#     # if y_pred_mask is not None:
#     #     y_dates = RV_fit[y_pred_mask.values].index
#     # else:
#     # y_dates = RV_fit.index

#     X = X_train

#     # # Create stratified random shuffle which keeps together years as blocks.
#     kwrgs_cv = ['kfold', 'seed']
#     kwrgs_cv = {k:i for k, i in kwrgs.items() if k in kwrgs_cv}
#     [kwrgs.pop(k) for k in kwrgs_cv.keys()]
#     if len(kwrgs_cv) >= 1:
#         cv = utils.get_cv_accounting_for_years(y_train, **kwrgs_cv)
#         kwrgs['store_cv_values'] = False
#     else:
#         cv = None
#         kwrgs['store_cv_values'] = True
#     model = RidgeCV(cv=cv,
#                     **kwrgs)

#     if feat_sel is not None:
#         if feat_sel['model'] is None:
#             feat_sel['model'] = model
#         model, new_features, rfecv = utils.feature_selection(X_train, y_train.values, **feat_sel)
#         X_pred = X_pred[new_features]
#     else:
#         model.fit(X_train, y_train)


#     y_pred = model.predict(X_pred)

#     prediction = pd.DataFrame(y_pred, index=y_pred_mask.index, columns=[0])
#     model.X_pred = X_pred
#     model.name = 'Ridge Regression'
#     #%%
#     return prediction, model
