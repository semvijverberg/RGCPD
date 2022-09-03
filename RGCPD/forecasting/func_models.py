#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:05:28 2020

@author: semvijverberg
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit, StratifiedKFold

import itertools

flatten = lambda l: list(itertools.chain.from_iterable(l))
import properscoring as ps
from sklearn import metrics, preprocessing

from .. import functions_pp


def get_cv_accounting_for_years(y_train=pd.DataFrame, kfold: int=5,
                                seed: int=1, groups=None):
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
    # if dealing with subseasonal data, there is a lot of autocorrelation.
    # it is best practice to keep the groups of target dates within a year well
    # seperated, therefore:
    if groups is None and np.unique(y_train.index.year).size != y_train.size:
        # find where there is a gap in time, indication of seperate RV period
        gapdays = (y_train.index[1:] - y_train.index[:-1]).days
        adjecent_dates = gapdays > (np.median(gapdays)+gapdays/2)
        n_gr = gapdays[gapdays > (np.median(gapdays)+gapdays/2)].size + 1
        dategroupsize = np.argmax(adjecent_dates) + 1
        groups = np.repeat(np.arange(0,n_gr), dategroupsize)
        if groups.size != y_train.size: # else revert to keeping years together
            groups = y_train.index.year
    elif groups is None and np.unique(y_train.index.year).size == y_train.size:
        groups = y_train.index.year # annual data, no autocorrelation groups
    else:
        pass


    high_normal_low = y_train.groupby(groups).sum()
    high_normal_low[(high_normal_low > high_normal_low.quantile(q=.66)).values] = 1
    high_normal_low[(high_normal_low < high_normal_low.quantile(q=.33)).values] = -1
    high_normal_low[np.logical_and(high_normal_low!=1, high_normal_low!=-1)] = 0
    # high_normal_low = high_normal_low.groupby(groups).sum()
    freq  = high_normal_low
    # freq = y_train.groupby(groups).sum()
    # freq = (freq > freq.mean()).astype(int)

    # all_years = np.unique(freq.index) Folds may be of different size
    # while all_years.size % kfold != 0:
    #     kfold += 1


    cv_strat = StratifiedKFold(n_splits=kfold, shuffle=True,
                               random_state=seed)
    test_gr = []
    for i, j in cv_strat.split(X=freq.index, y=freq.values):
        test_gr.append(j)
        # test_gr.append(freq.index[j].values)

    label_test = np.zeros( y_train.size , dtype=int)
    for i, test_fold in enumerate(test_gr):
        for j, gr in enumerate(groups):
            if j in list(test_fold):
                label_test[j] = i

    cv = PredefinedSplit(label_test)
    cv.uniqgroups = test_gr
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
    fit_masks = fit_masks.copy() # make copy of potential slice of df
    if 'fit_model_mask' not in fit_masks.columns:
        fit_masks.loc[:,'fit_model_mask'] = fit_masks['RV_mask'].copy()

    RV_mask = fit_masks['RV_mask'].copy()
    x_pred = RV_mask.shift(periods=-int(lag_i))
    x_pred[~x_pred.notna()] = False

    # y_fit, accounting for if test data of X is used to predict train data y
    # due to lag

    y_fit = fit_masks['fit_model_mask'].copy()
    # if lag_i > 0:
    #     # y_date cannot be predicted elsewise mixing
    #     # test test test train train train, cannot use dates test -> train
    #     # dates left boundary
    #     left_boundary = fit_masks['TrainIsTrue'].shift(periods=-lag_i,
    #                                         fill_value=fit_masks['TrainIsTrue'][-1])
    #     # train train test test test train train, cannot use dates train -> test
    #     # dates right boundary
    #     right_boundary = fit_masks['TrainIsTrue'].shift(periods=lag_i,
    #                                         fill_value=fit_masks['TrainIsTrue'][-1])
    #     diff_left = left_boundary.astype(int) - fit_masks['TrainIsTrue'].astype(int)
    #     diff_right = right_boundary.astype(int) - fit_masks['TrainIsTrue'].astype(int)
    #     diff_traintest = np.logical_or(diff_left, diff_right)
    #     dates_boundary_due_to_lag = diff_traintest[diff_traintest.astype(bool)].index
    #     y_fit.loc[dates_boundary_due_to_lag] = False


    x_fit = y_fit.shift(periods=-int(lag_i))
    n_nans = x_fit[~x_fit.notna()].size
    # set last x_fit date to False if x_fit caused nan
    if n_nans > 0:
        # take into account that last x_fit_train should be False to have
        # equal length y_train & x_fit
        x_fit[~x_fit.notna()] = False

    # cannot predict first values of y because there is no X
    if lag_i >= 0:
        dates_no_X_info = RV_mask.index[:int(lag_i)]
    elif lag_i < 0:
        dates_no_X_info = RV_mask.index[int(lag_i):]
    y_pred = RV_mask.copy()
    y_pred.loc[dates_no_X_info] = False
    y_fit.loc[dates_no_X_info] = False


    fit_masks.loc[:,'x_fit'] = x_fit == 1
    fit_masks.loc[:,'y_fit'] = y_fit == 1
    fit_masks.loc[:,'x_pred'] = x_pred
    fit_masks.loc[:,'y_pred'] = y_pred
    fit_masks = fit_masks.drop(['RV_mask'], axis=1)
    fit_masks = fit_masks.drop(['fit_model_mask'], axis=1)
    return fit_masks

def get_masks(df_norm):
    '''
    x_fit and y_fit can be encompass more data then x_pred, therefore they
    are split from x_pred and y_pred.
    y_pred is needed in the special occasion no past (lagged) data for X avaible
    if these are not given, then model x_fit, y_fit, x_pred & y_pred are
    fitted according to TrainIsTrue at lag=0.
    '''
    TrainIsTrue = df_norm['TrainIsTrue']==1
    if 'x_fit' in df_norm.columns:
        x_fit_mask = np.logical_and(TrainIsTrue, df_norm['x_fit']==True)
    else:
        x_fit_mask = TrainIsTrue
    if 'y_fit' in df_norm.columns:
        y_dates = df_norm['y_fit'][df_norm['y_fit']==True].index
        TrainIsTrue_yfit = TrainIsTrue.loc[y_dates]
        y_fit_mask = np.logical_and(TrainIsTrue_yfit==True, df_norm['y_fit'].loc[y_dates]==True)
        y_fit_mask = y_fit_mask
    else:
        y_fit_mask = TrainIsTrue
    if 'x_pred' in df_norm.columns:
        x_pred_mask = df_norm['x_pred']==True
    else:
        x_pred_mask = pd.Series(np.repeat(True, x_fit_mask.size),
                                index=x_fit_mask.index)
    if 'y_pred' in df_norm.columns:
        y_pred_dates = df_norm['y_pred'][df_norm['y_pred']==True].index
        y_pred_mask = df_norm['y_pred'].loc[y_pred_dates]
    else:
        y_pred_mask = None
    return x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask

def _standardize_sklearn(c, TrainIsTrue):
    standardize = preprocessing.StandardScaler()
    standardize.fit(c[TrainIsTrue.values].values.reshape(-1,1))
    return pd.Series(standardize.transform(c.values.reshape(-1,1)).squeeze(),
                     name=c.columns[0], index=c.index)
# pd.Series(standardize.transform(c.values.reshape(-1,1)).squeeze(),
#                      index=c.index, name=c.columns[0])

def standardize_on_train(c, TrainIsTrue):
    return ((c - c[TrainIsTrue.values==True].mean()) \
            / c[TrainIsTrue.values==True].std()).squeeze()

def standardize_on_train_and_RV(c, df_splits_s, lag, mask='x_fit'):
    fit_masks = apply_shift_lag(df_splits_s, lag)
    TrainIsTrue = fit_masks['TrainIsTrue']
    x_fit = fit_masks[mask]
    TrainRVmask = np.logical_and(TrainIsTrue==True, x_fit==True)
    return ((c - c[TrainRVmask.values].mean()) \
            / c[TrainRVmask.values].std()).squeeze()

def robustscaling_on_train(c, TrainIsTrue):
    return (c - c[TrainIsTrue.values==True].quantile(q=.25)) \
            / (c[TrainIsTrue.values==True].quantile(q=.75) - c[TrainIsTrue.values==True].quantile(q=.25))

def minmaxscaler_on_train(c, TrainIsTrue):
    return (c - c[TrainIsTrue.values==True].min()) \
            / (c[TrainIsTrue.values==True].max() - c[TrainIsTrue.values==True].min())

def corrcoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]

def r2_score(y_true, y_pred, multioutput='variance_weighted'):
    return metrics.r2_score(y_true, y_pred, multioutput=multioutput)


class ErrorSkillScore:
    def __init__(self, constant_bench: float=False, squared=False):
        '''
        Parameters
        ----------
        y_true : pd.DataFrame or pd.Series or np.ndarray
        y_pred : pd.DataFrame or pd.Series or np.ndarray
        benchmark : float, optional
            DESCRIPTION. The default is None.
        squared : boolean value, optional (default = True)
            If True returns MSE value, if False returns RMSE value

        Returns
        -------
        RMSE (Skill Score).

        '''
        if type(constant_bench) in [float, int, np.float_]:
            self.benchmark = float(constant_bench)
        elif type(constant_bench) in [np.ndarray, pd.Series, pd.DataFrame]:
            self.benchmark = np.array(constant_bench, dtype=float)
        else:
            print('benchmark is set to False')
            self.benchmark = False

        self.squared = squared
        # if type(self.benchmark) is not None:


    def RMSE(self, y_true, y_pred):
        self.RMSE_score = metrics.mean_squared_error(y_true, y_pred,
                                              squared=self.squared)
        if self.benchmark is False:
            return self.RMSE_score
        elif type(self.benchmark) is float:
            b_ = np.zeros(y_true.size) ; b_[:] = self.benchmark
        elif type(self.benchmark) is np.ndarray:
            b_  = self.benchmark
        self.RMSE_bench = metrics.mean_squared_error(y_true,
                                           b_,
                                           squared=self.squared)
        return (self.RMSE_bench - self.RMSE_score) / self.RMSE_bench

    def MAE(self, y_true, y_pred):
        fc_score = metrics.mean_absolute_error(y_true, y_pred)
        if self.benchmark is False:
            return fc_score
        elif type(self.benchmark) is float:
            b_ = np.zeros(y_true.size) ; b_[:] = self.benchmark
        elif type(self.benchmark) is np.ndarray:
            b_  = self.benchmark
        self.MAE_bench = metrics.mean_absolute_error(y_true, b_)
        return (self.MAE_bench - fc_score) / self.MAE_bench

    def BSS(self, y_true, y_pred):
        self.brier_score = metrics.brier_score_loss(y_true, y_pred)
        if self.benchmark is False:
            return self.brier_score
        elif type(self.benchmark) is float:
            self.b_ = np.zeros(y_true.size) ; self.b_[:] = self.benchmark
        elif type(self.benchmark) is np.ndarray:
            self.b_  = self.benchmark
        self.BS_bench = metrics.brier_score_loss(y_true, self.b_)
        return (self.BS_bench - self.brier_score) / self.BS_bench

class binary_score:
    def __init__(self, threshold: float=0.5):
        self.threshold = threshold

    def precision(self, y_true, y_pred):
        y_pred_b = y_pred > self.threshold
        return round(metrics.precision_score(y_true, y_pred_b)*100,0)

    def accuracy(self, y_true, y_pred):
        #  P(class=0) * P(prediction=0) + P(class=1) * P(prediction=1)
        y_pred_b = y_pred > self.threshold
        return round(metrics.accuracy_score(y_true, y_pred_b)*100,0)


def AUC_SS(y_true, y_pred):
    # from http://bibliotheek.knmi.nl/knmipubIR/IR2018-01.pdf eq. 1
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    auc_bench = .5
    return (auc_score - auc_bench) / (1-auc_bench)

class CRPSS_vs_constant_bench:
    def __init__(self, constant_bench: float=False, return_mean=True,
                 weights: np.ndarray=None):
        '''
        Parameters
        ----------
        y_true : pd.DataFrame or pd.Series or np.ndarray
        y_pred : pd.DataFrame or pd.Series or np.ndarray
        benchmark : float, optional
            DESCRIPTION. The default is None.
        return_mean: boolean value, optional (default = True)
            If True mean CRPSS instead of array of size len(y_true)
        weights : array_like, optional
            If provided, the CRPS is calculated exactly with the assigned
            probability weights to each forecast. Weights should be positive, but
            do not need to be normalized. By default, each forecast is weighted
            equally.
        Returns
        -------
        if return_mean == False (default):
            mean CRPSS versus benchmark
        if return_mean:
            mean CRPSS versus benchmark and continuous evaluation of forecasts

        '''
        self.benchmark = constant_bench
        self.return_mean = return_mean
        self.weights = weights
        # if type(self.benchmark) is not None:

        # return metrics.mean_squared_error(y_true, y_pred, squared=root
    def CRPSS(self, y_true, y_pred):
        fc_score = ps.crps_ensemble(y_true, y_pred,
                                    weights=self.weights)
        if self.return_mean:
            fc_score = fc_score.mean()
        if self.benchmark is False:
            return fc_score
        elif type(self.benchmark) in [float, int]:
            b_ = np.zeros_like(y_true) ; b_[:] = self.benchmark
            bench = ps.crps_ensemble(y_true, b_,
                                    weights=self.weights)
            if self.return_mean:
                bench = bench.mean()
            return (bench - fc_score) / bench


def get_scores(prediction, df_splits: pd.DataFrame=None, score_func_list: list=None,
               score_per_test=False, n_boot: int=1, blocksize: int=1,
               rng_seed=1):
    '''


    Parameters
    ----------
    prediction : TYPE
        DESCRIPTION.
    df_splits : pd.DataFrame, optional
        DESCRIPTION. The default is None.
    score_func_list : list, optional
        DESCRIPTION. The default is None.
    score_per_test : TYPE, optional
        DESCRIPTION. The default is True.
    n_boot : int, optional
        DESCRIPTION. The default is 1.
    blocksize : int, optional
        DESCRIPTION. The default is 1.
    rng_seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    pd.DataFrames format:
    index [opt. splits]
    Multi-index columns [lag, metric name]
    df_trains, df_test_s, df_tests, df_boots.

    '''
    #%%
    if df_splits is None and 'TrainIsTrue' not in prediction.columns:
        # assuming all is test data
        TrainIsTrue = np.zeros((prediction.index.size, 1))
        RV_mask  = np.ones((prediction.index.size, 1))
        df_splits = pd.DataFrame(np.concatenate([TrainIsTrue,RV_mask], axis=1),
                                   index=prediction.index,
                                   dtype=bool,
                                   columns=['TrainIsTrue', 'RV_mask'])
    elif df_splits is None and 'TrainIsTrue' in prediction.columns:
        # TrainIsTrue columns are part of prediction
        df_splits = prediction[['TrainIsTrue', 'RV_mask']]



    # add empty multi-index to maintain same data format
    if hasattr(df_splits.index, 'levels')==False:
        df_splits = pd.concat([df_splits], keys=[0])

    if hasattr(prediction.index, 'levels')==False:
        prediction = pd.concat([prediction], keys=[0])

    columns = [c for c in prediction.columns[:] if c not in ['TrainIsTrue', 'RV_mask']]
    if 'TrainIsTrue' not in prediction.columns:
        pred = prediction.merge(df_splits,
                                left_index=True,
                                right_index=True)
    else:
        pred = prediction


    # score on train and per test split
    if score_func_list is None:
        score_func_list = [metrics.mean_squared_error, corrcoef]
    splits = pred.index.levels[0]
    columns = np.array(columns[1:])
    df_trains = np.zeros( (columns.size), dtype=object)
    df_tests_s = np.zeros( (columns.size), dtype=object)
    for c, col in enumerate(columns):
        df_train = pd.DataFrame(np.zeros( (splits.size, len(score_func_list))),
                            columns=[f.__name__ for f in score_func_list])
        df_test_s = pd.DataFrame(np.zeros( (splits.size, len(score_func_list))),
                            columns=[f.__name__ for f in score_func_list])
        for s in splits:
            sp = pred.loc[s]
            not_constant = True
            if np.unique(sp.iloc[:,0]).size == 1:
                not_constant = False
            trainRV = np.logical_and(sp['TrainIsTrue']==1, sp['RV_mask']==True)
            testRV  = np.logical_and(sp['TrainIsTrue']==0, sp['RV_mask']==True)
            for f in score_func_list:
                name = f.__name__
                if (~trainRV).all()==False and not_constant: # training data exists
                    train_score = f(sp[trainRV].iloc[:,0], sp[trainRV].loc[:,col])
                else:
                    train_score  = np.nan
                if score_per_test and testRV.any() and not_constant:
                    test_score = f(sp[testRV].iloc[:,0], sp[testRV].loc[:,col])
                else:
                    test_score = np.nan

                df_train.loc[s,name] = train_score
                df_test_s.loc[s,name] = test_score
        df_trains[c] = df_train
        df_tests_s[c]  = df_test_s
    df_trains = pd.concat(df_trains, keys=columns, axis=1)
    df_tests_s = pd.concat(df_tests_s, keys=columns, axis=1)


    # score on complete test
    df_tests = np.zeros( (columns.size), dtype=object)
    pred_test = functions_pp.get_df_test(pred).iloc[:,:-2]
    if pred_test.size != 0 : # ensure test data is available
        for c, col in enumerate(columns):
            df_test = pd.DataFrame(np.zeros( (1,len(score_func_list))),
                                    columns=[f.__name__ for f in score_func_list])
            for f in score_func_list:
                name = f.__name__
                y_true = pred_test.iloc[:,0]
                y_pred = pred_test.loc[:,col]
                if np.unique(y_true).size >= 2:
                    df_test[name] = f(y_true, y_pred)
                else:
                    if c == 0:
                        print('Warning: y_true is constant. Returning NaN.')
                    df_test[name] = np.nan
            df_tests[c]  = df_test
        df_tests = pd.concat(df_tests, keys=columns, axis=1)


    # Bootstrapping with replacement
    df_boots = np.zeros( (columns.size), dtype=object)
    if pred_test.size != 0: # ensure test data is available
        for c, col in enumerate(columns):
            old_index = range(0,len(y_true),1)
            n_bl = blocksize
            chunks = [old_index[n_bl*i:n_bl*(i+1)] for i in range(int(len(old_index)/n_bl))]
            if np.unique(y_true).size > 1 or n_boot==0:
                score_list = _bootstrap(pred_test.iloc[:,[0,c+1]], n_boot,
                                        chunks, score_func_list,
                                        rng_seed=rng_seed)
            else:
                score_list  = np.repeat(np.nan,
                                        n_boot*len(score_func_list)).reshape(n_boot, -1)

            df_boot = pd.DataFrame(score_list,
                                   columns=[f.__name__ for f in score_func_list])
            df_boots[c] = df_boot
        df_boots = pd.concat(df_boots, keys=columns, axis=1)

    out = (df_trains, df_tests_s, df_tests, df_boots)

#%%
    return out

def cond_fc_verif(df_predict: pd.DataFrame,
                  df_forcing: pd.DataFrame,
                  df_splits: pd.DataFrame,
                  score_func_list: list=None,
                  quantiles:list =[.25],
                  n_boot: int=0):
    ''' Calculate metrics on seperate time indices. Split in time indices is
    determined by anomalous states of the df_forcing timeseries. The quantiles
    determine 'how anomalous' the seperation is.

    Parameters
    ----------
    df_predict : pd.DataFrame
        Out of sample prediction with multi-index [split, time].
    df_forcing : pd.DataFrame
        Out of sample forcing timeseries with multi-index [split, time].
        Calculates an equal weighted mean over columns to get 1-d timeseries
    df_splits : pd.DataFrame
        Train-test split masks with multi-index [split, time].
    score_func_list : list, optional
        list with scoring metrics. The default is None.
    quantiles : list, optional
        list with quantiles (q) to split the time indices.
        e.g., when q=0.25, time indices will be split based on df_forcing
        being below the 0.25q and above 0.75q, i.e. anomalous.
        The default is [.25].
    n_boot : int, optional
        n times bootstrapping skill metrics.
        The default is 0.

    Returns
    -------
    pd.DataFrame, metric names are the index (rows) and columns are the strong
    and weak quantile subsets. For example, [strong 50%, weak 50%] for q=.25.

    '''
    #%%
    df_forctest = functions_pp.get_df_test(df_forcing.mean(axis=1),
                                           df_splits=df_splits)

    df_test = functions_pp.get_df_test(df_predict,
                                       df_splits=df_splits)

    metrics = [s.__name__ for s in score_func_list]
    if n_boot > 0:
        cond_df = np.zeros((len(metrics), len(quantiles)*2, n_boot))
    else:
        cond_df = np.zeros((len(metrics), len(quantiles)*2))
    stepsize = 1 if len(quantiles)==1 else len(quantiles)*2
    for i, met in enumerate(metrics):
        for k, l in enumerate(range(0,stepsize,2)):
            q = quantiles[k]

            # =============================================================
            # Strong forcing
            # =============================================================
            # extrapolate quantile values based on training data
            q_low = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':q})
            # Extract out-of-sample quantile values
            q_low = functions_pp.get_df_test(q_low,
                                               df_splits=df_splits)

            q_high = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':1-q})
            q_high = functions_pp.get_df_test(q_high,
                                               df_splits=df_splits)

            low = df_forctest < q_low.values.ravel()
            high = df_forctest > q_high.values.ravel()
            mask_anomalous = np.logical_or(low, high)
            # anomalous Boundary forcing
            condfc = df_test[mask_anomalous.values]
            # condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
            cond_verif_tuple = get_scores(condfc,
                                                   score_func_list=score_func_list,
                                                   n_boot=n_boot,
                                                   score_per_test=False,
                                                   blocksize=1,
                                                   rng_seed=1)

            df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
            cond_verif_tuple  = cond_verif_tuple
            if n_boot == 0:
                cond_df[i, l] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            else:
                cond_df[i, l, :] = df_boot[df_boot.columns[0][0]][met]
            # =============================================================
            # Weak forcing
            # =============================================================
            q_higher_low = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':.5-q})
            q_higher_low = functions_pp.get_df_test(q_higher_low,
                                               df_splits=df_splits)


            q_lower_high = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':.5+q})
            q_lower_high = functions_pp.get_df_test(q_lower_high,
                                               df_splits=df_splits)

            higher_low = df_forctest > q_higher_low.values.ravel()
            lower_high = df_forctest < q_lower_high.values.ravel()

            mask_anomalous = np.logical_and(higher_low, lower_high)

            condfc = df_test[mask_anomalous.values]

            cond_verif_tuple = get_scores(condfc,
                                                   score_func_list=score_func_list,
                                                   n_boot=n_boot,
                                                   score_per_test=False,
                                                   blocksize=1,
                                                   rng_seed=1)
            df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
            if n_boot == 0:
                cond_df[i, l+1] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            else:
                cond_df[i, l+1, :] = df_boot[df_boot.columns[0][0]][met]
    columns = [[f'strong {int(q*200)}%', f'weak {int(q*200)}%'] for q in quantiles]
    columns = functions_pp.flatten(columns)
    if n_boot > 0:
        columns = pd.MultiIndex.from_product([columns, list(range(n_boot))])

    df_cond_fc = pd.DataFrame(cond_df.reshape((len(metrics), -1)),
                              index=list(metrics),
                              columns=columns)
    #%%
    return df_cond_fc


def _bootstrap(pred_test, n_boot_sub, chunks, score_func_list, rng_seed: int=1):

    y_true = pred_test.iloc[:,0]
    y_pred = pred_test.iloc[:,1]
    score_l = []
    rng = np.random.RandomState(rng_seed) ; i = 0 ; r = 0
    while i != n_boot_sub:
        i += 1 # loop untill n_boot
        # bootstrap by sampling with replacement on the prediction indices
        ran_ind = rng.randint(0, len(chunks) - 1, len(chunks))
        ran_blok = [chunks[i] for i in ran_ind] # random selection of blocks
        indices = list(itertools.chain.from_iterable(ran_blok)) #blocks to list

        if len(np.unique(y_true[indices])) < 2:
            i -= 1 ; r += 1 # resample and track # of resamples with r
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if r <= 100:
            score_l.append([f(y_true[indices],
                              y_pred[indices]) for f in score_func_list])
        else: # after 100 resamples, plug in NaNs
            score_l.append([np.nan for i in range(len(score_func_list))])
            if i == n_boot_sub:
                print(f'Too many ({r}) resample attempts to get both negative '
                      'and positive samples of truth, returning NaNs')

    return score_l

def SciKitModel_coeff(model, lag):
    '''
    Wrapper function to cast feature_importance or regression coeff. of any
    SciKitModel into pandas DataFrame

    Parameters
    ----------
    model : TYPE
        A fitted SciKitModel instance.
    lag : int

    Returns
    -------
    pd.DataFrame

    '''
    if hasattr(model, 'best_estimator_'): # GridSearchCV instance
        model = model.best_estimator_
    if hasattr(model, 'feature_importances_'): # for GBR
        name = 'Relative Feature Importance'
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        name = 'Coefficients'
        importances = model.coef_
    df = pd.DataFrame(importances.reshape(-1),
                        index=model.X_pred.columns,
                        columns=[lag])
    return df.rename_axis(name, axis=1)


