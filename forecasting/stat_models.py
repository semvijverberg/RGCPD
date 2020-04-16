#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:01:23 2019

@author: semvijverberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit, StratifiedKFold
from numpy.random import default_rng
from sklearn.inspection import partial_dependence
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import RFECV
import multiprocessing
max_cpu = multiprocessing.cpu_count()
from matplotlib.lines import Line2D
import itertools
flatten = lambda l: list(itertools.chain.from_iterable(l))

logit = ('logit', None)

#GBR_logitCV = ('GBR-logitCV',
#              {'max_depth':3,
#               'learning_rate':1E-3,
#               'n_estimators' : 750,
#               'max_features':'sqrt',
#               'subsample' : 0.6,
#               'random_state':60,
#               'min_impurity_decrease':1E-7} )

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

def logit_skl(y_ts, df_norm, keys=None, kwrgs_logit=None):

    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries
    '''

    if keys is None:
            no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
            keys = df_norm.columns
            keys = [k for k in keys if k not in no_data_col]
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    warnings.filterwarnings("ignore", category=FutureWarning)

    if kwrgs_logit == None:
        # use Bram settings
        kwrgs_logit = { 'class_weight':{ 0:1, 1:1},
                'scoring':'brier_score_loss',
                'penalty':'l2',
                'solver':'lbfgs'}


    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_logit.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_logit.copy()
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

    RV_bin_fit = y_ts['bin']
    # y_ts dates no longer align with x_fit  y_fit masks
    y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values
    y_train = RV_bin_fit[y_fit_mask].squeeze()

    # if y_pred_mask is not None:
    #     y_dates = RV_bin_fit[y_pred_mask.values].index
    # else:
    y_dates = RV_bin_fit.index
    
    X = X_train

    # Create stratified random shuffle which keeps together years as blocks.
    kwrgs_cv = ['kfold', 'seed']
    kwrgs_cv = {k:i for k, i in kwrgs.items() if k in kwrgs_cv}
    [kwrgs.pop(k) for k in kwrgs_cv.keys()]
   
    cv = get_cv_accounting_for_years(y_train, **kwrgs_cv)
    model = LogisticRegressionCV(fit_intercept=True,
                                 cv=cv,
                                 n_jobs=1, 
                                 **kwrgs)
    if feat_sel is not None:
        if feat_sel['model'] is None:
            feat_sel['model'] = model
        model, new_features, rfecv = feature_selection(X_train, y_train.values, **feat_sel)
        X_pred = X_pred[new_features]
    else:
        model.fit(X_train, y_train)


    y_pred = model.predict_proba(X_pred)[:,1]

    prediction = pd.DataFrame(y_pred, index=y_dates, columns=[0])
    model.X_pred = X_pred
    #%%
    return prediction, model



def GBC(y_ts, df_norm, keys, kwrgs_GBM=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if kwrgs_GBM == None:
        # use Bram settings
        kwrgs_GBM = {'max_depth':3,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5,
                 'min_samples_split':.15}

    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBM.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBM.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    if 'scoringCV' in kwrgs.keys():
        scoring = kwrgs.pop('scoringCV')
         # sorted(sklearn.metrics.SCORERS.keys())
         # scoring   = 'neg_mean_squared_error'
         # scoring='roc_auc'
    if 'feat_sel' in kwrgs:
        feat_sel = kwrgs.pop('feat_sel')
    else:
        feat_sel = None

    # Get training years
    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = get_masks(df_norm)

    X = df_norm[keys]
    X_train = X[x_fit_mask.values]
    X_pred  = X[x_pred_mask.values]



    RV_bin_fit = y_ts['bin']
    # y_ts dates no longer align with x_fit  y_fit masks
    y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values
    y_train = RV_bin_fit[y_fit_mask].squeeze()

    # if y_pred_mask is not None:
    #     y_dates = RV_bin_fit[y_pred_mask.values].index
    # else:
    y_dates = RV_bin_fit.index


    model = GradientBoostingClassifier(**kwrgs)

    if feat_sel is not None:
        if feat_sel['model'] is None:
            feat_sel['model'] = model
        model, new_features, rfecv = feature_selection(X_train, y_train.values.ravel(), **feat_sel)
        X_pred = X_pred[new_features] # subset predictors
        X_train = X_train[new_features] # subset predictors
    else:
        model.fit(X_train, y_train.values.ravel())



    if len(kwrgs_gridsearch) != 0:
        # get cross-validation splitter
        if 'kfold' in kwrgs.keys():
            kfold = kwrgs.pop('kfold')
        else:
            kfold = 5 
        cv = get_cv_accounting_for_years(len(y_train), kfold, seed=1)
        
        model = GridSearchCV(model,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=cv, refit=scoring,
                  return_train_score=True, iid=False)
        model = model.fit(X_train, y_train.values.ravel())
        if verbosity == 1:
            results = model.cv_results_
            scores = results['mean_test_score']
            greaterisbetter = model.scorer_._sign
            improv = int(100* greaterisbetter*(max(scores)- min(scores)) / max(scores))
            print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
                  "best params {}".format(
                    improv, model.best_score_, model.best_params_))
    else:
        model.fit(X_train, y_train.values.ravel())

    if len(kwrgs_gridsearch) != 0:
        prediction = pd.DataFrame(model.best_estimator_.predict_proba(X_pred)[:,1],
                              index=y_dates, columns=['GBR'])
    else:
        prediction = pd.DataFrame(model.predict_proba(X_pred)[:,1],
                              index=y_dates, columns=['GBR'])

    model.X_pred = X_pred


#    logit_pred.plot() ; plt.plot(RV.RV_bin)
#    plt.figure()
#    prediction.plot() ; plt.plot(RV.RV_ts)
#    metrics_sklearn(RV.RV_bin, logit_pred.values, y_pred_c)
    #%%
    return prediction, model


def logit(y_ts, df_norm, keys):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries
    '''
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if keys is None:
        no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
        keys = df_norm.columns
        keys = [k for k in keys if k not in no_data_col]

    # Get training years
    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = get_masks(df_norm)

    X = df_norm[keys]
    X = add_constant(X)
    X_train = X[x_fit_mask.values]
    X_pred  = X[x_pred_mask.values]

    RV_bin_fit = y_ts['bin']
    # y_ts dates no longer align with x_fit  y_fit masks
    y_fit_mask = df_norm['TrainIsTrue'].loc[y_fit_mask.index].values
    y_train = RV_bin_fit[y_fit_mask].squeeze()


    # if y_pred_mask is not None:
    #     y_dates = RV_bin_fit[y_pred_mask.values].index
    # else:
    y_dates = RV_bin_fit.index

    # Statsmodel wants the dataframes and that the indices are aligned.
    # Therefore making new dataframe for X_train
    try:
        model_set = sm.Logit(y_train,
                         pd.DataFrame(X_train.values, index=y_train.index), disp=0)
    except:
        print(x_fit_mask)
        print(X_train)
        print(y_train)

    try:
        model = model_set.fit( disp=0, maxfun=60 )
        prediction = model.predict(X_pred)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            model = model_set.fit(method='bfgs', disp=0 )
            prediction = model.predict(X_pred)
        else:
            raise
    except Exception as e:
        print(e)
        model = model_set.fit(method='bfgs', disp=0 )
        prediction = model.predict_proba(X_pred)

    prediction = pd.DataFrame(prediction.values, index=y_dates, columns=[0])
    model.X_pred = X_pred
    #%%
    return prediction, model


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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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

# =============================================================================
# Plotting
# =============================================================================
import seaborn as sns
from matplotlib import cycler
nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
colors_nice = cycler('color',
                nice_colors)
colors_datasets = sns.color_palette('deep')

line_styles = ['solid', 'dashed', (0, (3, 5, 1, 5, 1, 5)), 'dotted']

def plot_importances(models_splits_lags, lag=0, keys=None, cutoff=6,
                     plot=True):
    #%%
    # keys = ['autocorr', '10_1_sst']
    if type(lag) is int:
        df_all = _get_importances(models_splits_lags, lag=lag)

        if plot:
            import matplotlib as mpl
            mpl.rcParams.update(mpl.rcParamsDefault)
            # fig, ax = plt.subplots(constrained_layout=True)
            lag_d = df_all.index[0][0]
            df_r = df_all.loc[lag_d]
            if keys is not None:
                # take show up to cutoff most important features
                df_r = df_r[keys]
 
            g = sns.catplot(data=df_r, palette=sns.color_palette(["#e74c3c"]), 
                            orient='h', kind='box', ax=None, height=7)
            
            ax = g.ax
            # ax.set_facecolor('white')
            ax.grid(which='both')
            ax.set_title(f"{df_all.columns.name}")
            # ax.barh(np.arange(df_r.size), df_r.squeeze().values, tick_label=df_r.columns)
            ax.text(0.98, 0.03, f'lead time: {lag_d} days',
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat',
                              edgecolor='black', alpha=0.5),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
            lim = df_r.apply(abs).max().max()
            lim = lim + .1*lim
            ax.set_xlim((-lim,lim))
            fig = g.fig
    elif type(lag) is list or type(lag) is np.ndarray:

        dfs = []
        for i, l in enumerate(lag):
            df_ = _get_importances(models_splits_lags, lag=l)
            dfs.append(df_)
        all_vars = []
        all_vars.append([list(df.columns.values) for df in dfs])
        all_vars = np.unique(flatten(flatten(all_vars)))
        df_all = pd.DataFrame(columns=all_vars)
        for i, l in enumerate(lag):
            df_n = dfs[i]
            df_all = df_all.append(df_n, sort=False)
        sort_index = df_all.mean(0).sort_values(ascending=False).index
        df_all = df_all.reindex(sort_index, axis='columns')
        df_all = df_all.reindex(pd.MultiIndex.from_tuples(df_all.index))
        if plot:

            if keys is None:
                # take show up to cutoff most important features
                df_pl = df_all.reindex(df_all.mean(0).abs().sort_values(
                                                                ascending=False).index, axis=1)
                df_pl = df_pl.iloc[:,:cutoff]

            else:
                df_pl = df_all.loc[:,[k for k in sort_index if k in keys]]
            # plot vs lags
            fig, ax = plt.subplots(constrained_layout=True)
            styles_ = [['solid'], ['dashed']]
            styles = flatten([6*s for i, s in enumerate(styles_)])[:df_pl.size]
            linewidths = np.linspace(abs(cutoff)/4, 2, abs(cutoff))[::-1]
            lags_df = df_all.index.levels[0]
            for col, style, lw, cm in zip(df_pl.columns, styles, linewidths, colors_datasets):
                splits = df_all.loc[:,col].index.levels[1]
                df_var = df_all.loc[:,col]
                print(col)
                for s in splits:                   
                    ax.plot(lags_df.values, df_var.loc[:,s].values, 
                             linestyle=style,
                             linewidth=1,
                             color=cm, alpha=.3,
                             label=None)
                ax.plot(lags_df.values, 
                        df_var.mean(axis=0, level=0).values, 
                        linestyle=style,
                        linewidth=lw,
                        color=cm,
                        label=col)
                    # df_var.loc[:,s].plot(figsize=(8,5),
                    #                       linestyle=style,
                    #                       linewidth=1,
                    #                       label=label,
                    #                       ax=ax)
                ax.set_xticks(lags_df)
            ax.hlines(y=0, xmin=lags_df[0], xmax=lags_df[-1])
            ax.legend()
            ax.set_title(f'{df_.columns.name} vs. lead time')
            ax.set_xlabel('lead time [days]')
    #%%
    return df_all, fig



def _get_importances(models_splits_lags, lag=0):

    #%%
    '''
    get feature importance for single lag
    '''

    models_splits = models_splits_lags[f'lag_{lag}']
    splits = np.arange(len(models_splits_lags[f'lag_{lag}']))
    feature_importances = {}
    

    # if keys is None:
    keys = set()
    [keys.update(list(r.X_pred.columns)) for k, r in models_splits.items()]
    masks = ['TrainIsTrue', 'x_fit', 'x_pred', 'y_fit', 'y_pred']
    keys = [k for k in keys if k not in masks]
    tuples_multiindex = []
    for i, k in enumerate(keys):
        np_import = np.zeros( (splits.size))
        for splitkey, regressor in models_splits.items():
            s = int(splitkey.split('_')[1])
            keys_s = list(regressor.X_pred.columns[(regressor.X_pred.dtypes != bool)])
            if hasattr(regressor, 'feature_importances_'): # for GBR
                name_values = 'Relative Feature Importance'
                importances = regressor.feature_importances_
            elif hasattr(regressor, 'coef_'): # for logit
                name_values = 'Logistic Regression Coefficients'
                importances = regressor.coef_.squeeze(0)
            
            if k not in feature_importances.keys():
                feature_importances[k] = []
            if k not in keys_s:
                np_import[s] = np.nan
            else:
                # coeff belonging to var
                 idx = keys_s.index(k)
                 np_import[s] = importances[idx]
            tuples_multiindex.append((k, s))
        
        feature_importances[k] = np_import
        
            # for name, importance in zip(keys_s, importances):
            #     if name not in feature_importances:
            #         if name not in feature_importances.keys():
            #             feature_importances[name] = []
            #     feature_importances[name].append( importance )
                # feature_importances[name][1] += 1

    # remnant from Bram, importance by amount of time precursor was in model.
    # robust precursors get divided by 10, while other precursors are divided
    # by 1. == silly
    # names, importances = [], []
    # for name, (importance, count) in feature_importances.items():
    #     names.append(name)
    #     importances.append(float(importance) / float(count))
        
    df = pd.DataFrame(feature_importances)
    df_mean = df.apply(np.nanmean).apply(abs)
    columns = df_mean.sort_values(ascending=False).index
    df = df[columns]
    # add info lags
    df = pd.concat([df], keys=[lag])
    df = df.rename_axis(name_values, axis=1)
    # names, importances = [], []

    # for name, importances_splits in feature_importances.items():
    #     names.append(name)
    #     importances.append(np.mean(importances_splits))
    # if hasattr(regressor, 'feature_importances_'):
    #     importances = np.array(importances) / np.sum(importances)
    # elif hasattr(regressor, 'coef_'): # for logit
    #     importances = np.array(importances)
    # order = np.argsort(importances)
    # names_order = [names[index] for index in order] ; names_order.reverse()
    # zz = np.zeros( (len(names_order)), dtype=object)
    # for i, k in enumerate(names_order):
    #     zz[i] = feature_importances[k]
    # # freq = (regressor.X_pred.index[1] - regressor.X_pred.index[0]).days
    # # lags_tf = [l*freq for l in [lag]]
    # # if freq != 1:
    # #     # the last day of the time mean bin is tfreq/2 later then the centerered day
    # #     lags_tf = [l_tf- int(freq/2) if l_tf!=0 else 0 for l_tf in lags_tf]
    # df = pd.DataFrame(zz, index=names_order, columns=[lag])
    # # df = pd.DataFrame([sorted(importances, reverse=True)], columns=names_order,
    # #                   index=[lag])
    

    #%%
    return df


def plot_oneway_partial_dependence(GBR_models_split_lags, keys=None, lags=None,
                                   grid_resolution=20):
    #%%
    sns.set_style("whitegrid")
    sns.set_style(rc = {'axes.edgecolor': 'black'})


    if lags is None:
        lag_keys = GBR_models_split_lags.keys()
        lags = [int(l.split('_')[1]) for l in lag_keys][:3]


    if keys is None:
        keys = set()
        for l, lag in enumerate(lags):
            # get models at lag
            GBR_models_split = GBR_models_split_lags[f'lag_{lag}']
        [keys.update(list(r.X_pred.columns)) for k, r in GBR_models_split.items()]
        masks = ['TrainIsTrue', 'x_fit', 'x_pred', 'y_fit', 'y_pred']
        keys = [k for k in keys if k not in masks]
    keys = keys

    df_lags = []
    for l, lag in enumerate(lags):
        # get models at lag
        GBR_models_split = GBR_models_split_lags[f'lag_{lag}']

        df_keys = []
        keys_in_lag = []
        for i, key in enumerate(keys):
            y = [] ; x = []
            for splitkey, regressor in GBR_models_split.items():
                if key in list(regressor.X_pred.columns):
                    X_pred = regressor.X_pred
                    index = list(X_pred.columns).index(key)
                    TrainIsTrue = regressor.df_norm['TrainIsTrue']
                    TestIsTrue = TrainIsTrue.loc[X_pred.index] == False
                    X_test = X_pred[TestIsTrue]
                    # X_test = regressor.X_pred.loc[:,all_keys][regressor.X_pred['x_pred']]
                    _y, _x = partial_dependence(regressor, X=X_test,
                                                features=[index],
                                                grid_resolution=grid_resolution)
                    y.append(_y[0])
                    x.append(_x[0])
                    keys_in_lag.append(key)
            if len(y) != 0:
                # y has shape (grid_res, splits_key_present)
                y_mean = np.array(y).mean(0)
                y_std = np.std(y, 0).ravel()
                x_vals = np.mean(x, 0)
                count_splits = np.repeat(np.array(y).shape[0], y_mean.shape)
                data = [y_mean[:,None], y_std[:,None], x_vals[:,None], count_splits[:,None]]
                data = np.concatenate(data, axis=1)
                df_key = pd.DataFrame(data, columns=['y_mean', 'y_std', 'x_vals', 'count splits'])
                df_keys.append(df_key)
        df_keys = pd.concat(df_keys, keys=np.unique(keys_in_lag))
        df_lags.append(df_keys)
    df_lags = pd.concat(df_lags, keys=lags)
    # =============================================================================
    # Plotting
    # =============================================================================
    #%%
    col_wrap = 4
    g = sns.FacetGrid(pd.DataFrame(data=keys), col=0, col_wrap=col_wrap,
                      aspect=1.5, sharex=False)
    custom_lines = [] ; _legend = []
    for l, lag in enumerate(lags):

        style = line_styles[l]
        color = colors_datasets[l]
        custom_lines.append(Line2D([0],[0],linestyle=style, color=color, lw=4,
                                   markersize=10))
        _legend.append(f'lag {lag}')
#        text_lag = []
        for i, key in enumerate(keys):
            ax = g.axes[i]
            df_plot = df_lags.loc[lag, key]
            y_mean = df_plot['y_mean']
            y_std  = 2 * df_plot['y_std']
            x_vals = df_plot['x_vals']
            ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std,
                            color=color, linestyle=style, alpha=0.2)
            ax.plot(x_vals, y_mean, color=color, linestyle=style)
            ax.set_title(key)
            if i == 0:
                ax.legend(custom_lines, _legend, handlelength=3)

    return df_lags, g.fig

    #%%

def _get_twoway_pairdepend(GBR_models_split, i, pair, grid_resolution):
    y = [] ; x = []
    for split, regressor in GBR_models_split.items():
        check_pair = [True for p in pair if p in list(regressor.X_pred.columns)]
        if all(check_pair):
            # retrieve index of two variables
            index = [list(regressor.X_pred.columns).index(p) for p in pair]
            all_keys = regressor.X_pred.columns[(regressor.X_pred.dtypes != bool)]
            X_test = regressor.X_pred.loc[:,all_keys][regressor.X_pred['x_pred']]
            _y, _x = partial_dependence(regressor, X=X_test, features=[index],
                                        grid_resolution=grid_resolution)
            y.append(_y.squeeze())
            x.append(_x)
    result = np.mean(y, axis=0)
    x_tick = np.mean(x, axis=0)
    return result, x_tick

def plot_twoway_partial_dependence(GBR_models_split_lags, lag_i=0, keys=None,
                                   plot_pairs=None, min_corrcoeff=0.1,
                                   grid_resolution=20):
    '''
    Parameters
    ----------
    keys : list or tuple.
        if keys is list: find all pairwise relationships based on correlation.
        Criteria for corr matrix: relation must be (sign corr and > min_corrcoeff)
        if key is tuple, will plot pairwise dependencies among the given variables
    plot_pairs : list of tuples.
        if list of tuples is supplied, only plot the pairs inside the tuples.
        if plot_pairs is not None, keys is overwritten.

    '''

    import df_ana
    #%%
    GBR_models_split = GBR_models_split_lags[f'lag_{lag_i}']

    if plot_pairs is None and type(keys) is list:
        # collect all pairwise relationship that pass criteria
        # Criteria: relation must be (sign corr and > min_corrcoeff)
        all_pairs = set()
        # plot two way depend. if timeseries are correlated
        # first calculating cross corr matrix per split
        for splitkey, regressor in GBR_models_split.items():
            all_keys = regressor.X_pred.columns[(regressor.X_pred.dtypes != bool)]
            X_test = regressor.X_pred.loc[:,all_keys][regressor.X_pred['x_pred']]
            cross_corr, sig_mask = df_ana.corr_matrix_pval(X_test)[:2]
            np.fill_diagonal(sig_mask, False)
            mask = np.logical_and(sig_mask, cross_corr.values > min_corrcoeff)
            sig_cross = cross_corr.where(mask)

            for index, row in sig_cross.iterrows():
                notnan = row.dropna()
                corr_vs_index = [key for key, v in notnan.iteritems() if v != 1.0]
                pairwise = [sorted([index, k]) for k in corr_vs_index]
                if len(pairwise) != 0:
    #                keep_s[index] = corr_vs_index
                    all_pairs.update([tuple(pair) for pair in pairwise])
        all_pairs = list(all_pairs)

    if keys is None and plot_pairs is None:
        plot_pairs = all_pairs
    elif type(keys) is list and plot_pairs is None:
        plot_pairs = [p for p in all_pairs if any(x in p for x in keys)]
    elif type(keys) is tuple and plot_pairs is None:
        zz = [tuple(sorted((x,y))) for x in keys for y in keys if x != y]
        plot_pairs = [tuple(t) for t in np.unique(zz, axis=0)]
    if plot_pairs is not None:
        plot_pairs = plot_pairs

    from time import time
    t0 = time()
    results = np.zeros( (len(plot_pairs), grid_resolution, grid_resolution) )
    x_ticks = np.zeros( (len(plot_pairs)), dtype=object)
    futures = {}
    with ProcessPoolExecutor(max_workers=max_cpu) as pool:
        for i, pair in enumerate(plot_pairs):
            futures[i] = pool.submit(_get_twoway_pairdepend, GBR_models_split,
                                       i, pair, grid_resolution)

    for key, future in enumerate(futures.items()):
        results[key], x_ticks[key] = future[1].result()
    print(round(time() - t0, 1))

    df_all = {}
    for i, pair in enumerate(plot_pairs):
        x_coord = np.round(x_ticks[i][0],3)
        y_coord = np.round(x_ticks[i][1],3)[::-1]
        img2d = results[i]
        img2d = np.flip(img2d, axis=1)
        df = pd.DataFrame(data=np.array([np.repeat(x_coord,len(y_coord)),
                                         np.tile(y_coord,len(x_coord)),
                                         img2d.flatten()]).T,
                                         columns=[pair[0], pair[1], 'vals'])

        df_p = df.pivot(pair[0], pair[1], 'vals').round(3)
        df_all[str(pair)] = df_p
#    df_all = pd.concat(dfs, keys=plot_pairs)

    #%%

    df_temp = pd.DataFrame(data=range(len(plot_pairs)), index=plot_pairs)
    g = sns.FacetGrid(df_temp, col=0, col_wrap=3, aspect=1.5,
                      sharex=False, sharey=False)

    for i, pair in enumerate(plot_pairs):

        ax = g.axes[i]
        df_p = df_all[str(pair)]
        vmin = np.min(results) ; vmax = np.max(results)
        ax = sns.heatmap(df_p, vmin=-max(abs(vmin),vmax),
                         vmax=max(abs(vmin),vmax), center=0,
                         cmap=plt.cm.afmhot_r,
                         yticklabels=4,
                         xticklabels=4,
                         ax=ax)
        ax.invert_yaxis()
        ax.tick_params(axis='both')
        orig = [float(item.get_text()) for item in ax.get_xticklabels()]
#        x_want = [-1,0,1]
#        x_close = [find_nearest(orig, t)[0] for t in x_want]
#        x_tl = np.array(np.repeat('', len(orig)), dtype=object)
#        for i, ind in enumerate(x_close):
#            x_tl[ind] = str(x_want[i])
        ax.set_xticklabels(np.round(orig, 1))
        orig = [float(item.get_text()) for item in ax.get_yticklabels()]
#        y_want = [-1,0,1]
#        y_close = [find_nearest(orig, t)[0] for t in y_want]
#        y_tl = np.array(np.repeat('', len(orig)), dtype=object)
#        for i, ind in enumerate(y_close):
#            y_tl[ind] = str(y_want[i])
        ax.set_yticklabels(np.round(orig, 1))
#        vmin = np.min(results) ; vmax = np.max(results)
#        im = ax.pcolormesh(x_coord, y_coord, img2d, cmap=plt.cm.afmhot_r,
#                      vmin=-max(abs(vmin),vmax), vmax=max(abs(vmin),vmax))
#        ax.set_ylabel(pair[0], labelpad=-3)
#        ax.set_xlabel(pair[1], labelpad=-1.5)
#    cax = g.fig.add_axes([0.4, -0.03, 0.2, 0.02]) # [left, bottom, width, height]
#    g.fig.colorbar(im, cax=cax, orientation='horizontal')
    g.fig.subplots_adjust(wspace=0.3)
    g.fig.subplots_adjust(hspace=0.3)
    #%%
    return df_all, g.fig

def _get_synergy(GBR_models_split_lags, lag_i=0, plot_pairs=None,
                 grid_resolution=20):
    #%%
    i=0
    pair = plot_pairs[i]
    df_single = plot_oneway_partial_dependence(GBR_models_split_lags,
                                               keys=list(pair), lags=[lag_i],
                                               grid_resolution=grid_resolution).loc[lag_i]
    df_key1 = df_single.loc[pair[0]]
    df_key2 = df_single.loc[pair[1]]
    x_coord = df_key1['x_vals']
    y_coord = df_key2['x_vals'][::-1]
    v1, v2 = np.meshgrid(df_key1['y_mean'], df_key2['y_mean'])
    # flip v2 to match df_pair
    v2 = np.flip(v2, axis=0)
    img2d = v1 + v2
    df_flat = pd.DataFrame(data=np.array([np.repeat(x_coord,len(y_coord)),
                                         np.tile(y_coord,len(x_coord)),
                                         img2d.flatten()]).T,
                                         columns=[pair[0], pair[1], 'vals'])
    df_p = df_flat.pivot(pair[0], pair[1], 'vals').round(3)

    plt.figure()
    plt.imshow(df_p, cmap=plt.cm.RdBu) ; plt.colorbar()


    df_pair = plot_twoway_partial_dependence(GBR_models_split_lags, lag_i=lag_i,
                                   plot_pairs=plot_pairs, grid_resolution=grid_resolution)

    plt.figure()
    plt.imshow(df_pair.values, cmap=plt.cm.RdBu) ; plt.colorbar()
#%%

def _get_regularization(models_splits_lags, lag_i=0):

    models_splits = models_splits_lags[f'lag_{lag_i}']

    # np.array( (len(models_splits), )
    result_splits = []
    # best = []
    for splitkey, model in models_splits.items():
        Cs_ = models_splits[splitkey].Cs_
        # low Cs_ is strong regulazation
        scores = models_splits[splitkey].scores_[1]
        Cs_format = ['{:.0E}'.format(v) for v in Cs_]
        df_ = pd.DataFrame(scores, columns=Cs_format)
        df_ = df_.append(pd.Series(model.Cs_ == model.C_[0],
                             index=Cs_format,
                             dtype=bool),
                             ignore_index=True)

        # df['best'] = m.Cs_ == m.C_[0]
        result_splits.append(df_)
    df = pd.concat(result_splits, keys=range(len(models_splits)))
    df = df.rename_axis(models_splits[splitkey].scoring, axis=1)
    # df.rename_axis(models_splits[splitkey].C_, axis=0)

    return df

def plot_regularization(models_splits_lags, lag_i=0):
    #%%
    df = _get_regularization(models_splits_lags, lag_i=lag_i)
    n_spl = np.unique(df.index.get_level_values(0)).size
    g = sns.FacetGrid(pd.DataFrame(range(n_spl), index=range(n_spl)),
                      col=0, col_wrap=2, aspect=1.5,
                      sharex=True, sharey=True)

    for i, ax in enumerate(g.axes):
        df_p = df.loc[i].iloc[:-1].copy()
        lines = np.array(['--'] * df_p.columns.size)
        lines[df.loc[i].iloc[-1] == 1.] = '-'
        lw = np.array([1] * df_p.columns.size)
        lw[df.loc[i].iloc[-1] == 1.] = 3
        df_p.plot(cmap=plt.cm.Reds_r, kind='line',ax=ax,
                       style=list(lines),
                       legend=False)
        if i == 0:
            ax.legend(fontsize=7, mode='expand')
        for i, l in enumerate(ax.lines):
            plt.setp(l, linewidth=lw[i])

        ax.set_ylim(-.3, 0)
        ax.set_ylabel(df_p.columns.name)
        ax.set_xlabel('LogitRegr CV folds')
    g.fig.suptitle('Inverse Regularization strength (low is strong)', y=1.00)
    return g.fig

