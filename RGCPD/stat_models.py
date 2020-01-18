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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import partial_dependence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
max_cpu = multiprocessing.cpu_count()
from matplotlib.lines import Line2D
import itertools
flatten = lambda l: list(itertools.chain.from_iterable(l))

logit = ('logit', None)

GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 750,
               'max_features':'sqrt',
               'subsample' : 0.6,
               'random_state':60} )



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
    RV_bin_fit = RV_bin_fit.loc[y_fit_mask.index].copy()
    y_train = RV_bin_fit[y_fit_mask.values].squeeze()     

    if y_pred_mask is not None:
        y_dates = RV_bin_fit[y_pred_mask.values].index
    else:
        y_dates = X.index

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
        prediction = model.predict(X_pred)
    
    prediction = pd.DataFrame(prediction.values, index=y_dates, columns=[0])                          
    #%%
    return prediction, model

def GBR_logitCV(y_ts, df_norm, keys, kwrgs_GBR=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
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

    # Get training years
    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = get_masks(df_norm)
    
    X = df_norm[keys]
    X_train = X[x_fit_mask.values]
    X_pred  = X[x_pred_mask.values]

    RV_ts_fit = y_ts['cont']
    RV_ts_fit = RV_ts_fit.loc[y_fit_mask.index]
    y_train = RV_ts_fit[y_fit_mask.values].squeeze() 

    if y_pred_mask is not None:
        y_dates = RV_ts_fit[y_pred_mask.values].index
    else:
        y_dates = X.index

    y_train = RV_ts_fit[y_fit_mask.values] 
    
    if y_pred_mask is not None:
        y_dates = RV_ts_fit[y_pred_mask.values].index
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
        if verbosity == 1:
            results = regressor.cv_results_
            scores = results['mean_test_score'] 
            improv = int(100* (min(scores)-max(scores)) / max(scores))
            print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
                  "best params {}".format(
                    improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    
        
    prediction = pd.DataFrame(regressor.predict(X_pred), 
                              index=y_dates, columns=[0])
    # add masks for second fit with logitCV
    TrainIsTrue_ = df_norm['TrainIsTrue'].loc[y_pred_mask[y_pred_mask.values].index]
    prediction['TrainIsTrue'] = pd.Series(TrainIsTrue_.values, 
                                  index=TrainIsTrue_.index)

    logit_pred, model_logit = logit_skl(y_ts, prediction, keys=None)
    
    
    
#    logit_pred.plot() ; plt.plot(RV.RV_bin)
#    plt.figure()
#    prediction.plot() ; plt.plot(RV.RV_ts)
#    metrics_sklearn(RV.RV_bin, logit_pred.values, y_pred_c)
    #%%
    return logit_pred, regressor


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
    
    # Get training years
    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = get_masks(df_norm)
    
    X = df_norm[keys]
#    X = add_constant(X)
    X_train = X[x_fit_mask.values]
    X_pred  = X[x_pred_mask.values]
    
    RV_bin_fit = y_ts['bin']
    RV_bin_fit = RV_bin_fit.loc[y_fit_mask.index]
    RV_bin_fit = RV_bin_fit.loc[y_fit_mask.index]
    y_train = RV_bin_fit[y_fit_mask.values].squeeze()   

    if y_pred_mask is not None:
        y_dates = RV_bin_fit[y_pred_mask.values].index
    else:
        y_dates = X.index
    
    # sample weight not yet supported by GridSearchCV (august, 2019)
    strat_cv = StratifiedKFold(5, shuffle=False)
    model = LogisticRegressionCV(Cs=10, fit_intercept=True, 
                                 cv=strat_cv,
                                 n_jobs=1, 
                                 **kwrgs_logit)
                                 
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_pred)[:,1]

    prediction = pd.DataFrame(y_pred, index=y_dates, columns=[0])

    #%%
    return prediction, model

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

def plot_importances(GBR_models_split_lags, lag=1, keys=None, cutoff=6, 
                     plot=True):
    #%%
#    keys = ['autocorr', '10_1_sst']
    if type(lag) is int:
        df_all = _get_importances(GBR_models_split_lags, lag=lag)
        
        if plot:
            fig, ax = plt.subplots(constrained_layout=True)
            lag_d = df_all.index[0]
            if keys is None:
                # take show up to cutoff most important features
                df_r = df_all.loc[lag_d].sort_values()[-cutoff:]
            else:
                df_r = df_all.loc[lag_d].loc[keys]
            ax.set_title(f"Relative Feature Importances")
            ax.barh(np.arange(df_r.size), df_r.squeeze().values, tick_label=df_r.index)
            ax.text(0.97, 0.07, f'lead time: {lag_d} days',
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', 
                              edgecolor='black', alpha=0.5),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
    elif type(lag) is list or type(lag) is np.ndarray:
        
        dfs = []
        for i, l in enumerate(lag):
            dfs.append(_get_importances(GBR_models_split_lags, lag=l))
        all_vars = []
        all_vars.append([list(df.columns.values) for df in dfs])
        all_vars = np.unique(flatten(flatten(all_vars)))
        df_all = pd.DataFrame(columns=all_vars)
        for i, l in enumerate(lag):
            df_n = dfs[i] 
            df_all = df_all.append(df_n, sort=False)
        sort_index = df_all.mean(0).sort_values(ascending=False).index
        df_all = df_all.reindex(sort_index, axis='columns')
        
        if plot:
            
            if keys is None:
                # take show up to cutoff most important features
                df_pl = df_all.loc[:,sort_index[:cutoff]]
            else:
                df_pl = df_all.loc[:,[k for k in sort_index if k in keys]]
            # plot vs lags
            fig, ax = plt.subplots(constrained_layout=True)
            styles_ = [['solid'], ['dashed']]
            styles = flatten([6*s for i, s in enumerate(styles_)])[:df_pl.size]
            linewidths = np.linspace(cutoff/3, 1, cutoff)
            for col, style, lw in zip(df_pl.columns, styles, linewidths):
                df_all.loc[:,col].plot(figsize=(8,5), 
                                      linestyle=style,
                                      linewidth=lw,
                                      ax=ax)
            ax.legend()
            ax.set_title('Relative Feature importance vs. lead time')
            ax.set_xlabel('lead time [days]')
    #%%
    return df_all


def _get_importances(GBR_models_split_lags, lag=1, _ax=None):
                     
    #%%
    '''
    get feature importance for single lag
    '''

    GBR_models_split = GBR_models_split_lags[f'lag_{lag}']
    feature_importances = {}
    
    for splitkey, regressor in GBR_models_split.items():
        all_keys = list(regressor.X.columns[(regressor.X.dtypes != bool)])
        importances = regressor.feature_importances_
        for name, importance in zip(all_keys, importances):
            if name not in feature_importances:
                feature_importances[name] = [0, 0]
            feature_importances[name][0] += importance
            feature_importances[name][1] += 1

    names, importances = [], []

    
    for name, (importance, count) in feature_importances.items():

        names.append(name)
        importances.append(float(importance) / float(count))
    

    importances = np.array(importances) / np.sum(importances)
    order = np.argsort(importances)
    names_order = [names[index] for index in order] ; names_order.reverse()
    freq = (regressor.X.index[1] - regressor.X.index[0]).days
    lags_tf = [l*freq for l in [lag]]
    if freq != 1:
        # the last day of the time mean bin is tfreq/2 later then the centerered day
        lags_tf = [l_tf- int(freq/2) if l_tf!=0 else 0 for l_tf in lags_tf]
    df = pd.DataFrame([sorted(importances, reverse=True)], columns=names_order, index=lags_tf)      

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
        [keys.update(list(r.X.columns)) for k, r in GBR_models_split.items()]
        masks = ['TrainIsTrue', 'x_fit', 'x_pred', 'y_fit', 'y_pred']
        keys = [k for k in keys if k not in masks]
    keys = keys
    
    df_lags = []
    for l, lag in enumerate(lags):
        # get models at lag
        GBR_models_split = GBR_models_split_lags[f'lag_{lag}']

        df_keys = []
        for i, key in enumerate(keys):
            y = [] ; x = []
            for splitkey, regressor in GBR_models_split.items():
                if key in list(regressor.X.columns):
                    index = list(regressor.X.columns).index(key)
                    all_keys = regressor.X.columns[(regressor.X.dtypes != bool)]
                    X_test = regressor.X.loc[:,all_keys][regressor.X['x_pred']]
                    _y, _x = partial_dependence(regressor, X=X_test, features=[index],
                                                grid_resolution=grid_resolution)
                    y.append(_y[0])
                    x.append(_x[0])
            
            y_mean = np.array(y).mean(0)
            y_std = np.std(y, 0).ravel()
            x_vals = np.mean(x, 0)
            data = np.concatenate([y_mean[:,None], y_std[:,None], x_vals[:,None]], axis=1)
            df_key = pd.DataFrame(data, columns=['y_mean', 'y_std', 'x_vals'])
            df_keys.append(df_key)
        df_keys = pd.concat(df_keys, keys=keys)
        df_lags.append(df_keys)
    df_lags = pd.concat(df_lags, keys=lags)
    # =============================================================================
    # Plotting    
    # =============================================================================
    #%%
    g = sns.FacetGrid(pd.DataFrame(data=keys), col=0, col_wrap=3, 
                      aspect=1.5, sharex=False)   
    custom_lines = [] ; _legend = []
    for l, lag in enumerate(lags):
        
        style = line_styles[l]
        color = colors_datasets[l]
        custom_lines.append(Line2D([0],[0],linestyle=style, color=color, lw=4,
                                   markersize=10))
        _legend.append(f'lag {lag}')
        
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

    return df_lags
            
    #%%


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def _get_twoway_pairdepend(GBR_models_split, i, pair, grid_resolution): 
    y = [] ; x = []
    for split, regressor in GBR_models_split.items():
        check_pair = [True for p in pair if p in list(regressor.X.columns)]
        if all(check_pair):
            # retrieve index of two variables
            index = [list(regressor.X.columns).index(p) for p in pair]
            all_keys = regressor.X.columns[(regressor.X.dtypes != bool)]
            X_test = regressor.X.loc[:,all_keys][regressor.X['x_pred']]
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
            all_keys = regressor.X.columns[(regressor.X.dtypes != bool)]
            X_test = regressor.X.loc[:,all_keys][regressor.X['x_pred']]
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
    return df_all 

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
                                   

#    plot_twoway_partial_dependence()
            
#        keep.append( {index : corr_vs_index} )
#    zz = sig_cross.loc[row]



#def GBR(y_ts, df_norm, keys=None, kwrgs_GBR=None, verbosity=0):
#    #%%
#    '''
#    X contains all precursor data, incl train and test
#    X_train, y_train are split up by TrainIsTrue
#    Preciction is made for whole timeseries    
#    '''
#    import warnings
#    warnings.filterwarnings("ignore", category=DeprecationWarning) 
#    warnings.simplefilter(action='ignore', category=FutureWarning)
#    
#    
#    if keys is None:
#        no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
#        keys = df_norm.columns
#        keys = [k for k in keys if k not in no_data_col]
#        
#    if kwrgs_GBR == None:
#        # use Bram settings
#        kwrgs_GBR = {'max_depth':3,
#                 'learning_rate':0.001,
#                 'n_estimators' : 1250,
#                 'max_features':'sqrt',
#                 'subsample' : 0.5}
#    
#    # find parameters for gridsearch optimization
#    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
#    # only the constant parameters are kept
#    kwrgs = kwrgs_GBR.copy()
#    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
#    
#    X = df_norm[keys]
#    X = add_constant(X)
#    RV_ts = RV.RV_ts_fit
#    # Get training years
#    TrainIsTrue = df_norm['TrainIsTrue'] 
#    # Get mask to make only prediction for RV_mask dates
#    pred_mask   = df_norm['RV_mask']
#  
#    X_train = X[TrainIsTrue]
#    y_train = RV_ts[TrainIsTrue.values] 
#    
#    # add sample weight mannually
##    y_train[y_train > y_train.mean()] = 10 * y_train[y_train > y_train.mean()]
#    
#    # sample weight not yet supported by GridSearchCV (august, 2019)
##    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
#    regressor = GradientBoostingRegressor(**kwrgs)
#
#    if len(kwrgs_gridsearch) != 0:
##        scoring   = 'r2'
#        scoring   = 'neg_mean_squared_error'
#        regressor = GridSearchCV(regressor,
#                  param_grid=kwrgs_gridsearch,
#                  scoring=scoring, cv=5, refit=scoring, 
#                  return_train_score=False)
#        regressor.fit(X_train, y_train.values.ravel())
#        results = regressor.cv_results_
#        scores = results['mean_test_score'] 
#        improv = int(100* (min(scores)-max(scores)) / max(scores))
#        print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
#              "best params {}".format(
#                improv, regressor.best_score_, regressor.best_params_))
#    else:
#        regressor.fit(X_train, y_train.values.ravel())
#    
#
#    prediction = pd.DataFrame(regressor.predict(X[pred_mask]),
#                              index=X[pred_mask].index, columns=[0])
#    prediction['TrainIsTrue'] = pd.Series(TrainIsTrue.values, index=X.index)
#    prediction['RV_mask'] = pd.Series(pred_mask.values, index=pred_mask.index)
#    
#    logit_pred, model_logit = logit(RV, prediction, keys=None)
#    #%%
#    return logit_pred, (model_logit, regressor)