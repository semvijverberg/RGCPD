#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:15:50 2019

@author: semvijverberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from sklearn import metrics
import functions_pp
from sklearn.calibration import calibration_curve
import sklearn.metrics as metrics
import seaborn as sns
import itertools 
flatten = lambda l: list(itertools.chain.from_iterable(l))

import matplotlib as mpl
from matplotlib import cycler
nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
colors_nice = cycler('color',
                nice_colors)
colors_datasets = [np.array(c) for c in sns.color_palette('deep')]

line_styles = ['solid', 'dashed', (0, (3, 5, 1, 5, 1, 5)), 'dotted']
# dashdotdotted = (0, (3, 5, 1, 5, 1, 5)))


plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors_nice)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='black')
plt.rc('ytick', direction='out', color='black')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

mpl.rcParams['figure.figsize'] = [7.0, 5.0]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600

mpl.rcParams['font.size'] = 13
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'


def get_scores_improvement(m_splits, fc, s, lag, metric=None):
#    import stat_models
    import warnings
    warnings.filterwarnings("ignore")
    m = m_splits[f'split_{s}']
       
#    assert hasattr(m, 'n_estimators'), print(m.cv_results_)

        
#    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = stat_models.get_masks(m.X)
    
#    keys = m.X.columns[m.X.dtypes != bool]
    X_pred = m.X_pred
    if hasattr(m, 'n_estimators')==False:   
        m = m.best_estimator_
    
    if hasattr(m, 'predict_proba'):
        y_true = fc.TV.RV_bin
    else:
        y_true = fc.TV.RV_ts
    
    
    TrainIsTrue = m.df_norm['TrainIsTrue']
    X_train = X_pred[TrainIsTrue.loc[X_pred.index]]
#    X_test = pd.to_datetime([d for d in X_pred.index if d not in X_train[X_train].index])
    X_test = X_pred[~TrainIsTrue.loc[X_pred.index]]
    y_maskTrainIsTrue = TrainIsTrue.loc[fc.TV.dates_RV]
    y_test = y_true[~y_maskTrainIsTrue].values.squeeze()

    test_scores = np.zeros(m.n_estimators) 
    train_scores = np.zeros(m.n_estimators) 
    for i, y_pred in enumerate(m.staged_predict(X_test)):
        if metric is None:
            test_scores[i] = m.loss_(y_test, y_pred)
        else:
            test_scores[i] = metric(y_test, y_pred)
            
    if metric is not None:
        y_train = y_true[y_maskTrainIsTrue].values.squeeze()
        for i, y_pred in enumerate(m.staged_predict(X_train)):
            train_scores[i] = metric(y_train, y_pred)
    return train_scores, test_scores


def plot_deviance(fc, lag=None, split='all', model=None, 
                  metric=metrics.brier_score_loss):
    #%%
    if model is None:
        model = [n[0] for n in fc.stat_model_l if n[0][:2]=='GB'][0]
        
    if lag is None:
        lag = int(list(fc.dict_models[model].keys())[0].split('_')[1])
    
    m_splits = fc.dict_models[model][f'lag_{lag}']
    
#    assert hasattr(m_splits['split_0'], 'n_estimators'), '{}'.format(
#            m_splits['split_0'].cv_results_)
        
    if split == 'all':
        splits = [int(k.split('_')[-1]) for k in m_splits.keys()]
    else:
        splits = [split]
    
    g = sns.FacetGrid(pd.DataFrame(splits), col=0, col_wrap=2, aspect=2)
    for col, s in enumerate(splits):
        
        m_splits = fc.dict_models[model][f'lag_{lag}']
        train_score_, test_scores = get_scores_improvement(m_splits, fc, s, lag,
                                                           metric=metric)
        
        ax = g.axes[col]
        ax.plot(np.arange(train_score_.size) + 1, train_score_, 'b-',
                 label='Training Deviance')
        ax.plot(np.arange(train_score_.size) + 1, test_scores, 'r-',
                 label='Test Deviance')
        ax.legend()
        ax.set_xlabel('Boosting Iterations')
        ax.set_ylabel('Deviance')
    #%%
    return g.fig
    #%%
def get_pred_split(m_splits, fc, s, lag):

    import warnings
    warnings.filterwarnings("ignore")
    m = m_splits[f'split_{s}']
        
    if hasattr(m, 'predict_proba'):
        X_pred = m.X_pred
        y_true = fc.TV.RV_bin
        prediction = pd.DataFrame(m.predict_proba(X_pred)[:,1], 
                              index=y_true.index, columns=[lag])  
        
    else:
        y_true = fc.TV.RV_ts
        prediction = pd.DataFrame(m.predict(X_pred), 
                              index=y_true.index, columns=[lag])  
        
    
    TrainIsTrue = fc.TV.TrainIsTrue.loc[s].loc[prediction.index]
    
    pred_train = prediction.loc[TrainIsTrue[TrainIsTrue].index].squeeze()
    pred_test = prediction.loc[TrainIsTrue[~TrainIsTrue].index]
    y_train = y_true.loc[TrainIsTrue[TrainIsTrue].index]
    y_test  = y_true.loc[TrainIsTrue[~TrainIsTrue].index]
    train_score = metrics.mean_squared_error(y_train, pred_train)
    test_score  = metrics.mean_squared_error(y_test, pred_test)
    return prediction, y_true, train_score, test_score, m

def visual_analysis(fc, model=None, lag=None, split='all', col_wrap=4,
                    wspace=0.02):
    '''
    

    Parameters
    ----------
    fc : class_fc
        DESCRIPTION.
    model : str, optional
        DESCRIPTION. The default is None.
    lag : int, optional
        lag in days. The default is None.
    split : int or 'all', optional
        DESCRIPTION. The default is 'all'.
    col_wrap : int, optional
        DESCRIPTION. The default is 4.
    wspace : TYPE, optional
        DESCRIPTION. The default is 0.02.

    Returns
    -------
    None.

    '''
    #%%
    
    if model is None:
        model = list(fc.dict_models.keys())[0]
    if lag is None:
        lag = int(list(fc.dict_models[model].keys())[0].split('_')[1])
    
    m_splits = fc.dict_models[model][f'lag_{lag}']

    if split == 'all':
        s = 0
    else:
        s = split
    
    prediction, y_true, train_score, test_score, m = get_pred_split(m_splits, fc, s, lag)
    
    
    import matplotlib.dates as mdates
    import datetime
    prediction['year'] = prediction.index.year
    years = np.unique(prediction.index.year)[:]
    g = sns.FacetGrid(prediction, col='year', sharex=False,  sharey=True, 
                      col_wrap=col_wrap, aspect=1.5)
    proba = float(prediction[lag].max()) <= 1 and float(prediction[lag].min()) >= 0
        
    clim = y_true.mean()
    y_max = max(prediction[lag].max(), y_true.max().values) - clim.mean()
    y_min = min(prediction[lag].min(), y_true.min().values) + clim.mean()
    dy = max(y_max, y_min)
    
    for col, yr in enumerate(years):
        ax = g.axes[col]
        if split == 'all':
            splits = [int(k.split('_')[-1]) for k in m_splits.keys()]
            train_scores = [train_score]
            test_scores = [test_score]
            for s in splits[1:]:
                prediction, y_true, train_score, test_score, m = get_pred_split(m_splits, fc, s, lag)
                pred = prediction[(prediction.index.year==yr)][lag]
                testyrs = np.unique(fc.TrainIsTrue.loc[s][~fc.TrainIsTrue.loc[s]].index.year)
                
                    
#                else:
#                    label='_nolegend_'
                dates = pred.index
                ax.scatter(dates, pred)
                if yr in testyrs:
                    testplt = ax.plot(dates, pred, linewidth=1.5, linestyle='solid')
                else:
                    ax.plot(dates, pred, linewidth=1, linestyle='dashed')
                train_scores.append(train_score)
                test_scores.append(test_score)
            trainscorestr = 'MSE train: {:.2f} ± {:.2f}'.format(
                    np.mean(train_scores), np.std(train_scores))
            testscorestr = 'MSE test: {:.2f} ± {:.2f}'.format(
                    np.mean(test_scores), np.std(test_scores))
            text = trainscorestr+'\n'+testscorestr
        else:
            pred = prediction[(prediction['year']==yr).values][lag]
            dates = pred.index
            ax.scatter(dates, pred)
            ax.plot(dates, pred)
            trainscorestr = 'MSE train: {:.2f}'.format(train_score)
            testscorestr = 'MSE test: {:.2f}'.format(test_score)
            text = trainscorestr+'\n'+testscorestr
        ax.legend( (testplt), (['test year']))
        
        if col==0 or col == len(years)-1:
            props = dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.5)
            ax.text(0.05, 0.95, text,
                        fontsize=12,
                        bbox=props,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
        ax.plot(dates, y_true[y_true.index.year==yr], color='black')
        ax.hlines(clim, dates.min(), dates.max(), linestyle='dashed')
#        dt_years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y-%m')
        
        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        datemin = datetime.date(dates.min().year, dates.min().month, 1)
        try:
            datemax = datetime.date(dates.max().year, dates.max().month, 31)
        except:
            datemax = datetime.date(dates.max().year, dates.max().month, 30)
        ax.set_xlim(datemin, datemax)
        ax.grid(True)
        ax.tick_params(labelsize=10)
        if proba:
            ax.set_ylim(0, 1)
        else:            
            ax.set_ylim(float(clim-dy), float(clim+dy))
    
    g.fig.suptitle(model, y=1.02)
    g.fig.subplots_adjust(wspace=wspace)
        #%%
    return g.fig

def get_score_matrix(d_expers=dict, model=str, metric=str, lags_t=None, 
                     file_path=None):
    #%%
    percen = np.array(list(d_expers.keys()))
    tfreqs = np.array(list(d_expers[percen[0]].keys()))
    npscore = np.zeros( shape=(percen.size, tfreqs.size) )
    np_sig = np.zeros( shape=(percen.size, tfreqs.size), dtype=object )

    for j, pkey in enumerate(percen):
        dict_freq = d_expers[pkey]
        for k, tkey in enumerate(tfreqs):
            df_valid = dict_freq[tkey][0]
            df_metric = df_valid.loc[metric]
            npscore[j,k] = float(df_metric.loc[metric].values)
            con_low = df_metric.loc['con_low'].values
            con_high = float(df_metric.loc['con_high'].values)
            if type(con_low) is np.ndarray:
                con_low = np.quantile(con_low[0], 0.025) # alpha is 0.05
            else:
                con_low = float(df_metric.loc['con_low'].values)
            np_sig[j,k] = '{:.2f} - {:.2f}'.format(con_low, con_high)

    data = npscore
    df_data = pd.DataFrame(data, index=percen, columns=tfreqs)
    df_data = df_data.rename_axis(f'lead time: {lags_t} days', axis=1)
    df_sign = pd.DataFrame(np_sig, index=percen, columns=tfreqs)
    
    dict_of_dfs = {f'df_data_{metric}':df_data,'df_sign':df_sign}
    
    path_data = functions_pp.store_hdf_df(dict_of_dfs, file_path=file_path)
    return path_data, dict_of_dfs

def plot_score_matrix(path_data=str, col=0,
                      x_label=None, ax=None):
    #%%
    dict_of_dfs = functions_pp.load_hdf5(path_data=path_data)
    datakey = [k for k in dict_of_dfs.keys() if k[:7] == 'df_data'][0]
    metric = datakey.split('_')[-1]
    df_data = dict_of_dfs[datakey]
    df_sign = dict_of_dfs['df_sign']

    np_arr = df_sign.to_xarray().to_array().values
    np_sign = np_arr.swapaxes(0,1)
    annot = np.zeros_like(df_data.values, dtype="f8").tolist()
    for i1, r in enumerate(df_data.values):
        for i2, c in enumerate(r):
            round_val = np.round(df_data.values[i1,i2], 2).astype('f8')
            # lower confidence bootstrap higer than 0.0
            sign = np_sign[i1,i2] 

            annot[i1][i2] = '{}={:.2f} \n {}'.format(metric, 
                                                     round_val, 
                                                     sign)
    
    ax = None
    if ax==None:
        print('ax == None')
        fig, ax = plt.subplots(constrained_layout=True, figsize=(20,13))
    
    ax = sns.heatmap(df_data, ax=ax, vmin=0, vmax=round(max(df_data.max())+0.05, 1), cmap=sns.cm.rocket_r,
                     annot=np.array(annot), 
                     fmt="", cbar_kws={'label': f'{metric}'})
    ax.set_yticklabels(labels=df_data.index, rotation=0)
    ax.set_ylabel('Percentile threshold [-]')
    ax.set_xlabel(x_label)
    ax.set_title(df_data.columns.name)
    #%%
                
    return fig
    
   
def plot_score_expers(d_expers=dict, model=str, metric=str, lags_t=None,
                      color='red', style='solid', col=0,
                      x_label=None, x_label2=None, ax=None):
    #%%
    ax = None
    if ax==None:
        print('ax == None')
        fig, ax = plt.subplots(constrained_layout=True, figsize=(10,5))
    
    folds = np.array(list(d_expers.keys()))
    spread_size = 0.3
    steps = np.linspace(-spread_size,spread_size, folds.size)
#    freqs = len(d_expers.items()[])
    for i, fold_key in enumerate(folds):
        dict_freq = d_expers[fold_key]
        x_vals_freq = list(dict_freq.keys())
        x_vals = np.arange(len(x_vals_freq))
        y_vals = [] ; y_mins = [] ; y_maxs = []
        for x in x_vals_freq:
            df_valid = dict_freq[x][model][0]
            df_metric = df_valid.loc[metric]
            y_vals.append(float(df_metric.loc[metric].values))
            y_mins.append(float(df_metric.loc['con_low'].values))
            y_maxs.append(float(df_metric.loc['con_high'].values))
        
        x_vals_shift = x_vals+steps[i]

        
        ax.scatter(x_vals_shift, y_vals, color=color, linestyle=style,
                        linewidth=3, alpha=1 )
        # C.I. inteval

        ax.scatter(x_vals_shift, y_mins, s=70, marker="_", color='black')
        ax.scatter(x_vals_shift, y_maxs, s=70, marker="_", color='black')
        ax.vlines(x_vals_shift, y_mins, y_maxs, color='black', linewidth=1)
                  
        for x_t,y_t in zip(x_vals_shift, y_mins):
            ax.text(x_t, y_t-.005, f'{fold_key}', horizontalalignment='center',
                    verticalalignment='top')
        
    
        ax.hlines(y=0, xmin=min(x_vals)-spread_size, xmax=max(x_vals)+spread_size, linestyle='dotted', linewidth=0.75)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals_freq)
    
        
        if lags_t is not None:
            if np.unique(lags_t).size > 1 and i==0:
        
                ax2 = ax.twiny()
                ax2.set_xbound(ax.get_xbound())
                ax2.set_xticks(x_vals)
                ax2.set_xticklabels(lags_t)
                ax2.grid(False)
                ax2.set_xlabel(x_label2)
                text = f'Lead time varies'
                props = dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.5)
                ax.text(0.5, 0.983, text,
                        fontsize=15,
                        bbox=props,
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)
    
            if np.unique(lags_t).size == 1 and i==0:
                text = f'Lead time: {int(np.unique(lags_t))} days'
                props = dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.5)
                ax.text(0.5, 0.983, text,
                        fontsize=15,
                        bbox=props,
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)
    
    
        if metric == 'BSS':
            y_lim = (-0.4, 0.6)
        elif metric[:3] == 'AUC':
            y_lim = (0,1.0)
        elif metric == 'EDI':
            y_lim = (-1.,1.0)
        ax.set_ylim(y_lim)
        ax.set_ylabel(metric)
        ax.set_xlabel(x_label)
    ax.plot()
    #%%
    return fig

def plot_score_lags(df_metric, metric, color, lags_tf, linestyle='solid',
                    clim=None, cv_lines=False, col=0, threshold_bin=None,
                    ax=None):

    #%%
    # ax=None
    if ax==None:
        print('ax == None')
        ax = plt.subplot(111)

    if metric == 'BSS':
        y_lim = (-0.4, 0.6)
    elif metric[:3] == 'AUC':
        y_lim = (0,1.0)
    elif metric in ['Precision', 'Accuracy']:
        y_lim = (0,1)
        y_b = clim
    y = np.array(df_metric.loc[metric])
    y_min = np.array(df_metric.loc['con_low'])
    y_max = np.array(df_metric.loc['con_high'])
    if cv_lines == True:
        y_cv  = [0]


    x = lags_tf

    ax.fill_between(x, y_min, y_max, linestyle='solid',
                            edgecolor='black', facecolor=color, alpha=0.3)
    ax.plot(x, y, color=color, linestyle=linestyle,
                    linewidth=2, alpha=1 )
    ax.scatter(x, y, color=color, linestyle=linestyle,
                    linewidth=2, alpha=1 )
    if cv_lines == True:
        for f in range(y_cv.shape[1]):
            linestyle = 'loosely dotted'
            ax.plot(x, y_cv[f,:], color=color, linestyle=linestyle,
                         alpha=0.35 )
    ax.set_xlabel('Lead time [days]', fontsize=13, labelpad=0.1)
    ax.grid(b=True, which='major')    
    xticks = x

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_ylabel(metric)
    ax.set_ylim(y_lim)
    if metric == 'BSS':
        y_major = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
        ax.set_yticks(y_major, minor=False)
        ax.set_yticklabels(y_major)
        ax.set_yticks(np.arange(-0.6,0.6+1E-9, 0.1), minor=True)
        ax.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=1)
        ax.text(max(x), 0 - 0.05, 'Benchmark clim. pred.',
                horizontalalignment='right', fontsize=12,
                verticalalignment='center',
                rotation=0, rotation_mode='anchor', alpha=0.5)
    elif metric == 'AUC-ROC':
        y_b = 0.5
        ax.set_yticks(np.arange(0.5,1+1E-9, 0.1), minor=True)
        ax.hlines(y=y_b, xmin=min(x), xmax=max(x), linewidth=1)
    elif metric in ['AUC-PR', 'Precision', 'Accuracy']:
        ax.set_yticks(np.arange(0.5,1+1E-9, 0.1), minor=True)
        y_b = clim
        ax.hlines(y=y_b, xmin=min(x), xmax=max(x), linewidth=1)

    if metric in ['Precision', 'Accuracy'] and threshold_bin is not None:
        if threshold_bin == 'clim':
        # binary metrics calculated for clim prevailance
            ax.text(0.00, 0.05, r'Event pred. when fc $\geq$ clim. prob.',
                    horizontalalignment='left', fontsize=10,
                    verticalalignment='center', transform=ax.transAxes,
                    rotation=0, rotation_mode='anchor', alpha=0.5)
            # old : percentile_t = 100 * clim_prev
        elif threshold_bin == 'upper_clim':
            # binary metrics calculated for top 75% of 'above clim prob'
            ax.text(0.00, 0.05, r'Event pred. when fc$\geq$1.25 * clim. prob.',
                    horizontalalignment='left', fontsize=10,
                    verticalalignment='center', transform=ax.transAxes,
                    rotation=0, rotation_mode='anchor', alpha=0.5)
            # old: bin_threshold = 100 * (1 - 0.75*clim_prev)
            # old:  percentile_t = bin_threshold
        elif isinstance(threshold_bin, int) or isinstance(threshold_bin, float):
            if threshold_bin < 1:
                threshold_bin = int(100*threshold_bin)
            else:
                threshold_bin = threshold_bin
            ax.text(0.00, 0.05, r'Event pred. when fc$\geq${}'.format(threshold_bin),
                    horizontalalignment='left', fontsize=10,
                    verticalalignment='center', transform=ax.transAxes,
                    rotation=0, rotation_mode='anchor', alpha=0.5)
        elif isinstance(threshold_bin, tuple):
            times = threshold_bin[0]
            ax.text(0.00, 0.05, r'Event pred. when fc$\geq${} * clim. prob.'.format(times),
                    horizontalalignment='left', fontsize=10,
                    verticalalignment='center', transform=ax.transAxes,
                    rotation=0, rotation_mode='anchor', alpha=0.5)
            
    if metric in ['AUC-ROC', 'AUC-PR', 'Precision']:
        ax.text(max(x), y_b-0.05, 'Benchmark rand. pred.',
                horizontalalignment='right', fontsize=12,
                verticalalignment='center',
                rotation=0, rotation_mode='anchor', alpha=0.5)
    if col != 0 :
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)



#    str_freq = str(x).replace(' ' ,'')
    #%%
    return ax


def rel_curve_base(df_RV, lags_tf, n_bins=5, col=0, ax=None):
    #%%



    if ax==None:
        print('ax == None')
        fig, ax = plt.subplots(1, facecolor='white')

    ax.set_fc('white')

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('0.5')
    ax.grid(b=True, which = 'major', axis='both', color='black',
            linestyle='--', alpha=0.2)


    # perfect forecast
    perfect = np.arange(0,1+1E-9,(1/n_bins))
    pos_text = np.array((0.50, 0.50+0.025))
    ax.plot(perfect,perfect, color='black', alpha=0.5)
    trans_angle = plt.gca().transData.transform_angles(np.array((44.3,)),
                                                       pos_text.reshape((1, 2)))[0]
    ax.text(pos_text[0], pos_text[1], 'perfectly reliable', fontsize=14,
                   rotation=trans_angle, rotation_mode='anchor')
    obs_clim = df_RV['prob_clim'].mean()
    ax.text(obs_clim+0.2, obs_clim-0.05, 'Obs. clim',
                horizontalalignment='center', fontsize=14,
         verticalalignment='center', rotation=0, rotation_mode='anchor')
    ax.hlines(y=obs_clim, xmin=0, xmax=1, label=None, color='grey',
              linestyle='dashed')
    ax.vlines(x=obs_clim, ymin=0, ymax=1, label=None, color='grey',
              linestyle='dashed')

    # forecast clim
#    pred_clim = y_pred_all.mean().values
#    ax.vlines(x=np.mean(pred_clim), ymin=0, ymax=1, label=None)
#    ax.vlines(x=np.min(pred_clim), ymin=0, ymax=1, label=None, alpha=0.2)
#    ax.vlines(x=np.max(pred_clim), ymin=0, ymax=1, label=None, alpha=0.2)
    ax.text(np.min(obs_clim)-0.025, obs_clim.mean()+0.3, 'Obs. clim',
            horizontalalignment='center', fontsize=14,
     verticalalignment='center', rotation=90, rotation_mode='anchor')
    # resolution = reliability line
    BSS_clim_ref = perfect - obs_clim
    dist_perf = (BSS_clim_ref / 2.) + obs_clim
    x = np.arange(0,1+1E-9,1/n_bins)
    ax.plot(x, dist_perf, c='grey')
    def get_angle_xy(x, y):
        import math
        dy = np.mean(dist_perf[1:] - dist_perf[:-1])
        dx = np.mean(x[1:] - x[:-1])
        angle = np.rad2deg(math.atan(dy/dx))
        return angle
    angle = get_angle_xy(x, dist_perf)
    pos_text = (x[int(4/n_bins)], dist_perf[int(2/n_bins)]+0.04)
    trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                      np.array(pos_text).reshape((1, 2)))[0]
#    ax.text(pos_text[0], pos_text[1], 'resolution=reliability',
#            horizontalalignment='center', fontsize=14,
#     verticalalignment='center', rotation=trans_angle, rotation_mode='anchor')
    # BSS > 0 ares
    ax.fill_between(x, dist_perf, perfect, color='grey', alpha=0.5)
    ax.fill_betweenx(perfect, x, np.repeat(obs_clim, x.size),
                    color='grey', alpha=0.5)
    # Better than random
    ax.fill_between(x, dist_perf, np.repeat(obs_clim, x.size), color='grey', alpha=0.2)
    if col == 0:
        ax.set_ylabel('Observed frequency')
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel('Forecast probability')
    ax.set_ylim(-0.02,1.02)
    ax.set_xlim(-0.02,1.02)
    #%%
    return ax, n_bins
    #%%
def rel_curve(df_RV, y_pred_all, color, lags_tf, n_bins, linestyle='solid', mean_lags=True, ax=None):
    #%%

    if ax==None:
        ax, n_bins = rel_curve_base(df_RV, lags_tf)

    strategy = 'uniform' # 'quantile' or 'uniform'
    fop = [] ; mpv = []
    for l, lag in enumerate(lags_tf):

        out = calibration_curve(df_RV['RV_binary'],   y_pred_all[lag],
                                n_bins=n_bins, strategy=strategy)
        fraction_of_positives, mean_predicted_value = out
        fop.append(fraction_of_positives)
        mpv.append(mean_predicted_value)
    fop = np.array(fop)
    mpv = np.array(mpv)
    if len(fop.shape) == 2:
        # al bins are present, we can take a mean over lags
        # plot forecast
        mean_mpv = np.mean(mpv, 0) ; mean_fop = np.mean(fop, 0)
        fop_std = np.std(fop, 0)
    else:
        bins = np.arange(0,1+1E-9,1/n_bins)
        b_prev = 0
        dic_mpv = {}
        dic_fop = {}
        for i, b in enumerate(bins[1:]):

            list_mpv = []
            list_fop = []
            for i_lags, m_ in enumerate(mpv):
                m_ = list(m_)
                # if mpv falls in bin, it is added to the list, which will added to
                # the dict_mpv

                list_mpv.append([val for val in m_ if (val < b and val > b_prev)])
                list_fop.append([fop[i_lags][m_.index(val)] for idx,val in enumerate(m_) if (val < b and val > b_prev)])
            dic_mpv[i] = flatten(list_mpv)
            dic_fop[i] = flatten(list_fop)
            b_prev = b
        mean_mpv = np.zeros( (n_bins) )
        mean_fop = np.zeros( (n_bins) )
        fop_std  = np.zeros( (n_bins) )
        for k, item in dic_mpv.items():
            mean_mpv[k] = np.mean(item)
            mean_fop[k] = np.mean(dic_fop[k])
            fop_std[k]  = np.std(dic_fop[k])

    ax.plot(mean_mpv, mean_fop, color=color, linestyle=linestyle, label=None) ;

    ax.fill_between(mean_mpv, mean_fop+fop_std,
                    mean_fop-fop_std, label=None,
                    alpha=0.2, color=color) ;
    ax.set_ylim(-0.02,1.02)
    ax.set_xlim(-0.02,1.02)
    color_line = ax.lines[-1].get_c() # get color
    # determine size freq
    freq = np.histogram(y_pred_all[lag], bins=n_bins)[0]
    n_freq = freq / df_RV.index.size
    ax.scatter(mean_mpv, mean_fop, s=n_freq*2000,
               color=color_line, alpha=0.5)


    #%%
    return ax

def plot_oneyr_events(df, event_percentile, test_year):
    #%%
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)

    linestyles = ['solid', 'dashed', 'stippled']

    for i, col in enumerate(df.columns):
        if event_percentile == 'std':
            # binary time serie when T95 exceeds 1 std
            threshold = df[col].mean() + df[col].std()
        else:
            percentile = event_percentile
            threshold = np.percentile(df[col], percentile)

        testyear = df[df.index.year == test_year]
        freq = pd.Timedelta(testyear.index.values[1] - testyear.index.values[0])
        plotpaper = df.loc[pd.date_range(start=testyear.index.values[0],
                                                    end=testyear.index.values[-1],
                                                    freq=freq )]


        color = nice_colors[i]
        ax.plot_date(plotpaper.index, plotpaper[col].values, color=color,
                     linewidth=3, linestyle=linestyles[i], label=col, alpha=0.8)
        ax.axhline(y=threshold, color=color, linewidth=2 )
        if i == 0:
            label = 'Events'
        else:
            label = None
        ax.fill_between(plotpaper.index, threshold, plotpaper[col].values.squeeze(),
                         where=(plotpaper[col].values.squeeze() > threshold),
                     interpolate=True, color="crimson", label=label)
        ax.legend(fontsize='x-large', fancybox=True, facecolor='grey',
                  frameon=True, framealpha=0.3)
        ax.set_title('Timeseries and Events', fontsize=18)
        ax.set_ylabel('Temperature anomalies [K]', fontsize=15)
        ax.set_xlabel('')
    #%%
    return

def plot_events(RV, color, n_yrs = 10, col=0, ax=None):
    #%%
#    ax=None

    if type(n_yrs) == int:
        years = []
        sorted_ = RV.freq_per_year.sort_values().index
        years.append(sorted_[:int(n_yrs/2)])
        years.append(sorted_[-int(n_yrs/2):])
        years = flatten(years)
        dates_ts = []
        for y in years:
            d = RV.dates_RV[RV.dates_RV.year == y]
            dates_ts.append(d)

        dates_ts = pd.to_datetime(flatten(dates_ts))
    else:
        dates_ts = RV.RV_bin.index

    if ax==None:
        print('ax == None')
        fig, ax = plt.subplots(1, subplot_kw={'facecolor':'white'})

    else:
        ax.axes.set_facecolor('white')
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('0.5')

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    y = RV.RV_bin.loc[dates_ts]

    years = np.array(years)
    x = np.linspace(0, years.size, dates_ts.size)
    ax.bar(x, y.values.ravel(), alpha=0.75, color='silver', width=0.1, label='events')
    clim_prob = RV.RV_bin.sum() / RV.RV_bin.size
    ax.hlines(clim_prob, x[0], x[-1], label='Clim prob.')


    means = []
    for chnk in list(chunks(x, int(dates_ts.size/2.))):
        means.append( chnk.mean() )
    ax.margins(0.)
    cold_hot = means
    labels =['Lowest \nn-event years', 'Highest \nn-event years']
    ax.set_xticks(cold_hot)
    ax.set_xticklabels(labels)
    minor_ticks = np.linspace(0, x.max(), dates_ts.size)
    ax.set_xticks(minor_ticks, minor=True);
    ax.grid(b=True,which='major', axis='y', color='grey', linestyle='dotted',
            linewidth=1)
    if col == 0:
        ax.legend(facecolor='white', markerscale=2, handlelength=0.75)
        ax.set_ylabel('Probability', labelpad=-3)
        probs = [f"{int(i*100)}%" for i in np.arange(0,1+1E-9,0.2)]
        ax.set_yticklabels(probs)
    else:
        ax.tick_params(labelleft=False)


    #%%
    return ax, dates_ts

def plot_ts(RV, y_pred_all, dates_ts, color='blue', linestyle='solid', lag_i=1, ax=None):
    #%%

    if ax == None:
        ax, dates_ts = plot_events(RV, color, ax=None)

#    dates = y_pred_all.index
    n_yrs = np.unique(dates_ts.year).size
    x = np.linspace(0, n_yrs, dates_ts.size)
    y = y_pred_all.iloc[:,lag_i].loc[dates_ts]
    ax.plot(x, y.values.ravel() , linestyle=linestyle, marker=None,
            linewidth=1, color=color)
    #%%
    return ax

def plot_freq_per_yr(RV):
    #%%
    dates_RV = RV.RV_bin.index
    all_yrs = np.unique(dates_RV.year)
    freq = pd.DataFrame(data= np.zeros(all_yrs.size),
                        index = all_yrs, columns=['freq'])
    for i, yr in enumerate(all_yrs):
        oneyr = RV.RV_bin.loc[functions_pp.get_oneyr(dates_RV, yr)]
        freq.loc[yr] = oneyr.sum().values
    plt.figure( figsize=(8,6) )
    plt.bar(freq.index, freq['freq'])
    plt.ylabel('Events p/y', fontdict={'fontsize':14})
    #%%


def merge_valid_info(list_of_fc, store=True):
    dict_merge_all = {}
    for fc in list_of_fc:
        datasetname = fc.dataset.replace(' ', '__')
        expername   = fc.experiment.replace(' ', '__')
        uniq_label = datasetname+'..'+fc.stat_model[0]+'..'+expername
        dict_merge_all[uniq_label+'...df_valid'] = fc.dict_sum[0]
        dict_merge_all[uniq_label+'...df_RV'] = fc.dict_sum[1]
        dict_merge_all[uniq_label+'...y_pred_all'] = fc.dict_sum[2]
    
    if store:
        if hasattr(fc, 'filename')==False:
            fc._get_outpaths()
        functions_pp.store_hdf_df(dict_merge_all, file_path=fc.filename+'.h5')
    return dict_merge_all

def valid_figures(dict_merge_all, line_dim='model', group_line_by=None,
                  met='default', wspace=0.08, col_wrap=None):
   
    '''
    3 dims to plot: [metrics, experiments, stat_models]
    2 can be assigned to row or col, the third will be lines in the same axes.
    '''
    
    # group_line_by=None; met='default'; wspace=0.08; col_wrap=None; threshold_bin=fc.threshold_pred
    #%%
    dims = ['exper', 'models', 'met']
    col_dim = [s for s in dims if s not in [line_dim, 'met']][0]
    if met == 'default':
        met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Precision', 'Rel. Curve']
    
   
    all_expers = list(dict_merge_all.keys())
    uniq_labels = [k.split('...')[0] for k in all_expers][::3]
    dict_all = {}
    for k in uniq_labels:
        # retrieve sub dataframe of forecast
        df_valid = dict_merge_all[k+'...df_valid']
        df_RV = dict_merge_all[k+'...df_RV']
        y_pred_all = dict_merge_all[k+'...y_pred_all']
        dict_all[k] = (df_valid, df_RV, y_pred_all)
        
    comb   = list(dict_all.keys())
    
    datasets = [c.split('..')[0] for c in comb]
    models   = [c.split('..')[1] for c in comb]
    expers   = [c.split('..')[2] for c in comb]

   

    def line_col_arrangement(lines_req, option1, option2, comb):
        
        duplicates = [lines_req.count(l) != 1 for l in lines_req]
        if any(duplicates) == False:
            # not all duplicates in lines_req
            if np.unique(option1).size==1 and np.unique(option2).size!=1:
                # option1 has only 1 repeated value
                # append option2 to line, and create col for option1
                lines = [option2[i]+'..'+l for i, l in enumerate(lines_req)]
                cols = np.unique(option1)
            if np.unique(option1).size==1 and np.unique(option2).size==1:
                # option1 and option2 have both only 1 repeated value
                # create single col for this combination:
                lines = np.unique(lines_req)
                cols = [option1[i]+'..'+option2[i] for i in range(len(comb))]
                cols = np.unique(cols)
            elif np.unique(option2).size!=1 and np.unique(option2).size==1:
                lines = [option1[i]+'..'+l for i, l in enumerate(lines_req)]
                cols = np.unique(option2)
            # else:
            #     cols= list(itertools.product(np.unique(option1), 
            #                              np.unique(option2)))
            #     cols = [c[0] + '..' + c[1] for c in cols]
        elif all([lines_req.count(l) == len(lines_req) for l in lines_req]):
            # all duplicates
           cols = []
           for c in comb:
               cols.append('..'.join([p for p in c.split('..') if p not in lines_req]))
           cols = np.unique(cols)   
           lines = np.unique(lines_req)
        else:
            cols = []
            for c in comb:
                cols.append('..'.join([p for p in c.split('..') if p not in lines_req]))
            cols = np.unique(cols)
            if np.unique(option2).size > 1:
                sor = [option2.count(v) for v in option2]
                cols = [x for _,x in sorted(zip(sor,cols), reverse=False)]
            elif np.unique(option1).size > 1:
                sor = [option1.count(v) for v in option1]
                cols = [x for _,x in sorted(zip(sor,cols), reverse=False)]

            lines = np.unique(lines_req)
        return lines, cols 
        
    if line_dim == 'model':
        lines_req = models
        left = [datasets, expers]
    elif line_dim == 'exper':
        lines_req = expers
        left = [datasets, models]
    elif line_dim == 'dataset':
        lines_req = datasets
        left = [models, expers]

    
    assert line_dim in ['model', 'exper', 'dataset'], ('illegal key for line_dim, '
                           'choose \'exper\' or \'model\'')
    
    lines, cols = line_col_arrangement(lines_req, *left, comb)


    if len(cols) == 1 and group_line_by is not None:
        group_s = len(group_line_by)
        cols = group_line_by
        lines_grouped = []
        for i in range(0,len(lines),group_s):
            lines_grouped.append(lines[i:i+group_s])



    grid_data = np.zeros( (2, len(met)), dtype=str)
    grid_data = np.stack( [np.repeat(met, len(cols)),
                           np.repeat(cols, len(met))])


    df = pd.DataFrame(grid_data.T, columns=['met', col_dim])
    if len(cols) != 1 or col_wrap is None:
        g = sns.FacetGrid(df, row='met', col=col_dim, height=3, aspect=1.4,
                      sharex=False,  sharey=False)
        # Only if 1 column is requisted, col_wrap is allowed
    if len(cols) == 1 and col_wrap is not None:

        g = sns.FacetGrid(df, col='met', height=3, aspect=1.4,
                      sharex=False,  sharey=False, col_wrap=col_wrap)




    for col, c_label in enumerate(cols):

        if col_wrap is None:
            g.axes[0,col].set_title(c_label.replace('__',' ').replace('..', ' '))
        if len(models) == 1 and group_line_by is not None:
            lines = lines_grouped[col]


        for row, metric in enumerate(met):

            if col_wrap is None:
                ax = g.axes[row,col]
            else:
                ax = g.axes[row]


            for l, line in enumerate(lines):

                if line_dim == 'model' or line_dim=='dataset':
                    color = nice_colors[l]
                    # model = line
                    # exper = c_label
                    
                elif line_dim == 'exper':
                    color = colors_datasets[l]
                    # model = c_label
                    # exper = line
                    # if len(models) == 1 and group_line_by is not None:
                    #     exper = line
                    #     model = models[0]
                    
                # match_list = []
                
                string_exp = line +'..'+ c_label.replace(' ','..') 
                got_it = False ; 
                for k in comb:
                    match = np.array([i in k for i in string_exp])
                    match = np.insert(match, obj=0, 
                                      values=np.array([i in string_exp for i in k]))
                    match = match[match].size / match.size
                    # match_list.append(match)
                    # first try full match (experiments differ only in 1 dim)
                    if match==1:
                        df_valid, df_RV, y_pred_all = dict_all[k]
                        # df_RV = RV.prob_clim.merge(RV.RV_bin, left_index=True, right_index=True)
                        # print(string_exp, k, '\n', df_valid.loc['BSS'].loc['BSS'])
                        got_it = True 
                        break
                if got_it == True:
                    # if experiment not present, continue loop, but skip this
                    # string_exp

                

                    lags_tf     = y_pred_all.columns.astype(int)
    
    
                    # =========================================================
                    # # plot metrics in df_valid
                    # =========================================================
                    if metric in ['AUC-ROC', 'AUC-PR', 'BSS', 'Precision', 'Accuracy']:
                        df_metric = df_valid.loc[metric]
                        if metric in ['AUC-PR', 'Precision', 'Accuracy']:
                            RV_bin = df_RV['RV_binary']
                            clim = RV_bin.values[RV_bin==1].size / RV_bin.size
                            if metric == 'Accuracy':
                                import validation as valid
                                # threshold upper 3/4 of above clim
                                threshold = int(100 * (1 - 0.75*clim))
                                f_ = valid.get_metrics_confusion_matrix
                                df_ran = f_(RV_bin.squeeze().values, 
                                            y_pred_all.loc[:,:0],
                                            thr=[threshold], n_shuffle=400)
                                clim = df_ran[threshold/100]['fc shuf'].loc[:,'Accuracy'].mean()
    
                        else:
                            clim = None
                        plot_score_lags(df_metric, metric, color, lags_tf,
                                        linestyle=line_styles[l], clim=clim,
                                        cv_lines=False, col=col, 
                                        threshold_bin=df_valid.index.name, ax=ax)
                    # =========================================================
                    # # plot reliability curve
                    # =========================================================
                    if metric == 'Rel. Curve':
                        if l == 0:
                            ax, n_bins = rel_curve_base(df_RV, lags_tf, col=col, ax=ax)
                        # print(l,line)
    
                        rel_curve(df_RV, y_pred_all, color, lags_tf, n_bins,
                                  linestyle=line_styles[l], mean_lags=True,
                                  ax=ax)
    
                    # if metric == 'ts':
                    #     if l == 0:
                    #         ax, dates_ts = plot_events(RV, color=nice_colors[-1], n_yrs=6,
                    #                          col=col, ax=ax)
                    #     plot_ts(RV, y_pred_all, dates_ts, color, line_styles[l], lag_i=lags_tf[0], ax=ax)
    
                    # legend conditions
                    same_models = all([row==0, col==0, line==lines[-1]])
                    grouped_lines = np.logical_and(row==0, group_line_by is not None)
                    if same_models or grouped_lines:
                        ax.legend(ax.lines, lines,
                              loc = 'lower left', fancybox=True,
                              handletextpad = 0.2, markerscale=0.1,
                              borderaxespad = 0.1,
                              handlelength=2.5, handleheight=1, prop={'size': 12})

    #%%
    g.fig.subplots_adjust(wspace=wspace)

    return g.fig


