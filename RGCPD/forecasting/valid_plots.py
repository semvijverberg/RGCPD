#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:15:50 2019

@author: semvijverberg
"""

from itertools import chain, permutations, product

#import matplotlib.patches as patches
#from sklearn import metrics
from .. import functions_pp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve

flatten = lambda l: list(chain.from_iterable(l))

import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import cycler
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

nice_colors = ['#EE6666', '#3388BB', '#9988DD', '#EECC55',
                '#88BB44', '#FFBBBB']
colors_nice = cycler('color',
                nice_colors)
colors_datasets = [np.array(c) for c in sns.color_palette('deep')]

# line_styles = ['solid', 'dashed', (0, (3, 5, 1, 5, 1, 5)), 'dotted']
line_styles = ['-', '--', '-.', ':', '']
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
    from . import stat_models
    import warnings
    warnings.filterwarnings("ignore")
    m = m_splits[f'split_{s}']

#    assert hasattr(m, 'n_estimators'), print(m.cv_results_)


#    x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = stat_models.get_masks(m.X)

#    keys = m.X.columns[m.X.dtypes != bool]
    X_pred = m.X_pred
    TrainIsTrue = m.df_norm['TrainIsTrue']
    if hasattr(m, 'n_estimators')==False:
        m = m.best_estimator_

    if hasattr(m, 'predict_proba'):
        y_true = fc.TV.RV_bin
    else:
        y_true = fc.TV.RV_ts



    X_train = X_pred[TrainIsTrue.loc[X_pred.index]]
#    X_test = pd.to_datetime([d for d in X_pred.index if d not in X_train[X_train].index])
    X_test = X_pred[~TrainIsTrue.loc[X_pred.index]]
    y_maskTrainIsTrue = fc.TV.TrainIsTrue.loc[s].loc[fc.TV.dates_RV]
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

def visual_analysis(fc, model=None, lag=None, split='all', col_wrap=5,
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


    import datetime

    import matplotlib.dates as mdates
    prediction['year'] = prediction.index.year
    years = np.unique(prediction.index.year)[:]
    g = sns.FacetGrid(prediction, col='year', sharex=False,  sharey=True,
                      col_wrap=col_wrap, aspect=1.5, size=1.5)
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
            for s in splits[:]:
                prediction, y_true, train_score, test_score, m = get_pred_split(m_splits, fc, s, lag)
                pred = prediction[(prediction.index.year==yr)][lag]
                testyrs = np.unique(fc.TrainIsTrue.loc[s][~fc.TrainIsTrue.loc[s]].index.year)
                dates = pred.index

                if yr in testyrs:
                    testplt = ax.plot(dates, pred, linewidth=1.5, linestyle='solid',
                                      color='red')
                    ax.scatter(dates, pred, color='red', s=8)
                else:
                    ax.plot(dates, pred, linewidth=1, linestyle='dashed',
                            color='grey')
                    ax.scatter(dates, pred, color='grey', s=2)
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
        ax.legend( (testplt), (['test year']), fontsize='x-small')

        if col==0 or col == len(years)-1:
            props = dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.5)
            ax.text(0.05, 0.95, text,
                        fontsize=8,
                        bbox=props,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
        ax.plot(dates, y_true[y_true.index.year==yr], color='black')
        ax.hlines(clim, dates.min(), dates.max(), linestyle='dashed',
                  linewidth=1.5)
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

    g.fig.suptitle(model + f' lag {lag}', y=1.0)
    g.fig.subplots_adjust(wspace=wspace)
        #%%
    return g.fig

def get_score_matrix(d_expers=dict, metric=str, lags_t=None,
                     file_path=None):
    #%%
    percen = np.array(list(d_expers.keys()))
    tfreqs = np.array(list(d_expers[percen[0]].keys()))
    folds  = np.array(list(d_expers[percen[0]][tfreqs[0]].keys()))
    npscore = np.zeros( shape=(folds.size, percen.size, tfreqs.size) )
    np_sig = np.zeros( shape=(folds.size, percen.size, tfreqs.size),
                      dtype=object )

    for j, pkey in enumerate(percen):
        dict_freq = d_expers[pkey]
        for k, tkey in enumerate(tfreqs):
            for i, fold in enumerate(folds):
                df_valid = dict_freq[tkey][fold][0]
                df_metric = df_valid.loc[metric]
                mean = float(df_metric.loc[metric].values)
                CI   = (df_metric.loc['con_low'].values,
                        float(df_metric.loc['con_high'].values))
                npscore[i,j,k] = mean
                con_low = CI[0]
                con_high = CI[1]
                if type(con_low) is np.ndarray:
                    con_low = np.quantile(con_low[0], 0.025) # alpha is 0.05
                else:
                    con_low = float(con_low)
                np_sig[i,j,k] = '{:.2f} - {:.2f}'.format(con_low, con_high)

    data = npscore
    index = pd.MultiIndex.from_product([folds, percen],names=['fold', 'percen'])

    df_data = pd.DataFrame(data.reshape(-1, len(tfreqs)), index=index,
                           columns=tfreqs)
    df_data = df_data.rename_axis(f'lead time: {lags_t} days', axis=1)
    df_sign = pd.DataFrame(np_sig.reshape(-1, len(tfreqs)),
                           index=index, columns=tfreqs)

    dict_of_dfs = {f'df_data_{metric}_{lags_t}':df_data,'df_sign':df_sign}

    path_data = functions_pp.store_hdf_df(dict_of_dfs, file_path=file_path)
    #%%
    return path_data, dict_of_dfs

def plot_score_matrix(path_data=str, x_label=None, ax=None):

    #%%
    dict_of_dfs = functions_pp.load_hdf5(path_data=path_data)
    datakey = [k for k in dict_of_dfs.keys() if k[:7] == 'df_data'][0]
    metric = datakey.split('_')[-2]
    df_data = dict_of_dfs[datakey]
    df_sign = dict_of_dfs['df_sign']
    first_fold = df_data.index.get_level_values(0)[0]

    df_data = df_data.xs(first_fold, level='fold')
    df_sign = df_sign.xs(first_fold, level='fold')

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
        h = 4 * df_data.index.get_level_values(1).size
        fig, ax = plt.subplots(constrained_layout=True, figsize=(20,h))
    df_data = df_data.sort_index(axis=0, level=1, ascending=False)
    ax = sns.heatmap(df_data, ax=ax, vmin=0, vmax=round(max(df_data.max())+0.05, 1), cmap=sns.cm.rocket_r,
                     annot=np.array(annot),
                     fmt="", cbar_kws={'label': f'{metric}'})
    ax.set_yticklabels(labels=df_data.index, rotation=0)
    ax.set_ylabel('Percentile threshold [-]')
    ax.set_xlabel(x_label)
    ax.set_title(df_data.columns.name)
    #%%

    return fig


def plot_score_expers(path_data=str, x_label=None):


    #%%

    dict_of_dfs = functions_pp.load_hdf5(path_data=path_data)
    datakey = [k for k in dict_of_dfs.keys() if k[:7] == 'df_data'][0]
    metric  = datakey.split('_')[-2]
    lag     = datakey.split('_')[-1]
    df_data = dict_of_dfs[datakey]

    mean = df_data.mean(axis=0, level=1)
    y_min = df_data.min(axis=0, level=1)
    y_max = df_data.max(axis=0, level=1)

    # index to rows in FacetGrid
    grid_data = np.stack( [np.repeat(mean.index[:,None], 1),
                           np.repeat(1, mean.index.size)])
    df = pd.DataFrame(grid_data.T, columns=['index', 'None'])
    g = sns.FacetGrid(df, row='index', height=3, aspect=3.5,
                      sharex=False,  sharey=False)
    color='red'
    for r, row_label in enumerate(mean.index):
        ax = g.axes[r,0]
        df_freq = mean.loc[row_label]
        df_min  = y_min.loc[row_label]
        df_max  = y_max.loc[row_label]

        for f in np.unique(df_data.index.get_level_values(0)):
            ax.scatter(df_freq.index, df_data.loc[f].loc[row_label].values,
                       color='black', marker="_", s=70,
                       alpha=.3 )

        ax.scatter(df_freq.index, df_freq.values, s=90, marker="_",
                   color=color, alpha=1 )

        ax.scatter(df_freq.index, df_min.values, s=90,
                   marker="_", color='black')
        ax.scatter(df_freq.index, df_max.values, s=90,
                   marker="_", color='black')
        # ax.vlines(df_freq.index, df_min.values, df_max.values, color='black', linewidth=1)


        if r == 0:
            text = f'Lead time: {int(np.unique(lag))} days'
            props = dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.5)
            ax.text(0.5, 1.05, text,
                    fontsize=14,
                    bbox=props,
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes)
        if row_label == 'std':
            text = '+1 std (~84th percentile)'
        else:
            text = f'{row_label}th percentile'
        props = dict(boxstyle='round', facecolor=None, edgecolor='black', alpha=0.5)
        ax.text(0.015, .98, text,
                fontsize=10,
                bbox=props,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
        if metric == 'BSS':
            y_lim = (-0.4, 0.4)
        elif metric[:3] == 'AUC':
            y_lim = (0,1.0)
        elif metric == 'EDI':
            y_lim = (-1.,1.0)
        ax.set_ylim(y_lim)
        ax.set_ylabel(metric, labelpad=-2)
        if r == mean.index.size-1:
            ax.set_xlabel(x_label, labelpad=4)

    #%%
    return g.fig

def plot_score_lags(df_metric, metric, color, lags_tf, linestyle='solid',
                    clim=None, cv_lines=False, col=0, threshold_bin=None,
                    fontbase=12, ax=None):

    #%%
    # ax=None
    if ax is None:
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

    ax.fill_between(x, y_min, y_max,
                    edgecolor=color, facecolor=color, alpha=0.3,
                    linestyle=linestyle, linewidth=2)
    ax.plot(x, y, color=color, linestyle=linestyle,
                    linewidth=2, alpha=1 )
    ax.scatter(x, y, color=color, linestyle=linestyle,
                    linewidth=2, alpha=1 )
    if cv_lines == True:
        for f in range(y_cv.shape[1]):
            linestyle = 'loosely dotted'
            ax.plot(x, y_cv[f,:], color=color, linestyle=linestyle,
                         alpha=0.35 )
    ax.set_xlabel('Lead time [days]', fontsize=fontbase, labelpad=4)
    ax.grid(b=True, which='major')
    xticks = x

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.tick_params(labelsize=fontbase-2)
    ax.set_ylabel(metric, fontsize=fontbase)
    ax.set_ylim(y_lim)
    if metric == 'BSS':
        y_major = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
        ax.set_yticks(y_major, minor=False)
        ax.set_yticklabels(y_major)
        ax.set_yticks(np.arange(-0.6,0.6+1E-9, 0.1), minor=True)
        ax.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=1)
        ax.text(max(x), 0 - 0.05, 'Benchmark clim. pred.',
                horizontalalignment='right', fontsize=fontbase,
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
        with_numbers = any(char.isdigit() for char in threshold_bin)
        if with_numbers == False:
            # it is a string
            if 'clim' in threshold_bin:
            # binary metrics calculated for clim prevailance
                ax.text(0.00, 0.05, r'Event pred. when fc $\geq$ clim. prob.',
                        horizontalalignment='left', fontsize=fontbase-2,
                        verticalalignment='center', transform=ax.transAxes,
                        rotation=0, rotation_mode='anchor', alpha=0.5)
            elif 'upper_clim' in threshold_bin:
                # binary metrics calculated for top 75% of 'above clim prob'
                ax.text(0.00, 0.05, r'Event pred. when fc$\geq$1.25 * clim. prob.',
                        horizontalalignment='left', fontsize=fontbase-2,
                        verticalalignment='center', transform=ax.transAxes,
                        rotation=0, rotation_mode='anchor', alpha=0.5)
            # old: bin_threshold = 100 * (1 - 0.75*clim_prev)
            # old:  percentile_t = bin_threshold

        elif '(' in threshold_bin and with_numbers:
            # dealing with tuple, deciphering..
            times = float(threshold_bin.split('(')[1].split(',')[0])
            ax.text(0.00, 0.05, r'Event pred. when fc$\geq${} * clim. prob.'.format(times),
                    horizontalalignment='left', fontsize=fontbase-2,
                    verticalalignment='center', transform=ax.transAxes,
                    rotation=0, rotation_mode='anchor', alpha=0.5)
        elif with_numbers:
            threshold_bin = threshold_bin.replace('b', '').replace('\'', '')
            if float(threshold_bin) < 1:
                threshold_bin = int(100*threshold_bin)
            else:
                threshold_bin = int(threshold_bin)
            ax.text(0.00, 0.05, r'Event pred. when fc$\geq${}%'.format(threshold_bin),
                    horizontalalignment='left', fontsize=fontbase-2,
                    verticalalignment='center', transform=ax.transAxes,
                    rotation=0, rotation_mode='anchor', alpha=0.5)

    if metric in ['AUC-ROC', 'AUC-PR', 'Precision', 'Accuracy']:
        ax.text(max(x), y_b-0.05, 'Benchmark rand. pred.',
                horizontalalignment='right', fontsize=fontbase,
                verticalalignment='center',
                rotation=0, rotation_mode='anchor', alpha=0.5)
    if col != 0 :
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)



#    str_freq = str(x).replace(' ' ,'')
    #%%
    return ax


def rel_curve_base(df_RV, n_bins=10, col=0, fontbase=12, ax=None):
    #%%

    # ax=None

    if ax==None:
        print('ax == None')
        fig, ax = plt.subplots(1, facecolor='white')
        ax.set_xticklabels(['']*n_bins)
        ax.set_fc('white')

    divider = make_axes_locatable(ax)
    axhist = divider.append_axes("bottom", size="50%", pad=0.25)
    axhist.set_fc('white')
    ax.set_fc('white')

    ax.patch.set_edgecolor('black')
    axhist.patch.set_edgecolor('black')
    ax.patch.set_linewidth('0.5')
    axhist.patch.set_linewidth('0.5')
    ax.grid(b=True, which = 'major', axis='both', color='black',
            linestyle='--', alpha=0.2)


    # perfect forecast
    perfect = np.arange(0,1+1E-9,(1/n_bins))
    pos_text = np.array((0.42, 0.42+0.025))
    ax.plot(perfect,perfect, color='black', alpha=0.5)
    trans_angle = plt.gca().transData.transform_angles(np.array((30.,)),
                                                       pos_text.reshape((1, 2)))[0]
    ax.text(pos_text[0], pos_text[1], 'perfectly reliable', fontsize=fontbase,
                   rotation=trans_angle, rotation_mode='anchor')

    obs_clim = df_RV['prob_clim'].mean()
    ax.text(obs_clim+0.2, obs_clim-0.09, 'Obs. clim',
                horizontalalignment='center', fontsize=fontbase-2,
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
    ax.text(np.min(obs_clim)-0.025, obs_clim.mean()+0.32, 'Obs. clim',
            horizontalalignment='center', fontsize=fontbase-2,
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
        # hardcode adapting code due to splitting axes, pythagoras fails
        return angle
    angle = get_angle_xy(x, dist_perf)
    pos_text = (x[int(4/n_bins)], dist_perf[int(2/n_bins)]+0.04)
    trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                      np.array(pos_text).reshape((1, 2)))[0]

    # BSS > 0 area
    ax.fill_between(x, dist_perf, perfect, color='grey', alpha=0.5)
    ax.fill_betweenx(perfect, x, np.repeat(obs_clim, x.size),
                    color='grey', alpha=0.5)
    # Better than random
    ax.fill_between(x, dist_perf, np.repeat(obs_clim, x.size), color='grey', alpha=0.2)
    if col == 0:
        ax.set_ylabel('Obs. frequencies', fontsize=fontbase)
        axhist.set_ylabel('Count', labelpad=15, fontsize=fontbase)
    else:
        ax.tick_params(labelleft=False)
        axhist.tick_params(labelleft=False)
    ax.set_ylim(-0.02,1.02)
    ax.set_xlim(-0.02,1.02)

    #%%
    return [ax, axhist], n_bins
    #%%
def rel_curve(df_RV, y_pred_all, lags_relcurve, n_bins, color, line_style=None,
              legend='single', fontbase=12, ax=None):

    #%%
    # ax=None
    if ax==None:
        axes, n_bins = rel_curve_base(df_RV)
        ax, axhist = axes
    else:
        ax, axhist = ax
        ax.set_xticklabels(['']*n_bins)


    strategy = 'uniform' # 'quantile' or 'uniform'
    fop = [] ; mpv = []
    for l, lag in enumerate(lags_relcurve):

        out = calibration_curve(df_RV['RV_binary'],   y_pred_all[lag],
                                n_bins=n_bins, strategy=strategy)
        fraction_of_positives, mean_predicted_value = out
        fop.append(fraction_of_positives)
        mpv.append(mean_predicted_value)
    fop = np.array(fop)
    mpv = np.array(mpv)


    for l, lag in enumerate(lags_relcurve):
        if len(lags_relcurve) > 1:
            line_style = line_styles[l]
        ax.plot(mpv[l], fop[l], color=color, linestyle=line_style,
                label=f'lag {lag}', marker='s', markersize=3)
        # print(line_styles)
        # print(l)
        # print(line_styles[l])
        axhist.hist(y_pred_all[lag], range=(0,1), bins=n_bins, color=color,
                    histtype="step", density=True, linestyle=line_style, label=None)

        ## Normalized to 1
        # bin_height,bin_boundary = np.histogram(y_pred_all[lag],bins=2*n_bins)
        # #define width of each column
        # width = bin_boundary[1]-bin_boundary[0]
        # #standardize each column by dividing with the maximum height
        # bin_height = bin_height/float(max(bin_height))
        # #plot
        # axhist.bar(bin_boundary[:-1],bin_height,width = width, edgecolor=color,
        #            linestyle=line_style, label=None, fill=False, linewidth=1)

        axhist.set_xlim(-0.02,1.02)
    if legend == 'single':
        color_leg = color
    else:
        color_leg = 'grey'

    lines = [line_styles[l] for l in range(len(lags_relcurve))]
    lines = [Line2D([0], [0], color=color_leg, linewidth=2, linestyle=l) for l in lines]
    ax.legend(lines, [f'lag {lag}' for lag in lags_relcurve], fontsize=fontbase-3)

    # axhist.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    # axhist.tick_params(axis='y', labelsize=8)
    axhist.ticklabel_format(style='sci', scilimits=(0, 1), useMathText=True)
    axhist.yaxis.offsetText.set_fontsize(fontbase-4)
    ax.tick_params(labelsize=fontbase-2)

    ax.set_ylim(-0.02,1.02)
    ax.set_xlim(-0.02,1.02)
    axhist.set_xlabel('Forecast probabilities', fontsize=fontbase,
                      labelpad=2)




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
                     interpolate=True, color=color, label=label, alpha=.5)
        ax.legend(fontsize='x-large', fancybox=True, facecolor='grey',
                  frameon=True, framealpha=0.3)
        ax.set_title('Timeseries and Events', fontsize=18)
        ax.set_ylabel('Temperature anomalies [K]', fontsize=15)
        ax.set_xlabel('')
        ax.tick_params(labelsize=15)
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

def plot_freq_per_yr(RV_bin):
    #%%
    dates_RV = RV_bin.index
    all_yrs = np.unique(dates_RV.year)
    freq = pd.DataFrame(data= np.zeros(all_yrs.size),
                        index = all_yrs, columns=['freq'])
    for i, yr in enumerate(all_yrs):
        oneyr = RV_bin.loc[functions_pp.get_oneyr(dates_RV, yr)]
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
        fc = list_of_fc[0]
        if hasattr(fc, 'pathexper')==False:
            fc._get_outpaths()
        functions_pp.store_hdf_df(dict_merge_all, file_path=fc.pathexper+'/data.h5')
    return dict_merge_all

def valid_figures(dict_merge_all, line_dim='model', group_line_by=None,
                  met='default', wspace=0.08, hspace=0.25, col_wrap=None,
                  skip_redundant_title=False,figaspect=1.4, lines_legend=None,
                  lags_relcurve: list=None, fontbase=12):

    '''
    3 dims to plot: [metrics, experiments, stat_models]
    2 can be assigned to row or col, the third will be lines in the same axes.
    '''

    # group_line_by=None; met='default'; wspace=0.08; col_wrap=None; skip_redundant_title=True; lags_relcurve=None; figaspect=2; fontbase=12; lines_legend=None
    #%%
    dims = ['exper', 'models', 'met']
    col_dim = [s for s in dims if s not in [line_dim, 'met']][0]
    if met == 'default':
        met = ['AUC-ROC', 'BSS', 'Rel. Curve', 'Precision']


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
    dims     = [datasets, models, expers]



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


    if line_dim is not None:
        # compare different lines by:
        assert line_dim in ['model', 'exper', 'dataset'], ('illegal key for line_dim, '
                               'choose \'exper\' or \'model\'')

        if line_dim == 'model':
            lines_req = models
            left = [datasets, expers]
        elif line_dim == 'exper':
            lines_req = expers
            left = [datasets, models]
        elif line_dim == 'dataset':
            lines_req = datasets
            left = [models, expers]


        lines, cols = line_col_arrangement(lines_req, *left, comb)
        cols = cols[::-1]

    elif group_line_by is not None:
        # group lines with same dimension {group_line_by}:
        assert group_line_by in ['model', 'exper', 'dataset'], ('illegal key for line_dim, '
                               'choose \'exper\' or \'model\'')

        if group_line_by == 'model':
            cols_req = models
            left = [datasets, expers]
        elif group_line_by == 'exper':
            cols_req = expers
            left = [datasets, models]
        elif group_line_by == 'dataset':
            cols_req = datasets
            left = [models, expers]
        group_s = np.unique(cols_req).size
        lines_per_group = int(len(comb) / group_s)
        cols = np.unique(cols_req)
        lines_grouped = []
        for i, gs in enumerate(range(0,len(comb),lines_per_group)):
            lines_full = comb[gs:gs+lines_per_group]
            lines_col = [l.replace(cols[i], '') for l in lines_full]
            lines_grouped.append(lines_col)





    grid_data = np.zeros( (2, len(met)), dtype=str)
    grid_data = np.stack( [np.repeat(met, len(cols)),
                           np.repeat(cols, len(met))])


    df = pd.DataFrame(grid_data.T, columns=['met', col_dim])
    if len(cols) != 1 or col_wrap is None:
        g = sns.FacetGrid(df, row='met', col=col_dim, height=3, aspect=figaspect,
                      sharex=False,  sharey=False)
        # Only if 1 column is requisted, col_wrap is allowed
    if len(cols) == 1 and col_wrap is not None:

        g = sns.FacetGrid(df, col='met', height=3, aspect=figaspect,
                      sharex=False,  sharey=False, col_wrap=col_wrap)


    def style_label(dims, label, all_labels, skip_redundant_title):
        if skip_redundant_title and len(all_labels) > 1:
            # remove string if it is always the same, i.e. redundant
            for dim in dims:
                if np.unique(dim).size == 1:
                    d = dim[0]
                    if d in label:
                        label = label.replace(d, '')
        if label[:2] == '..':
            label = label.replace(label[:2], '')
        if label[-2:] == '..':
            label = label.replace(label[-2:], '')
        label = label.replace('__',' ').replace('..', ' ')
        return label.replace('__',' ').replace('..', ' ')

    for col, c_label in enumerate(cols):

        if col_wrap is None:

            styled_col_label = style_label(dims, c_label, cols, skip_redundant_title)

            g.axes[0,col].set_title(styled_col_label)

        if group_line_by is not None:
            lines = lines_grouped[col]


        for row, metric in enumerate(met):

            if col_wrap is None:
                ax = g.axes[row,col]
            else:
                ax = g.axes[row]


            for l, line in enumerate(lines):

                if line_dim == 'model' or line_dim=='dataset':
                    color = nice_colors[l]
                elif line_dim == 'exper':
                    color = colors_datasets[l]
                elif group_line_by is not None:
                    color = nice_colors[l]



                string_exp = line +'..'+ c_label.replace(' ','..')
                diff_order = list(permutations(string_exp.split('..'), 3))
                diff_order = ['..'.join(sublist) for sublist in diff_order]
                got_it = False ;
                for av, req in product(comb,diff_order):
                    if av == req:
                        # print(av,string_exp)
                        df_valid, df_RV, y_pred_all = dict_all[av]
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
                                from . import validation as valid

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
                                        threshold_bin=df_valid.index.name,
                                        fontbase=fontbase, ax=ax)
                    # =========================================================
                    # # plot reliability curve
                    # =========================================================
                    if metric == 'Rel. Curve':
                        if l == 0:
                            doubleax, n_bins = rel_curve_base(df_RV, col=col,
                                                              fontbase=fontbase,
                                                              ax=ax)
                        if len(lines) > 1:
                            legend = 'multiple'
                        else:
                            legend = 'single'
                        if lags_relcurve is None:
                            lags_relcurve = [lags_tf[int(lags_tf.size/2)]]
                        rel_curve(df_RV, y_pred_all, lags_relcurve, n_bins,
                                  color=color, line_style=line_styles[l],
                                  legend=legend, fontbase=fontbase, ax=doubleax)


                    # legend conditions
                    if lines_legend is None:
                        lines_leg = [style_label(dims, l, lines, skip_redundant_title) for l in lines]
                    else:
                        lines_leg = lines_legend[col]
                    same_models = all([row==0, col==0, line==lines[-1]])
                    grouped_lines = np.logical_and(row==0, group_line_by is not None)
                    if same_models or grouped_lines:
                        ax.legend(ax.lines, lines_leg,
                              loc = 'lower left', fancybox=True,
                              handletextpad = 0.2, markerscale=0.1,
                              borderaxespad = 0.1,
                              handlelength=2.5, handleheight=1, prop={'size': 12})

    #%%
    g.fig.subplots_adjust(wspace=wspace, hspace=hspace)

    return g.fig
