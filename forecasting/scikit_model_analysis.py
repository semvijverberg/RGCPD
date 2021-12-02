#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:12:11 2021

@author: semvijverberg
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
import func_models as fc_utils
from typing import Union

nice_colors = ['#EE6666', '#3388BB', '#9988DD', '#EECC55',
                '#88BB44', '#FFBBBB']
line_styles = ['-', '--', '-.', ':', '']

cl_combs = np.array(np.meshgrid(line_styles, nice_colors),
                     dtype=object).T.reshape(-1,2)

def ensemble_error_estimators(model, kwrgs_model: dict,
                                     min_estimators=15,
                                     max_estimators=200,
                                     steps=5):
    '''
    Takes in best_estimator_ that has already been fitted.
    best_estimator_ instance should contain df_norm (i.e., DataFrame with
    x_fit and x_fit masks).
    '''
    #%%
    from collections import OrderedDict

    x_train_mask, y_fit_mask = fc_utils.get_masks(model.df_norm)[:2]
    x_fit = model.df_norm[model.df_norm.columns[:-5]][x_train_mask]
    y_fit = model.target[y_fit_mask.values]

    if 'n_estimators' in kwrgs_model.keys():
        max_estimators = kwrgs_model.pop('n_estimators')
        if type(max_estimators) is list: max_estimators = max(max_estimators)

    kwrgs_gridsearch = {k:i for k, i in kwrgs_model.items() if type(i) == list}
    kwrgs = kwrgs_model.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]

    if 'scoringCV' in kwrgs.keys():
        kwrgs.pop('scoringCV')

    # get CV args
    kwrgs_cv = ['kfold', 'seed']
    kwrgs_cv = {k:i for k, i in kwrgs.items() if k in kwrgs_cv}
    [kwrgs.pop(k) for k in kwrgs_cv.keys()]

    name = model.__str__().split('(')[0]
    model.set_params(**kwrgs) # set static params
    # get params to 'search'
    vals = [list(v) for v in kwrgs_gridsearch.values()]
    keys = list(kwrgs_gridsearch.keys()) # keys arguments
    combs = np.array(np.meshgrid(*[v for v in vals]),
                     dtype=object).T.reshape(-1,len(keys))
    params = [dict(zip(keys, c)) for c in combs]


    models = [(f'{name} {str(p)}', p) for p in params]

    error_rate = OrderedDict((label, []) for label, _ in models)
    for label, p in models:
        for i in np.arange(min_estimators, max_estimators + 1, 5):
            model.set_params(n_estimators=i, **p)
            model.fit(x_fit.dropna(axis=1), y_fit.values.ravel())
            oob_error = 1 - model.oob_score_
            error_rate[label].append((i, oob_error))

    f, ax = plt.subplots(1)
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for i, (label, clf_err) in enumerate(error_rate.items()):
        xs, ys = zip(*clf_err)
        ax.plot(xs, ys, label=label, c=cl_combs[i][1], ls=cl_combs[i][0])

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error score")
    plt.legend(bbox_to_anchor=(1.0, .5), fontsize=10)
    plt.show()
    return f

def GridSearch_summary_xlxs(models_lags, filename):
    '''

    Parameters
    ----------
    models_lags : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    engine : .xlsx or .csv, optional
        DESCRIPTION. The default is '.csv'.

    Returns
    -------
    None.

    '''
    #%%
    lag_keys = list(models_lags.keys())
    split_keys = list(models_lags[lag_keys[0]].keys())
    if filename.endswith('.csv'):
        filename = filename[:-4]
    first_order = ['mean_test_score', 'mean_train_score']
    not_interesting = ['std_fit_time', 'mean_score_time', 'std_score_time']
    for lag_key in lag_keys:
        dfs_lag = []
        for split_key in split_keys:

            gs = models_lags[lag_key][split_key]
            paramcols = list(gs.param_grid.keys())
            dataparams = np.array(np.meshgrid(*list(gs.param_grid.values()))).\
                            T.reshape(-1,len(paramcols))
            df_p = pd.DataFrame(dataparams, columns=paramcols)
            df_r = pd.DataFrame({**gs.cv_results_})
            metrics = list(gs.cv_results_.keys())
            metrics = [m for m in metrics if m not in ['params', 'param_max_depth']]
            metrics = [m for m in metrics if m not in not_interesting]
            metrics = first_order + [m for m in metrics if m not in first_order]
            dfs_lag.append(pd.concat([df_p, df_r[metrics]], axis=1))
        dfs = pd.concat(dfs_lag, keys=split_keys)
        df_mean = pd.concat([dfs.mean(0, level=1)], keys=['mean'])
        df_std = df_p.merge(dfs[metrics].std(0, level=1),
                                left_index=True, right_index=True)
        df_std = pd.concat([df_std],
                          keys=['std'])
        df = pd.concat([df_mean, df_std, dfs])
        subfilename = filename + f'_{lag_key}'
        df.to_csv(subfilename + '.csv')

    # if engine == '.csv':
    #     writer = ExcelWriter(filename + engine)
    #     folds = [f[0] for f in df.index[::df.index.levels[1].size]]
    #     for fold in folds:
    #         sheet_name = 'l{}_s{}'.format(lag_key.split('_')[-1],
    #                                    split_key.split('_')[-1])
    #         df.loc[fold].to_excel(writer, sheet_name)
    #     writer.save()




