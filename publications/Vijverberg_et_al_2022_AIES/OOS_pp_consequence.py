#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:52:40 2022

@author: semvijverberg
"""
import os, inspect, sys
if '/Users/semvijverberg/surfdrive/Scripts/jannes_code/RGCPD' in sys.path:
    sys.path.remove('/Users/semvijverberg/surfdrive/Scripts/jannes_code/RGCPD')
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
    n_cpu = 8
else:
    n_cpu = 3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import argparse
# from matplotlib.lines import Line2D
# import csv
# import re

user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/Vijverberg_et_al_2022_AIES/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
if main_dir not in sys.path:
    sys.path.append(main_dir)

import utils_paper3

path_main = '/Users/semvijverberg/Desktop/cluster/surfdrive/output_paper3'
standard_subfolder = 'a9943/USDA_Soy_clusters__1/timeseriessplit_30/s1'
model_name = 'LogisticRegression'
fc_month = 'April'


fc_types = [0.31, 0.33, 0.35]
fc_type = fc_types[0]

path_no_OOS = f'test_no_areaw_oosT_False/{standard_subfolder}/df_data_fctpFalse_all'
path_OOS_no_q = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctpFalse_all'
path_OOS = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctp_T_all'


def plot_table(subtitle, path, ax):

    df_scores = []
    for fc_type in fc_types:
        orig_fc_type = path.split('df_data_')[1][:4]
        path = path.replace(orig_fc_type, str(fc_type))
        target_options = [['Target', 'Target | PPS']]
        out = utils_paper3.load_scores(target_options[0], model_name, model_name, 2000,
                                       path, condition=['strong 30%'])
        df_scores_, df_boots, df_preds = out

        # correct order ['All, '50%', '30%']
        df_scores_ = [df_scores_[i] for i in [0,1]]
        df_test_m = [d[fc_month] for d in df_scores_]
        table_col = [d.index[0] for d in df_test_m][1:]
        order_verif = [f'All (q={fc_type})'] + ['Top '+t.split(' ')[1] + f' (q={fc_type})' for t in table_col]
        for i, df_test_skill in enumerate(df_test_m):
            r = {df_test_skill.index[0]:order_verif[i]}
            df_scores.append( df_test_skill.rename(r, axis=0) )
    df_scores = pd.concat(df_scores, axis=1)
    df_scores = df_scores.T.reset_index().pivot_table(index='index')
    # df_scores = df_scores[order_verif]

    rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr.', 'MAE':'MAE-SS',
                  'BSS':'BSS', 'roc_auc_score':'AUC', 'r2_score':'$r^2$',
                  'mean_absolute_percentage_error':'MAPE', 'AUC_SS':'AUC-SS',
                  'precision':'Precision', 'accuracy':'Accuracy'}
    metrics_plot = ['BSS', 'precision']


    tsc = df_scores.loc[metrics_plot].values
    tscs = np.zeros(tsc.shape, dtype=object)
    for i, m in enumerate(metrics_plot):
        _sc_ = tsc[i]
        if 'prec' in m or 'acc' in m:
            sc_f = ['{:.0f}%'.format(round(s,0)) for s in _sc_.ravel()]
            # if df_boots_list is not None:
            #     for j, s in enumerate(zip(tbl[i], tbh[i])):
            #         sc_f[j] = sc_f[j].replace('%', r'$_{{{:.0f}}}^{{{:.0f}}}$%'.format(*s))
        else:
            sc_f = ['{:.2f}'.format(round(s,2)) for s in tsc[i].ravel()]

            # if df_boots_list is not None:
            #     for j, s in enumerate(zip(tbl[i], tbh[i])):
            #         sc_f[j] = sc_f[j] + r'$_{{{:.2f}}}^{{{:.2f}}}$'.format(*s)
        tscs[i] = sc_f


    table = ax.table(cellText=tscs,
                      cellLoc='center',
                      rowLabels=[rename_met[m] for m in metrics_plot],
                      colLabels=df_scores.columns, loc='center',
                      edges='closed')
    table.set_fontsize(30)

    table.scale(1, 3)
    ax.axis('off')
    ax.axis('tight')
    ax.set_facecolor('white')
    ax.set_title(subtitle, fontsize=16)
    return ax

fig, ax = plt.subplots(3, 1, figsize=(10,7.2))
# plt.subplots_adjust(hspace=5)
ax1 = plot_table('No OOS pre-processing', os.path.join(path_main, path_no_OOS), ax[0])
ax2 = plot_table('OOS pre-processing, in-sample quantiles', os.path.join(path_main, path_OOS_no_q), ax[1])
ax3 = plot_table('OOS pre-processing', os.path.join(path_main, path_OOS), ax[2])
fig.suptitle(f'{fc_month}, quantile: {fc_type}', y=0.96, fontsize=22)