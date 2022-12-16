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
standard_subfolder = 'a9943/USDA_Soy_clusters__1/timeseriessplit_25/s1'
model_names = ['RandomForestClassifier', 'LogisticRegression']
model_name = model_names[0]
fc_months = ['August', 'July', 'June', 'May', 'April', 'March', 'February']
model_rename = {'LogisticRegression' : 'Logist. Regr.', 'RandomForestClassifier':'Random Forest'}

fc_types = [0.31, 0.33, 0.35]

path_no_OOS = f'test_no_areaw_oosT_False/{standard_subfolder}/df_data_fctpFalse_all_CD'
path_OOS_no_q = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctpFalse_all_CD'
path_OOS = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctp_T_all_CD'


def get_data(path, subset):

    metric = 'BSS'
    df_scores = [] ; df_boots = []
    for fc_type in fc_types:
        orig_fc_type = path.split('df_data_')[1][:4]
        path = path.replace(orig_fc_type, str(fc_type))
        target_options = [['Target', 'Target | PPS']]
        out = utils_paper3.load_scores(target_options[0], model_name, model_name, 2000,
                                       path, condition=['strong 30%'])
        df_scores_, df_boots_, df_preds = out

        # correct order ['All, '50%', '30%']
        df_scores_ = [df_scores_[i] for i in [0,1]]
        df_test_m = df_scores_ #[d[fc_month] for d in df_scores_]
        table_col = [d.index[0] for d in df_test_m][1:]
        order_verif = ['All'] + ['Top '+t.split(' ')[1] for t in table_col]
        # order_verif = [f'All\nq={fc_type}'] + ['Top '+t.split(' ')[1] + f'\nq={fc_type}' for t in table_col]
        for i, df_test_skill in enumerate(df_test_m):
            r = {df_test_skill.index[0]:order_verif[i]}
            df_test_m[i] = df_test_skill.rename(r, axis=0)
            # df_scores.append( df_test_skill[metric].rename(r, axis=0) )


        df_boots_ = pd.concat(df_boots_, keys=order_verif).xs(metric, axis=1, level=1)
        df_boots.append( df_boots_ )
        df_scores_ = pd.concat(df_test_m).xs(metric, axis=1, level=1)
        df_scores.append( df_scores_ )


    df_boots = pd.concat(df_boots, axis=1, keys = fc_types)
    df_scores = pd.concat(df_scores, axis=1, keys=fc_types)
    df_scores.columns = df_scores.columns.set_names(['quantiles', 'month'])
    df_scores_all = df_scores.loc[subset].reset_index().pivot_table(index='quantiles', columns='month')[subset].T
    df_scores_all = df_scores_all.reindex(fc_months)

    return df_scores_all

    # metrics_plot = ['BSS', 'precision']
    # tsc = df_scores.loc[metrics_plot].values
    # tscs = np.zeros(tsc.shape, dtype=object)
    # for i, m in enumerate(metrics_plot):
    #     _sc_ = tsc[i]
    #     if 'prec' in m or 'acc' in m:
    #         sc_f = ['{:.0f}%'.format(round(s,0)) for s in _sc_.ravel()]
    #         # if df_boots_list is not None:
    #         #     for j, s in enumerate(zip(tbl[i], tbh[i])):
    #         #         sc_f[j] = sc_f[j].replace('%', r'$_{{{:.0f}}}^{{{:.0f}}}$%'.format(*s))
    #     else:
    #         sc_f = ['{:.2f}'.format(round(s,2)) for s in tsc[i].ravel()]

    #         # if df_boots_list is not None:
    #         #     for j, s in enumerate(zip(tbl[i], tbh[i])):
    #         #         sc_f[j] = sc_f[j] + r'$_{{{:.2f}}}^{{{:.2f}}}$'.format(*s)
    #     tscs[i] = sc_f


    # table = ax.table(cellText=tscs,
    #                   cellLoc='center',
    #                   rowLabels=[rename_met[m] for m in metrics_plot],
    #                   colLabels=df_scores.columns, loc='center',
    #                   edges='closed')
    # table.set_fontsize(30)

    # table.scale(1, 4)
    # ax.axis('off')
    # ax.axis('tight')
    # ax.set_facecolor('white')
    # ax.set_title(subtitle, fontsize=16)

df_collect_all = [] ; df_collect_top = []
paths = {'No OOS pre-processing'                    : os.path.join(path_main, path_no_OOS),
         'OOS pre-processing, in-sample quantiles'  : os.path.join(path_main, path_OOS_no_q),
         'OOS pre-processing'                       : os.path.join(path_main, path_OOS)}

for path in paths.values():
    df_collect_all.append(get_data(path, 'All'))
    df_collect_top.append(get_data(path, 'Top 30%'))
#%%
fig, axes = plt.subplots(2, 3, figsize=(14,6), sharey=True, sharex=True)

colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c'][:len(fc_types)] * len(fc_months)
for i, df_scores in enumerate(df_collect_all):

    legend = True if i == len(df_collect_all)-1 else False
    df_scores.plot(y=fc_types, kind='bar', ax=axes[0,i], legend=legend, color=colors)
    axes[0,i].set_title(list(paths.keys())[i], fontsize=12)
    axes[0,i].set_ylim(0,1)
    axes[0,i].set(xlabel=None)
    if i==0: axes[0,i].set_ylabel('All datapoints', fontsize=12)

for i, df_scores in enumerate(df_collect_top):

    df_scores.plot(y=fc_types, kind='bar', ax=axes[1,i], legend=False, color=colors)
    # axes[1,i].set_title(list(paths.keys())[i], fontsize=12)
    axes[1,i].set_ylim(0,1)
    axes[1,i].set(xlabel=None)
    if i==0: axes[1,i].set_ylabel('Window of opportunity (Top 30%)', fontsize=12)

plt.subplots_adjust(wspace=0.1, hspace=.1)
fig.suptitle(f'Brier Skill Score {model_rename[model_name]}', y=0.965, fontsize=22)
fig.savefig(os.path.join(path_main, 'test_no_areaw_oosT_True', standard_subfolder, 'OOS_sensitivity.pdf'),
            bbox_inches='tight')
# ax1 = plot_table('No OOS pre-processing', os.path.join(path_main, path_no_OOS), ax[0])
# ax2 = plot_table('OOS pre-processing, in-sample quantiles', os.path.join(path_main, path_OOS_no_q), ax[1])
# ax3 = plot_table('OOS pre-processing', os.path.join(path_main, path_OOS), ax[2])
# fig.suptitle(f'{fc_month}', y=0.96, fontsize=22)