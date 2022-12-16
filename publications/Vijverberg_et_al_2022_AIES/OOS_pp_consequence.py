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

fc_months = ['August', 'July', 'June', 'May', 'April', 'March', 'February']
model_rename = {'LogisticRegression' : 'Logist. Regr.', 'RandomForestClassifier':'Random Forest'}

fc_types = [0.31, 0.33, 0.35]

path_no_OOS = f'test_no_areaw_oosT_False/{standard_subfolder}/df_data_fctpFalse_all_CD'
path_OOS_no_q = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctpFalse_all_CD'
path_OOS = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctp_T_all_CD'


def get_data(path, subset, model_name):

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



paths = {'No OOS pre-processing'                    : os.path.join(path_main, path_no_OOS),
          # 'OOS pre-processing, in-sample quantiles'  : os.path.join(path_main, path_OOS_no_q),
         'OOS pre-processing'                       : os.path.join(path_main, path_OOS)}

model_out = {}
for model_name in model_names:

    df_collect_all = [] ; df_collect_top = []
    for path in paths.values():
        df_collect_all.append(get_data(path, 'All', model_name))
        df_collect_top.append(get_data(path, 'Top 30%', model_name))
    model_out[model_name] = (df_collect_all, df_collect_top)
#%%
model_name = model_names[0]
df_collect_all, df_collect_top = model_out[model_name]

fig, axes = plt.subplots(2, 2, figsize=(9,6), sharey=True, sharex=True)
plt.rc('legend', fontsize=10)
colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c'][:len(fc_types)] * len(fc_months)
for i, df_scores in enumerate(df_collect_all):

    legend = True if i == len(df_collect_all)-1 else False
    df_scores.plot(y=fc_types, kind='bar', ax=axes[0,i], legend=legend, color=colors)
    axes[0,i].set_title(list(paths.keys())[i], fontsize=12)
    axes[0,i].set_ylim(-.2,1)
    axes[0,i].axhline(df_scores[0.33].mean(), xmax=1.2, color=colors[1], ls='dashed', lw=1)
    axes[0,i].set(xlabel=None)
    if i==0:
        axes[0,i].set_ylabel('All datapoints', fontsize=12)
    if i == 0 and model_names.index(model_name) == 1:
        axes[0,i].annotate('Perfect skill', xy=(-.5, 1), xytext=(0.2, .9),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=9,
                transform=axes[1,i].transAxes)
        axes[0,i].annotate('Climatological skill', xy=(-.5, 0), xytext=(0.2, -.1),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=9,
                transform=axes[1,i].transAxes)
axes[0,1].annotate('drop in skill', xy=(-0.5, df_collect_all[1][0.33].mean()),
                   xytext=(-.5, df_collect_all[0][0.33].mean()+.05),
                   horizontalalignment="center",
        arrowprops=dict(arrowstyle="->", color='black'),
        fontsize=9, bbox = dict(boxstyle ="round", fc ="0.8"))


for i, df_scores in enumerate(df_collect_top):

    df_scores.plot(y=fc_types, kind='bar', ax=axes[1,i], legend=False, color=colors)
    axes[1,i].axhline(df_scores[0.33].mean(), color=colors[1], ls='dashed', lw=1)
    axes[1,i].set_ylim(-.2,1)
    axes[1,i].set(xlabel=None)
    if i==0:
        axes[1,i].set_ylabel('Window of opportunity (Top 30%)', fontsize=12)
axes[1,1].annotate('drop in skill', xy=(-0.5, df_collect_top[1][0.33].mean()),
                   xytext=(-.5, df_collect_top[0][0.33].mean()+.05),
                   horizontalalignment="center",
        arrowprops=dict(arrowstyle="->", color='black'),
        fontsize=9, bbox = dict(boxstyle ="round", fc ="0.8"))

plt.subplots_adjust(wspace=0.1, hspace=.1)
fig.suptitle(f'Brier Skill Score {model_rename[model_name]}', y=0.965, fontsize=22)
fig.savefig(os.path.join(path_main, 'test_no_areaw_oosT_True', standard_subfolder, f'OOS_sensitivity_{model_name}.pdf'),
            bbox_inches='tight')
# ax1 = plot_table('No OOS pre-processing', os.path.join(path_main, path_no_OOS), ax[0])
# ax2 = plot_table('OOS pre-processing, in-sample quantiles', os.path.join(path_main, path_OOS_no_q), ax[1])
# ax3 = plot_table('OOS pre-processing', os.path.join(path_main, path_OOS), ax[2])
# fig.suptitle(f'{fc_month}', y=0.96, fontsize=22)