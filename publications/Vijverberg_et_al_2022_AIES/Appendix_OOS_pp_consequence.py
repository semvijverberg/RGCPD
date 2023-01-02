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

fc_types = [0.31, 0.32, 0.33, 0.34, 0.35]

dataset = 'all_CD'
path_no_OOS = f'test_no_areaw_oosT_False/{standard_subfolder}/df_data_fctpFalse_' + dataset
path_OOS_no_q = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctpFalse_' + dataset
path_OOS = f'test_no_areaw_oosT_True/{standard_subfolder}/df_data_fctp_T_' + dataset


def get_data(path, subset, model_name, metric = 'precision'):

    if subset != 'All':
        condition = [subset.replace('Top', 'strong')]
    else:
        condition = ['strong 30%']

    df_scores = [] ; df_boots = []
    for fc_type in fc_types:
        orig_fc_type = path.split('df_data_')[1][:4]
        path = path.replace(orig_fc_type, str(fc_type))
        target_options = [['Target', 'Target | PPS']]
        out = utils_paper3.load_scores(target_options[0], model_name, model_name, 2000,
                                       path, condition=condition)
        df_scores_, df_boots_, df_preds = out

        df_scores_ = [df_scores_[i] for i in [0,1]]
        df_test_m = df_scores_
        table_col = [d.index[0] for d in df_test_m][1:]
        order_verif = ['All'] + ['Top '+t.split(' ')[1] for t in table_col]
        for i, df_test_skill in enumerate(df_test_m):
            r = {df_test_skill.index[0]:order_verif[i]}
            df_test_m[i] = df_test_skill.rename(r, axis=0)

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



paths = {'in-sample'                    : os.path.join(path_main, path_no_OOS),
          # 'OOS pre-processing, in-sample quantiles'  : os.path.join(path_main, path_OOS_no_q),
         'out-of-sample (OOS)'                       : os.path.join(path_main, path_OOS)}

metric = 'BSS'
cond = 'Top 30%'
metric_rename = {'BSS': 'Brier Skill Score', 'precision': 'Precision'}
model_out = {}
for model_name in model_names:
    df_collect_all = [] ; df_collect_top = []
    for path in paths.values():
        df_collect_all.append(get_data(path, 'All', model_name, metric = metric))
        df_collect_top.append(get_data(path, cond, model_name, metric = metric))
    model_out[model_name] = (df_collect_all, df_collect_top)
#%%
for j in range(2):
    # RF, LR
    df_collect_all, df_collect_top  = model_out[model_names[1]][j], model_out[model_names[0]][j]
    # df_collect_top = model_out[model_names[0]][1] # LR

    fig, axes = plt.subplots(2, 3, figsize=(9,6), sharey=False, sharex=False,
                             gridspec_kw={'width_ratios': [3, 3, 1.5]})
    plt.rc('legend', fontsize=8)
    colors = ['#ef476f', '#f4a261', '#06d6a0', '#118ab2', '#073b4c'][:len(fc_types)] * len(fc_months)
    for i, df_scores in enumerate(df_collect_all):

        legend = True if i == len(df_collect_all)-1 else False
        df_scores.plot(y=fc_types, kind='bar', ax=axes[0,i], legend=legend, color=colors)
        axes[0,i].set_title(list(paths.keys())[i], fontsize=12)
        if metric == 'BSS':
            axes[0,i].set_ylim(-.1,1)
        else:
            axes[0,i].set_ylim(-10,100)
        # axes[0,i].axhline(df_scores.mean().mean(), xmax=1.2, color='black', alpha=.8, ls='dashed', lw=1)
        axes[0,i].set(xlabel=None)

        if i==0:
            axes[0,i].set_ylabel('Logistic Regression', fontsize=12)
        if i == 0 and model_names.index(model_name) == 1 and metric == 'BSS':
            axes[0,i].annotate('Perfect skill', xy=(-.5, 1), xytext=(0.2, .93),
                    arrowprops=dict(arrowstyle="->", color='black'),
                    fontsize=9,
                    transform=axes[1,i].transAxes)
            axes[0,i].annotate('Climatological skill', xy=(-.5, 0), xytext=(0.2, -.07),
                    arrowprops=dict(arrowstyle="->", color='black'),
                    fontsize=9,
                    transform=axes[1,i].transAxes)

    diff = df_collect_all[1] - df_collect_all[0]
    diff.mean(axis=0).plot(ax=axes[0,2], y=fc_types, kind='bar', color=colors)
    if metric == 'BSS':
        axes[0,2].set_ylim(-.3,.3)
    else:
        axes[0,2].set_ylim(-30,30)
    axes[0,2].set(xlabel=None)
    axes[0,2].set_ylabel(f'drop in {metric}')
    axes[0,2].yaxis.tick_right() ; axes[0,2].yaxis.set_label_position("right")
    axes[0,2].set_xticklabels([])
    axes[0,2].set_title('Difference\nOOS minus in-sample', fontsize=12)

    # diff = df_collect_all[0].mean().mean() - df_collect_all[1].mean().mean()
    # if diff > .02:
    #     axes[0,1].annotate('drop in skill', xy=(-0.5, df_collect_all[1].mean().mean()),
    #                     xytext=(-.5, df_collect_all[0].mean().mean()+.05),
    #                     horizontalalignment="center",
    #                     arrowprops=dict(arrowstyle="->", color='black'),
    #                     fontsize=9, bbox = dict(boxstyle ="round", fc ="0.8"))


    for i, df_scores in enumerate(df_collect_top):

        df_scores.plot(y=fc_types, kind='bar', ax=axes[1,i], legend=False, color=colors)
        # axes[1,i].axhline(df_scores.mean().mean(), color='black', alpha=.8, ls='dashed', lw=1)
        if metric == 'BSS':
            axes[1,i].set_ylim(-.1,1)
        else:
            axes[1,i].set_ylim(-10,100)
        axes[1,i].set(xlabel=None)
        if i==0:
            axes[1,i].set_ylabel('Random Forest', fontsize=12)
    diff = df_collect_top[1] - df_collect_top[0]
    diff.mean(axis=0).plot(ax=axes[1,2], y=fc_types, kind='bar', color=colors)
    if metric == 'BSS':
        axes[1,2].set_ylim(-.3,.3)
    else:
        axes[1,2].set_ylim(-30,30)
    axes[1,2].set_ylabel(f'drop in {metric}')
    axes[1,2].yaxis.tick_right() ; axes[1,2].yaxis.set_label_position("right")

    # diff = df_collect_top[0].mean().mean() - df_collect_top[1].mean().mean()
    # if diff > .02:
    #     axes[1,1].annotate('drop in skill', xy=(-0.5, df_collect_top[1].mean().mean()),
    #                     xytext=(-.5, df_collect_top[0].mean().mean()+.05),
    #                     horizontalalignment="center",
    #                     arrowprops=dict(arrowstyle="->", color='black'),
    #                     fontsize=9, bbox = dict(boxstyle ="round", fc ="0.8"))


    axes[0,0].set_xticklabels([]) ; axes[0,1].set_xticklabels([])
    axes[0,1].set_yticklabels([]) ; axes[1,1].set_yticklabels([])
    axes[1,0].set_xticks(range(df_scores.index.size), list(df_scores.index))
    axes[1,1].set_xticks(range(df_scores.index.size), list(df_scores.index))
    plt.subplots_adjust(wspace=0.1, hspace=.1)
    if j == 0:
        title = f'{metric_rename[metric]} (all datapoints)'
    elif j == 1:
        title = f'{metric_rename[metric]} ({cond})'
    fig.suptitle('In-sample vs. out-of-sample preprocessing of target\n'+title, y=1.02, fontsize=15)
    fig.savefig(os.path.join(path_main, 'test_no_areaw_oosT_True', standard_subfolder, f'OOS_sensitivity_{j}_{dataset}_{metric}_{cond[-3:-1]}.pdf'),
                bbox_inches='tight')
#%%
subset_verifs = ['All', 'Top 50%', 'Top 30%']
model_name = model_names[0] # RF
df_collect_BSS = [] ; df_collect_prec = []
for subset_verif in subset_verifs:
    df_collect_BSS.append(get_data(os.path.join(path_main, path_OOS),
                                   subset_verif, model_name, metric = 'BSS'))
    df_collect_prec.append(get_data(os.path.join(path_main, path_OOS),
                                   subset_verif, model_name, metric = 'precision'))
#%%
fig, axes = plt.subplots(2, 3, figsize=(9,6.5), sharey=False, sharex=False)
plt.rc('legend', fontsize=8)
colors = ['#ef476f', '#f4a261', '#06d6a0', '#118ab2', '#073b4c'][:len(fc_types)] * len(fc_months)
for i, df_scores in enumerate(df_collect_BSS):

    if i == 0:
        df_scores.plot(y=fc_types, kind='bar', ax=axes[0,i], legend=True, color=colors).legend(loc='upper left', fontsize=9, title='quantiles')
    else:
        df_scores.plot(y=fc_types, kind='bar', ax=axes[0,i], legend=False, color=colors)
    axes[0,i].set_title(subset_verifs[i], fontsize=12)
    axes[0,i].set_ylim(-.1,1)
    axes[0,i].set(xlabel=None)
    # axes[0,i].set_xticks(range(df_scores.index.size), list(df_scores.index))
    axes[0,i].set_xticklabels([])
    if i != 0:
        axes[0,i].set_yticklabels([])
axes[0,0].set_ylabel('Brier Skill Score')

for i, df_scores in enumerate(df_collect_prec):
    df_scores.plot(y=fc_types, kind='bar', ax=axes[1,i], legend=False, color=colors)
    # axes[1,i].set_title(subset_verifs[i], fontsize=12)
    axes[1,i].set_ylim(-10,100)
    axes[1,i].set_yticks(np.arange(0,101, 15))
    axes[1,i].set_yticklabels(np.arange(0,101, 15))
    axes[1,i].set(xlabel=None)
    if i != 0:
        axes[1,i].set_yticklabels([])

plt.subplots_adjust(wspace=0.1, hspace=.1)
axes[1,0].set_ylabel('Precision')
fig.savefig(os.path.join(path_main, 'test_no_areaw_oosT_True', standard_subfolder, f'skill_vs_qs_{dataset}_{model_rename[model_name]}.pdf'),
            bbox_inches='tight')
