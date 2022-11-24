#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
else:
    mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
#     # Optionally set font to Computer Modern to avoid common missing font errors
#     mpl.rc('font', family='serif', serif='cm10')

#     mpl.rc('text', usetex=True)

import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import pandas as pd
# import xarray as xr
# import sklearn.linear_model as scikitlinear
import argparse

os.chdir('/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper_Raed/')
user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-3])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
# from RGCPD import BivariateMI
# import class_BivariateMI
import func_models as fc_utils
import functions_pp, df_ana, find_precursors
import plot_maps; import core_pp


experiments = ['seasons']
target_datasets = ['USDA_Soy', 'USDA_Wheat', 'GDHY_Soy']
combinations = np.array(np.meshgrid(target_datasets, experiments)).T.reshape(-1,2)
i_default = 1


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-i", "--intexper", help="intexper", type=int,
                        default=i_default)
    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseArguments()
    out = combinations[args.intexper]
    target_dataset = int(out[0])
    experiment = out[1]
    print(f'arg {args.intexper} {out}')
else:
    target_dataset = 'USDA_Soy'
    experiment = 'seasons'


if target_dataset == 'GDHY_Soy':
    # GDHY dataset 1980 - 2015
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/q50_nc4_dendo_707fb.nc')
    cluster_label = 3
    name_ds='ts'
    start_end_year = (1980, 2015)
elif target_dataset == 'USDA_Soy':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts.nc')
    name_ds='Soy_Yield'
    start_end_year = (1950, 2019)



method     = 'leave_4'
n_boot = 500
if experiment == 'seasons':
    corlags = np.array([
                        ['12-01', '02-28'], # DJF
                        ['03-01', '05-31'], # MAM
                        ['06-01', '08-31'], # JJA
                        ['09-01', '11-30'] # SON
                        ])
    periodnames = ['DJF','MAM','JJA','SON']
elif experiment == 'semestral':
    corlags = np.array([
                        ['12-01', '05-31'], # DJFMAM
                        ['03-01', '08-31'], # MAMJJA
                        ['06-01', '11-30'], # JJASON
                        ['01-01', '12-31']  # annual
                        ])
    periodnames = ['DJFMAM', 'MAMJJA', 'JJASON', 'annual']

append_main = target_dataset
path_out_main = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/output'
PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)

#%% run RGPD
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
list_of_name_path = [(cluster_label, TVpath),
                      ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc'))]
                      # ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI = None



rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=start_end_year,
           path_outmain=path_out_main,
           append_pathsub=append_main)

rg.pp_precursors()
rg.pp_TV(anomaly=False, detrend=True)
rg.traintest(method)

ds = core_pp.import_ds_lazy(rg.list_precur_pp[0][1])
season = ds.resample(time='QS-DEC').mean()
#%% on post-processed (anomaly, detrended) SST
import climate_indices
df_ENSO, ENSO_yrs, df_states = climate_indices.ENSO_34(rg.list_precur_pp[0][1],
                                  rg.df_splits.copy(),
                                  get_ENSO_states=True)

cycle = df_states[[f'EN_cycle']].loc[0]
print('El Nino yrs', list(cycle[cycle=='EN0'].dropna().index.year))
cycle = df_states[[f'LN_cycle']].loc[0]
print('La Nina yrs', list(cycle[cycle=='LN0'].dropna().index.year))

#%% Composites of Anderson 2017 ENSO states

for title in ['EN-1', 'EN0', 'EN+1', 'LN-1', 'LN0', 'LN+1']:
    cycle = df_states[[f'{title[:2]}_cycle']].loc[0]
    selyrs = cycle[cycle==title].dropna().index.year
    # print(title, selyrs)

    kwrgs = {'hspace':0.2, 'aspect':4, 'cbar_vert':0.04, 'clevels':np.arange(-.75, .76, .25),
             'title_fontdict':{'y':.95}, 'y_ticks':False, 'x_ticks':False}
    comp = [d for d in pd.to_datetime(season.time.values) if d.year in selyrs]
    ds_plot = season.sel(time=pd.to_datetime(comp)).groupby('time.month').mean()
    ds_plot = ds_plot.rename({'month':'season'})
    ds_plot['season'] = ['DJF', 'MAM', 'JJA', 'SON']
    plot_maps.plot_corr_maps(ds_plot, row_dim='season',
                             title=title+f' (n={selyrs.size})', **kwrgs)


#%% Composites of SST during Low yield
lowyield = rg.dates_TV[(rg.TV_ts < rg.TV_ts.quantile(.33)).values].year

comp = [d for d in pd.to_datetime(season.time.values) if d.year in lowyield]
ds_plot = season.sel(time=pd.to_datetime(comp)).groupby('time.month').mean()
ds_plot = ds_plot.rename({'month':'season'})
ds_plot['season'] = ['DJF', 'MAM', 'JJA', 'SON']
plot_maps.plot_corr_maps(ds_plot, row_dim='season', title='low yield', **kwrgs)

# low_prior = lowyield - 1
# lwyield = [d for d in pd.to_datetime(season.time.values) if d.year in low_prior]
# plot_maps.plot_corr_maps(season.sel(time=pd.to_datetime(lwyield)).groupby('time.month').mean(),
#                          row_dim='month', title='low yield prior', **kwrgs)
#%% Composites of SST during High yield
highyield = rg.dates_TV[(rg.TV_ts > rg.TV_ts.quantile(.66)).values].year

comp = [d for d in pd.to_datetime(season.time.values) if d.year in highyield]
ds_plot = season.sel(time=pd.to_datetime(comp)).groupby('time.month').mean()
ds_plot = ds_plot.rename({'month':'season'})
ds_plot['season'] = ['DJF', 'MAM', 'JJA', 'SON']
plot_maps.plot_corr_maps(ds_plot, row_dim='season', title='high yield', **kwrgs)

# high_prior = highyield - 1
# high_d = [d for d in pd.to_datetime(season.time.values) if d.year in high_prior]
# plot_maps.plot_corr_maps(season.sel(time=pd.to_datetime(high_d)).groupby('time.month').mean(),
                         # row_dim='month', title='high yield prior', **kwrgs)

#%% ENSO states to predict Soy Yield with RandomForest
from sklearn.ensemble import RandomForestClassifier
from stat_models_cont import ScikitModel
logit_skl = ScikitModel(RandomForestClassifier, verbosity=0)
kwrgs_model={'n_estimators':200,
            'max_depth':[5,7,10],
            'scoringCV':'neg_brier_score',
            'oob_score':True,
            'min_samples_leaf':2,
            'random_state':0,
            'max_samples':.4}



rg.get_ts_prec()
rg.df_data = rg.merge_df_on_df_data(pd.get_dummies(df_states, prefix='',
                                                   prefix_sep=''))
rg.df_data  = rg.merge_df_on_df_data(df_ENSO)

fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
target_ts = (target_ts > target_ts.mean()).astype(int)


out = rg.fit_df_data_ridge(target=target_ts,
                            keys=['ENSO34', 'states', 'EN+1', 'LN+1', 'EN-1', 'EN0', 'LN-1', 'LN0'],
                            fcmodel=logit_skl,
                            kwrgs_model=kwrgs_model,
                            transformer=False,
                            tau_min=0, tau_max=0)
predict, weights, model_lags = out

weights_norm = weights.mean(axis=0, level=1)
weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
score_func_list = [BSS, fc_utils.metrics.roc_auc_score]


df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                 rg.df_data.iloc[:,-2:],
                                                                 score_func_list,
                                                                 n_boot = n_boot,
                                                                 score_per_test=False,
                                                                 blocksize=1,
                                                                 rng_seed=1)
print(logit_skl.scikitmodel.__name__, '\n', 'Test score\n',
      'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
      'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
      '\nTrain score\n',
      'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
      'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))

# [model_lags['lag_0'][f'split_{i}'].best_params_ for i in range(17)]

#%% ENSO states to predict Soy Yield with Logistic regr.

from sklearn.linear_model import LogisticRegressionCV
from stat_models_cont import ScikitModel
logit_skl = ScikitModel(LogisticRegressionCV, verbosity=0)
kwrgs_model = {'kfold':10}

out = rg.fit_df_data_ridge(target=target_ts,
                            keys=['ENSO34', 'states'],
                            fcmodel=logit_skl,
                            kwrgs_model=kwrgs_model,
                            transformer=False,
                            tau_min=0, tau_max=0)
predict, weights, model_lags = out

weights_norm = weights.mean(axis=0, level=1)
weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                 rg.df_data.iloc[:,-2:],
                                                                 score_func_list,
                                                                 n_boot = n_boot,
                                                                 score_per_test=False,
                                                                 blocksize=1,
                                                                 rng_seed=1)

print(logit_skl.scikitmodel.__name__, '\n', 'Test score\n',
      'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
      'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
      '\nTrain score\n',
      'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
      'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))




#%% Correlating both the gradient and absolute timeseries of ENSO with target
df_ENSO_s = df_ENSO.loc[0]
grad_w = 3
gap = 6
for month in range(1,13):
    grad_ENSO = df_ENSO_s.shift(int(1+grad_w/2+gap)).rolling(int(grad_w/2),
                                                       center=True,
                                         min_periods=1).mean() - \
                        df_ENSO_s.rolling(int(grad_w/2), center=True,
                                        min_periods=1).mean()
    X_dates = core_pp.get_subdates(df_ENSO_s.index,
                                   start_end_date=(f'{month}-01',f'{month}-28'),
                                   start_end_year=(1951, 2019))
    target_data = rg.TV_ts[1:].values
    # df_ENSO_s.loc[X_dates].plot()
    corr_grad = np.corrcoef(grad_ENSO.loc[X_dates].values.squeeze(),
                                             target_data)[0][1]
    corr_abs = np.corrcoef(df_ENSO_s.loc[X_dates].values.squeeze(),
                                             target_data)[0][1]
    print('{:02d}'.format(month),
          'Gradient ENSO {:.2f}\n'.format(corr_grad),
          '  Absolute values {:.2f}'.format(corr_abs) )


