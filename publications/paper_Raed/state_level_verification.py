#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:24:43 2021

@author: semvijverberg
"""


import os, inspect, sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
    n_cpu = 5
else:
    n_cpu = 3

import numpy as np
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import pandas as pd
from joblib import Parallel, delayed
import argparse
from matplotlib.lines import Line2D
import csv
import re
from sklearn.linear_model import LinearRegression
import xarray as xr


user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
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


from RGCPD.forecasting import func_models as fc_utils
from RGCPD import functions_pp, find_precursors, plot_maps, core_pp
import RGCPD.forecasting.stat_models_cont as sm
import utils_paper3


target_datasets = ['States']
seeds = [1] # ,5]
models = ['LR', 'RF']
methods = ['timeseriessplit_25', 'leave_1'] # ['ranstrat_20'] timeseriessplit_30, 'timeseriessplit_20', 'timeseriessplit_30',
training_datas = ['all', 'all_CD']
combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    models,
                                    methods,
                                    training_datas)).T.reshape(-1,5)
i_default = -1
load = 'all'
save = True
# training_data = 'onelag' # or 'all_CD' or 'onelag' or 'all'


model_combs_cont = [['Ridge', 'Ridge'],
                    ['Ridge', 'RandomForestRegressor'],
                    ['RandomForestRegressor', 'RandomForestRegressor']]
model_combs_bina = [['LogisticRegression', 'LogisticRegression']]
                    # ['LogisticRegression', 'RandomForestClassifier'],
                    # ['RandomForestClassifier', 'RandomForestClassifier']]

model_combs_bina = [['LogisticRegression', 'LogisticRegression'],
                    ['RandomForestClassifier', 'RandomForestClassifier']]


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
    comb = combinations[args.intexper]
    target_dataset = comb[0]
    seed = int(comb[1])
    model = comb[2]
    method = comb[3]
    training_data = comb[4]
    print(f'arg {args.intexper} {comb}')
else:
    out = combinations[i_default]
    target_dataset = out[0]
    seed = int(out[1])
    model = out[2]
    method = out[3]



noseed = np.logical_or(method.lower()[:-3] == 'timeseriessplit',
                       method.split('_')[0] == 'leave')
if noseed and seed > 1:
    print('stop run')
    sys.exit()



# path to raw Soy Yield dataset
if sys.platform == 'linux':
    root_data = user_dir+'/surfdrive/Scripts/RGCPD/publications/paper_Raed/data/'
else:
    root_data = user_dir+'/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/'
raw_filename = os.path.join(root_data, 'masked_rf_gs_county_grids.nc')



Soy_state_path =  os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')



calc_ts= 'region mean' # 'pattern cov'
alpha_corr = .05
alpha_CI = .05
n_boot = 2000
append_pathsub = f'/{method}/s{seed}'
if method == 'leave_1': append_pathsub += 'gp_prior_1_after_1'



path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'fc_extra2lags')
path_out_main = os.path.join(path_out_main, 'a9943')

PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)
USBox = (225, 300, 20, 60)


#%% load and pre-process data

def df_oos_lindetrend(df_fullts: pd.DataFrame,
                      df_splits: pd.DataFrame):
    '''
    Pre-process column data (of State level yield).
    Cluster mask specified by dsclust
    '''

    fcmodel = sm.ScikitModel(LinearRegression)

    preds = []
    for s in df_splits.index.levels[0]:
        fit_masks = df_splits.loc[s].iloc[:,-2:]
        x_time = pd.DataFrame(df_fullts.index.year,
                              index=df_fullts.index,
                              columns=['time'])
        x_time = x_time.merge(fit_masks, left_index=True,right_index=True)

        pred, model = fcmodel.fit_wrapper({'ts':df_fullts},
                                          x_time)
        preds.append(pred)
    preds = pd.concat(preds, keys=df_splits.index.levels[0])

    old = pd.concat([df_fullts]*df_splits.index.levels[0].size,
                          keys=df_splits.index.levels[0])

    fig, ax = plt.subplots(2)
    for s in df_splits.index.levels[0]:
        ax[0].plot(old.loc[s], color='black')
        ax[0].plot(preds.loc[s])
        ax[1].plot(old.loc[s] - preds.loc[s].values)

    return old - preds.values



#%% get data associated with forecast (made in forecast.py)

fc_type = 0.33
if 'timeseries' in method:
    btoos = '_T' # if btoos=='_T': binary target out of sample.
else:
    btoos = ''

path_input_main = os.path.join(path_out_main, 'USDA_Soy_clusters__1' + append_pathsub)
pathsub_df = f'df_data_{str(fc_type)}{btoos}'
pathsub_verif = f'verif_{str(fc_type)}{btoos}'
if training_data != 'CL':
    pathsub_df  += '_'+training_data
    pathsub_verif += '_'+training_data
filepath_df_datas = os.path.join(path_input_main, pathsub_df)
filepath_verif = os.path.join(path_input_main, pathsub_verif)

model_name = 'LogisticRegression' # 'RandomForestClassifier'
fc_month = 'February'

# Get forecasts
fc_months_periodnames = {'August': 'JJ', 'July': 'MJ', 'June': 'AM',
                         'May': 'MA','April': 'FM', 'March': 'JF',
                         'December': 'SO', 'February': 'DJ'}
filepath_df_output = os.path.join(path_input_main,
                                  f'df_output_{fc_months_periodnames[fc_month]}.h5')

df_output = functions_pp.load_hdf5(filepath_df_output)
df_data  = df_output['df_data']
splits = df_data.index.levels[0]
if training_data == 'all':
    df_input = df_data
    keys_dict = {s:df_input.columns[1:-2] for s in range(splits.size)}
# use all RG-DR timeseries that are C.D. for (final) prediction
elif training_data == 'all_CD':
    df_input = df_data
    df_pvals = df_output['df_pvals'].copy()
    keys_dict = utils_paper3.get_CD_df_data(df_pvals, alpha_CI)
df_splits = df_data.iloc[:,-2:].copy()

out = utils_paper3.load_scores(['Target'], model_name, model_name,
                               2000, filepath_df_datas,
                               condition='strong 50%')
df_scores, df_boots, df_preds = out


df_test_m = [d[fc_month] for d in df_scores]
df_boots_list = [d[fc_month] for d in df_boots]
df_test  = df_preds[0][['Target', fc_month]]
# df_test = functions_pp.get_df_test(df_test, df_splits=df_splits)

#%% get pre-processed soy yield xarray

def read_csv_Raed(path):
    orig = pd.read_csv(path)
    orig = orig.drop(labels='Unnamed: 0', axis=1)
    orig.index = pd.to_datetime([f'{y}-01-01' for y in orig.Year])
    orig.index.name = 'time'
    return orig.drop(labels='Year', axis=1)

def read_csv_State(path, State: str=None, col='obs_yield'):
    orig = read_csv_Raed(path)
    orig = orig.set_index('State', append=True)
    orig = orig.pivot_table(index='time', columns='State')[col]
    if State is None:
        State = orig.columns
    return orig[State]

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']

scoringCV = 'neg_brier_score'
kwrgs_model1 = {'scoringCV':scoringCV,
                'C':list([1E-3, 1E-2, 5E-2, 1E-1,
                           .5,1,1.2,4,7,10,20]), # Smaller C, strong regul.
                # 'C':list([1E-1,10,50]), # Small C set for test runs
                'random_state':seed,
                'penalty':'l2',
                'solver':'lbfgs',
                'kfold':15,
                'max_iter':200,
                'n_jobs':n_cpu}
model1_tuple = (sm.ScikitModel(sm.lm.LogisticRegression, verbosity=0),
                kwrgs_model1)


from sklearn.ensemble import RandomForestClassifier
kwrgs_model2={'n_estimators':300,
              'max_depth':[2, 5, 8, 15],
              'scoringCV':scoringCV,
              # 'criterion':'mse',
              'oob_score':True,
              'random_state':0,
              'min_impurity_decrease':0,
              'max_features':[0.4,0.8],
              'max_samples':[0.4, 0.7],
              'kfold':10,
              'n_jobs':n_cpu}
model2_tuple = (sm.ScikitModel(RandomForestClassifier, verbosity=0),
                kwrgs_model2)

if model == 'LR':
    fcmodel, kwrgs_model = model1_tuple
elif model == 'RF':
    fcmodel, kwrgs_model = model2_tuple

#%%


fc_type = .33
btoos = '_T'


path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'STATES')
os.makedirs(path_out_main, exist_ok=True)
path_save = os.path.join(path_out_main,
                          f'{model}_{method}_{training_data}_{fc_type}')

if os.path.exists(os.path.join(path_save,'summary.h5')):
    d_dfs = functions_pp.load_hdf5(os.path.join(path_save,'summary.h5'))
    skill_summary = d_dfs['skill_summary']
    skill_summary_cond_50 = d_dfs['skill_summary_cond_50']
    skill_summary_cond_30 = d_dfs['skill_summary_cond_30']
else:

    forecast_months = ['August', 'July', 'June', 'May',
                        'April', 'March', 'February']
    regions_forcing = ['Pacific+SM', 'Pacific+SM', 'only_Pacific',
                        'only_Pacific', 'only_Pacific', 'only_Pacific',
                        'only_Pacific']

    forecast_months = ['April', 'March', 'February']
    skill_summary = []
    skill_summary_cond_50 = []
    skill_summary_cond_30 = []
    for im, fc_month in enumerate(forecast_months):


        filepath_df_output = os.path.join(path_input_main,
                                          f'df_output_{fc_months_periodnames[fc_month]}.h5')

        df_output = functions_pp.load_hdf5(filepath_df_output)
        df_data  = df_output['df_data']
        splits = df_data.index.levels[0]
        if training_data == 'all':
            df_input = df_data
            keys_dict = {s:df_input.columns[1:-2] for s in range(splits.size)}
        # use all RG-DR timeseries that are C.D. for (final) prediction
        elif training_data == 'all_CD':
            df_input = df_data
            df_pvals = df_output['df_pvals'].copy()
            keys_dict = utils_paper3.get_CD_df_data(df_pvals, alpha_CI)
        df_splits = df_data.iloc[:,-2:].copy()
        # get forcing
        region_labels = [1,0] if fc_month == 'August' else [1]
        # 1 = horseshoe Pacific region and 0 = SM pattern ts.
        keys = [k for k in df_data.columns[1:-2] if int(k.split('..')[1]) in region_labels]
        df_forcing = df_data[keys]

        # load state level observed yield
        dates_verif = df_splits.index.levels[1]
        skill_states = []
        skill_states_cond_50 = [] ;
        skill_states_cond_30 = []
        for STATE in All_states[:]:
            df_verif = read_csv_State(Soy_state_path, [STATE]).loc[dates_verif]
            if float(np.isnan(df_verif).sum()) != 0:
                continue
            df_verif_pp = df_oos_lindetrend(df_verif, df_splits)

            # Define poor yield events out or in sample
            _target_ts = df_verif_pp.iloc[:,[0]].copy()
            if btoos == '_T': # OOS event threshold
                quantile = functions_pp.get_df_train(_target_ts,
                                                      df_splits=df_splits,
                                                      s='extrapolate',
                                                      function='quantile',
                                                      kwrgs={'q':fc_type})
                quantile = quantile.values
                target_ts = df_verif_pp.iloc[:,[0]].copy()
            else: # in sample event threshold
                _target_ts = _target_ts.groupby(level=1).mean()
                _target_ts = (_target_ts - _target_ts.mean()) / _target_ts.std()
                quantile = float(_target_ts.quantile(fc_type))
            if fc_type >= 0.5:
                target_ts = (_target_ts > quantile).astype(int)
            elif fc_type < .5:
                target_ts = (_target_ts < quantile).astype(int)
            target_ts

            # define skill scores
            if fc_type == 'continuous':
                # benchmark is climatological mean (after detrending)
                bench = float(target_ts.mean())
                RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=bench).RMSE
                MAE_SS = fc_utils.ErrorSkillScore(constant_bench=bench).MAE
                score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS,
                                    fc_utils.r2_score]
            else:
                # benchmark is clim. probability
                BSS = fc_utils.ErrorSkillScore(constant_bench=fc_type).BSS
                score_func_list = [BSS, fc_utils.metrics.roc_auc_score,
                                fc_utils.binary_score(threshold=fc_type).accuracy,
                                fc_utils.binary_score(threshold=fc_type).precision]
            metric_names = [s.__name__ for s in score_func_list]

            # fit model
            predict, weights, models_lags = sm.fit_df_data_sklearn(df_data, keys=keys_dict,
                                                                    tau_min=0,
                                                                    tau_max=0,
                                                                    target=target_ts,
                                                                    fcmodel=fcmodel,
                                                                    kwrgs_model=kwrgs_model)
            predict = predict.rename({0:fc_month}, axis=1)

            # calculate skill scores
            out_verification = fc_utils.get_scores(predict,
                                                    df_splits,
                                                    score_func_list,
                                                    n_boot=0,
                                                    blocksize=1,
                                                    rng_seed=1)
            df_train_m, df_test_s_m, df_test_m, df_boot = out_verification
            df_test_m = df_test_m.rename({0:STATE}, axis=0)
            skill_states.append(df_test_m)


            # get skill during strong horseshoe state years
            df_cond_fc = fc_utils.cond_fc_verif(df_predict = predict,
                                                df_forcing = df_forcing,
                                                df_splits = df_splits,
                                                score_func_list=score_func_list,
                                                quantiles=[.25, .15])
            # 50% strong boundary forcing subset
            df_cond_fc_50 = df_cond_fc.T.loc[['strong 50%']]
            df_cond_fc_50.columns = df_test_m.columns
            df_cond_fc_50 = df_cond_fc_50.rename({'strong 50%':STATE}, axis=0)
            skill_states_cond_50.append(df_cond_fc_50)
            # 30% strong boundary forcing subset
            df_cond_fc_30 = df_cond_fc.T.loc[['strong 30%']]
            df_cond_fc_30.columns = df_test_m.columns
            df_cond_fc_30 = df_cond_fc_30.rename({'strong 30%':STATE}, axis=0)
            skill_states_cond_30.append(df_cond_fc_30)

        skill_summary.append(pd.concat(skill_states, axis=0))
        skill_summary_cond_50.append(pd.concat(skill_states_cond_50, axis=0))
        skill_summary_cond_30.append(pd.concat(skill_states_cond_30, axis=0))

    skill_summary = pd.concat(skill_summary, axis=1)
    skill_summary_cond_50 = pd.concat(skill_summary_cond_50, axis=1)
    skill_summary_cond_30 = pd.concat(skill_summary_cond_30, axis=1)


    #%%
    os.makedirs(path_save, exist_ok=True)
    functions_pp.store_hdf_df({'skill_summary':skill_summary,
                               'skill_summary_cond_50':skill_summary_cond_50,
                               'skill_summary_cond_30':skill_summary_cond_30},
                              os.path.join(path_save,'summary.h5'))
    skill_summary.to_csv(os.path.join(path_save,'all_data.csv'))
    skill_summary_cond_50.to_csv(os.path.join(path_save,'cond_50.csv'))
    skill_summary_cond_30.to_csv(os.path.join(path_save,'cond_30.csv'))


#%%
from RGCPD.clustering import make_country_mask
from itertools import product
raw_filename = os.path.join(root_data, 'masked_rf_gs_county_grids.nc')
selbox = [253,290,28,52] ; years = list(range(1975, 2020))
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
    root_data = user_dir+'/surfdrive/Scripts/RGCPD/publications/paper_Raed/data/'
else:
    root_data = user_dir+'/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/'
raw_filename = os.path.join(root_data, 'masked_rf_gs_county_grids.nc')
da = core_pp.import_ds_lazy(raw_filename,
                            **{'selbox':selbox,
                               'var':'variable'}).isel(z=0).drop('z')
xarray, df_codes = make_country_mask.create_mask(da,
                                                 level='US_States')

#%%
months = skill_summary.columns[::skill_summary.columns.levels[1].size]
months = [m[0] for m in months]
metrics = skill_summary.columns[::skill_summary.columns.levels[0].size]
metrics = [m[1] for m in metrics]
subsets = ['all', '50%', '30%']
xr_skill_sum = xarray.copy()
xr_skill_sum.values = np.zeros_like(xarray)
xr_skill_sum = core_pp.expand_dim(xr_skill_sum, subsets, 'subset')
xr_skill_sum = core_pp.expand_dim(xr_skill_sum, metrics, 'metric')
xr_skill_sum = core_pp.expand_dim(xr_skill_sum, months, 'month')



xr_skill_sum.name = 'skill' ; xr_skill_sum.attrs = {}

leaveoutstates = ['NORTH DAKOTA', 'MINNESOTA', 'WISCONSIN']
selstates = [s for s in skill_summary.index if s not in leaveoutstates]

def skill_to_state(skill_summary, df_codes, selstates: list=None, metric='BSS',
                   month='April'):
    if selstates is None:
        selstates = skill_summary.index
    vals = skill_summary.loc[:,month][metric]
    labels = []
    for s in selstates:
        df_s = df_codes[df_codes['name'].str.match(s, case=False)]
        labels.append(int(df_s['label']))
    return list(labels), list(vals)


def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

np_skill_sum = np.zeros_like(xr_skill_sum, dtype=float)
np_skill_sum[:] = np.nan
for index, momesu in enumerated_product(months, metrics, subsets):
    mo, me, su = momesu
    i, j, k = index
    if su == 'all':
        _skill = skill_summary
    elif su == '50%':
        _skill = skill_summary_cond_50
    elif su == '30%':
            _skill = skill_summary_cond_30
    labels, vals = skill_to_state(_skill, df_codes, selstates,
                                  metric=me, month=mo)

    replace_labels = find_precursors.view_or_replace_labels
    xr_out = replace_labels(xarray.copy(),
                                       regions=labels,
                                       replacement_labels=vals)
    np_skill_sum[i,j,k] = xr_out.copy()

xr_skill_sum.values = np_skill_sum
xr_skill_sum = xr_skill_sum.where(xr_skill_sum.values!=0)


#%%

metric='roc_auc_score'
metric='BSS'
#ee9b00, #ca6702, #bb3e03, #ae2012, #9b2226
cmp = ["ade8f4","e9d8a6","ee9b00","bb3e03","ae2012","9b2226"]
cmp = plot_maps.get_continuous_cmap(cmp,
                float_list=list(np.linspace(0,1,6)))
if metric == 'BSS':
    clevels = np.arange(0, .51, .1)
elif metric == 'roc_auc_score':
    clevels = np.arange(.5, 1.01, .1)

fg = plot_maps.plot_corr_maps(xr_skill_sum.sel(metric=metric).drop('metric'),
                              col_dim='subset', row_dim='month',
                              hspace=.4, clevels=clevels,
                              clabels=clevels,
                              cmap=cmp,
                              kwrgs_cbar = {'orientation':'horizontal', 'extend':'min'})


facecolorocean = '#caf0f8' ; facecolorland='white'
for ax in fg.fig.axes[:-1]:
    ax.add_feature(cfeature.STATES, zorder=1, linewidth=.5, edgecolor='black')
    ax.add_feature(plot_maps.cfeature.__dict__['LAND'],
                   facecolor=facecolorland,
                   zorder=0)
    ax.add_feature(plot_maps.cfeature.__dict__['OCEAN'],
                   facecolor=facecolorocean,
                   zorder=0)
cbar = fg.fig.axes[-1]




