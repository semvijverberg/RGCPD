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
import pandas as pd
from joblib import Parallel, delayed
import argparse
from matplotlib.lines import Line2D
import csv
import re

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


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import func_models as fc_utils
import functions_pp, find_precursors
import plot_maps, core_pp
import wrapper_PCMCI
import utils_paper3
from stat_models import plot_importances
from stat_models_cont import ScikitModel
import scikit_model_analysis as sk_ana
from sklearn.linear_model import LinearRegression

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']


target_datasets = ['USDA_Soy_clusters__1']
seeds = [1] # ,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
methods = ['ranstrat_20', 'timeseriessplit_20', 'timeseriessplit_30', 'timeseriessplit_25', 'leave_1'] # ['ranstrat_20'] timeseriessplit_30
training_datas = ['onelag', 'all', 'all_CD']
combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    yrs,
                                    methods,
                                    training_datas)).T.reshape(-1,5)
i_default = -2
load = 'all'
save = True
# training_data = 'onelag' # or 'all_CD' or 'onelag' or 'all'
fc_types = [0.33, 'continuous']
fc_types = [0.33]

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
    out = combinations[args.intexper]
    target_dataset = out[0]
    seed = int(out[1])
    start_end_year = (int(out[2][:4]), int(out[2][-4:]))
    method = out[3]
    training_data = out[4]
    print(f'arg {args.intexper} {out}')
else:
    out = combinations[i_default]
    target_dataset = out[0]
    seed = int(out[1])
    start_end_year = (int(out[2][:4]), int(out[2][-4:]))
    method = out[3]

def read_csv_Raed(path):
    orig = pd.read_csv(path)
    orig = orig.drop('Unnamed: 0', axis=1)
    orig.index = pd.to_datetime([f'{y}-01-01' for y in orig.Year])
    orig.index.name = 'time'
    return orig.drop('Year', 1)

noseed = np.logical_or(method.lower()[:-3] == 'timeseriessplit',
                       method.split('_')[0] == 'leave')
if noseed and seed > 1:
    print('stop run')
    sys.exit()

def read_csv_State(path, State: str=None, col='obs_yield'):
    orig = read_csv_Raed(path)
    orig = orig.set_index('State', append=True)
    orig = orig.pivot_table(index='time', columns='State')[col]
    if State is None:
        State = orig.columns
    return orig[State]

# path to raw Soy Yield dataset
if sys.platform == 'linux':
    root_data = user_dir+'/surfdrive/Scripts/RGCPD/publications/paper_Raed/data/'
else:
    root_data = user_dir+'/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/'
raw_filename = os.path.join(root_data, 'masked_rf_gs_county_grids.nc')

if target_dataset.split('__')[0] == 'USDA_Soy_clusters':
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/linkage_ward_nc2_dendo_0d570.nc')
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/linkage_ward_nc2_dendo_lindetrendgc_a9943.nc')
    # TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/linkage_ward_nc2_dendo_detrgc_int_c88c0.nc')
    # TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/linkage_ward_nc2_dendo_interp_ff5d6.nc')
    cluster_label = int(target_dataset.split('__')[1]) ; name_ds = 'ts'
elif target_dataset == 'Aggregate_States':
    path =  os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')
    States = ['KENTUCKY', 'TENNESSEE', 'MISSOURI', 'ILLINOIS', 'INDIANA']
    TVpath = read_csv_State(path, State=States, col='obs_yield').mean(1)
    TVpath = pd.DataFrame(TVpath.values, index=TVpath.index, columns=['KENTUCKYTENNESSEEMISSOURIILLINOISINDIANA'])
    name_ds='Soy_Yield' ; cluster_label = ''



calc_ts= 'region mean' # 'pattern cov'
alpha_corr = .05
alpha_CI = .05
n_boot = 2000
append_pathsub = f'/{method}/s{seed}'
extra_lag = True

append_main = target_dataset
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'fc_extra2lags')
if target_dataset.split('__')[0] == 'USDA_Soy_clusters': # add cluster hash
    path_out_main = os.path.join(path_out_main, TVpath.split('.')[0].split('_')[-1])
elif target_dataset.split('__')[0] == 'All_State_average': # add cluster hash
    path_out_main = os.path.join(path_out_main, 'All_State_Average')
elif target_dataset in All_states: # add cluster hash
    path_out_main = os.path.join(path_out_main, 'States')

PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)
USBox = (225, 300, 20, 60)



list_of_name_path = [(cluster_label, TVpath),
                       ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500_1950-2019_1_12_monthly_1.0deg.nc')),
                       ('smi', os.path.join(path_raw, 'SM_ownspi_gamma_2_1950-2019_1_12_monthly_1.0deg.nc'))]


def df_oos_lindetrend(df_fullts, df_splits):
    fcmodel = ScikitModel(LinearRegression)

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


def ds_oos_lindetrend(dsclust, df_splits, path):

    kwrgs_NaN_handling={'missing_data_ts_to_nan':False,
                        'extra_NaN_limit':False,
                        'inter_method':False,
                        'final_NaN_to_clim':False}
    years = list(range(1950, 2020))
    selbox = [253,290,28,52]
    ds_raw = core_pp.import_ds_lazy(raw_filename, var='variable', selbox=selbox,
                                    kwrgs_NaN_handling=kwrgs_NaN_handling).rename({'z':'time'})
    ds_raw.name = 'Soy_Yield'
    ds_raw['time'] = pd.to_datetime([f'{y+1949}-01-01' for y in ds_raw.time.values])
    ds_raw = ds_raw.sel(time=core_pp.get_oneyr(ds_raw, *years))

    label = int(target_dataset.split('__')[-1])
    clusmask = dsclust['xrclustered'] == label
    ds_raw = ds_raw.where(clusmask)
    ds_out = utils_paper3.detrend_oos_3d(ds_raw, min_length=30,
                                         df_splits=df_splits,
                                         standardize=True,
                                         path=path)
    return ds_out

path_input_main = os.path.join(path_out_main, target_dataset + append_pathsub)
fc_type = 0.33
if 'timeseries' in method:
    btoos = '_T' # if btoos=='_T': binary target out of sample.
    # btoos = '_theor' # binary target based on gaussian quantile
else:
    btoos = ''


pathsub_df = f'df_data_{str(fc_type)}{btoos}'
pathsub_verif = f'verif_{str(fc_type)}{btoos}'
if training_data != 'CL':
    pathsub_df  += '_'+training_data
    pathsub_verif += '_'+training_data
filepath_df_datas = os.path.join(path_input_main, pathsub_df)
filepath_verif = os.path.join(path_input_main, pathsub_verif)

model_name = 'LogisticRegression' # 'RandomForestClassifier'
fc_month = 'February'

#%%
fc_months_periodnames = {'August': 'JJ', 'July': 'MJ', 'June': 'AM',
                         'May': 'MA','April': 'FM', 'March': 'JF',
                         'December': 'SO', 'February': 'DJ'}
filepath_df_output = os.path.join(path_input_main,
                                  f'df_output_{fc_months_periodnames[fc_month]}.h5')

df_output = functions_pp.load_hdf5(filepath_df_output)
df_data  = df_output['df_data']
df_splits = df_data.iloc[:,-2:]

out = utils_paper3.load_scores(['Target'], model_name, model_name,
                               2000, filepath_df_datas,
                               condition='strong 50%')
df_scores, df_boots, df_preds = out


df_test_m = [d[fc_month] for d in df_scores]
df_boots_list = [d[fc_month] for d in df_boots]
df_test  = df_preds[0][['Target', fc_month]]
df_test = functions_pp.get_df_test(df_test, df_splits=df_splits)
