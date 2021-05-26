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
# else:
#     # Optionally set font to Computer Modern to avoid common missing font errors
#     mpl.rc('font', family='serif', serif='cm10')

#     mpl.rc('text', usetex=True)
#     mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xarray as xr
import csv
# import sklearn.linear_model as scikitlinear
import argparse

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
import functions_pp, df_ana, climate_indices, find_precursors
from stat_models import plot_importances
import plot_maps; import core_pp


experiments = ['bimonthly'] #['seasons', 'bimonthly', 'semestral']
target_datasets = ['USDA_Soy', 'USDA_Maize']#, 'GDHY_Soy']
seeds = seeds = [1] # [1,2,3,4,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
add_prev = [True, False]
feature_sel = [True, False]
combinations = np.array(np.meshgrid(target_datasets,
                                    experiments,
                                    seeds,
                                    yrs,
                                    add_prev,
                                    feature_sel)).T.reshape(-1,6)
i_default = 4


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
    experiment = out[1]
    seed = int(out[2])
    start_end_year = (int(out[3][:4]), int(out[3][-4:]))
    feature_selection = out[4] == 'True'
    add_previous_periods = out[5] == 'True'
    print(f'arg {args.intexper} {out}')
else:
    out = combinations[i_default]
    target_dataset = out[0]
    experiment = out[1]
    seed = int(out[2])
    start_end_year = (int(out[3][:4]), int(out[3][-4:]))



if target_dataset == 'GDHY_Soy':
    # GDHY dataset 1980 - 2015
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/q50_nc4_dendo_707fb.nc')
    cluster_label = 3
    name_ds='ts'
    # start_end_year = (1980, 2015)
elif target_dataset == 'USDA_Soy':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts.nc')
    name_ds='Soy_Yield' ; cluster_label = None
    # start_end_year = (1950, 2019)
elif target_dataset == 'USDA_Maize':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_maize_spatial_mean_ts.nc')
    name_ds='Maize_Yield' ; cluster_label = None
    # start_end_year = (1950, 2019)



calc_ts='region mean' # pattern cov
alpha_corr = .05
method     = 'ran_strat10' ;
n_boot = 2000

tfreq = 2
corlags = np.array([3,2,1,0])
periodnames = ['MA', 'MJ', 'JA', 'SO']
start_end_TVdate = ('09-01', '10-31')
start_end_date = ('01-01', '12-31')

append_main = target_dataset
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3')
PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-20,60)
USBox = (225, 300, 20, 60)



#%% run RGPD

list_of_name_path = [(cluster_label, TVpath),
                       ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc')),
                      # ('z500', os.path.join(path_raw, 'z500_1950-2019_1_12_monthly_1.0deg.nc')),
                       ('smi', os.path.join(path_raw, 'SM_spi_gamma_01_1950-2019_1_12_monthly_1.0deg.nc'))]
                      # ('swvl1', os.path.join(path_raw, 'swvl1_1950-2019_1_12_monthly_1.0deg.nc')),
                      # ('swvl1', os.path.join(path_raw, 'swvl1_1950-2019_1_12_monthly_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={}, group_lag=False,
                            distance_eps=400, min_area_in_degrees2=7,
                            calc_ts=calc_ts, selbox=GlobalBox,
                            lags=corlags),
                 BivariateMI(name='smi', func=class_BivariateMI.corr_map,
                             alpha=alpha_corr, FDR_control=True,
                             kwrgs_func={}, group_lag=False,
                             distance_eps=300, min_area_in_degrees2=4,
                             calc_ts=calc_ts, selbox=USBox,
                             lags=corlags)
                 ]


rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=start_end_year,
           tfreq=tfreq,
           path_outmain=path_out_main)
precur = rg.list_for_MI[0] ;

subfoldername = target_dataset+'_'.join(['', experiment, str(method),
                                         's'+ str(seed)] +
                                        list(np.array(start_end_year, str)))


rg.pp_precursors(detrend=[True, {'tp':False, 'smi':False, 'swvl1':False, 'swvl3':False}],
                 anomaly=[True, {'tp':False, 'smi':False, 'swvl1':False, 'swvl3':False}],
                 auto_detect_mask=[False, {'swvl1':True, 'swvl2':True}])
rg.pp_TV(detrend=True)
rg.traintest(method, seed=seed, subfoldername=subfoldername)
n_spl = rg.df_splits.index.levels[0].size

ds = core_pp.import_ds_lazy(rg.list_of_name_path[2][1], auto_detect_mask=True)
#%%
# ds = ds.where(ds.values>0)
# ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
# ds.attrs['units'] = 'mm'
# ds.to_netcdf(functions_pp.get_download_path() +'/SM_for_SPI.nc')

# ds = ds.transpose('time', 'latitude', 'longitude')
# ds.to_netcdf(functions_pp.get_download_path() +'/SMI.nc')
# ds = core_pp.import_ds_lazy(functions_pp.get_download_path() +'/SMI.nc')
#%%
rg.calc_corr_maps()
for p in rg.list_for_MI:
    p.corr_xr['lag'] = ('lag', periodnames)


#%%
save = False
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':0.2,
                  'wspace':-.15, 'size':3, 'cbar_vert':0.04,
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                   'y_ticks':np.arange(-10,61,20), #'x_ticks':np.arange(130, 280, 25),
                  'title':'',
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('sst', kwrgs_plot=kwrgs_plotcorr_sst, save=save)
#%%
SM = rg.list_for_MI[1]
# SM.adjust_significance_threshold(alpha_corr)
# SM.corr_xr['mask'] = ~SM.corr_xr['mask']
kwrgs_plotcorr_SM = {'row_dim':'lag', 'col_dim':'split','aspect':2, 'hspace':0.2,
                      'wspace':0, 'size':3, 'cbar_vert':0.04,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                       'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                      'title':'',
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('smi', kwrgs_plot=kwrgs_plotcorr_SM, min_detect_gc=1, save=save)


#%%
# precur = rg.list_for_MI[0]
# precur.distance_eps = 425 ; precur.min_area_in_degrees2 = 4
rg.cluster_list_MI()
# rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plotcorr_sst)

# Ensure that Caribean is alway splitted from Pacific
# sst = rg.list_for_MI[0]
# copy_labels = sst.prec_labels.copy()
# all_labels = copy_labels.values[~np.isnan(copy_labels.values)]
# uniq_labels = np.unique(all_labels)
# prevail = {l:list(all_labels).count(l) for l in uniq_labels}
# prevail = functions_pp.sort_d_by_vals(prevail, reverse=True)
# label = list(prevail.keys())[0]
# sst.prec_labels, _ = find_precursors.split_region_by_lonlat(sst.prec_labels.copy(),
#                                                       label=int(label), plot_l=0,
#                                                       kwrgs_mask_latlon={'bottom_left':(95,15)})



# copy_labels = SM.prec_labels.copy()
# all_labels = copy_labels.values[~np.isnan(copy_labels.values)]
# uniq_labels = np.unique(all_labels)
# prevail = {l:list(all_labels).count(l) for l in uniq_labels}
# prevail = functions_pp.sort_d_by_vals(prevail, reverse=True)
# two_largest = list(prevail.keys())[:2]
# df_coords = find_precursors.labels_to_df(SM.prec_labels.copy())
# label = df_coords.loc[two_largest].longitude.idxmax()
# SM.prec_labels, _ = find_precursors.split_region_by_lonlat(SM.prec_labels.copy(),
#                                                       label=int(label), plot_l=2,
#                                                       kwrgs_mask_latlon={'bottom_left':(285,45)})

rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plotcorr_sst, save=save)

rg.quick_view_labels('smi', kwrgs_plot=kwrgs_plotcorr_SM, save=save)


#%%
# for p in rg.list_for_MI:
#     p.prec_labels['lag'] = ('lag', periodnames)
rg.get_ts_prec(precur_aggr=tfreq)
rg.df_data = rg.df_data.rename({'Nonets':target_dataset},axis=1)


#%%
def feature_selection_CondDep(df_data, keys, alpha_CI=.05):
    # Feature selection Cond. Dependence
    keys = list(keys) # must be list
    corr, pvals = wrapper_PCMCI.df_data_Parcorr(df_data.copy(), z_keys=keys,
                                                keys=keys)
    # removing all keys that are Cond. Indep. in each trainingset
    keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
    for s in rg.df_splits.index.levels[0]:
        for k_i in keys:
            onekeyCI = (np.nan_to_num(pvals.loc[k_i][s],nan=alpha_CI) > alpha_CI).mean()>0
            keyisNaN = np.isnan(pvals.loc[k_i][s]).all()
            if onekeyCI or keyisNaN:
                k_ = keys_dict[s].copy() ; k_.pop(k_.index(k_i))
                keys_dict[s] = k_

    return corr, pvals, keys_dict.copy()

#%%
keys = list(rg.df_data.columns[:-2])
# keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='sst']
# keys = [k for k in keys if k.split('..')[1] == '1']
# keys = [k for k in keys if k.split('..')[0]=='JA'] ;
# keys.insert(0, target_dataset)
# keys = None


tigr_function_call='run_bivci'
kwrgs_tigr={'tau_min':0, 'tau_max':3,
            'val_only':False,
            'remove_links':[(0,-1), (0,-2), (0,-3)]}


rg.PCMCI_init(keys=keys)
t_min = 0; t_max = 3
selected_links_splits = {}
for s, pcmci_dict in rg.pcmci_dict.items():
    var_names = pcmci_dict.var_names
    selected_links = {}
    for i, y in enumerate(var_names):
        links = []
        for j, z in enumerate(var_names):
            if y != z and 'sst' in y:
                links.append([(j,-l) for l in range(t_min, t_max)])
            else:
                links.append([(j,-l) for l in range(t_min, t_max)])
        selected_links[i] = core_pp.flatten(links)
    selected_links_splits[s] = selected_links

tigr_function_call='run_pcmci'
kwrgs_tigr={'tau_min':t_min, 'tau_max':t_max,
            'pc_alpha':[.01, .05, .1, .2],
            'max_conds_py':2,
            'max_conds_px':0,
            'selected_links':selected_links_splits}

rg.PCMCI_df_data(keys=keys,
                  tigr_function_call=tigr_function_call,
                  kwrgs_tigr=kwrgs_tigr,
                  verbosity=2)
alpha_CI = .05
rg.PCMCI_get_links(var=target_dataset, alpha_level=alpha_CI)
rg.df_links.mean(axis=0,level=1)


#%%
keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='sst']
# keys = [k for k in keys if k.split('..')[1] == '1']
keys = [k for k in keys if k.split('..')[0]=='JA'] ;
corr, pvals, keys_dict = feature_selection_CondDep(rg.df_data.copy(),
                                                           keys, alpha_CI)
# always C.D. (every split)
keys = [k for k in keys if (np.nan_to_num(pvals.loc[k],nan=alpha_CI) <= alpha_CI).mean()== 1]





#%% forecasting
import wrapper_PCMCI
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from stat_models_cont import ScikitModel

# fcmodel = ScikitModel(RandomForestRegressor, verbosity=0)
# kwrgs_model={'n_estimators':200,
#             'max_depth':[2,5,7],
#             'scoringCV':'neg_mean_squared_error',
#             'oob_score':True,
#             'min_samples_leaf':2,
#             'random_state':0,
#             'max_samples':.6,
#             'n_jobs':1}
fcmodel = ScikitModel(RidgeCV, verbosity=0)
kwrgs_model = {'scoring':'neg_mean_absolute_error',
                'alphas':np.concatenate([[0],np.logspace(-5,0, 6),
                                          np.logspace(.01, 2.5, num=25)]), # large a, strong regul.
                'normalize':False,
                'fit_intercept':False,
                'store_cv_values':True,
                'kfold':5}

def append_dict(month, df_test_m, df_train_m):
    dkeys = [f'{month} RMSE', f'{month} RMSE tr',
              f'{month} Corr.', f'{month} Corr. tr',
              f'{month} MAE test', f'{month} MAE tr']
    append_dict = {dkeys[0]:float(df_test_m.iloc[:,0].round(3)),
                    dkeys[1]:df_train_m.mean().iloc[0].round(3),
                    dkeys[2]:float(df_test_m.iloc[:,1].round(3)),
                    dkeys[3]:float(df_train_m.mean().iloc[1].round(3)),
                    dkeys[4]:float(df_test_m.iloc[:,2].round(3)),
                    dkeys[5]:float(df_train_m.mean().iloc[2].round(3))}
    dict_v.update(append_dict)
    return



blocksize=1
lag = 0
alpha_CI = .1
variables = ['sst', 'smi']
variables = ['sst']

feature_selection = True
# add_previous_periods = False

add_PDO = False
if add_PDO:
    # get PDO
    SST_pp_filepath = rg.list_precur_pp[0][1]
    if 'df_PDOsplit' not in globals():
        df_PDO, PDO_patterns = climate_indices.PDO(SST_pp_filepath,
                                                    None)
        df_PDOsplit = df_PDO.loc[0]
        plot_maps.plot_corr_maps(PDO_patterns)

    df_PDOsplit = df_PDOsplit[['PDO']].apply(fc_utils.standardize_on_train,
                          args=[df_PDO.loc[0]['TrainIsTrue']],
                          result_type='broadcast')
    PDO_aggr_periods = np.array([
                                ['03-01', '02-28'],
                                ['06-01', '05-31'],
                                ['09-01', '08-31'],
                                ['12-01', '11-30']
                                ])


dict_v = {'target':target_dataset, 'method':method,'S':f's{seed}',
          'yrs':start_end_year, 'fs':str(feature_selection),
          'addprev':add_previous_periods}

fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
# CRPSS = fc_utils.CRPSS_vs_constant_bench(constant_bench=clim_mean_temp).CRPSS
score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
metric_names = [s.__name__ for s in score_func_list]

list_test = []
list_train = []
list_test_b = []
list_pred_test = []
no_info_fc = []

CondDepKeys = {} ; CondDepKeysDict = {}
for i, months in enumerate(periodnames[:]): #!!!
    print(f'forecast using precursors of {months}')
    # i=0; months, start_end_TVdate = periodnames[i], corlags[i]

    keys = [k for k in rg.df_data.columns[:-2] if k.split('..')[-1] in variables]
    keys = [k for k in keys if months == k.split('..')[0]]


    if add_PDO:
        se = PDO_aggr_periods[i]
        df_PDOagr = functions_pp.time_mean_periods(df_PDOsplit,
                                                    start_end_periods=se)
        # remove old PDO timeseries (if present)
        rg.df_data = rg.df_data[[k for k in rg.df_data.columns if k != 'PDO']]
        rg.df_data = rg.merge_df_on_df_data(df_PDOagr)
        keys.append('PDO')


    if len(keys) != 0 and 'PDO' not in keys and feature_selection:
        k_c = rg.df_links[months]
        keys_dict = {}
        for s in range(n_spl):
            ks = list(k_c.loc[s][k_c.loc[s]==True].index)
            if len(ks) == 0: # choose most robust precursor across lags
                ks = [rg.df_links.mean(axis=0,level=1).mean(axis=1).idxmax()]
            keys_dict[s] = ks
        # corr, pvals, keys_dict = feature_selection_CondDep(rg.df_data.copy(),
        #                                                     keys, alpha_CI)
        # # always C.D. (every split)
        # keys = [k for k in keys if (np.nan_to_num(pvals.loc[k],nan=alpha_CI) <= alpha_CI).mean()== 1]
    else:
        keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars

    CondDepKeys[months] = keys
    CondDepKeysDict[months] = keys_dict

    if add_previous_periods and periodnames.index(months) != 0:
        # merging CD keys that were found for each split seperately to keep
        # clean train-test split
        for s in range(n_spl):
            pm = periodnames[periodnames.index(months) - 1 ]
            pmk = [k for k in CondDepKeysDict[pm][s] if k.split('..')[0]==pm]
            keys_dict[s] = keys_dict[s] + pmk


    if len(keys) != 0:
        lag_  = corlags[periodnames.index(months)]
        out = rg.fit_df_data_ridge(target=target_ts,
                                    keys=keys_dict,
                                    tau_min=lag_, tau_max=lag_,
                                    kwrgs_model=kwrgs_model,
                                    fcmodel=fcmodel,
                                    transformer=None)

        predict, weights, models_lags = out
        prediction = predict.rename({predict.columns[0]:'target',lag:'Prediction'},
                                    axis=1)

        if i==0:
            weights_norm = weights.mean(axis=0, level=1)
            weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')


        df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                                  rg.df_data.iloc[:,-2:],
                                                                  score_func_list,
                                                                  n_boot = n_boot,
                                                                  blocksize=1,
                                                                  rng_seed=seed)

        m = models_lags[f'lag_{lag_}'][f'split_{0}']
        if months == 'JA':
            print(models_lags[f'lag_{lag_}'][f'split_{0}'].X_pred.iloc[0])
            print(lag_, df_test_m)
        cvfitalpha = [models_lags[f'lag_{lag_}'][f'split_{s}'].alpha_ for s in range(n_spl)]
        if kwrgs_model['alphas'].max() in cvfitalpha: print('Max a reached')
        if kwrgs_model['alphas'].min() in cvfitalpha: print('Min a reached')
        # assert kwrgs_model['alphas'].min() not in cvfitalpha, 'decrease min a'

        df_test = functions_pp.get_df_test(predict.rename({lag_:months}, axis=1),
                                            cols=[months],
                                            df_splits=rg.df_splits)
        # appending results
        list_pred_test.append(df_test)

    else:
        print('no precursor timeseries found, scores all 0')

        index = pd.MultiIndex.from_product([['Prediction'],metric_names])
        df_boot = pd.DataFrame(data=np.zeros((n_boot, len(score_func_list))),
                            columns=index)
        df_test_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
                                  columns=index)
        df_train_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
                                  columns=index)


    df_test_m.index = [months] ;
    df_test_m.columns = pd.MultiIndex.from_product([['Prediction'],metric_names])
    columns = pd.MultiIndex.from_product([np.array([months]),
                                        df_train_m.columns.levels[1]])

    df_train_m.columns = columns
    df_boot.columns = columns

    list_test_b.append(df_boot)
    list_test.append(df_test_m)
    list_train.append(df_train_m)
    append_dict(months, df_test_m, df_train_m)
    # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
    #                       function=df_ana.plot_timeseries,
    #                       kwrgs={'timesteps':rg.fullts.size,
    #                                   'nth_xyear':5})


#%%

import matplotlib.patches as mpatches


def boxplot_scores(list_scores, list_test_b, alpha=.1):

    df_scores = pd.concat(list_test)
    df_test_b = pd.concat(list_test_b,axis=1)

    yerr = [] ; quan = [] ;
    monmet = np.array(df_test_b.columns)
    for i, (mon, met) in enumerate(monmet):
        Eh = 1 - alpha/2 ; El = alpha/2
        # met = rename_metrics_cont[met]
        _scores = df_test_b[mon][met]
        tup = [_scores.quantile(El), _scores.quantile(Eh)]
        quan.append(tup)
        mean = df_scores.values.flatten()[i] ;
        tup = abs(mean-tup)
        yerr.append(tup)

    _yerr = np.array(yerr).reshape(df_scores.columns.size,len(list(CondDepKeys.keys()))*2,
                                    order='F').reshape(df_scores.columns.size,2,len(list(CondDepKeys.keys())))
    ax = df_scores.plot.bar(rot=0, yerr=_yerr,
                            capsize=8, error_kw=dict(capthick=1),
                            color=['blue', 'green', 'purple'],
                            legend=False,
                            figsize=(10,8))
    for noinfo in no_info_fc:
        # first two children are not barplots
        idx = list(CondDepKeys.keys()).index(noinfo) + 3
        ax.get_children()[idx].set_color('r') # RMSE bar
        idx = list(CondDepKeys.keys()).index(noinfo) + 15
        ax.get_children()[idx].set_color('r') # RMSE bar



    ax.set_ylabel('Skill Score', fontsize=16, labelpad=-5)
    # ax.tick_params(labelsize=16)
    # ax.set_xticklabels(months, fontdict={'fontsize':20})
    title = 'U.S. {} forecast {}\n'.format(target_dataset.split('_')[-1],
                                            str(start_end_year))
    if experiment == 'semestral':
        title += 'from half-year mean '
    elif experiment == 'seasons':
        title += 'from seasonal mean '
    title += '{}'.format(' and '.join([v.upper() for v in variables]))

    ax.set_title(title,
                  fontsize=18)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=16)


    patch1 = mpatches.Patch(color='blue', label='RMSE-SS')
    patch2 = mpatches.Patch(color='green', label='Corr. Coef.')
    patch3 = mpatches.Patch(color='purple', label='MAE-SS')
    handles = [patch1, patch2, patch3]
    legend1 = ax.legend(handles=handles,
              fontsize=16, frameon=True, facecolor='grey',
              framealpha=.5)
    ax.add_artist(legend1)

    ax.set_ylim(-0.3, 1)

    append_str = '-'.join(periodnames) #+'_'+'_'.join(np.array(start_end_year, str))
    plt.savefig(os.path.join(rg.path_outsub1,
              f'skill_fs{feature_selection}_addprev{add_previous_periods}_'
              f'ab{alpha}_ac{alpha_corr}_afs{alpha_CI}_nb{n_boot}_blsz{blocksize}_{append_str}.pdf'))


boxplot_scores(list_test, list_test_b, alpha=.1)

#%%
if feature_selection:
    append_str = '-'.join(periodnames)
    for ip, precur in enumerate(rg.list_for_MI):
        CDlabels = precur.prec_labels.copy() ; CDl = np.zeros_like(CDlabels)
        CDcorr = precur.corr_xr.copy()
        for i, month in enumerate(CondDepKeys):
            CDkeys = CondDepKeys[month]
            region_labels = [int(l.split('..')[1]) for l in CDkeys if l.split('..')[-1] == precur.name]
            f = find_precursors.view_or_replace_labels
            CDlabels[:,i] = f(CDlabels[:,i].copy(), region_labels)
        mask = (np.isnan(CDlabels)).astype(bool)
        if ip == 0:
            kwrgs_plot = kwrgs_plotcorr_sst
        elif ip == 1:
            kwrgs_plot = kwrgs_plotcorr_SM

        plot_maps.plot_labels(CDlabels.mean(dim='split'), kwrgs_plot=kwrgs_plot)
        if save:
            plt.savefig(os.path.join(rg.path_outsub1,
                                  f'CondDep_labels_{precur.name}_{append_str}_ac{alpha_corr}_eps{precur.distance_eps}'
                                  f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}'+rg.figext))

        fig = plot_maps.plot_corr_maps(CDcorr.mean(dim='split'),
                                  mask_xr=mask, **kwrgs_plot)
        if save:
            fig.savefig(os.path.join(rg.path_outsub1,
                                  f'CondDep_{precur.name}_{append_str}_ac{alpha_corr}_aCI{alpha_CI}'+rg.figext))


#%%
keys  = core_pp.flatten(list(CondDepKeys.values()))
corrf, pvalf, keys_dictf = feature_selection_CondDep(rg.df_data.copy(), keys)
final_keys = [k for k in keys if (np.nan_to_num(pvalf.loc[k],nan=alpha_CI) <= alpha_CI).mean()== 1]

#%%
# # code run with or without -i
# if sys.flags.inspect:
name_csv = f'output_regression_{experiment}_sensivity.csv'
name_csv = os.path.join(rg.path_outmain, name_csv)
for csvfilename, dic in [(name_csv, dict_v)]:
    # create .csv if it does not exists
    if os.path.exists(csvfilename) == False:
        with open(csvfilename, 'a', newline='') as csvfile:

            writer = csv.DictWriter(csvfile, list(dic.keys()))
            writer.writerows([{f:f for f in list(dic.keys())}])

    # write
    with open(csvfilename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, list(dic.keys()))
        writer.writerows([dic])

#%% EXIT


sys.exit()


