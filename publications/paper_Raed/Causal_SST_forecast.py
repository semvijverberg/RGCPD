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
import plot_maps; import core_pp


target_datasets = ['USDA_Soy', 'USDA_Maize']#, 'GDHY_Soy']
seeds = seeds = [1] # [1,2,3,4,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
add_prev = [True, False]
feature_sel = [True, False]
combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    yrs,
                                    add_prev,
                                    feature_sel)).T.reshape(-1,5)
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
    seed = int(out[1])
    start_end_year = (int(out[2][:4]), int(out[2][-4:]))
    feature_selection = out[3] == 'True'
    add_previous_periods = out[4] == 'True'
    print(f'arg {args.intexper} {out}')
else:
    out = combinations[i_default]
    target_dataset = out[0]
    seed = int(out[1])
    start_end_year = (int(out[2][:4]), int(out[2][-4:]))



if target_dataset == 'GDHY_Soy':
    # GDHY dataset 1980 - 2015
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/q50_nc4_dendo_707fb.nc')
    cluster_label = 3
    name_ds='ts'
    # start_end_year = (1980, 2015)
elif target_dataset == 'USDA_Soy':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts.nc')
    name_ds='Soy_Yield' ; cluster_label = ''
    # TVpath = '/Users/semvijverberg/surfdrive/VU_Amsterdam/GDHY_MIRCA2000_Soy/USDA/ts_spatial_avg_midwest.h5'
    # TVpath = '/Users/semvijverberg/surfdrive/VU_Amsterdam/GDHY_MIRCA2000_Soy/USDA/ts_spatial_avg.h5'
    # name_ds='USDA_Soy'
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


SM_lags = np.array([
                    ['04-01', '04-30'], # MA
                    ['06-01', '06-30'], # MJ
                    ['08-01', '08-31'], # JA
                    ['10-01', '10-31']  # SO
                    ])
SM_periodnames = ['MA','MJ','JA','SO']
SST_lags = np.array([
                    ['09-01', '04-01'],
                    ['10-01', '05-01'],
                    ['11-01', '06-01'],
                    ['12-01', '07-01'],
                    ['01-01', '08-01'],
                    ['02-01', '09-01'], # Mar-Oct
                    ['03-01', '10-01'] # Mar-Oct
                    ])
SST_periodnames = ['April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct']


append_main = target_dataset
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3')
PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)
USBox = (225, 300, 20, 60)



#%% run RGPD

list_of_name_path = [(cluster_label, TVpath),
                       ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc')),
                      # ('z500', os.path.join(path_raw, 'z500_1950-2019_1_12_monthly_1.0deg.nc')),
                       ('smi', os.path.join(path_raw, 'SM_ownspi_gamma_2_1950-2019_1_12_monthly_1.0deg.nc'))]
                      # ('swvl1', os.path.join(path_raw, 'swvl1_1950-2019_1_12_monthly_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={},
                            distance_eps=400, min_area_in_degrees2=7,
                            calc_ts=calc_ts, selbox=GlobalBox,
                            lags=SST_lags, group_split=True,
                            use_coef_wghts=True),
                  BivariateMI(name='smi', func=class_BivariateMI.corr_map,
                             alpha=.05, FDR_control=True,
                             kwrgs_func={},
                             distance_eps=300, min_area_in_degrees2=4,
                             calc_ts=calc_ts, selbox=USBox,
                             lags=SM_lags, use_coef_wghts=True)]


rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=None,
           start_end_date=None,
           start_end_year=start_end_year,
           tfreq=None,
           path_outmain=path_out_main)
precur = rg.list_for_MI[0] ; lag = precur.lags[0]

subfoldername = target_dataset+'_'.join(['', str(method),
                                         's'+ str(seed)] +
                                        list(np.array(start_end_year, str)))


rg.pp_precursors(detrend=[True, {'tp':False, 'smi':False, 'swvl1':False, 'swvl3':False}],
                 anomaly=[True, {'tp':False, 'smi':False, 'swvl1':False, 'swvl3':False}],
                 auto_detect_mask=[False, {'swvl1':True, 'swvl2':True}])
rg.pp_TV(name_ds=name_ds, detrend=True, ext_annual_to_mon=False)
rg.traintest(method, seed=seed, subfoldername=subfoldername)
n_spl = rg.df_splits.index.levels[0].size

ds = core_pp.import_ds_lazy(rg.list_of_name_path[1][1])
#%%
# ds = ds.where(ds.values>0)
# ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
# ds.attrs['units'] = 'mm'
# ds.to_netcdf(functions_pp.get_download_path() +'/SM_for_SPI.nc')

# ds = ds.transpose('time', 'latitude', 'longitude')
# ds.to_netcdf(functions_pp.get_download_path() +'/SMI.nc')
# ds = core_pp.import_ds_lazy(functions_pp.get_download_path() +'/SMI.nc')
#%%
rg.calc_corr_maps('sst')
sst = rg.list_for_MI[0]
sst.corr_xr['lag'] = ('lag', SST_periodnames)

#%%
rg.calc_corr_maps('smi')
SM = rg.list_for_MI[1]
SM.corr_xr['lag'] = ('lag', SM_periodnames)

#%%
save = False
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':0,
                  'wspace':-.15, 'size':3, 'cbar_vert':0.05,
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                   'y_ticks':np.arange(-10,61,20), #'x_ticks':np.arange(130, 280, 25),
                  'title':'',
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('sst', kwrgs_plot=kwrgs_plotcorr_sst, min_detect_gc=.1, save=save)

#%%
kwrgs_plotcorr_SM = {'row_dim':'lag', 'col_dim':'split','aspect':2, 'hspace':0.2,
                      'wspace':0, 'size':3, 'cbar_vert':0.04,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                       'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                      'title':'',
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('smi', kwrgs_plot=kwrgs_plotcorr_SM, save=save)


#%%

sst.distance_eps = 225 ; sst.min_area_in_degrees2 = 4
rg.cluster_list_MI('sst')
sst_prec_labels = sst.prec_labels.copy()
# rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plotcorr_sst)

# # Ensure that Caribean is alway splitted from Pacific
# sst = rg.list_for_MI[0]
# copy_labels = sst.prec_labels.copy()
# all_labels = copy_labels.values[~np.isnan(copy_labels.values)]
# uniq_labels = np.unique(all_labels)
# prevail = {l:list(all_labels).count(l) for l in uniq_labels}
# prevail = functions_pp.sort_d_by_vals(prevail, reverse=True)
# label = list(prevail.keys())[0]
# sst.prec_labels, _ = find_precursors.split_region_by_lonlat(sst.prec_labels.copy(),
#                                                       label=int(label), plot_l=3,
#                                                       kwrgs_mask_latlon={'bottom_left':(95,15)})

merge = find_precursors.merge_labels_within_lonlatbox

# # Ensure that what is in Atlantic is one precursor region
# lonlatbox = [260, 350, 17, 35]
# sst.prec_labels = merge(sst, lonlatbox)
# Indonesia_oceans = [110, 150, 0, 10]
# sst.prec_labels = merge(sst, Indonesia_oceans)
# Japanese_sea = [100, 140, 30, 50]
# sst.prec_labels = merge(sst, Japanese_sea)
Mediterrenean_sea = [0, 45, 30, 50]
sst.prec_labels = merge(sst, Mediterrenean_sea)
# East_Tropical_Atlantic = [330, 20, -10, 10]
# sst.prec_labels = merge(sst, East_Tropical_Atlantic)
rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plotcorr_sst, min_detect_gc=1, save=save)
#%%
SM = rg.list_for_MI[1]
SM.distance_eps = 270 ; SM.min_area_in_degrees2 = 4
rg.cluster_list_MI('smi')

lonlatbox = [220, 240, 25, 50] # eastern US
SM.prec_labels = merge(SM, lonlatbox)
lonlatbox = [270, 280, 25, 45] # mid-US
SM.prec_labels = merge(SM, lonlatbox)

rg.quick_view_labels('smi', kwrgs_plot=kwrgs_plotcorr_SM, min_detect_gc=1, save=save)

#%%

rg.get_ts_prec()
rg.df_data = rg.df_data.rename({rg.df_data.columns[0]:target_dataset},axis=1)
# # fill first value of smi (NaN because of missing December when calc smi
# # on month februari).
# keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='smi']
# rg.df_data[keys] = rg.df_data[keys].fillna(value=0)

df_data = rg.get_subdates_df(years=(1951, 2019))


#%% Causal Inference
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

    # if add_previous_periods and periodnames.index(months) != 0:
    #     # merging CD keys that were found for each split seperately to keep
    #     # clean train-test split
    #     for s in range(n_spl):
    #         pm = periodnames[periodnames.index(months) - 1 ]
    #         pmk = [k for k in CondDepKeysDict[pm][s] if k.split('..')[0]==pm]
    #         keys_dict[s] = keys_dict[s] + pmk


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




RFmodel = ScikitModel(RandomForestClassifier, verbosity=0)
kwrgs_model={'n_estimators':200,
            'max_depth':2,
            'scoringCV':'neg_brier_score',
            'oob_score':True,
            # 'min_samples_leaf':None,
            'random_state':0,
            'max_samples':.3,
            'n_jobs':1}

# keys = rg.df_data.columns[:-2]
# keys = ['MAM..4..sst', 'JJA..4..sst']

out = rg.fit_df_data_ridge(target=target_ts,
                            keys=keys,
                            fcmodel=RFmodel,
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
print(RFmodel.scikitmodel.__name__, '\n', 'Test score\n',
      'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
      'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
      '\nTrain score\n',
      'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
      'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))

# [model_lags['lag_0'][f'split_{i}'].best_params_ for i in range(17)]

#%% Event prediciton with logistic regressiong


logit_skl = ScikitModel(LogisticRegressionCV, verbosity=0)
kwrgs_model = {'kfold':10,
               'scoring':'neg_brier_score'}

target_ts = rg.TV.RV_ts
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
target_ts = (target_ts > target_ts.quantile(.60)).astype(int)

mon = 8
keys = [str(mon)+k for k in CondDepKeys['JJA']]

out = rg.fit_df_data_ridge(target=target_ts,
                            keys=keys,
                            fcmodel=logit_skl,
                            kwrgs_model=kwrgs_model,
                            transformer=fc_utils.standardize_on_train,
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


