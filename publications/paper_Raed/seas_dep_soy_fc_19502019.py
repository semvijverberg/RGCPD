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

os.chdir('/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper_Raed/')
user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
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


experiments = ['seasons', 'bimonthly', 'semestral']
target_datasets = ['USDA_Soy', 'USDA_Maize', 'GDHY_Soy']
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
    target_dataset = out[0]
    experiment = out[1]
    print(f'arg {args.intexper} {out}')
else:
    out = combinations[i_default]
    target_dataset = out[0]
    experiment = out[1]


if target_dataset == 'GDHY_Soy':
    # GDHY dataset 1980 - 2015
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/q50_nc4_dendo_707fb.nc')
    cluster_label = 3
    name_ds='ts'
    start_end_year = (1980, 2015)
elif target_dataset == 'USDA_Soy':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts.nc')
    name_ds='Soy_Yield' ; cluster_label = None
    start_end_year = (1950, 2019)
elif target_dataset == 'USDA_Maize':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_maize_spatial_mean_ts.nc')
    name_ds='Maize_Yield' ; cluster_label = None
    start_end_year = (1950, 2019)


tfreq = 90 if experiment == 'seasons' else None
calc_ts='region mean' # pattern cov
alpha_corr = .05
method     = 'ran_strat10' ; seed = 1
n_boot = 2000
if experiment == 'seasons':
    corlags = np.array([
                        ['12-01', '02-28'], # DJF
                        ['03-01', '05-31'], # MAM
                        ['06-01', '08-31'], # JJA
                        ['09-01', '11-30'] # SON
                        ])
    periodnames = ['DJF','MAM','JJA','SON']
elif experiment == 'bimonthly':
    corlags = np.array([
                        ['03-01', '04-30'], # MA
                        ['05-01', '06-30'], # MJ
                        ['07-01', '08-31'], # JA
                        ['09-01', '10-31']  # SO
                        ])
    periodnames = ['March-April','May-June','July-August','Sept-Oct']
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
USBox = (225, 300, 20, 60)

start_end_year = (1960, 2019) #!!!

#%% run RGPD
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
list_of_name_path = [(cluster_label, TVpath),
                       ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc')),
                      # ('z500', os.path.join(path_raw, 'z500_1950-2019_1_12_monthly_1.0deg.nc')),
                       ('smi3', os.path.join(path_raw, 'SM_spi_gamma_03_1950-2019_1_12_monthly_1.0deg.nc')),
                      # ('swvl1', os.path.join(path_raw, 'swvl1_1950-2019_1_12_monthly_1.0deg.nc')),
                      ('swvl1', os.path.join(path_raw, 'swvl1_1950-2019_1_12_monthly_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={},
                            distance_eps=350, min_area_in_degrees2=4,
                            calc_ts=calc_ts, selbox=GlobalBox,
                            lags=corlags),
                 BivariateMI(name='smi3', func=class_BivariateMI.corr_map,
                             alpha=.05, FDR_control=True,
                             kwrgs_func={},
                             distance_eps=290, min_area_in_degrees2=4,
                             calc_ts=calc_ts, selbox=USBox,
                             lags=np.array(
                                 [[l[0].replace(l[0][:2],l[1][:2]),l[1]] for l in corlags]
                                 )
                             )]


rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=start_end_year,
           tfreq=tfreq,
           path_outmain=path_out_main)
precur = rg.list_for_MI[0] ; lag = precur.lags[0]

subfoldername = target_dataset+'_'.join(['', experiment, str(method), 's'+ str(seed)])


rg.pp_precursors(detrend=[True, {'tp':False, 'smi3':False, 'swvl1':False, 'swvl3':False}],
                 anomaly=[True, {'tp':False, 'smi3':False, 'swvl1':False, 'swvl3':False}],
                 auto_detect_mask=[False, {'swvl1':True, 'swvl2':True}])
rg.pp_TV(detrend=True)
rg.traintest(method, subfoldername=subfoldername)
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
save = True
kwrgs_plotcorr = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':0,
                  'wspace':-.15, 'size':3, 'cbar_vert':0.05,
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  # 'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(130, 280, 25),
                  'title':'',
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr(plotlags=periodnames[:4], kwrgs_plot=kwrgs_plotcorr, save=save)

#%%

kwrgs_plotlabels = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':0,
              'wspace':-.15, 'size':3, 'cbar_vert':0.05,
              'units':'Corr. Coeff. [-]',
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              # 'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(130, 280, 25),
              'title':'',
              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
# precur.distance_eps = 350 ; precur.min_area_in_degrees2 = 4
rg.cluster_list_MI()
rg.quick_view_labels(kwrgs_plot=kwrgs_plotlabels)
# prec_labels, _ = find_precursors.split_region_by_lonlat(precur.prec_labels.copy(),
#                                                      label=3, plot_l=2,
#                                                      kwrgs_mask_latlon={'latmax':32})
# prec_labels, _ = find_precursors.split_region_by_lonlat(prec_labels.copy(), label=3,
#                                        plot_l=2, kwrgs_mask_latlon={'lonmax':190})
# precur.prec_labels = prec_labels
rg.quick_view_labels(kwrgs_plot=kwrgs_plotlabels, save=save)

#%%
rg.get_ts_prec()
rg.df_data = rg.df_data.rename({'Nonets':target_dataset},axis=1)
#%% get PDO
# SST_pp_filepath = rg.list_precur_pp[0][1]

# if 'df_PDOsplit' not in globals():
#     df_PDO, PDO_patterns = climate_indices.PDO(SST_pp_filepath,
#                                                None) #rg.df_splits)
#     # summerdates = core_pp.get_subdates(dates, start_end_TVdate)
#     df_PDOsplit = df_PDO.loc[0]#.loc[summerdates]
#     plot_maps.plot_corr_maps(PDO_patterns)

# df_PDOsplit = df_PDOsplit[['PDO']].apply(fc_utils.standardize_on_train,
#                      args=[df_PDO.loc[0]['TrainIsTrue']],
#                      result_type='broadcast')


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
                'alphas':np.logspace(.1, 1.5, num=25), # large a, strong regul.
                'normalize':False}


def append_dict(month, df_test_m):
    dkeys = [f'{month} RMSE-SS', f'{month} Corr.']
    append_dict = {dkeys[0]:float(df_test_m.iloc[:,0].round(3)),
                   dkeys[1]:float(df_test_m.iloc[:,1].round(3))}
    dict_v.update(append_dict)
    return



blocksize=1
lag = 0
alpha_CI = .05
variables = ['sst', 'smi3']
add_previous_periods = False
add_PDO = False
PDO_aggr_periods = np.array([
                            ['03-01', '02-28'],
                            ['06-01', '05-31'],
                            ['09-01', '08-31'],
                            ['12-01', '11-30']
                            ])


dict_v = {'lag':lag,'Seed':f's{seed}'}

fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
# CRPSS = fc_utils.CRPSS_vs_constant_bench(constant_bench=clim_mean_temp).CRPSS
score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]

list_test = []
list_test_b = []
no_info_fc = []
CondDepKeys = {} ; CondDepKeysDict = {}
for i, months in enumerate(periodnames[:]):
    # months, start_end_TVdate = periodnames[2], corlags[2]
    # i = 0; months = periodnames[0]

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

    keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
    if len(keys) != 0 and 'PDO' not in keys:
        corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data.copy(), z_keys=keys, keys=keys)
        # removing all keys that are Cond. Indep. in each trainingset

        for s in rg.df_splits.index.levels[0]:
            for k in keys:
                if (np.nan_to_num(pvals.loc[k][s],nan=alpha_CI) > alpha_CI).mean()>0:
                    k_ = keys_dict[s].copy() ; k_.pop(k_.index(k))
                    keys_dict[s] = k_
                    # keys_dict[s] = k
        # always C.D. (every split)
        keys = [k for k in keys if (np.nan_to_num(pvals.loc[k],nan=alpha_CI) <= alpha_CI).mean()== 1]
    CondDepKeys[months] = keys
    CondDepKeysDict[months] = keys_dict

    if add_previous_periods:
        keys = functions_pp.flatten([d for d in CondDepKeys.values()])
        print(months, keys)


    if len(keys) != 0:
        out = rg.fit_df_data_ridge(target=target_ts,
                                   keys=keys_dict,
                                   tau_min=0, tau_max=0,
                                   kwrgs_model=kwrgs_model,
                                   fcmodel=fcmodel,
                                   transformer=fc_utils.standardize_on_train)

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
                                                                 rng_seed=1)

        # n_splits = rg.df_data.index.levels[0].size
        # cvfitalpha = [models_lags[f'lag_{lag}'][f'split_{s}'].alpha_ for s in range(n_splits)]
        # print('mean alpha {:.0f}'.format(np.mean(cvfitalpha)))
        # maxalpha_c = list(cvfitalpha).count(alphas[-1])
        # if maxalpha_c > n_splits/3:
        #     print(f'\nalpha {int(np.mean(cvfitalpha))}')
        #     print(f'{maxalpha_c} splits are max alpha\n')
        #     # maximum regularization selected. No information in timeseries
        #     df_test_m['Prediction']['corrcoef'][:] = 0
        #     df_boot['Prediction']['corrcoef'][:] = 0
        #     no_info_fc.append(months)
        df_test = functions_pp.get_df_test(prediction.merge(rg.df_data.iloc[:,-2:],
                                                        left_index=True,
                                                        right_index=True)).iloc[:,:2]

    else:
        print('no precursor timeseries found, scores all 0')
        metric_names = [s.__name__ for s in score_func_list]
        index = pd.MultiIndex.from_product([['Prediction'],metric_names])
        df_boot = pd.DataFrame(data=np.zeros((n_boot, len(score_func_list))),
                            columns=index)
        df_test_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
                                  columns=index)


    list_test_b.append(df_boot)
    list_test.append(df_test_m)
    # append_dict(month, df_test_m)
    # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
    #                       function=df_ana.plot_timeseries,
    #                       kwrgs={'timesteps':rg.fullts.size,
    #                                   'nth_xyear':5})


#%%

import matplotlib.patches as mpatches
corrvals = [test.values[0,1] for test in list_test]
MSE_SS_vals = [test.values[0,0] for test in list_test]
MAE_SS_vals = [test.values[0,-1] for test in list_test]
rename_metrics = {'RMSE-SS':'RMSE',
                  'Corr. Coef.':'corrcoef',
                  'MAE-SS':'MAE'}
df_scores = pd.DataFrame({'RMSE-SS':MSE_SS_vals,
                          'Corr. Coef.':corrvals,
                          'MAE-SS':MAE_SS_vals},
                         index=list(CondDepKeys.keys()))
df_test_b = pd.concat(list_test_b, keys = list(CondDepKeys.keys()),axis=1)

yerr = [] ; quan = [] ; alpha = .1
# for i in range(len(periodnames) * df_scores.columns.size):
monmet = np.array(np.meshgrid(list(CondDepKeys.keys()),
                              df_scores.columns)).T.reshape(-1,2) ;
for i, (mon, met) in enumerate(monmet):
    Eh = 1 - alpha/2 ; El = alpha/2
    met = rename_metrics[met]
    _scores = df_test_b[mon]['Prediction'][met]
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
             fontsize=16)
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

# # manually define a new patch
# if len(no_info_fc) != 0:
#     patch = mpatches.Patch(color='red', label=r'$\alpha=$'+f' ${int(alphas[-1])}$')
#     legend2 = ax.legend(loc='upper left', handles=[patch],
#           fontsize=16, frameon=True, facecolor='grey',
#           framealpha=.5)
#     ax.add_artist(legend2)

ax.set_ylim(-0.3, 1)

append_str +'_'+'_'.join(np.array(start_end_year, str))
plt.savefig(os.path.join(rg.path_outsub1,
              f'skill_score_vs_months_a{alpha}_nb{n_boot}_blsz{blocksize}_ac{alpha_corr}_{append_str}.pdf'))

#%%
CDlabels = precur.prec_labels.copy() ; CDl = np.zeros_like(CDlabels)
CDcorr = precur.corr_xr.copy()
for i, month in enumerate(CondDepKeys):
    CDkeys = CondDepKeys[month]
    region_labels = [int(l.split('..')[1]) for l in CDkeys]
    f = find_precursors.view_or_replace_labels
    CDlabels[:,i] = f(CDlabels[:,i].copy(), region_labels)
mask = (np.isnan(CDlabels)).astype(bool)
plot_maps.plot_labels(CDlabels.mean(dim='split'), kwrgs_plot=kwrgs_plotlabels)
if save:
    plt.savefig(os.path.join(rg.path_outsub1,
                         f'CondDep_labels_{append_str}_ac{alpha_corr}_eps{precur.distance_eps}'
                         f'minarea{precur.min_area_in_degrees2}'+rg.figext))

fig = plot_maps.plot_corr_maps(CDcorr.mean(dim='split'),
                         mask_xr=mask, **kwrgs_plotcorr)
if save:
    fig.savefig(os.path.join(rg.path_outsub1,
                         f'CondDep_{append_str}_ac{alpha_corr}'+rg.figext))

#%%

# rg.cluster_list_MI()
# zz1 = precur.prec_labels.copy()
# zz2 = find_precursors.view_or_replace_labels(zz1, [1,2])

#%%
# rg.plot_maps_corr(plotlags=month, region_labels=region_labels,
#                       kwrgs_plot=kwrgs_plotcorr)
#%% Re-aggregated SST data of regions with time_mean_bins

def reaggregated_regions(precur, rg, start_end_date1, precur_aggr):
    splits = rg.df_data.index.levels[0]
    rg.kwrgs_load['start_end_date'] = start_end_date1
    rg.kwrgs_load['closed_on_date'] = start_end_date1[-1]
    df_data1 = find_precursors.spatial_mean_regions(precur, precur_aggr=precur_aggr,
                                                   kwrgs_load=rg.kwrgs_load)
    df_data1 = pd.concat(df_data1, keys = splits)

    CondDepKeysList = core_pp.flatten([l for l in CondDepKeys.values()])
    CondDepKeysList = [k for k in CondDepKeysList if k.split('..')[-1] == precur.name]
    df_sub1 = df_data1[CondDepKeysList]

    ts_corr = np.zeros( (splits.size), dtype=object)
    for s in splits:
        l = []
        for yr in rg.dates_TV.year:
            singleyr1 = df_sub1.loc[s].loc[functions_pp.get_oneyr(df_sub1.loc[s], yr)].T
            newcols = []
            for col in singleyr1.index:
                newcols.append([str(m)+col for m in singleyr1.columns.month])
            singleyr1 = pd.DataFrame(singleyr1.values.reshape(1,-1),
                         columns=core_pp.flatten(newcols),
                         index=pd.to_datetime([f'{yr}-01-01']))
            l.append(singleyr1)
            df_s = pd.concat(l)
        ts_corr[s] = df_s
    df_data_yrs = pd.merge(pd.concat(ts_corr, keys=splits),
                           rg.df_splits, left_index=True, right_index=True)
    return df_data_yrs

df_data_yrs_sst1 = reaggregated_regions(rg.list_for_MI[0], rg, ('01-01', '08-01'), 2)
df_data_yrs_sst2 = reaggregated_regions(rg.list_for_MI[0], rg, ('01-01', '09-01'), 2)

df_data_yrs_smi = reaggregated_regions(rg.list_for_MI[1], rg, ('02-01', '09-01'), 1)
df_data_yrs = pd.concat([df_data_yrs_sst1.iloc[:,:-2],
                         df_data_yrs_sst2.iloc[:,:-2],
                         df_data_yrs_smi], axis=1)

rg.df_data = df_data_yrs
fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
target_ts = rg.TV.RV_ts
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
target_ts = (target_ts > target_ts.mean()).astype(int)


#%% forecast Yield as function of months

assert all([CondDepKeysDict['JJA'][s]==CondDepKeys['JJA'] for s in range(n_spl)]), 'Yo'

prediction = 'continuous' ; q = None
prediction = 'events' ; q = .4

if prediction == 'continuous':
    model = ScikitModel(verbosity=0)
    kwrgs_model = {'scoring':'neg_mean_squared_error',
                    'alphas':np.logspace(.01, 1, num=25), # large a, strong regul.
                    'normalize':False}
elif prediction == 'events':
    model = ScikitModel(LogisticRegressionCV, verbosity=0)
    kwrgs_model = {'kfold':5,
                   'scoring':'neg_brier_score'}
target_ts = rg.TV.RV_ts ; target_ts.columns = [target_dataset]
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
if prediction == 'events':
    if q >= 0.5:
        target_ts = (target_ts > target_ts.quantile(q)).astype(int)
    elif q < .5:
        target_ts = (target_ts < target_ts.quantile(q)).astype(int)
    BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS

rg.df_data = df_data_yrs.copy() ;
tau_min=0 ; tau_max=0
list_df_test = [] ; list_df_train = [] ; list_df_boot = [] ; list_pred_test = []
for mon in range(5,10):
# mon = 10
    keys = [str(mon)+k for k in CondDepKeys['JJA']]
    keys = [k for k in keys if k in rg.df_data.columns]

    out = rg.fit_df_data_ridge(target=target_ts,
                                keys=keys,
                                fcmodel=model,
                                kwrgs_model=kwrgs_model,
                                transformer=fc_utils.standardize_on_train,
                                tau_min=tau_min, tau_max=tau_max)
    predict, weights, model_lags = out

    weights_norm = weights.mean(axis=0, level=1)
    weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

    if prediction == 'continuous':
        score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
    elif prediction == 'events':
        score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                     rg.df_data.iloc[:,-2:],
                                                                     score_func_list,
                                                                     n_boot = n_boot,
                                                                     score_per_test=False,
                                                                     blocksize=1,
                                                                     rng_seed=1)
    if prediction == 'continuous':
        [model_lags['lag_0'][f'split_{i}'].alpha_ for i in range(n_spl)]
        print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'RMSE {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
              'MAE {:.2f}\n'.format(df_test_m.loc[0][0]['MAE']),
              'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
              '\nTrain score\n',
              'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
              'MAE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['MAE']),
              'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))
    elif prediction == 'events':
        [model_lags['lag_0'][f'split_{i}'].Cs for i in range(n_spl)]
        print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
              'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
              '\nTrain score\n',
              'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
              'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))


    list_pred_test.append(functions_pp.get_df_test(predict.rename({0:mon}, axis=1),
                                                 cols=[mon],
                                                 df_splits=rg.df_splits))
    df_test_m.index = [mon] ;
    columns = pd.MultiIndex.from_product([np.array([mon]),
                                        df_train_m.columns.levels[1]])
    df_train_m.columns = columns
    df_boot.columns = columns
    list_df_test.append(df_test_m) ; list_df_train.append(df_train_m)
    list_df_boot.append(df_boot)

df_scores = pd.concat(list_df_test).T.loc[0]
df_train_score = pd.concat(list_df_train, axis=1)
df_boots = pd.concat(list_df_boot, axis=1)
list_pred_test.insert(0, target_ts)
df_pred_test = pd.concat(list_pred_test, axis=1)

plot_vs_lags(df_scores, target_ts, df_boots, alpha=.1)

#%% Cascaded forecast


# part 1
include_obs = True ; max_multistep = 2
list_df_test_c = [] ; list_df_boot_c = [] ;
list_df_train_c = [] ; list_pred_test_c = []
fc_month = {} ;
for forecast_month in range(5,8):
    print(f'Forecast month {forecast_month}')
    target_month = 8

    rg.df_data = df_data_yrs.copy()
    lag_max = 2 ;
    precur_keys = {}
    for i, mon in enumerate(range(forecast_month, target_month)):
        model = ScikitModel(verbosity=0)
        kwrgs_model = {'scoring':'neg_mean_absolute_error',
                        'alphas':np.logspace(-1, 1, num=20), # large a, strong regul.
                        'normalize':False}
        print(f'predicting precursors month {mon+1}')
        if (mon - forecast_month) == max_multistep: # if max lag, stop multi-step forecasting
            continue

        predict_precursors = []
        precur_target_keys = [str(mon+1)+k for k in CondDepKeys['JJA']]
        for j, k in enumerate(precur_target_keys):

            # if i == 0:
            keys = [str(mon)+k for k in CondDepKeys['JJA']] #+ ['PDO']
            for l in range(1, lag_max):
                keys += [str(mon-l*2)+k for k in CondDepKeys['JJA']]
            keys = [k for k in keys if k in rg.df_data.columns] # check if present
            if i > 0:
                # add precursor timeseries of month? not yet implemented

                # add future precursor target to train model
                rg.df_data = pd.merge(df_data_yrs[[k]],
                                      rg.df_data, left_index=True, right_index=True)
            corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data.copy(), z_keys=keys,
                                            keys=keys,
                                            target=k)
            # removing all keys that are Cond. Indep. in each trainingset
            keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
            for s in rg.df_splits.index.levels[0]:
                for k_i in keys:
                    if (np.nan_to_num(pvals.loc[k_i][s],nan=alpha_CI) > alpha_CI).mean()>0:
                        k_ = keys_dict[s].copy() ; k_.pop(k_.index(k_i))
                        keys_dict[s] = k_
                if include_obs:
                    k_ = keys_dict[s].copy() ;
                    keys_dict[s] = k_

            precur_keys[k] = keys_dict
            keys = keys_dict


            if len([True for k,v in keys_dict.items() if len(keys_dict[k])==0])!=0:
                print('No keys left')
                # k_ += [str(forecast_month)+k for k in CondDepKeys['JJA']]
                keys = [str(forecast_month)+k for k in CondDepKeys['JJA']]
                rg.df_data = pd.merge(df_data_yrs[[k]],
                                      df_data_yrs[keys + ['TrainIsTrue', 'RV_mask']],
                                      left_index=True, right_index=True)


            target_ts = df_data_yrs[[k]].mean(0,level=1)
            out = rg.fit_df_data_ridge(target=target_ts,
                                        keys=keys,
                                        fcmodel=model,
                                        kwrgs_model=kwrgs_model,
                                        transformer=fc_utils.standardize_on_train,
                                        tau_min=0, tau_max=0)
            predict, weights, model_lags = out
            # weights_norm = weights.mean(axis=0, level=1)
            # weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')
            score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
            df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                             rg.df_data.iloc[:,-2:],
                                                                             score_func_list,
                                                                             n_boot = 1,
                                                                             score_per_test=False,
                                                                             blocksize=1,
                                                                             rng_seed=1)
            print(model.scikitmodel.__name__+' '+k, '\t', 'Test score: ',
                  # 'RMSE {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
                  'MAE {:.2f}\t'.format(df_test_m.loc[0][0]['MAE']),
                  # 'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
                  'Train score: ',
                  # 'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
                  'MAE {:.2f}\t'.format(df_train_m.mean(0).loc[0]['MAE']))
                  # 'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))

            [model_lags['lag_0'][f'split_{i}'].alpha_ for i in range(n_spl)]
            predict_precursors.append(predict.iloc[:,1:].rename({0:k},axis=1))
        cascade_fc = pd.concat(predict_precursors, axis=1)
        rg.df_data = pd.merge(cascade_fc.iloc[:,:],
                               rg.df_splits, left_index=True, right_index=True)

# =============================================================================
#     #% Part 2 of Cascaded forecast
# =============================================================================

    if prediction == 'continuous':
        model = ScikitModel(verbosity=0)
        kwrgs_model = {'scoring':'neg_mean_squared_error',
                        'alphas':np.logspace(-1, .5, num=25), # large a, strong regul.
                        'normalize':False}
    elif prediction == 'events':
        model = ScikitModel(LogisticRegressionCV, verbosity=0)
        kwrgs_model = {'kfold':5,
                       'Cs':np.logspace(-1.5, .5, num=25),
                       'scoring':'neg_brier_score'}

    target_ts = rg.TV.RV_ts ; target_ts.columns = [target_dataset]
    target_ts = (target_ts - target_ts.mean()) / target_ts.std()
    if prediction == 'events':
        if q >= 0.5:
            target_ts = (target_ts > target_ts.quantile(q)).astype(int)
        elif q < .5:
            target_ts = (target_ts < target_ts.quantile(q)).astype(int)
        BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS


    rg.df_data = pd.merge(cascade_fc.iloc[:,:],
                               rg.df_splits, left_index=True, right_index=True)
    rg.df_data = pd.merge(pd.concat([rg.TV.RV_ts]*n_spl, keys=range(n_spl)),
                          rg.df_data, left_index=True, right_index=True)

    keys = list(rg.df_data.columns[1:-2])
    corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data.copy(), z_keys=keys,
                                                keys=keys)

    # removing all keys that are Cond. Indep. in each trainingset
    keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
    for s in rg.df_splits.index.levels[0]:
        for k_i in keys:
            if (np.nan_to_num(pvals.loc[k_i][s],nan=alpha_CI) > alpha_CI).mean()>0:
                k_ = keys_dict[s].copy() ; k_.pop(k_.index(k_i))
                keys_dict[s] = k_

    fc_month[mon] = keys_dict


    out = rg.fit_df_data_ridge(target=target_ts,
                                keys=keys_dict,
                                fcmodel=model,
                                kwrgs_model=kwrgs_model,
                                transformer=fc_utils.standardize_on_train,
                                tau_min=0, tau_max=0)
    predict, weights, model_lags = out

    weights_norm = weights.mean(axis=0, level=1)
    weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')


    if prediction == 'continuous':
        score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
    elif prediction == 'events':
        score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                     rg.df_data.iloc[:,-2:],
                                                                     score_func_list,
                                                                     n_boot = n_boot,
                                                                     score_per_test=False,
                                                                     blocksize=1,
                                                                     rng_seed=1)
    if prediction == 'continuous':
        [model_lags['lag_0'][f'split_{i}'].alpha_ for i in range(n_spl)]
        print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'RMSE {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
              'MAE {:.2f}\n'.format(df_test_m.loc[0][0]['MAE']),
              'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
              '\nTrain score\n',
              'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
              'MAE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['MAE']),
              'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))
    elif prediction == 'events':
        [model_lags['lag_0'][f'split_{i}'].C_ for i in range(n_spl)]
        print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
              'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
              '\nTrain score\n',
              'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
              'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))

    list_pred_test_c.append(functions_pp.get_df_test(predict.rename({0:forecast_month},
                                                                    axis=1),
                                                 cols=[forecast_month],
                                                 df_splits=rg.df_splits))
    df_test_m.index = [forecast_month] ;
    columns = pd.MultiIndex.from_product([np.array([forecast_month]),
                                        df_train_m.columns.levels[1]])
    df_train_m.columns = columns
    df_boot.columns = columns
    list_df_test_c.append(df_test_m) ; list_df_train_c.append(df_train_m)
    list_df_boot_c.append(df_boot)


df_scores_c = pd.concat(list_df_test_c).T.loc[0]
df_train_score_c = pd.concat(list_df_train_c, axis=1)
df_boots_c = pd.concat(list_df_boot_c, axis=1)
list_pred_test_c.insert(0, target_ts)
df_pred_test_c = pd.concat(list_pred_test_c, axis=1)


plot_vs_lags(df_scores_c, target_ts, df_boots_c)

#%%
df_scores_dict = {'forecast':[df_scores,df_boots],
                 'cascaded forecast': [df_scores_c,df_boots_c]}


def plot_vs_lags(df_scores_dict, target_ts, df_boots_dict=None, rename_m: dict=None, orientation='vertical',
                 colorlist: list=['#3388BB', '#EE6666', '#9988DD'], alpha=.05):

    if rename_m is None:
        rename_m = {i:i for i in df_scores.index}
    else:
        rename_m = rename_m
    metrics_cols = list(rename_m.values())
    if orientation=='vertical':
        f, ax_ = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                         sharex=True) ;
    else:
        f, ax_ = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
                         sharey=False) ;
    if type(df_scores_dict) is not dict:
        df_scores_dict = {'forecast': [df_scores_dict, df_boots_dict]}

    for i, forecast_label in enumerate(df_scores_dict):

        df_scores_ = df_scores_dict[forecast_label][0]
        df_boots_ = df_scores_dict[forecast_label][1]
        for j, m in enumerate(metrics_cols):

            ax = ax_[j]
            ax.plot(df_scores_.columns, df_scores_.T[m],
                    label=forecast_label,
                    color=colorlist[i],
                    linestyle='solid')
            ax.set_xticks(df_scores_.columns)
            ax.set_xticklabels(df_scores_.columns)
            ax.set_xlabel('Month of forecast', fontsize=15)
            ax.set_ylabel(rename_m[m], fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.legend()
            ax.set_ylim(0,1)
            ax.fill_between(df_scores_.columns,
                            df_boots_.reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                            df_boots_.reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                            edgecolor=colorlist[i], facecolor=colorlist[i], alpha=0.3,
                            linestyle='solid', linewidth=2)

            if m == 'corrcoef':
                ax.set_ylim(-.3,1)
            elif m == 'roc_auc_score':
                ax.set_ylim(0,1)
            else:
                ax.set_ylim(-.2,.8)

    f.tight_layout()

plot_vs_lags(df_scores_dict, target_ts, alpha=.1)

#%% Event prediciton with RF


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


