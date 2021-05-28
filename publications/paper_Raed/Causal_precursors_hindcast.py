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

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']


target_datasets = All_states
seeds = seeds = [1,2] # ,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
methods = ['random_20'] # ['ranstrat_20']
feature_sel = [True]
combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    yrs,
                                    methods,
                                    feature_sel)).T.reshape(-1,5)
i_default = 0


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



def read_csv_State(path, State: str=None, col='obs_yield'):
    orig = read_csv_Raed(path)
    orig = orig.set_index('State', append=True)
    orig = orig.pivot_table(index='time', columns='State')[col]
    if State is None:
        State = orig.columns
    return orig[State]

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
elif target_dataset == 'USDA_Soy_always_data':
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts_allways_data.nc')
    name_ds='Soy_Yield' ; cluster_label = ''
elif target_dataset == 'USDA_Soy_csv_midwest':
    path = os.path.join(main_dir, 'publications/paper_Raed/data/ts_spatial_avg_midwest.csv')
    TVpath = read_csv_Raed(path)
elif target_dataset.split('__')[0] == 'USDA_Soy_clusters':
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/linkage_ward_nc4_dendo_ee0e9.nc')
    cluster_label = int(target_dataset.split('__')[1]) ; name_ds = 'ts'
elif target_dataset == 'USDA_Maize':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_maize_spatial_mean_ts.nc')
    name_ds='Maize_Yield' ; cluster_label = None
else:
    path =  os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')
    TVpath = read_csv_State(path, State=target_dataset, col='obs_yield')
    TVpath = pd.DataFrame(TVpath.values, index=TVpath.index, columns=[TVpath.name])
    name_ds='Soy_Yield' ; cluster_label = ''


experiment = 'bimonthly'
calc_ts='region mean' # pattern cov
alpha_corr = .05
n_boot = 2000

if experiment == 'seasons':
    corlags = np.array([
                        ['12-01', '02-28'], # DJF
                        ['03-01', '05-31'], # MAM
                        ['06-01', '08-31'], # JJA
                        ['09-01', '11-30'] # SON
                        ])
    periodnames = ['DJF','MAM','JJA','SON']
    SM_lags = np.array([[l[0].replace(l[0][:2],l[1][:2]),l[1]] for l in corlags])
elif experiment == 'bimonthly':
    corlags = np.array([
                        ['03-01', '04-30'], # MA
                        ['05-01', '06-30'], # MJ
                        ['07-01', '08-31'], # JA
                        ['09-01', '10-31']  # SO
                        ])
    periodnames = ['MA','MJ','JA','SO']
    # SM_lags = corlags #np.array([
                        # ['05-01', '05-30'], # May
                        # ['06-01', '06-30'], # June
                        # ['07-01', '07-30'], # July
                        # ['08-01', '08-31']  # August
                        # ])
    # SM_periodnames = ['May', 'June', 'July', 'August']
    SM_lags = np.array([[l[0].replace(l[0][:2],l[1][:2]),l[1]] for l in corlags])
elif experiment == 'semestral':
    corlags = np.array([
                        ['12-01', '05-31'], # DJFMAM
                        ['03-01', '08-31'], # MAMJJA
                        ['06-01', '11-30'], # JJASON
                        ['01-01', '12-31']  # annual
                        ])
    # SM contains already 3 months aggregated value (aligned right), so below
    # calculates mean over 3 datapoints, encompassing 6 months of SMI.
    SM_lags = np.array([[l[0].replace(l[0][:2],str(int(l[1][:2])-2)),l[1]] for l in corlags])


    periodnames = ['DJFMAM', 'MAMJJA', 'JJASON', 'annual']

append_main = target_dataset
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'hindcast')
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
                            lags=corlags, use_coef_wghts=True),
                  BivariateMI(name='smi', func=class_BivariateMI.corr_map,
                             alpha=.05, FDR_control=True,
                             kwrgs_func={},
                             distance_eps=300, min_area_in_degrees2=4,
                             calc_ts='pattern cov', selbox=USBox,
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

subfoldername = target_dataset+'_hindcast/'+'_'.join([experiment, str(method),
                                         's'+ str(seed)])


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
rg.calc_corr_maps()
for p in rg.list_for_MI:
    p.corr_xr['lag'] = ('lag', periodnames)


#%%
save = True
min_detect_gc=.5
subtitles = np.array([['March-April mean'], ['May-June mean'],
                       ['July-Aug mean'], ['Sep-Oct mean']])
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4,
                      'hspace':.38, 'wspace':-.15, 'size':2, 'cbar_vert':0.07,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                      'y_ticks':False, 'x_ticks':False, #np.arange(-10,61,20), #'x_ticks':np.arange(130, 280, 25),
                      'title':'', 'subtitles':subtitles,
                      'subtitle_fontdict':{'fontsize':25},
                      'clevels':np.arange(-.8,.9,.1),
                      'clabels':np.arange(-.8,.9,.4),
                      'cbar_tick_dict':{'labelsize':25},
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('sst', kwrgs_plot=kwrgs_plotcorr_sst, save=save,
                  min_detect_gc=min_detect_gc)

#%%
kwrgs_plotcorr_SM = {'row_dim':'lag', 'col_dim':'split','aspect':2, 'hspace':0.25,
                      'wspace':0, 'size':3, 'cbar_vert':0.06,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                       # 'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                       'y_ticks':False, 'x_ticks':False,
                       'title':'', 'subtitles':subtitles,
                       'subtitle_fontdict':{'fontsize':30},
                       'clevels':np.arange(-.8,.9,.1),
                       'clabels':np.arange(-.8,.9,.4),
                       'cbar_tick_dict':{'labelsize':25},
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('smi', kwrgs_plot=kwrgs_plotcorr_SM, save=save,
                  min_detect_gc=min_detect_gc)


#%%
sst = rg.list_for_MI[0]
# sst.distance_eps = 425 ; sst.min_area_in_degrees2 = 7
sst.distance_eps = 200 ; sst.min_area_in_degrees2 = 3
rg.cluster_list_MI('sst')
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

# Ensure that what is in Atlantic is one precursor region
lonlatbox = [260, 330, 20, 40]
merge = find_precursors.merge_labels_within_lonlatbox
sst.prec_labels = merge(sst, lonlatbox)
# Indonesia_oceans = [110, 150, 0, 10]
# sst.prec_labels = merge(sst, Indonesia_oceans)
Japanese_sea = [100, 130, 30, 50]
sst.prec_labels = merge(sst, Japanese_sea)
Mediterrenean_sea = [0, 45, 30, 50]
sst.prec_labels = merge(sst, Mediterrenean_sea)

kwrgs_plotlabels_sst = kwrgs_plotcorr_sst.copy()
kwrgs_plotlabels_sst.pop('clevels'); kwrgs_plotlabels_sst.pop('clabels')
kwrgs_plotlabels_sst.pop('cbar_tick_dict')
kwrgs_plotlabels_sst['cbar_vert'] = 0
rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plotlabels_sst,
                     min_detect_gc=min_detect_gc, save=save)
#%%
SM = rg.list_for_MI[1]
SM.distance_eps = 200 ; SM.min_area_in_degrees2 = 3
rg.cluster_list_MI('smi')

lonlatbox = [220, 250, 25, 50] # eastern US
SM.prec_labels = merge(SM, lonlatbox)
lonlatbox = [270, 280, 25, 45] # mid-US
SM.prec_labels = merge(SM, lonlatbox)


kwrgs_plotlabels_SM = kwrgs_plotcorr_SM.copy()
kwrgs_plotlabels_SM.pop('clevels'); kwrgs_plotlabels_SM.pop('clabels')
kwrgs_plotlabels_SM.pop('cbar_tick_dict')
kwrgs_plotlabels_SM['cbar_vert'] = 0
rg.quick_view_labels('smi', kwrgs_plot=kwrgs_plotlabels_SM,
                     min_detect_gc=min_detect_gc, save=save)
#%%
load_sst = '{}_a{}_{}_{}_{}'.format(sst._name, sst.alpha,
                                            sst.distance_eps,
                                            sst.min_area_in_degrees2,
                                            periodnames[-1])
sst.store_netcdf(rg.path_outsub1, load_sst, add_hash=False)
load_SM = '{}_a{}_{}_{}_{}'.format(SM._name, SM.alpha,
                                           SM.distance_eps,
                                           SM.min_area_in_degrees2,
                                           periodnames[-1])
SM.store_netcdf(rg.path_outsub1, load_SM, add_hash=False)
for p in rg.list_for_MI:
    p.corr_xr['lag'] = ('lag', periodnames)
    p.prec_labels['lag'] = ('lag', periodnames)
#%%

rg.get_ts_prec()
rg.df_data = rg.df_data.rename({rg.df_data.columns[0]:target_dataset},axis=1)
# # fill first value of smi (NaN because of missing December when calc smi
# # on month februari).
# keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='smi']
# rg.df_data[keys] = rg.df_data[keys].fillna(value=0)



#%%
import wrapper_PCMCI
def feature_selection_CondDep(df_data, keys, z_keys=None, alpha_CI=.05, x_lag=0, z_lag=0):

    # Feature selection Cond. Dependence
    keys = list(keys) # must be list
    if z_keys is None:
        z_keys = keys
    corr, pvals = wrapper_PCMCI.df_data_Parcorr(df_data.copy(), keys=keys,
                                                z_keys=z_keys, z_lag=z_lag)
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


regress_autocorr_SM = False
unique_keys = np.unique(['..'.join(k.split('..')[1:]) for k in rg.df_data.columns[1:-2]])
list_pvals = [] ; list_corr = []
for k in unique_keys:
    z_keys = [z for z in rg.df_data.columns[1:-2] if k not in z]

    for mon in periodnames:
        keys = [mon+ '..'+k]
        if regress_autocorr_SM and 'sm' in k:
            z_keys = [z for z in rg.df_data.columns[1:-2] if keys[0] not in z]


        if keys[0] not in rg.df_data.columns:
            continue
        out = feature_selection_CondDep(rg.df_data.copy(), keys=keys,
                                        z_keys=z_keys, alpha_CI=.05)
        corr, pvals, keys_dict = out
        list_pvals.append(pvals.max(axis=0, level=0))
        list_corr.append(corr.mean(axis=0, level=0))


df_pvals = pd.concat(list_pvals,axis=0)
df_corr = pd.concat(list_corr,axis=0)
rg.df_pvals = df_pvals
rg.df_corr = df_corr

#%%

# from sklearn.ensemble import RandomForestClassifier

alpha_CI = .05
CondDepKeys = {} ;
for i, mon in enumerate(periodnames):
    list_mon = []
    _keys = [k for k in rg.df_pvals.index if mon in k] # month
    df_sig = rg.df_pvals[rg.df_pvals.loc[_keys] <= alpha_CI].dropna(axis=0, how='all') # significant

    for k in df_sig.index:
        corr_val = rg.df_corr.loc[k].mean()
        RB = (rg.df_pvals.loc[k]<alpha_CI).sum()
        list_mon.append((k, corr_val, RB))
    CondDepKeys[mon] = list_mon

from sklearn.linear_model import RidgeCV
def get_df_mean_SST(rg, mean_vars=['sst'], alpha_CI=.05,
                    n_strongest='all',
                    weights=True, labels=None,
                    fcmodel=None, kwrgs_model=None,
                    target_ts=None):


    periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)
    df_pvals = rg.df_pvals.copy()
    df_corr  = rg.df_corr.copy()
    unique_keys = np.unique(['..'.join(k.split('..')[1:]) for k in rg.df_data.columns[1:-2]])
    if labels is not None:
        unique_keys = [k for k in unique_keys if k in labels]

    # dict with strongest mean parcorr over growing season
    mean_SST_list = []
    # keys_dict = {s:[] for s in range(rg.n_spl)} ;
    keys_dict_meansst = {s:[] for s in range(rg.n_spl)} ;
    for s in range(rg.n_spl):
        mean_SST_list_s = []
        sign_s = df_pvals[s][df_pvals[s] <= alpha_CI].dropna(axis=0, how='all')
        for uniqk in unique_keys:
            # uniqk = '1..smi'
            # region label (R) for each month in split (s)
            keys_mon = [mon+ '..'+uniqk for mon in periodnames]
            # significant region label (R) for each month in split (s)
            keys_mon_sig = [k for k in keys_mon if k in sign_s.index] # check if sig.
            if uniqk.split('..')[-1] in mean_vars and len(keys_mon_sig)!=0:
                # mean over region if they have same correlation sign across months
                for sign in [1,-1]:
                    mask = np.sign(df_corr.loc[keys_mon_sig][[s]]) == sign
                    k_sign = np.array(keys_mon_sig)[mask.values.flatten()]
                    if len(k_sign)==0:
                        continue
                    # calculate mean over n strongest SST timeseries
                    if len(k_sign) > 1:
                        meanparcorr = df_corr.loc[k_sign][[s]].squeeze().sort_values()
                        if n_strongest == 'all':
                            keys_str = meanparcorr.index
                        else:
                            keys_str = meanparcorr.index[-n_strongest:]
                    else:
                        keys_str  = k_sign
                    if weights:
                        fit_masks = rg.df_data.loc[s].iloc[:,-2:]
                        df_d = rg.df_data.loc[s][keys_str].copy()
                        df_d = df_d.apply(fc_utils.standardize_on_train_and_RV,
                                          args=[fit_masks, 0])
                        df_d = df_d.merge(fit_masks, left_index=True,right_index=True)
                        # df_train = df_d[fit_masks['TrainIsTrue']]
                        df_mean, model = fcmodel.fit_wrapper({'ts':target_ts},
                                                          df_d, keys_str,
                                                          kwrgs_model)

                        # kwrgs = {'alphas':[1E-20, 1E-5, 1E-2, .1, 1, 10, 50, 100]}
                        # _m = fcmodel.scikitmodel(**kwrgs_model).fit(df_train,
                        #                                             target_ts)
                        # df_mean = pd.Series(_m.predict(df_d[keys_str]),
                        #                         index=df_d.index)
                    else:
                        df_mean = rg.df_data.loc[s][keys_str].copy().mean(1)
                    month_strings = [k.split('..')[0] for k in sorted(keys_str)]
                    df_mean = df_mean.rename({0:''.join(month_strings) + '..'+uniqk},
                                             axis=1)
                    keys_dict_meansst[s].append( df_mean.columns[0] )
                    mean_SST_list_s.append(df_mean)
            elif uniqk.split('..')[-1] not in mean_vars and len(keys_mon_sig)!=0:
                # use all timeseries (for each month)
                mean_SST_list_s.append(rg.df_data.loc[s][keys_mon_sig].copy())
                keys_dict_meansst[s] = keys_dict_meansst[s] + keys_mon_sig
            # # select strongest
            # if len(keys_mon_sig) != 0 and 'sst' in uniqk:
            #     # appending keys_dict for plotting causal regions
            #     df_corr.loc[keys_mon_sig].mean()
            #     keys_dict[s].append( df_corr.loc[keys_mon_sig][s].idxmax() )
            # if select_str_SM and len(keys_mon_sig) != 0 and 'sm' in uniqk:
            #     # use only strongest SM region
            #     df_corr.loc[keys_mon_sig].mean()
            #     keys_dict[s].append( df_corr.loc[keys_mon_sig][s].idxmax() )
            # elif select_str_SM==False and len(keys_mon_sig) != 0 and 'sm' in uniqk:
            #     # use all SM region
            #     keys_dict[s] = keys_dict[s] + keys_mon_sig
        df_s = pd.concat(mean_SST_list_s, axis=1)
        mean_SST_list.append(df_s)
    df_mean_SST = pd.concat(mean_SST_list, keys=range(rg.n_spl))
    df_mean_SST = df_mean_SST.merge(rg.df_splits.copy(),
                                    left_index=True, right_index=True)
    return df_mean_SST, keys_dict_meansst

keys_dict = {s:[] for s in range(rg.n_spl)} ;
periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)
df_pvals = rg.df_pvals.copy()
df_corr  = rg.df_corr.copy()
unique_keys = np.unique(['..'.join(k.split('..')[1:]) for k in rg.df_data.columns[1:-2]])
for s in range(rg.n_spl):
    sign_s = df_pvals[s][df_pvals[s] <= alpha_CI].dropna(axis=0, how='all')
    for uniqk in unique_keys:
        # region label (R) for each month in split (s)
        keys_mon = [mon+ '..'+uniqk for mon in periodnames]
        # significant region label (R) for each month in split (s)
        keys_mon_sig = [k for k in keys_mon if k in sign_s.index] # check if sig.
        keys_dict[s] = keys_dict[s] + keys_mon_sig

CondDepKeys_strongest = {}
for i, mon in enumerate(periodnames):
    strongest = np.unique([keys_dict[s] for s in range(rg.n_spl)])
    strongest = functions_pp.flatten([keys_dict[s] for s in range(rg.n_spl)])
    str_mon = []
    for l_mon in CondDepKeys[mon]:
        if l_mon[0] in strongest:
            str_mon.append(list(l_mon))
    # str_mon = [list(t) for t in CondDepKeys[mon] if t[0] in strongest]
    count = {}
    for j, k in enumerate(str_mon):
        c = functions_pp.flatten(list(keys_dict.values())).count(k[0])
        str_mon[j][-1] = c
    CondDepKeys_strongest[mon] = str_mon

# mean over SST regions instead of strongest:

#%%
# =============================================================================
# Plot Causal Links
# =============================================================================
plot_strongest = False
if plot_strongest:
    CondDepKeys = CondDepKeys_strongest.copy()

for ip, precur in enumerate(rg.list_for_MI):
    # ip=0; precur = rg.list_for_MI[ip]

    CDlabels = precur.prec_labels.copy()

    if precur.group_lag:
        CDlabels = xr.concat([CDlabels]*corlags.size, dim='lag')
        CDlabels['lag'] = ('lag', periodnames)
        CDcorr = precur.corr_xr_.copy()
    else:
        CDcorr = precur.corr_xr.copy()
    textinmap = []
    MCIstr = CDlabels.copy()
    for i, month in enumerate(CondDepKeys):

        CDkeys = [k[0] for k in CondDepKeys[month] if k[0].split('..')[-1]==precur.name]
        MCIv = [k[1] for k in CondDepKeys[month] if k[0].split('..')[-1]==precur.name]
        RB = [k[2] for k in CondDepKeys[month] if k[0].split('..')[-1]==precur.name]
        region_labels = [int(l.split('..')[1]) for l in CDkeys if l.split('..')[-1] == precur.name]
        f = find_precursors.view_or_replace_labels
        CDlabels[:,i] = f(CDlabels[:,i].copy(), region_labels)
        MCIstr[:,i]   = f(CDlabels[:,i].copy(), region_labels,
                          replacement_labels=MCIv)
        # get text on robustness:
        if len(CDkeys) != 0:
            temp = []
            df_labelloc = find_precursors.labels_to_df(CDlabels[:,i])
            for q, k in enumerate(CDkeys):
                l = int(k.split('..')[1])
                lat, lon = df_labelloc.loc[l].iloc[:2].values.round(1)
                if lon > 180: lon-360
                count = rg._df_count[k]
                text = f'{int(RB[q])}/{count}'
                temp.append([lon+10,lat+5, text, {'fontsize':15,
                                               'bbox':dict(facecolor='white', alpha=0.8)}])
            textinmap.append([(i,0), temp])

    mask = (np.isnan(CDlabels)).astype(bool)
    if ip == 0:
        kwrgs_plot = kwrgs_plotcorr_sst.copy()
    elif ip == 1:
        kwrgs_plot = kwrgs_plotcorr_SM.copy()
    # labels plot
    plot_maps.plot_labels(CDlabels.mean(dim='split'), kwrgs_plot=kwrgs_plot)
    if save:
        if method == 'pcmci':
            dirpath = rg.path_outsub2
        else:
            dirpath = rg.path_outsub1
        plt.savefig(os.path.join(dirpath,
                              f'{precur.name}_str{plot_strongest}_eps{precur.distance_eps}'
                              f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_labels'+rg.figext),
                     bbox_inches='tight')

    # MCI values plot
    kwrgs_plot.update({'clevels':np.arange(-0.8, 0.9, .1),
                       'textinmap':textinmap})
    fig = plot_maps.plot_corr_maps(MCIstr.mean(dim='split'),
                                   mask_xr=np.isnan(MCIstr.mean(dim='split')).astype(bool),
                                   **kwrgs_plot)
    if save:
        fig.savefig(os.path.join(dirpath,
                                  f'{precur.name}_str{plot_strongest}_eps{precur.distance_eps}'
                                  f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_MCI'+rg.figext),
                    bbox_inches='tight')


def df_predictions_for_plot(rg_list):
    df_preds = []
    for i, rg in enumerate(rg_list):
        rg.df_fulltso.index.name = None
        if i == 0:
            prediction = rg.prediction_tuple[0]
            prediction = rg.merge_df_on_df_data(rg.df_fulltso, prediction)
        else:
            prediction = rg.prediction_tuple[0].iloc[:,[1]]
        df_preds.append(prediction)
        if i+1 == len(rg_list):
            df_preds.append(rg.df_splits)
    df_preds  = pd.concat(df_preds, axis=1)
    return df_preds

def df_scores_for_plot(rg_list, name_object):
    df_scores = [] ; df_boot = [] ; df_tests = []
    for i, rg in enumerate(rg_list):
        verification_tuple = rg.__dict__[name_object]
        df_scores.append(verification_tuple[2])
        df_boot.append(verification_tuple[3])
        df_tests.append(verification_tuple[1])
    df_scores = pd.concat(df_scores, axis=1)
    df_boot = pd.concat(df_boot, axis=1)
    df_tests = pd.concat(df_tests, axis=1)
    return df_scores, df_boot, df_tests

#%% Plot regions with Corr value


# def plot_regions(rg, save, plot_parcorr=False):
#     # Get ConDepKeys
#     df_pvals = rg.df_pvals.copy()
#     df_corr  = rg.df_corr.copy()
#     periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)

#     CondDepKeys = {} ;
#     for i, mon in enumerate(periodnames):
#         list_mon = []
#         _keys = [k for k in df_pvals.index if mon in k] # month
#         df_sig = df_pvals[df_pvals.loc[_keys] <= alpha_CI].dropna(axis=0, how='all') # significant

#         for k in df_sig.index:
#             corr_val = df_corr.loc[k].mean()
#             RB = (df_pvals.loc[k]<alpha_CI).sum()
#             list_mon.append((k, corr_val, RB))
#         CondDepKeys[mon] = list_mon

#     for ip, precur in enumerate(rg.list_for_MI):
#         # ip=0; precur = rg.list_for_MI[ip]

#         CDlabels = precur.prec_labels.copy()

#         if precur.group_lag:
#             CDlabels = xr.concat([CDlabels]*len(periodnames), dim='lag')
#             CDlabels['lag'] = ('lag', periodnames)
#             CDcorr = precur.corr_xr_.copy()
#         else:
#             CDcorr = precur.corr_xr.copy()
#         textinmap = []
#         MCIstr = CDlabels.copy()
#         for i, month in enumerate(CondDepKeys):

#             CDkeys = [k[0] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
#             MCIv = [k[1] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
#             RB = [k[2] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
#             region_labels = [int(l.split('..')[1]) for l in CDkeys if precur.name in l.split('..')[-1]]
#             f = find_precursors.view_or_replace_labels
#             if len(CDkeys) != 0:
#                 if region_labels[0] == 0:
#                     region_labels = np.unique(CDlabels[:,i].values[~np.isnan(CDlabels[:,i]).values])
#                     region_labels = np.array(region_labels, dtype=int)
#                     MCIv = np.repeat(MCIv, len(region_labels))
#                     CDkeys = [CDkeys[0].replace('..0..', f'..{r}..') for r in region_labels]
#             CDlabels[:,i] = f(CDlabels[:,i].copy(), region_labels)
#             if plot_parcorr:
#                 MCIstr[:,i]   = f(CDlabels[:,i].copy(), region_labels,
#                                   replacement_labels=MCIv)
#             else:
#                 MCIstr[:,i]   = CDcorr[:,i].copy()


#             # get text on robustness:
#             if len(CDkeys) != 0:
#                 temp = []
#                 df_labelloc = find_precursors.labels_to_df(CDlabels[:,i])
#                 for q, k in enumerate(CDkeys):
#                     l = int(k.split('..')[1])
#                     if l == 0: # pattern cov
#                         lat, lon = df_labelloc.mean(0)[:2]
#                     else:
#                         lat, lon = df_labelloc.loc[l].iloc[:2].values.round(1)
#                     if lon > 180: lon-360
#                     if precur.calc_ts != 'pattern cov':
#                         count = rg._df_count[k]
#                         text = f'{int(RB[q])}/{count}'
#                         temp.append([lon+10,lat+5, text, {'fontsize':15,
#                                                'bbox':dict(facecolor='white', alpha=0.8)}])
#                     elif precur.calc_ts == 'pattern cov' and q == 0:
#                         count = rg._df_count[f'{month}..0..{precur.name}_sp']
#                         text = f'{int(RB[0])}/{count}'
#                         lon = float(CDlabels[:,i].longitude.mean())
#                         lat = float(CDlabels[:,i].latitude.mean())
#                         temp.append([lon,lat, text, {'fontsize':15,
#                                                'bbox':dict(facecolor='white', alpha=0.8)}])
#                 textinmap.append([(i,0), temp])

#         if ip == 0:
#             kwrgs_plot = kwrgs_plotcorr_sst.copy()
#         elif ip == 1:
#             kwrgs_plot = kwrgs_plotcorr_SM.copy()
#             kwrgs_plot.update({'cbar_vert':0.03})
#         # labels plot
#         kwrgs_plot_labels = kwrgs_plot
#         kwrgs_plot_labels.pop('clevels'); kwrgs_plot_labels.pop('clabels')
#         kwrgs_plot_labels.pop('cbar_tick_dict')
#         kwrgs_plot_labels['cbar_vert'] = 0
#         plot_maps.plot_labels(CDlabels.mean(dim='split'), kwrgs_plot=kwrgs_plot_labels)
#         if save:
#             if method == 'pcmci':
#                 dirpath = rg.path_outsub2
#             else:
#                 dirpath = rg.path_outsub1
#             plt.savefig(os.path.join(dirpath,
#                                   f'{precur.name}_eps{precur.distance_eps}'
#                                   f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_labels_'
#                                   f'{periodnames[-1]}'+rg.figext),
#                          bbox_inches='tight')

#         # MCI values plot
#         mask_xr = np.isnan(CDlabels).mean(dim='split') < 1.
#         kwrgs_plot.update({'clevels':np.arange(-0.8, 0.9, .1),
#                            'textinmap':textinmap})
#         fig = plot_maps.plot_corr_maps(MCIstr.where(mask_xr).mean(dim='split'),
#                                        mask_xr=mask_xr,
#                                        **kwrgs_plot)
#         if save:
#             fig.savefig(os.path.join(dirpath,
#                                       f'{precur.name}_eps{precur.distance_eps}'
#                                       f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_MCI_'
#                                       f'{periodnames[-1]}'+rg.figext),
#                         bbox_inches='tight')

def plot_regions(rg, save, plot_parcorr=False):
    # Get ConDepKeys
    df_pvals = rg.df_pvals.copy()
    df_corr  = rg.df_corr.copy()
    periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)

    CondDepKeys = {} ;
    for i, mon in enumerate(periodnames):
        list_mon = []
        _keys = [k for k in df_pvals.index if mon in k] # month
        df_sig = df_pvals[df_pvals.loc[_keys] <= alpha_CI].dropna(axis=0, how='all') # significant

        for k in df_sig.index:
            corr_val = df_corr.loc[k].mean()
            RB = (df_pvals.loc[k]<alpha_CI).sum()
            list_mon.append((k, corr_val, RB))
        CondDepKeys[mon] = list_mon

    for ip, precur in enumerate(rg.list_for_MI):
        # ip=0; precur = rg.list_for_MI[ip]

        CDlabels = precur.prec_labels.copy()

        if precur.group_lag:
            CDlabels = xr.concat([CDlabels]*len(periodnames), dim='lag')
            CDlabels['lag'] = ('lag', periodnames)
            CDcorr = precur.corr_xr_.copy()
        else:
            CDcorr = precur.corr_xr.copy()
        textinmap = []
        MCIstr = CDlabels.copy()
        for i, month in enumerate(CondDepKeys):

            CDkeys = [k[0] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
            MCIv = [k[1] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
            RB = [k[2] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
            region_labels = [int(l.split('..')[1]) for l in CDkeys if precur.name in l.split('..')[-1]]
            f = find_precursors.view_or_replace_labels
            if len(CDkeys) != 0:
                if region_labels[0] == 0: # pattern cov
                    region_labels = np.unique(CDlabels[:,i].values[~np.isnan(CDlabels[:,i]).values])
                    region_labels = np.array(region_labels, dtype=int)
                    MCIv = np.repeat(MCIv, len(region_labels))
                    CDkeys = [CDkeys[0].replace('..0..', f'..{r}..') for r in region_labels]
            CDlabels[:,i] = f(CDlabels[:,i].copy(), region_labels)
            if plot_parcorr:
                MCIstr[:,i]   = f(CDlabels[:,i].copy(), region_labels,
                                  replacement_labels=MCIv)
            else:
                MCIstr[:,i]   = CDcorr[:,i].copy()


            # get text on robustness:
            if len(CDkeys) != 0:
                temp = []
                df_labelloc = find_precursors.labels_to_df(CDlabels[:,i])
                for q, k in enumerate(CDkeys):
                    l = int(k.split('..')[1])
                    if l == 0: # pattern cov
                        lat, lon = df_labelloc.mean(0)[:2]
                    else:
                        lat, lon = df_labelloc.loc[l].iloc[:2].values.round(1)
                    if lon > 180: lon-360
                    if precur.calc_ts != 'pattern cov':
                        count = rg._df_count[k]
                        text = f'{int(RB[q])}/{count}'
                        temp.append([lon+10,lat+5, text, {'fontsize':15,
                                               'bbox':dict(facecolor='white', alpha=0.8)}])
                    elif precur.calc_ts == 'pattern cov' and q == 0:
                        count = rg._df_count[f'{month}..0..{precur.name}_sp']
                        text = f'{int(RB[0])}/{count}'
                        lon = float(CDlabels[:,i].longitude.mean())
                        lat = float(CDlabels[:,i].latitude.mean())
                        temp.append([lon,lat, text, {'fontsize':15,
                                               'bbox':dict(facecolor='white', alpha=0.8)}])
                textinmap.append([(i,0), temp])

        if ip == 0:
            kwrgs_plot = kwrgs_plotcorr_sst.copy()
        elif ip == 1:
            kwrgs_plot = kwrgs_plotcorr_SM.copy()
        # labels plot
        plot_maps.plot_labels(CDlabels.mean(dim='split'), kwrgs_plot=kwrgs_plot)
        if save:
            if method == 'pcmci':
                dirpath = rg.path_outsub2
            else:
                dirpath = rg.path_outsub1
            plt.savefig(os.path.join(dirpath,
                                  f'{precur.name}_eps{precur.distance_eps}'
                                  f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_labels_'
                                  f'{periodnames[-1]}'+rg.figext),
                         bbox_inches='tight')

        # MCI values plot
        mask_xr = np.isnan(CDlabels).mean(dim='split') < 1.
        kwrgs_plot.update({'clevels':np.arange(-0.8, 0.9, .1),
                           'textinmap':textinmap})
        fig = plot_maps.plot_corr_maps(MCIstr.where(mask_xr).mean(dim='split'),
                                       mask_xr=mask_xr,
                                       **kwrgs_plot)
        if save:
            fig.savefig(os.path.join(dirpath,
                                      f'{precur.name}_eps{precur.distance_eps}'
                                      f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_MCI_'
                                      f'{periodnames[-1]}'+rg.figext),
                        bbox_inches='tight')

plot_regions(rg, save=save, plot_parcorr=False)




#%% Continuous Forecast with Causal Precursors

from sklearn.linear_model import Ridge
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
                'alphas':np.concatenate([[1E-20],np.logspace(-5,0, 6),
                                         np.logspace(.01, 2.5, num=10)]), # large a, strong regul.
                'normalize':False,
                'fit_intercept':True,
                # 'store_cv_values':True}
                'kfold':5}

fcmodel = ScikitModel(Ridge, verbosity=0)
kwrgs_model = {'scoringCV':'neg_mean_absolute_error',
                'alpha':list(np.concatenate([[1E-20],np.logspace(-5,0, 6),
                                         np.logspace(.01, 2.5, num=10)])), # large a, strong regul.
                'normalize':False,
                'fit_intercept':True,
                'kfold':10}

mean_SST = True

# target
fc_mask = rg.df_data.iloc[:,-1].loc[0]
target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
# target_ts = (target_ts - target_ts.mean()) / target_ts.std()
# metrics
RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
metric_names = [s.__name__ for s in score_func_list]

if mean_SST:
    mean_vars=['sst', 'smi']
    for i, p in enumerate(rg.list_for_MI):
        if p.calc_ts == 'pattern cov':
            mean_vars[i] +='_sp'
    df_data, keys_dict = get_df_mean_SST(rg,
                                         mean_vars=mean_vars,
                                         alpha_CI=alpha_CI,
                                         n_strongest='all',
                                         weights=True,
                                         fcmodel=fcmodel,
                                         kwrgs_model=kwrgs_model,
                                         target_ts=target_ts)
else:
    df_data, keys_dict = get_df_mean_SST(rg,
                                         mean_vars=[],
                                         alpha_CI=alpha_CI,
                                         n_strongest='all',
                                         weights=True)

lag_ = 0 ; n_boot = 0
prediction_tuple = rg.fit_df_data_ridge(df_data=df_data,
                                        keys=keys_dict,
                                        target=target_ts,
                                        tau_min=0, tau_max=0,
                                        kwrgs_model=kwrgs_model,
                                        fcmodel=fcmodel,
                                        transformer=None)

predict, weights, models_lags = prediction_tuple
prediction = predict.rename({predict.columns[0]:'target',lag_:'Prediction'},
                            axis=1)
rg.prediction_tuple=prediction_tuple


weights_norm = weights.mean(axis=0, level=1)
# weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box', figsize=(15,5))


verification_tuple = fc_utils.get_scores(prediction,
                                         rg.df_data.iloc[:,-2:],
                                         score_func_list,
                                         n_boot = n_boot,
                                         blocksize=1,
                                         rng_seed=1)
df_train_m, df_test_s_m, df_test_m, df_boot = verification_tuple
rg.verification_tuple = verification_tuple


m = models_lags[f'lag_{lag_}'][f'split_{0}']
print(m.cv_results_['mean_test_score'].mean())

df_test = functions_pp.get_df_test(predict.rename({lag_:'causal'}, axis=1),
                                   df_splits=rg.df_splits)
print(df_test_m)

#%% Plot forecast
from matplotlib import gridspec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

df_preds_save = df_predictions_for_plot([rg])
d_dfs={'df_predictions':df_preds_save}
filepath_dfs = os.path.join(rg.path_outsub1, f'predictions_s{seed}_continuous.h5')
functions_pp.store_hdf_df(d_dfs, filepath_dfs)

df_scores, df_boot, df_tests = df_scores_for_plot([rg], name_object='verification_tuple')
d_dfs={'df_scores':df_scores, 'df_boot':df_boot, 'df_tests':df_tests}
filepath_dfs = os.path.join(rg.path_outsub1, f'scores_s{seed}_continuous.h5')
functions_pp.store_hdf_df(d_dfs, filepath_dfs)

def plot_forecast_ts(df_test_m, df_test):
    fontsize = 16

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 1, height_ratios=None)
    facecolor='white'
    ax0 = plt.subplot(gs[0], facecolor=facecolor)
    # df_test.plot(ax=ax0)
    ax0.plot_date(df_test.index, df_test[target_dataset], ls='-',
                  label='Observed', c='black')

    ax0.plot_date(df_test.index, df_test['causal'], ls='-', c='red',
                  label=r'Causal precursors ($\alpha=$'+f' {alpha_CI})')
    # ax0.set_xticks()
    # ax0.set_xticklabels(df_test.index.year,
    ax0.set_ylabel('Soy Yield [1/ha]', fontsize=fontsize)
    ax0.tick_params(labelsize=fontsize)
    ax0.axhline(y=0, color='black', lw=1)
    ax0.legend(fontsize=fontsize)

    df_scores = df_test_m.loc[0][df_test_m.columns[0][0]]
    Texts1 = [] ; Texts2 = [] ;
    textprops = dict(color='black', fontsize=fontsize+4, family='serif')
    rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr. Coeff.', 'MAE':'MAE-SS',
                  'BSS':'BSS', 'roc_auc_score':'ROC-AUC'}
    for k in df_scores.index:
        label = rename_met[k]
        val = round(df_scores[k], 2)
        Texts1.append(TextArea(f'{label}',textprops=textprops))
        Texts2.append(TextArea(f'{val}',textprops=textprops))
    texts_vbox1 = VPacker(children=Texts1,pad=0,sep=4)
    texts_vbox2 = VPacker(children=Texts2,pad=0,sep=4)

    ann1 = AnnotationBbox(texts_vbox1,(.02,.15),xycoords=ax0.transAxes,
                                box_alignment=(0,.5),
                                bboxprops = dict(facecolor='white',
                                                 boxstyle='round',edgecolor='white'))
    ann2 = AnnotationBbox(texts_vbox2,(.21,.15),xycoords=ax0.transAxes,
                                box_alignment=(0,.5),
                                bboxprops = dict(facecolor='white',
                                                 boxstyle='round',edgecolor='white'))
    ann1.set_figure(fig) ; ann2.set_figure(fig)
    fig.artists.append(ann1) ; fig.artists.append(ann2)
    return

plot_forecast_ts(df_test_m, df_test)
f_name = f'{method}_{seed}_continuous'
fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
if save:
    plt.savefig(fig_path, bbox_inches='tight')


#%%
# y_true = df_test['USDA_Soy']
# forecast = df_test['causal']

# cond_lags = corlags = np.array([['02-01','03-01'],  # FM
#                                 ['03-01', '04-30'], # MA
#                                 ['05-01', '06-30'], # MJ
#                                 ['07-01', '08-31'],
#                                 ['03-01', '08-01']]) # JA
# cond_periodnames = ['FM', 'MA', 'MJ', 'JA', 'March-Aug']

# cond_df  = np.zeros( (3, len(cond_periodnames), 3+1))
# for j, l in enumerate(cond_lags):

#     sst.lags = np.array([l]) # JA
#     ts_corr = find_precursors.spatial_mean_regions(sst,
#                                                    kwrgs_load=rg.kwrgs_load,
#                                                    force_reload=True,
#                                                    lags=['MJ'])
#     df_ts = pd.concat(ts_corr, keys=range(n_spl))

#     zz = df_ts[['MJ..1..sst']]
#     # zz = rg.df_data[['JA..1..sst']]

#     state_sst = functions_pp.get_df_test(zz.rename({lag_:cond_periodnames[j]}, axis=1),
#                                          df_splits=rg.df_splits)

#     df_test_m = rg.verification_tuple[2]
#     cond_df[:, j, 0] = df_test_m[df_test_m.columns[0][0]].loc[0]
#     quantiles = [.1, .2, .3]
#     for k, q in enumerate(quantiles):
#         low = state_sst < state_sst.quantile(q)
#         high = state_sst > state_sst.quantile(1-q)
#         mask_anomalous = np.logical_or(low, high)

#         condfc = df_test[mask_anomalous.values]
#         condfc = condfc.rename({'causal':cond_periodnames[j]}, axis=1)
#         cond_verif_tuple = fc_utils.get_scores(condfc,
#                                                score_func_list=score_func_list,
#                                                n_boot=0,
#                                                score_per_test=False,
#                                                blocksize=1,
#                                                rng_seed=seed)
#         df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple

#         metrics = df_test_m.columns.levels[1]
#         for i, met in enumerate(metrics):
#             # print(df_test_m)
#             cond_df[i, j, k+1] = df_test_m[cond_periodnames[j]].loc[0][met]

# df_cond_fc = pd.DataFrame(cond_df.reshape((len(metrics)*len(cond_periodnames), -1)),
#                           index=pd.MultiIndex.from_product([list(metrics), cond_periodnames]),
#                           columns=['all']+quantiles)



# df_cond_fc.to_excel(os.path.join(rg.path_outsub1, 'cond_fc_per_month.xlsx'))
#%%
def cond_forecast_table(rg_list):
    df_test_m = rg_list[0].verification_tuple[2]
    quantiles = [.15, .25]
    metrics = df_test_m.columns.levels[1]
    cond_df = np.zeros((metrics.size, len(rg_list), len(quantiles)*2))
    for i, met in enumerate(metrics):
        for j, rg in enumerate(rg_list):
            df_mean, keys_dict = get_df_mean_SST(rg, mean_vars=mean_vars,
                                                 n_strongest='all',
                                                 weights=True,
                                                 fcmodel=fcmodel,
                                                 kwrgs_model=kwrgs_model,
                                                 target_ts=target_ts)

            weights_norm = rg.prediction_tuple[1].mean(axis=0, level=1)
            weights_norm = weights_norm.sort_values(ascending=False, by=0)

            PacAtl = []
            df_labels = find_precursors.labels_to_df(rg.list_for_MI[0].prec_labels)
            dlat = df_labels['latitude'] - 29
            dlon = df_labels['longitude'] - 290
            zz = pd.concat([dlat.abs(),dlon.abs()], axis=1)
            Atlan = zz.query('latitude < 10 & longitude < 10')
            if Atlan.size > 0:
                PacAtl.append(int(Atlan.index[0]))
            PacAtl.append(int(df_labels['n_gridcells'].idxmax())) # Pacific SST

            keys = [k for k in weights_norm.index if int(k.split('..')[1]) in PacAtl]
            keys = [k for k in keys if 'sst' in k]


            PacAtl_ts = functions_pp.get_df_test(df_mean[keys],
                                              df_splits=rg.df_splits)

            weights_norm = weights_norm.div(weights_norm.loc[keys].max(axis=0))
            PacAtl_ts = weights_norm.loc[keys].T.loc[0] * PacAtl_ts # weigths
            PacAtl_ts = PacAtl_ts.mean(axis=1)

            prediction = rg.prediction_tuple[0]
            df_test = functions_pp.get_df_test(prediction,
                                               df_splits=rg.df_splits)

            # df_test_m = rg.verification_tuple[2]
            # cond_df[i, j, 0] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            for k, l in enumerate(range(0,4,2)):
                q = quantiles[k]
                low = PacAtl_ts < PacAtl_ts.quantile(q)
                high = PacAtl_ts > PacAtl_ts.quantile(1-q)
                mask_anomalous = np.logical_or(low, high)
                # anomalous Boundary forcing
                condfc = df_test[mask_anomalous.values]
                condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
                cond_verif_tuple = fc_utils.get_scores(condfc,
                                                       score_func_list=score_func_list,
                                                       n_boot=0,
                                                       score_per_test=False,
                                                       blocksize=1,
                                                       rng_seed=seed)
                df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
                rg.cond_verif_tuple  = cond_verif_tuple
                cond_df[i, j, l] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
                # mild boundary forcing
                higher_low = PacAtl_ts > PacAtl_ts.quantile(.5-q)
                lower_high = PacAtl_ts < PacAtl_ts.quantile(.5+q)
                mask_anomalous = np.logical_and(higher_low, lower_high) # changed 11-5-21

                condfc = df_test[mask_anomalous.values]
                condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
                cond_verif_tuple = fc_utils.get_scores(condfc,
                                                       score_func_list=score_func_list,
                                                       n_boot=0,
                                                       score_per_test=False,
                                                       blocksize=1,
                                                       rng_seed=seed)
                df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
                cond_df[i, j, l+1] = df_test_m[df_test_m.columns[0][0]].loc[0][met]

    columns = [[f'strong {int(q*200)}%', f'weak {int(q*200)}%'] for q in quantiles]
    df_cond_fc = pd.DataFrame(cond_df.reshape((len(metrics)*len(rg_list), -1)),
                              index=pd.MultiIndex.from_product([list(metrics), [rg.fc_month for rg in rg_list]]),
                              columns=functions_pp.flatten(columns))


    return df_cond_fc

rg.fc_month = 'November'
df_cond_fc = cond_forecast_table([rg])
# store as .xlsc
df_cond_fc.to_excel(os.path.join(rg.path_outsub1, f'cond_fc_{method}_s{seed}.xlsx'))
# Store as .h5
d_dfs={'df_cond_fc':df_cond_fc}
filepath_dfs = os.path.join(rg.path_outsub1, f'cond_fc_{method}_s{seed}.h5')
functions_pp.store_hdf_df(d_dfs, filepath_dfs)
print(df_cond_fc)

#%% Event Forecast with Causal Precursors

from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from stat_models_cont import ScikitModel


fcmodel1 = ScikitModel(Ridge, verbosity=0)
kwrgs_model1 = {'scoringCV':'neg_mean_absolute_error',
                'alpha':list(np.concatenate([np.logspace(-5,0, 6),
                                         np.logspace(.01, 2.5, num=10)])), # large a, strong regul.
                'normalize':False,
                'fit_intercept':True,
                'kfold':10}

fcmodel = ScikitModel(LogisticRegressionCV, verbosity=0)
kwrgs_model = {'scoringCV':'neg_brier_score',
                'Cs':np.concatenate([np.logspace(-5,0, 6),
                                          np.logspace(.3, 1.2, num=4)]), # large a, strong regul.
                'random_state':seed,
                'penalty':'l2',
                'solver':'lbfgs',
                'kfold':10,
                'max_iter':200}

fcmodel = ScikitModel(LogisticRegression, verbosity=0)
kwrgs_model = {'scoringCV':'neg_brier_score',
                'C':list([.1,.5,.8,1,1.2,4,7,10,20]), # large a, strong regul.
                'random_state':seed,
                'penalty':'l2',
                'solver':'lbfgs',
                'kfold':10,
                'max_iter':200}

mean_SST = True

q = .5
# target
fc_mask = rg.df_data.iloc[:,-1].loc[0]
target_ts_c = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
target_ts_c = (target_ts_c - target_ts_c.mean()) / target_ts.std()
if q >= 0.5:
    target_ts = (target_ts_c > target_ts_c.quantile(q)).astype(int)
elif q < .5:
    target_ts = (target_ts_c < target_ts_c.quantile(q)).astype(int)
# metrics
BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
score_func_list = [BSS, fc_utils.metrics.roc_auc_score]
metric_names = [s.__name__ for s in score_func_list]

if mean_SST:
    mean_vars=['sst', 'smi']
    for i, p in enumerate(rg.list_for_MI):
        if p.calc_ts == 'pattern cov':
            mean_vars[i] +='_sp'
    df_data, keys_dict = get_df_mean_SST(rg,
                                         mean_vars=mean_vars,
                                         alpha_CI=alpha_CI,
                                         n_strongest='all',
                                         weights=True,
                                         fcmodel=fcmodel,
                                         kwrgs_model=kwrgs_model,
                                         target_ts=target_ts)
else:
    df_data, keys_dict = get_df_mean_SST(rg,
                                         mean_vars=[],
                                         alpha_CI=alpha_CI,
                                         n_strongest='all',
                                         weights=True)

lag_ = 0 ; n_boot = 2000
prediction_tuple = rg.fit_df_data_ridge(df_data=df_data,
                                        keys=keys_dict,
                                        target=target_ts,
                                        tau_min=0, tau_max=0,
                                        kwrgs_model=kwrgs_model,
                                        fcmodel=fcmodel,
                                        transformer=False)

predict, weights, models_lags = prediction_tuple
prediction = predict.rename({predict.columns[0]:'target',lag_:'Prediction'},
                            axis=1)
rg.prediction_tuple=prediction_tuple


weights_norm = weights.mean(axis=0, level=1)
# weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box', figsize=(15,5))


verification_tuple = fc_utils.get_scores(prediction,
                                         rg.df_data.iloc[:,-2:],
                                         score_func_list,
                                         score_per_test=False,
                                         n_boot = n_boot,
                                         blocksize=1,
                                         rng_seed=1)
df_train_m, df_test_s_m, df_test_m, df_boot = verification_tuple
rg.verification_tuple = verification_tuple


[models_lags[f'lag_{lag_}'][f'split_{s}'].best_params_ for s in range(rg.n_spl)]
m = models_lags[f'lag_{lag_}'][f'split_{0}']
print(m.cv_results_['mean_test_score'].mean())

df_test = functions_pp.get_df_test(predict.rename({lag_:'causal'}, axis=1),
                                   df_splits=rg.df_splits)
print(df_test_m)
#%%
plot_forecast_ts(df_test_m, df_test)
f_name = f'{method}_{seed}_{q}'
fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
if save:
    plt.savefig(fig_path, bbox_inches='tight')

#%%


# sys.exit()

#%% forecasting



# def append_dict(month, df_test_m, df_train_m):
#     dkeys = [f'{month} RMSE', f'{month} RMSE tr',
#              f'{month} Corr.', f'{month} Corr. tr',
#              f'{month} MAE test', f'{month} MAE tr']
#     append_dict = {dkeys[0]:float(df_test_m.iloc[:,0].round(3)),
#                    dkeys[1]:df_train_m.mean().iloc[0].round(3),
#                    dkeys[2]:float(df_test_m.iloc[:,1].round(3)),
#                    dkeys[3]:float(df_train_m.mean().iloc[1].round(3)),
#                    dkeys[4]:float(df_test_m.iloc[:,2].round(3)),
#                    dkeys[5]:float(df_train_m.mean().iloc[2].round(3))}
#     dict_v.update(append_dict)
#     return

# def feature_selection_CondDep(pvals, alpha_CI=.05):
#     # removing all keys that are Cond. Indep. in each trainingset
#     keys = list(np.unique(pvals.index.levels[0]))
#     keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
#     for s in rg.df_splits.index.levels[0]:
#         for k_i in keys:
#             if (np.nan_to_num(pvals.loc[k_i][s],nan=alpha_CI) > alpha_CI).mean()>0:
#                 k_ = keys_dict[s].copy() ; k_.pop(k_.index(k_i))
#                 keys_dict[s] = k_
#     return keys_dict.copy()



# alpha_CI = 0.01 # [.01, .05, .1, .2]
# variables = ['sst', 'smi']


# feature_selection = True
# # add_previous_periods = False

# add_PDO = False
# if add_PDO:
#     # get PDO
#     SST_pp_filepath = rg.list_precur_pp[0][1]
#     if 'df_PDOsplit' not in globals():
#         df_PDO, PDO_patterns = climate_indices.PDO(SST_pp_filepath,
#                                                     None)
#         df_PDOsplit = df_PDO.loc[0]
#         plot_maps.plot_corr_maps(PDO_patterns)

#     df_PDOsplit = df_PDOsplit[['PDO']].apply(fc_utils.standardize_on_train,
#                           args=[df_PDO.loc[0]['TrainIsTrue']],
#                           result_type='broadcast')
#     PDO_aggr_periods = np.array([
#                                 ['03-01', '02-28'],
#                                 ['06-01', '05-31'],
#                                 ['09-01', '08-31'],
#                                 ['12-01', '11-30']
#                                 ])


# dict_v = {'target':target_dataset, 'method':method,'S':f's{seed}',
#           'yrs':start_end_year, 'fs':str(feature_selection),
#           'addprev':add_previous_periods}

# fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
# target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
# target_ts = (target_ts - target_ts.mean()) / target_ts.std()
# RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
# MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
# # CRPSS = fc_utils.CRPSS_vs_constant_bench(constant_bench=clim_mean_temp).CRPSS
# score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]

# list_test = []
# list_train = []
# list_test_b = []
# list_pred_test = []
# no_info_fc = []

# CondDepKeys = {} ; CondDepKeysDict = {} ; CV_test = {}
# for i, months in enumerate(periodnames[:]):
#     print(f'forecast using precursors of {months}')
#     # i=1; months, start_end_TVdate = periodnames[i], corlags[i]

#     keys = [k for k in rg.df_data.columns[:-2] if k.split('..')[-1] in variables]
#     keys = [k for k in keys if months == k.split('..')[0]]


#     if add_PDO:
#         se = PDO_aggr_periods[i]
#         df_PDOagr = functions_pp.time_mean_periods(df_PDOsplit,
#                                                    start_end_periods=se)
#         # remove old PDO timeseries (if present)
#         rg.df_data = rg.df_data[[k for k in rg.df_data.columns if k != 'PDO']]
#         rg.df_data = rg.merge_df_on_df_data(df_PDOagr)
#         keys.append('PDO')


#     if len(keys) != 0 and 'PDO' not in keys and feature_selection:
#         # Feature selection Cond. Dependence
#         keys = list(keys) # must be list
#         corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data.copy(), z_keys=keys,
#                                                     keys=keys)
#         alpha_CI_ = alpha_CI[-1] if type(alpha_CI) is list else alpha_CI
#         keys_dict = feature_selection_CondDep(pvals, alpha_CI_)

#         # always C.D. (every split)
#         keys = [k for k in keys if (np.nan_to_num(pvals.loc[k],nan=alpha_CI_) <= alpha_CI_).mean()== 1]
#     else:
#         keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars



#     if add_previous_periods and periodnames.index(months) != 0:
#         # merging CD keys that were found for each split seperately to keep
#         # clean train-test split
#         for s in range(n_spl):
#             pm = periodnames[periodnames.index(months) - 1 ]
#             pmk = [k for k in CondDepKeysDict[pm][s] if k.split('..')[0]==pm]
#             keys_dict[s] = keys_dict[s] + pmk

#     lag_ = 0
#     def model_fit(target_ts, keys_dict, kwrgs_model, fcmodel, lag_):
#         return rg.fit_df_data_ridge(target=target_ts,
#                                    keys=keys_dict,
#                                    tau_min=lag_, tau_max=lag_,
#                                    kwrgs_model=kwrgs_model,
#                                    fcmodel=fcmodel,
#                                    transformer=None)

#     if len(keys) != 0:
#         if feature_selection and type(alpha_CI) is list:
#             models_CV = {}
#             for a in alpha_CI:
#                 print(f'tuning alpha_CI {a}')
#                 new = feature_selection_CondDep(pvals, a)
#                 if new != keys_dict:
#                     keys_dict = new
#                     # print(keys_dict[0])
#                     models_CV[a] = model_fit(target_ts, keys_dict, kwrgs_model, fcmodel, lag_)

#             best_scores = {}
#             for a, out in models_CV.items():
#                 best_scores[a] = [out[-1][f'lag_{lag_}'][f'split_{s}'].best_score_ for s in range(n_spl)]
#             df_CV = pd.DataFrame(best_scores).mean(0) ; df_CV.columns = [months]
#             CV_test[months] = df_CV
#             best_alphaCI = df_CV.round(2).idxmax()
#             keys_dict = feature_selection_CondDep(pvals, best_alphaCI)
#             out = models_CV[best_alphaCI]
#         else:
#             out = model_fit(target_ts, keys_dict, kwrgs_model, fcmodel, lag_)


#         predict, weights, models_lags = out


#         prediction = predict.rename({predict.columns[0]:'target',lag_:'Prediction'},
#                                     axis=1)
#         # if i==0:
#             # weights_norm = weights.mean(axis=0, level=1)
#             # weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

#         df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
#                                                                  rg.df_data.iloc[:,-2:],
#                                                                  score_func_list,
#                                                                  n_boot = n_boot,
#                                                                  blocksize=1,
#                                                                  rng_seed=seed)

#         m = models_lags[f'lag_{lag_}'][f'split_{0}']
#         # if months == 'JA':
#             # print(models_lags[f'lag_{lag_}'][f'split_{0}'].X_pred.iloc[0])
#             # print(df_test_m)
#         models = [models_lags[f'lag_{lag_}'][f'split_{s}'] for s in range(n_spl)]
#         # a_idx = [int(np.argwhere(m.alphas==m.alpha_).squeeze()) for m in models]
#         # cvvalues = [m.cv_values_[:,a_idx[i]] for i, m in enumerate(models)]
#         # plt.plot(np.array(cvvalues).T.mean(0))
#         # cvfitalpha = [models_lags[f'lag_{lag}'][f'split_{s}'].alpha_ for s in range(n_spl)]
#         # if kwrgs_model['alphas'].max() in cvfitalpha: print('Max a reached')
#         # if kwrgs_model['alphas'].min() in cvfitalpha: print('Min a reached')
#         # assert kwrgs_model['alphas'].min() not in cvfitalpha, 'decrease min a'

#         df_test = functions_pp.get_df_test(predict.rename({0:months}, axis=1),
#                                             cols=[months],
#                                             df_splits=rg.df_splits)
#         # appending results
#         list_pred_test.append(df_test)

#     else:
#         print('no precursor timeseries found, scores all 0')
#         metric_names = [s.__name__ for s in score_func_list]
#         index = pd.MultiIndex.from_product([['Prediction'],metric_names])
#         df_boot = pd.DataFrame(data=np.zeros((n_boot, len(score_func_list))),
#                             columns=index)
#         df_test_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
#                                   columns=index)
#         df_train_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
#                                   columns=index)

#     CondDepKeys[months] = keys
#     CondDepKeysDict[months] = keys_dict

#     df_test_m.index = [months] ;
#     columns = pd.MultiIndex.from_product([np.array([months]),
#                                         df_train_m.columns.levels[1]])
#     df_train_m.columns = columns
#     df_boot.columns = columns

#     list_test_b.append(df_boot)
#     list_test.append(df_test_m)
#     list_train.append(df_train_m)
#     append_dict(months, df_test_m, df_train_m)
#     # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
#     #                       function=df_ana.plot_timeseries,
#     #                       kwrgs={'timesteps':rg.fullts.size,
#     #                                   'nth_xyear':5})


# #%%

# import matplotlib.patches as mpatches


# def boxplot_scores(list_scores, list_test_b, alpha=.1):

#     df_scores = pd.concat(list_test)
#     df_test_b = pd.concat(list_test_b,axis=1)

#     yerr = [] ; quan = [] ;
#     monmet = np.array(df_test_b.columns)
#     for i, (mon, met) in enumerate(monmet):
#         Eh = 1 - alpha/2 ; El = alpha/2
#         # met = rename_metrics_cont[met]
#         _scores = df_test_b[mon][met]
#         tup = [_scores.quantile(El), _scores.quantile(Eh)]
#         quan.append(tup)
#         mean = df_scores.values.flatten()[i] ;
#         tup = abs(mean-tup)
#         yerr.append(tup)

#     _yerr = np.array(yerr).reshape(df_scores.columns.size,len(list(CondDepKeys.keys()))*2,
#                                    order='F').reshape(df_scores.columns.size,2,len(list(CondDepKeys.keys())))
#     ax = df_scores.plot.bar(rot=0, yerr=_yerr,
#                             capsize=8, error_kw=dict(capthick=1),
#                             color=['blue', 'green', 'purple'],
#                             legend=False,
#                             figsize=(10,8))
#     for noinfo in no_info_fc:
#         # first two children are not barplots
#         idx = list(CondDepKeys.keys()).index(noinfo) + 3
#         ax.get_children()[idx].set_color('r') # RMSE bar
#         idx = list(CondDepKeys.keys()).index(noinfo) + 15
#         ax.get_children()[idx].set_color('r') # RMSE bar



#     ax.set_ylabel('Skill Score', fontsize=16, labelpad=-5)
#     # ax.tick_params(labelsize=16)
#     # ax.set_xticklabels(months, fontdict={'fontsize':20})
#     title = 'U.S. {} forecast {}\n'.format(target_dataset.split('_')[-1],
#                                            str(start_end_year))
#     if experiment == 'semestral':
#         title += 'from half-year mean '
#     elif experiment == 'seasons':
#         title += 'from seasonal mean '
#     title += '{}'.format(' and '.join([v.upper() for v in variables]))

#     ax.set_title(title,
#                  fontsize=18)
#     ax.tick_params(axis='y', labelsize=13)
#     ax.tick_params(axis='x', labelsize=16)


#     patch1 = mpatches.Patch(color='blue', label='RMSE-SS')
#     patch2 = mpatches.Patch(color='green', label='Corr. Coef.')
#     patch3 = mpatches.Patch(color='purple', label='MAE-SS')
#     handles = [patch1, patch2, patch3]
#     legend1 = ax.legend(handles=handles,
#               fontsize=16, frameon=True, facecolor='grey',
#               framealpha=.5)
#     ax.add_artist(legend1)

#     ax.set_ylim(-0.3, 1)

#     append_str = '-'.join(periodnames) #+'_'+'_'.join(np.array(start_end_year, str))
#     plt.savefig(os.path.join(rg.path_outsub1,
#               f'skill_fs{feature_selection}_addprev{add_previous_periods}_'
#               f'ab{alpha}_ac{alpha_corr}_afs{alpha_CI}_nb{n_boot}_{append_str}.pdf'))


# boxplot_scores(list_test, list_test_b, alpha=.1)



# #%%
# keys  = core_pp.flatten(list(CondDepKeys.values()))
# corrf, pvalf, keys_dictf = feature_selection_CondDep(rg.df_data.copy(), keys)
# final_keys = [k for k in keys if (np.nan_to_num(pvalf.loc[k],nan=alpha_CI) <= alpha_CI).mean()== 1]

# #%%
# # # code run with or without -i
# # if sys.flags.inspect:
# name_csv = f'output_regression_{experiment}_sensivity.csv'
# name_csv = os.path.join(rg.path_outmain, name_csv)
# for csvfilename, dic in [(name_csv, dict_v)]:
#     # create .csv if it does not exists
#     if os.path.exists(csvfilename) == False:
#         with open(csvfilename, 'a', newline='') as csvfile:

#             writer = csv.DictWriter(csvfile, list(dic.keys()))
#             writer.writerows([{f:f for f in list(dic.keys())}])

#     # write
#     with open(csvfilename, 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, list(dic.keys()))
#         writer.writerows([dic])

# #%% EXIT


# sys.exit()



# #%%
# # CondDep keys each month:
# alpha_CI = .05
# CondDepKeys = {} ;
# for i, mon in enumerate(periodnames):
#     list_mon = []
#     _keys = [k for k in df_pvals.index if mon in k] # month
#     df_sig = df_pvals[df_pvals.loc[_keys] <= alpha_CI].dropna(axis=0, how='all') # significant

#     for k in df_sig.index:
#         corr_val = df_corr.loc[k].mean()
#         RB = (df_pvals.loc[k]<alpha_CI).sum()
#         list_mon.append((k, corr_val, RB))
#     CondDepKeys[mon] = list_mon

# # dict with strongest mean parcorr over growing season
# keys_dict = {s:[] for s in range(n_spl)} ;
# for s in range(n_spl):
#     sign_s = df_pvals[s][df_pvals[s] <= alpha_CI].dropna(axis=0, how='all')
#     for uniqk in unique_keys:
#         keys_mon = [mon+ '..'+uniqk for mon in periodnames]
#         keys_mon = [k for k in keys_mon if k in sign_s.index] # check if sig.
#         if len(keys_mon) != 0:
#             df_corr.loc[keys_mon].mean()
#             keys_dict[s].append( df_corr.loc[keys_mon][s].idxmax() )

# #%% Re-aggregated SST data of regions with time_mean_bins

# def reaggregated_regions(precur, rg, start_end_date1, precur_aggr):
#     splits = rg.df_data.index.levels[0]
#     rg.kwrgs_load['start_end_date'] = start_end_date1
#     rg.kwrgs_load['closed_on_date'] = start_end_date1[-1]
#     df_data1 = find_precursors.spatial_mean_regions(precur, precur_aggr=precur_aggr,
#                                                    kwrgs_load=rg.kwrgs_load)
#     df_data1 = pd.concat(df_data1, keys = splits)

#     CondDepKeysList = core_pp.flatten([l for l in CondDepKeys.values()])
#     if type(CondDepKeysList) is list:
#         CondDepKeysList = [l[0] for l in CondDepKeysList]
#     CondDepKeysList = [k for k in CondDepKeysList if k.split('..')[-1] == precur.name]
#     df_sub1 = df_data1[CondDepKeysList]

#     ts_corr = np.zeros( (splits.size), dtype=object)
#     for s in splits:
#         l = []
#         for yr in rg.dates_TV.year:
#             singleyr1 = df_sub1.loc[s].loc[functions_pp.get_oneyr(df_sub1.loc[s], yr)].T
#             newcols = []
#             for col in singleyr1.index:
#                 newcols.append([str(m)+col for m in singleyr1.columns.month])
#             singleyr1 = pd.DataFrame(singleyr1.values.reshape(1,-1),
#                          columns=core_pp.flatten(newcols),
#                          index=pd.to_datetime([f'{yr}-01-01']))
#             l.append(singleyr1)
#             df_s = pd.concat(l)
#         ts_corr[s] = df_s
#     df_data_yrs = pd.merge(pd.concat(ts_corr, keys=splits),
#                            rg.df_splits, left_index=True, right_index=True)
#     return df_data_yrs

# df_data_yrs_sst1 = reaggregated_regions(rg.list_for_MI[0], rg, ('01-01', '08-01'), 2)
# df_data_yrs_sst2 = reaggregated_regions(rg.list_for_MI[0], rg, ('01-01', '09-01'), 2)

# df_data_yrs_smi = reaggregated_regions(rg.list_for_MI[1], rg, ('03-01', '09-01'), 1)
# df_data_yrs = pd.concat([df_data_yrs_sst1.iloc[:,:-2],
#                          df_data_yrs_sst2.iloc[:,:-2],
#                          df_data_yrs_smi], axis=1)

# df_data_yrs
# fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
# target_ts = rg.TV.RV_ts
# target_ts = (target_ts - target_ts.mean()) / target_ts.std()
# target_ts = (target_ts > target_ts.mean()).astype(int)

# #%%


# def plot_vs_lags(df_scores_dict, target_ts, df_boots_dict=None, rename_m: dict=None,
#                  orientation='vertical',
#                  colorlist: list=['#3388BB', '#EE6666', '#9988DD'], alpha=.05):

#     cropname = target_dataset.split('_')[1]
#     if (np.sum(target_ts).squeeze() / target_ts.size) < .45:
#         title = f'Skill predicting higher {cropname} yield years'
#     elif (np.sum(target_ts).squeeze() / target_ts.size) > .55:
#         title = f'Skill predicting lower {cropname} yield years'

#     if rename_m is None:
#         rename_m = {i:i for i in df_scores.index}
#     else:
#         rename_m = rename_m
#     metrics_cols = list(rename_m.values())
#     if orientation=='vertical':
#         f, ax_ = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
#                          sharex=True) ;
#     else:
#         f, ax_ = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
#                          sharey=False) ;
#     if type(df_scores_dict) is not dict:
#         df_scores_dict = {'forecast': [df_scores_dict, df_boots_dict]}

#     for i, forecast_label in enumerate(df_scores_dict):

#         df_scores_ = df_scores_dict[forecast_label][0]
#         df_boots_ = df_scores_dict[forecast_label][1]
#         for j, m in enumerate(metrics_cols):

#             ax = ax_[j]
#             ax.plot(df_scores_.columns, df_scores_.T[m],
#                     label=forecast_label,
#                     color=colorlist[i],
#                     linestyle='solid')
#             ax.set_xticks(df_scores_.columns)
#             ax.set_xticklabels(df_scores_.columns)
#             ax.set_xlabel('Month of forecast', fontsize=15)
#             ax.set_ylabel(rename_m[m], fontsize=15)
#             ax.tick_params(axis='both', which='major', labelsize=15)
#             ax.legend()
#             ax.set_ylim(0,1)
#             ax.fill_between(df_scores_.columns,
#                             df_boots_.reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
#                             df_boots_.reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
#                             edgecolor=colorlist[i], facecolor=colorlist[i], alpha=0.3,
#                             linestyle='solid', linewidth=2)

#             if m == 'corrcoef':
#                 ax.set_ylim(-.3,1)
#             elif m == 'roc_auc_score':
#                 ax.set_ylim(0,1)
#             else:
#                 ax.set_ylim(-.2,.8)
#     f.suptitle(title, x=.5, y=1.01, fontsize=18)
#     f.tight_layout()

# #%% forecast Yield as function of months

# use_mon_keys = 'JA'
# # if all([CondDepKeysDict[use_mon_keys][s]==CondDepKeys[use_mon_keys] for s in range(n_spl)]):
# #     print('Yo', 'keys not C.D. in every split')

# prediction = 'continuous' ; q = None
# # prediction = 'events' ; q = .4
# feature_selection = True

# if prediction == 'continuous':
#     model = ScikitModel(RidgeCV, verbosity=0)
#     kwrgs_model = {'scoring':'neg_mean_absolute_error',
#                     'alphas':np.concatenate([np.logspace(-5,0, 6),np.logspace(.01, 2.5, num=25)]), # large a, strong regul.
#                     'normalize':False,
#                     'fit_intercept':False,
#                     'kfold':5}
# elif prediction == 'events':
#     model = ScikitModel(LogisticRegressionCV, verbosity=0)
#     kwrgs_model = {'kfold':5,
#                    'scoring':'neg_brier_score'}


# target_ts = rg.TV.RV_ts ; target_ts.columns = [target_dataset]
# target_ts = (target_ts - target_ts.mean()) / target_ts.std()
# if prediction == 'events':
#     if q >= 0.5:
#         target_ts = (target_ts > target_ts.quantile(q)).astype(int)
#     elif q < .5:
#         target_ts = (target_ts < target_ts.quantile(q)).astype(int)
#     BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS

# rg.df_data = df_data_yrs.copy() ;
# tau_min=0 ; tau_max=0
# list_df_test = [] ; list_df_train = [] ; list_df_boot = [] ; list_pred_test = []
# for i, mon in enumerate(range(5,10)):
#     print(f'forecast at month {mon}')
# # mon = 10
#     keys = {}
#     for s in range(n_spl):
#         k_ = [str(mon)+k for k in CondDepKeysDict[use_mon_keys][s]]
#         keys[s] = [k for k in k_ if k in rg.df_data.columns]

#     if feature_selection:
#         if i == 0:
#             rg.df_data = pd.merge(pd.concat([rg.TV.RV_ts]*n_spl, keys=range(n_spl)),
#                                   rg.df_data, left_index=True, right_index=True)
#         keys = np.unique(core_pp.flatten(list(keys.values())))
#         corr, pvals, keys = feature_selection_CondDep(rg.df_data.copy(),
#                                                            keys)


#     out = rg.fit_df_data_ridge(target=target_ts,
#                                 keys=keys,
#                                 fcmodel=model,
#                                 kwrgs_model=kwrgs_model,
#                                 transformer=fc_utils.standardize_on_train,
#                                 tau_min=tau_min, tau_max=tau_max)
#     predict, weights, model_lags = out

#     # weights_norm = weights.mean(axis=0, level=1)
#     # weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

#     if prediction == 'continuous':
#         score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
#     elif prediction == 'events':
#         score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

#     df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
#                                                                      rg.df_data.iloc[:,-2:],
#                                                                      score_func_list,
#                                                                      n_boot = n_boot,
#                                                                      score_per_test=False,
#                                                                      blocksize=1,
#                                                                      rng_seed=1)
#     if prediction == 'continuous':
#         [model_lags['lag_0'][f'split_{i}'].alpha_ for i in range(n_spl)]
#         print(model.scikitmodel.__name__, '\n', 'Test score\n',
#               'RMSE {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
#               'MAE {:.2f}\n'.format(df_test_m.loc[0][0]['MAE']),
#               'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
#               '\nTrain score\n',
#               'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
#               'MAE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['MAE']),
#               'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))
#     elif prediction == 'events':
#         [model_lags['lag_0'][f'split_{i}'].Cs for i in range(n_spl)]
#         print(model.scikitmodel.__name__, '\n', 'Test score\n',
#               'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
#               'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
#               '\nTrain score\n',
#               'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
#               'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))


#     list_pred_test.append(functions_pp.get_df_test(predict.rename({0:mon}, axis=1),
#                                                  cols=[mon],
#                                                  df_splits=rg.df_splits))
#     df_test_m.index = [mon] ;
#     columns = pd.MultiIndex.from_product([np.array([mon]),
#                                         df_train_m.columns.levels[1]])
#     df_train_m.columns = columns
#     df_boot.columns = columns
#     list_df_test.append(df_test_m) ; list_df_train.append(df_train_m)
#     list_df_boot.append(df_boot)

# df_scores = pd.concat(list_df_test).T.loc[0]
# df_train_score = pd.concat(list_df_train, axis=1)
# df_boots = pd.concat(list_df_boot, axis=1)
# list_pred_test.insert(0, target_ts)
# df_pred_test = pd.concat(list_pred_test, axis=1)

# plot_vs_lags(df_scores, target_ts, df_boots, alpha=.1)




# #%%
# # =============================================================================
# # Cascaded forecast
# # =============================================================================

# # part 1
# include_obs = False ; max_multistep = 2
# list_df_test_c = [] ; list_df_boot_c = [] ;
# list_df_train_c = [] ; list_pred_test_c = []
# fc_month = {} ;
# for forecast_month in range(5,8):
#     print(f'Forecast month {forecast_month}')
#     target_month = forecast_month + 1

#     rg.df_data = df_data_yrs.copy()
#     lag_max = 1 ;
#     precur_keys = {}
#     for i, mon in enumerate(range(forecast_month, target_month)):
#         model = ScikitModel(RidgeCV, verbosity=0)
#         kwrgs_model = {'scoring':'neg_mean_absolute_error',
#                         'alphas':np.concatenate([[0],np.logspace(-5,0, 6),
#                                                  np.logspace(.01, 2.5, num=25)]), # large a, strong regul.
#                         'normalize':False,
#                         'fit_intercept':False,
#                         'kfold':5}
#         print(f'predicting precursors month {mon+1}')
#         if (mon - forecast_month) == max_multistep: # if max lag, stop multi-step forecasting
#             continue

#         predict_precursors = []
#         precur_target_keys = [str(mon+1)+k for k in CondDepKeys[use_mon_keys]]
#         for j, k in enumerate(precur_target_keys):

#             # if i == 0:
#             keys = [str(mon)+k for k in CondDepKeys[use_mon_keys]] #+ ['PDO']
#             for l in range(1, lag_max, 2):
#                 keys += [str(mon-l)+k for k in CondDepKeys[use_mon_keys]]
#             keys = [k for k in keys if k in rg.df_data.columns] # check if present
#             if i > 0:
#                 # add precursor timeseries of month? not yet implemented

#                 # add future precursor target to train model
#                 rg.df_data = pd.merge(df_data_yrs[[k]],
#                                       rg.df_data, left_index=True, right_index=True)
#             corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data.copy(), z_keys=keys,
#                                             keys=keys,
#                                             target=k)
#             # removing all keys that are Cond. Indep. in each trainingset
#             keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
#             for s in rg.df_splits.index.levels[0]:
#                 for k_i in keys:
#                     if (np.nan_to_num(pvals.loc[k_i][s],nan=alpha_CI) > alpha_CI).mean()>0:
#                         k_ = keys_dict[s].copy() ; k_.pop(k_.index(k_i))
#                         keys_dict[s] = k_

#             precur_keys[k] = keys_dict
#             keys = keys_dict


#             if len([True for k,v in keys_dict.items() if len(keys_dict[k])==0])!=0:
#                 print('No keys left')
#                 # k_ += [str(forecast_month)+k for k in CondDepKeys['JJA']]
#                 keys = [str(forecast_month)+k for k in CondDepKeys[use_mon_keys]]
#                 rg.df_data = pd.merge(df_data_yrs[[k]],
#                                       df_data_yrs[keys + ['TrainIsTrue', 'RV_mask']],
#                                       left_index=True, right_index=True)


#             target_ts = df_data_yrs[[k]].mean(0,level=1)
#             out = rg.fit_df_data_ridge(target=target_ts,
#                                         keys=keys,
#                                         fcmodel=model,
#                                         kwrgs_model=kwrgs_model,
#                                         transformer=fc_utils.standardize_on_train,
#                                         tau_min=0, tau_max=0)
#             predict, weights, model_lags = out
#             # weights_norm = weights.mean(axis=0, level=1)
#             # weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')
#             score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
#             df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
#                                                                              rg.df_data.iloc[:,-2:],
#                                                                              score_func_list,
#                                                                              n_boot = 1,
#                                                                              score_per_test=False,
#                                                                              blocksize=1,
#                                                                              rng_seed=1)

#             cvfitalpha = [models_lags[f'lag_{lag}'][f'split_{s}'].alpha_ for s in range(n_spl)]
#             if kwrgs_model['alphas'].max() in cvfitalpha: print('Max a reached')
#             if kwrgs_model['alphas'].min() in cvfitalpha: print('Min a reached')
#             print(model.scikitmodel.__name__+' '+k, '\t', 'Test score: ',
#                   # 'RMSE {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
#                   'MAE {:.2f}\t'.format(df_test_m.loc[0][0]['MAE']),
#                   # 'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
#                   'Train score: ',
#                   # 'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
#                   'MAE {:.2f}\t'.format(df_train_m.mean(0).loc[0]['MAE']))
#                   # 'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))


#             predict_precursors.append(predict.iloc[:,1:].rename({0:k},axis=1))
#         cascade_fc = pd.concat(predict_precursors, axis=1)
#         rg.df_data = pd.merge(cascade_fc.iloc[:,:],
#                                rg.df_splits, left_index=True, right_index=True)

# # =============================================================================
# #     #% Part 2 of Cascaded forecast
# # =============================================================================

#     if prediction == 'continuous':
#         model = ScikitModel(RidgeCV, verbosity=0)
#         kwrgs_model = {'scoring':'neg_mean_absolute_error',
#                         'alphas':np.concatenate([[0],
#                                                  np.logspace(-5,0, 6),
#                                                  np.logspace(.01, 2.5, num=25)]), # large a, strong regul.
#                         'normalize':False,
#                         'fit_intercept':False,
#                         'kfold':5}
#     elif prediction == 'events':
#         model = ScikitModel(LogisticRegressionCV, verbosity=0)
#         kwrgs_model = {'kfold':5,
#                        'Cs':np.logspace(-1.5, .5, num=25),
#                        'scoring':'neg_brier_score'}

#     target_ts = rg.TV.RV_ts ; target_ts.columns = [target_dataset]
#     target_ts = (target_ts - target_ts.mean()) / target_ts.std()
#     if prediction == 'events':
#         if q >= 0.5:
#             target_ts = (target_ts > target_ts.quantile(q)).astype(int)
#         elif q < .5:
#             target_ts = (target_ts < target_ts.quantile(q)).astype(int)
#         BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS


#     rg.df_data = pd.merge(cascade_fc.iloc[:,:],
#                                rg.df_splits, left_index=True, right_index=True)
#     if include_obs:
#         keys_obs = [str(mon)+k for k in CondDepKeys[use_mon_keys]] #+ ['PDO']
#         rg.df_data = pd.merge(df_data_yrs[keys_obs],
#                               rg.df_data, left_index=True, right_index=True)
#     rg.df_data = pd.merge(pd.concat([rg.TV.RV_ts]*n_spl, keys=range(n_spl)),
#                           rg.df_data, left_index=True, right_index=True)




#     keys = list(rg.df_data.columns[1:-2])
#     corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data.copy(), z_keys=keys,
#                                                 keys=keys)

#     # removing all keys that are Cond. Indep. in each trainingset
#     keys_dict = dict(zip(range(n_spl), [keys]*n_spl)) # all vars
#     for s in rg.df_splits.index.levels[0]:
#         for k_i in keys:
#             if (np.nan_to_num(pvals.loc[k_i][s],nan=alpha_CI) > alpha_CI).mean()>0:
#                 k_ = keys_dict[s].copy() ; k_.pop(k_.index(k_i))
#                 if len(k_) == 0:
#                     k_ = [str(mon)+k for k in CondDepKeys[use_mon_keys]] #+ ['PDO']
#                 keys_dict[s] = k_

#     fc_month[mon] = keys_dict


#     out = rg.fit_df_data_ridge(target=target_ts,
#                                 keys=keys_dict,
#                                 fcmodel=model,
#                                 kwrgs_model=kwrgs_model,
#                                 transformer=fc_utils.standardize_on_train,
#                                 tau_min=0, tau_max=0)
#     predict, weights, model_lags = out

#     # weights_norm = weights.mean(axis=0, level=1)
#     # weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')


#     if prediction == 'continuous':
#         score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
#     elif prediction == 'events':
#         score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

#     df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
#                                                                      rg.df_data.iloc[:,-2:],
#                                                                      score_func_list,
#                                                                      n_boot = n_boot,
#                                                                      score_per_test=False,
#                                                                      blocksize=1,
#                                                                      rng_seed=1)
#     if prediction == 'continuous':
#         cvfitalpha = [models_lags[f'lag_{lag}'][f'split_{s}'].alpha_ for s in range(n_spl)]
#         if kwrgs_model['alphas'].max() in cvfitalpha: print('Max a reached')
#         if kwrgs_model['alphas'].min() in cvfitalpha: print('Min a reached')
#         print(model.scikitmodel.__name__, '\n', 'Test score\n',
#               'RMSE {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
#               'MAE {:.2f}\n'.format(df_test_m.loc[0][0]['MAE']),
#               'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
#               '\nTrain score\n',
#               'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
#               'MAE {:.2f}\n'.format(df_train_m.mean(0).loc[0]['MAE']),
#               'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))
#     elif prediction == 'events':
#         [model_lags['lag_0'][f'split_{i}'].C_ for i in range(n_spl)]
#         print(model.scikitmodel.__name__, '\n', 'Test score\n',
#               'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
#               'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
#               '\nTrain score\n',
#               'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
#               'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))

#     list_pred_test_c.append(functions_pp.get_df_test(predict.rename({0:forecast_month},
#                                                                     axis=1),
#                                                  cols=[forecast_month],
#                                                  df_splits=rg.df_splits))
#     df_test_m.index = [forecast_month] ;
#     columns = pd.MultiIndex.from_product([np.array([forecast_month]),
#                                         df_train_m.columns.levels[1]])
#     df_train_m.columns = columns
#     df_boot.columns = columns
#     list_df_test_c.append(df_test_m) ; list_df_train_c.append(df_train_m)
#     list_df_boot_c.append(df_boot)


# df_scores_c = pd.concat(list_df_test_c).T.loc[0]
# df_train_score_c = pd.concat(list_df_train_c, axis=1)
# df_boots_c = pd.concat(list_df_boot_c, axis=1)
# list_pred_test_c.insert(0, target_ts)
# df_pred_test_c = pd.concat(list_pred_test_c, axis=1)


# plot_vs_lags(df_scores_c, target_ts, df_boots_c)



# #%%
# df_scores_dict = {'forecast':[df_scores,df_boots],
#                  'cascaded forecast': [df_scores_c,df_boots_c]}

# plot_vs_lags(df_scores_dict, target_ts, alpha=.1)

# #%% Event prediciton with RF


# RFmodel = ScikitModel(RandomForestClassifier, verbosity=0)
# kwrgs_model={'n_estimators':200,
#             'max_depth':2,
#             'scoringCV':'neg_brier_score',
#             'oob_score':True,
#             # 'min_samples_leaf':None,
#             'random_state':0,
#             'max_samples':.3,
#             'n_jobs':1}

# # keys = rg.df_data.columns[:-2]
# # keys = ['MAM..4..sst', 'JJA..4..sst']

# out = rg.fit_df_data_ridge(target=target_ts,
#                             keys=keys,
#                             fcmodel=RFmodel,
#                             kwrgs_model=kwrgs_model,
#                             transformer=False,
#                             tau_min=0, tau_max=0)
# predict, weights, model_lags = out

# weights_norm = weights.mean(axis=0, level=1)
# weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

# BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
# score_func_list = [BSS, fc_utils.metrics.roc_auc_score]


# df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
#                                                                  rg.df_data.iloc[:,-2:],
#                                                                  score_func_list,
#                                                                  n_boot = n_boot,
#                                                                  score_per_test=False,
#                                                                  blocksize=1,
#                                                                  rng_seed=1)
# print(RFmodel.scikitmodel.__name__, '\n', 'Test score\n',
#       'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
#       'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
#       '\nTrain score\n',
#       'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
#       'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))

# # [model_lags['lag_0'][f'split_{i}'].best_params_ for i in range(17)]

# #%% Event prediciton with logistic regressiong


# logit_skl = ScikitModel(LogisticRegressionCV, verbosity=0)
# kwrgs_model = {'kfold':10,
#                'scoring':'neg_brier_score'}

# target_ts = rg.TV.RV_ts
# target_ts = (target_ts - target_ts.mean()) / target_ts.std()
# target_ts = (target_ts > target_ts.quantile(.60)).astype(int)

# mon = 8
# keys = [str(mon)+k for k in CondDepKeys['JJA']]

# out = rg.fit_df_data_ridge(target=target_ts,
#                             keys=keys,
#                             fcmodel=logit_skl,
#                             kwrgs_model=kwrgs_model,
#                             transformer=fc_utils.standardize_on_train,
#                             tau_min=0, tau_max=0)
# predict, weights, model_lags = out

# weights_norm = weights.mean(axis=0, level=1)
# weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

# BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
# score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

# df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
#                                                                  rg.df_data.iloc[:,-2:],
#                                                                  score_func_list,
#                                                                  n_boot = n_boot,
#                                                                  score_per_test=False,
#                                                                  blocksize=1,
#                                                                  rng_seed=1)

# print(logit_skl.scikitmodel.__name__, '\n', 'Test score\n',
#       'BSS {:.2f}\n'.format(df_test_m.loc[0][0]['BSS']),
#       'AUC {:.2f}'.format(df_test_m.loc[0][0]['roc_auc_score']),
#       '\nTrain score\n',
#       'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
#       'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))


