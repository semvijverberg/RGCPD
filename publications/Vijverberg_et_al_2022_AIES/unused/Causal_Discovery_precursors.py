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
    # Optionally set font to Computer Modern to avoid common missing font errors
    # mpl.rc('font', family='serif', serif='cm10')

    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
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
seeds = seeds = [2] # [1,2,3,4,5]
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
corlags = np.array([6,5,4,3])
SM_lags = np.array([[l[0].replace(l[0][:2],str(int(l[1][:2])-2)),l[1]] for l in corlags])
# periodnames = ['MA', 'MJ', 'JA', 'SO']
start_end_TVdate = ('09-01', '10-31')
start_end_date = ('01-01', '12-31')

append_main = target_dataset
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3')
PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)
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
                            kwrgs_func={}, group_lag=True,
                            distance_eps=400, min_area_in_degrees2=7,
                            calc_ts=calc_ts, selbox=GlobalBox,
                            lags=corlags),
                 BivariateMI(name='smi', func=class_BivariateMI.corr_map,
                             alpha=alpha_corr, FDR_control=True,
                             kwrgs_func={}, group_lag=True,
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
# for p in rg.list_for_MI:
    # p.corr_xr['lag'] = ('lag', periodnames)



#%%
save = False
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':.15,
                  'wspace':-.15, 'size':3, 'cbar_vert':0.04,
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                   'y_ticks':np.arange(GlobalBox[2],61,20), #'x_ticks':np.arange(130, 280, 25),
                  'title':'',
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('sst', kwrgs_plot=kwrgs_plotcorr_sst, save=save)
#%%
SM = rg.list_for_MI[1]
# SM.adjust_significance_threshold(alpha_corr)
# SM.corr_xr['mask'] = ~SM.corr_xr['mask']
kwrgs_plotcorr_SM = {'row_dim':'lag', 'col_dim':'split','aspect':2, 'hspace':0.25,
                      'wspace':0, 'size':3, 'cbar_vert':0.04,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                       'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                      'title':'',
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr('smi', kwrgs_plot=kwrgs_plotcorr_SM, save=save)


#%%
# precur = rg.list_for_MI[1]
# precur.distance_eps = 400; precur.min_area_in_degrees2 = 7
rg.cluster_list_MI()
# rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plotcorr_sst)


# # Split ENSO?
# sst = rg.list_for_MI[0]
# copy_labels = sst.prec_labels.copy()
# all_labels = copy_labels.values[~np.isnan(copy_labels.values)]
# uniq_labels = np.unique(all_labels)
# prevail = {l:list(all_labels).count(l) for l in uniq_labels}
# prevail = functions_pp.sort_d_by_vals(prevail, reverse=True)
# label = list(prevail.keys())[0] # largest region
# sst.prec_labels, _ = find_precursors.split_region_by_lonlat(sst.prec_labels.copy(),
#                                                       label=int(label), plot_l=0,
#                                                       kwrgs_mask_latlon={'latmax':5})

# Ensure that what is in Atlantic is one precursor region
sst = rg.list_for_MI[0]
lonlatbox = [260, 350, 17, 40]
merge = find_precursors.merge_labels_within_lonlatbox
sst.prec_labels = merge(sst, lonlatbox)
sst.prec_labels = merge(sst, [110, 150, 0, 10])



lonlatbox = [220, 250, 25, 50] # eastern US
SM.prec_labels = merge(SM, lonlatbox)
lonlatbox = [250, 262, 25, 60] # mid-US
SM.prec_labels = merge(SM, lonlatbox)


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

kwrgs_plot_sst_group = kwrgs_plotcorr_sst.copy()
kwrgs_plot_sst_group.update({'cbar_vert':-0.1})
rg.quick_view_labels('sst', kwrgs_plot=kwrgs_plot_sst_group, save=save)

kwrgs_plot_SM_group = kwrgs_plotcorr_SM.copy()
kwrgs_plot_SM_group.update({'cbar_vert':-0.1})
rg.quick_view_labels('smi', kwrgs_plot=kwrgs_plot_SM_group, save=save)


#%%

rg.get_ts_prec(precur_aggr=tfreq)
rg.df_data = rg.df_data.rename({'Nonets':target_dataset},axis=1)
# fill first value of smi (NaN because of missing December when calc smi
# on month februari).
# keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='smi']
# rg.df_data[keys] = rg.df_data[keys].fillna(value=0, limit=1) #!!!
#%%
keys = list(rg.df_data.columns[:-2])
# keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='sst']
# keys.insert(0, target_dataset)

prior_knowledge_links = True

rg.PCMCI_init(keys=keys)
t_min = 0; t_max = 3
selected_links_splits = {}
for s, pcmci_dict_s in rg.pcmci_dict.items():
    var_names = pcmci_dict_s.var_names
    selected_links = {}
    for i, y in enumerate(var_names):
        links = []
        for j, z in enumerate(var_names):
            if z == target_dataset: # no lag available for target
                continue
            if prior_knowledge_links:
                if y == target_dataset:
                    links.append([(j,-l) for l in range(t_min, t_max+1)])
                elif y == z and 'sst' in y:
                    # no autocorr for SST, do not condition on it's own past
                    pass

                    #links.append([(j,-l) for l in range(t_min, 1)])
                elif 'sst' in y and 'smi' not in z:
                    # do not remove information of SMI from SST precursor, not physical
                    links.append([(j,-l) for l in range(t_min, t_max+1)])
                elif 'smi' in y:
                    links.append([(j,-l) for l in range(t_min, t_max+1)])
            else:
                links.append([(j,-l) for l in range(t_min, t_max+1)])


        selected_links[i] = core_pp.flatten(links)
    selected_links_splits[s] = selected_links

tigr_function_call='run_pcmci'
kwrgs_tigr={'tau_min':t_min, 'tau_max':t_max,
            'pc_alpha':.05,
            'max_conds_py':None,
            'max_conds_px':None,
            'max_combinations':3,
            'selected_links':selected_links_splits}

rg.PCMCI_df_data(keys=keys,
                 tigr_function_call=tigr_function_call,
                 kwrgs_tigr=kwrgs_tigr,
                 verbosity=2)
alpha_CI = .05
rg.PCMCI_get_links(var=target_dataset, alpha_level=alpha_CI)
rg.df_links.mean(axis=0,level=1)


#%% Manual (no time lags) CI tests


import wrapper_PCMCI

def feature_selection_CondDep(df_data, keys, alpha_CI=.05, x_lag=0, z_lag=0):
    # Feature selection Cond. Dependence
    keys = list(keys) # must be list
    corr, pvals = wrapper_PCMCI.df_data_Parcorr(df_data.copy(), z_keys=keys,
                                                keys=keys, z_lag=z_lag)
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


alpha_CI = .05
# keys = [k for k in rg.df_data.columns if k.split('..')[-1]=='sst']

corr = {} ; pval = {} ; keys_dict = {} ;
list_test = []
for lag in range(t_min, t_max+1):
    keys = list([k for k in rg.df_data.columns[1:-2]])
    corr_l, pval_l, keys_dict_l = feature_selection_CondDep(rg.df_data.copy(),
                                                       keys,
                                                       alpha_CI,
                                                       x_lag=lag,
                                                       z_lag=lag)
    # always C.D. (every split)
    keys = [k for k in keys if (np.nan_to_num(pval_l.loc[k],nan=alpha_CI) <= alpha_CI).mean()== 1]
    corr[lag] = corr_l
    pval[lag] = pval_l
    keys_dict[lag] = keys_dict_l


list_pval_splits = [] ; list_corr_splits = []
for s in range(n_spl):
    list_pval = [] ; list_corr = []
    for i, lag in enumerate(range(t_min, t_max+1)):
        pval_ = pval[lag][s].max(axis=0, level=0) # lowest significance
        streng = corr[lag][s].mean(axis=0, level=0) # mean parcorr value
        list_pval.append(pval_)
        list_corr.append(streng)
    df_pval_lag = pd.concat(list_pval, axis=1, keys=range(t_min, t_max+1))
    df_streng_lag = pd.concat(list_corr, axis=1, keys=range(t_min, t_max+1))
    df_streng_lag.columns = [f'coeff l{l}' for l in df_streng_lag.columns]
    list_pval_splits.append(df_pval_lag) ;
    list_corr_splits.append(df_streng_lag)
df_pvals_fs = pd.concat(list_pval_splits, keys=range(n_spl))
df_str_fs   = pd.concat(list_corr_splits, keys=range(n_spl))



#%%
# =============================================================================
# Plotting Causal links
# =============================================================================

method = 'pcmci'
# method = 'PC-like'

if method == 'pcmci':
    rg.PCMCI_get_links(target_dataset, alpha_level=alpha_CI)
    df_links = rg.df_links.mean(axis=0,level=1)
    df_MCI = rg.df_MCIc.mean(axis=0,level=1)
elif method == 'PC-like':
    df_links = (df_pvals_fs <= alpha_CI).mean(axis=0,level=1)
    df_MCI = df_str_fs.mean(axis=0,level=1)


CondDepKeys = {}
mon_to_lag = dict(zip(periodnames[::-1], df_links.columns))
for mon in periodnames:
    lag = mon_to_lag[mon]
    df_links[lag]
    causal = df_links[lag][df_links[lag] > 0.]
    MCI = df_MCI[[f'coeff l{lag}']][df_links[lag] > 0.]
    RB  = df_links[lag][df_links[lag] > 0.]
    CondDepKeys[mon] = list(zip(MCI.index,
                                MCI.values.reshape(-1),
                                RB.values.reshape(-1)))


if feature_selection:
    append_str = '-'.join(periodnames)
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
                    text = f'{int(RB[q]*n_spl)}/{count}'
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
                                  f'{precur.name}_eps{precur.distance_eps}'
                                  f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_labels'+rg.figext),
                         bbox_inches='tight')

        # MCI values plot
        kwrgs_plot.update({'clevels':np.arange(-0.8, 0.9, .2),
                           'textinmap':textinmap})
        fig = plot_maps.plot_corr_maps(MCIstr.mean(dim='split'),
                                       mask_xr=np.isnan(MCIstr.mean(dim='split')).astype(bool),
                                       **kwrgs_plot)
        if save:
            fig.savefig(os.path.join(dirpath,
                                      f'{precur.name}_eps{precur.distance_eps}'
                                      f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_MCI'+rg.figext),
                        bbox_inches='tight')


#%%
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
               'kfold':5}

method = 'pcmci'
# method = 'PC-like'


# get causal timeseries:
strongest_lag = False
df_causal = np.zeros(n_spl, dtype=object)
for s in range(n_spl):
    alpha_CI = .05
    if method == 'pcmci':
        rg.PCMCI_get_links(var=target_dataset, alpha_level=alpha_CI)
        # ensure there is one predictor
        while any(rg.df_links.loc[s].values.reshape(-1))==False:
            alpha_CI +=.05
            rg.PCMCI_get_links(var=target_dataset, alpha_level=alpha_CI)
        df_links_s = rg.df_MCIa.loc[s] <= alpha_CI
        df_links_s.columns = [int(k[-1]) for k in df_links_s.columns]
        df_str_s   = rg.df_MCIc.loc[s]
    elif method == 'PC-like':
        df_links_s = df_pvals_fs.loc[s] <= alpha_CI
        df_str_s   = df_str_fs.loc[s]

    ts_list = []
    df_MCIc_s = df_str_s
    df_data_s = rg.df_data.loc[s].copy()
    fit_masks = rg.df_data.loc[s][['TrainIsTrue', 'RV_mask']].copy()
    newfitmask = fit_masks[['TrainIsTrue','RV_mask']][fit_masks['RV_mask']]
    for i, k in enumerate(df_links_s.index):
        lags = df_links_s.loc[k][df_links_s.loc[k]].index
        if strongest_lag and len(lags) > 1:
            strngth = df_MCIc_s.loc[k][[f'coeff l{l}' for l in lags]].abs()
            lags = [int(strngth.idxmax()[-1])]
        for l in lags:
            m = fc_utils.apply_shift_lag(fit_masks, l)['x_fit']
            ts = df_data_s[[k]][m]
            ts.columns = [k.replace(k.split('..')[0], str(l))]
            ts.index = newfitmask.index
            ts_list.append(ts)
    df_s = pd.concat(ts_list, axis=1)
    df_s = df_s.merge(newfitmask, left_index=True, right_index=True)
    df_causal[s] = df_s
df_causal = pd.concat(df_causal, keys=range(n_spl))


# target
fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
# metrics
RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
metric_names = [s.__name__ for s in score_func_list]


#%%
lag_ = 0 ; n_boot = 0


out = rg.fit_df_data_ridge(df_data=df_causal,
                           target=target_ts,
                           tau_min=0, tau_max=0,
                           kwrgs_model=kwrgs_model,
                           fcmodel=fcmodel,
                           transformer=None)

# out = rg.fit_df_data_ridge(df_data=rg.df_data.copy(),
#                            keys=keys_dict_l,
#                            target=target_ts,
#                            tau_min=0, tau_max=0,
#                            kwrgs_model=kwrgs_model,
#                            fcmodel=fcmodel,
#                            transformer=None)


predict, weights, models_lags = out
prediction = predict.rename({predict.columns[0]:'target',lag_:'Prediction'},
                            axis=1)


weights_norm = weights.mean(axis=0, level=1)
weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box', figsize=(15,5))


df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                       rg.df_data.iloc[:,-2:],
                                                       score_func_list,
                                                       n_boot = n_boot,
                                                       blocksize=1,
                                                       rng_seed=seed)

m = models_lags[f'lag_{lag_}'][f'split_{0}']
cvfitalpha = [models_lags[f'lag_{lag_}'][f'split_{s}'].alpha_ for s in range(n_spl)]
if kwrgs_model['alphas'].max() in cvfitalpha: print('Max a reached')
if kwrgs_model['alphas'].min() in cvfitalpha: print('Min a reached')
# assert kwrgs_model['alphas'].min() not in cvfitalpha, 'decrease min a'

df_test = functions_pp.get_df_test(predict.rename({lag_:'causal'}, axis=1),
                                   df_splits=rg.df_splits)
print(df_test_m)


#%%
from matplotlib import gridspec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

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
ax0.set_ylabel('Standardized Soy Yield', fontsize=fontsize)
ax0.tick_params(labelsize=fontsize)
ax0.axhline(y=0, color='black', lw=1)
ax0.legend(fontsize=fontsize)

df_scores = df_test_m.loc[0]['Prediction']
Texts1 = [] ; Texts2 = [] ;
textprops = dict(color='black', fontsize=fontsize+4, family='serif')
rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr. Coeff.', 'MAE':'MAE-SS'}
for k, label in rename_met.items():
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
# if save:
#     plt.savefig(os.path.join(rg.path_outsub2,
#                          f'forecast_aCI{alpha_CI}.pdf'),
#             bbox_inches='tight')






#%% forecasting

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