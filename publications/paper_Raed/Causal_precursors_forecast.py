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
    n_cpu = 10
else:
    n_cpu = 3
# else:
#     # Optionally set font to Computer Modern to avoid common missing font errors
#     mpl.rc('font', family='serif', serif='cm10')

#     mpl.rc('text', usetex=True)
#     mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# import sklearn.linear_model as scikitlinear
import argparse
from time import sleep
import itertools, os, re
from matplotlib import gridspec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

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

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']


target_datasets = ['USDA_Soy_clusters__1', 'USDA_Soy_clusters__2']
seeds = [1,2,3,4] # ,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
methods = ['ranstrat_20'] # ['ranstrat_20'] timeseriessplit_30
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
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/init_usda_soy_spatial_mean_ts.nc')
    name_ds='Soy_Yield' ; cluster_label = ''
elif target_dataset == 'USDA_Soy_always_data':
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts_allways_data.nc')
    name_ds='Soy_Yield' ; cluster_label = ''
elif target_dataset == 'USDA_Soy_csv_midwest':
    path = os.path.join(main_dir, 'publications/paper_Raed/data/ts_spatial_avg_midwest.csv')
    TVpath = read_csv_Raed(path)
elif target_dataset.split('__')[0] == 'USDA_Soy_clusters':
    TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/linkage_ward_nc2_dendo_0d570.nc')
    cluster_label = int(target_dataset.split('__')[1]) ; name_ds = 'ts'
elif target_dataset == 'USDA_Maize':
    # USDA dataset 1950 - 2019
    TVpath =  os.path.join(main_dir, 'publications/paper_Raed/data/usda_maize_spatial_mean_ts.nc')
    name_ds='Maize_Yield' ; cluster_label = None
elif target_dataset == 'Aggregate_States':
    path =  os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')
    States = ['KENTUCKY', 'TENNESSEE', 'MISSOURI', 'ILLINOIS', 'INDIANA']
    TVpath = read_csv_State(path, State=States, col='obs_yield').mean(1)
    TVpath = pd.DataFrame(TVpath.values, index=TVpath.index, columns=['KENTUCKYTENNESSEEMISSOURIILLINOISINDIANA'])
    name_ds='Soy_Yield' ; cluster_label = ''
elif target_dataset == 'All_State_average':
    path =  os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')
    TVpath = read_csv_State(path, State=None, col='obs_yield').mean(1)
    TVpath = pd.DataFrame(TVpath.values, index=TVpath.index, columns=['All_State_average'])
    name_ds='Soy_Yield' ; cluster_label = ''
else:
    path =  os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')
    TVpath = read_csv_State(path, State=target_dataset, col='obs_yield')
    TVpath = pd.DataFrame(TVpath.values, index=TVpath.index, columns=[TVpath.name])
    name_ds='Soy_Yield' ; cluster_label = ''



calc_ts= 'region mean' # 'pattern cov'
alpha_corr = .05
alpha_CI = .05
n_boot = 2000
append_pathsub = f'/{method}/s{seed}'

append_main = target_dataset
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'forecast')
if target_dataset.split('__')[0] == 'USDA_Soy_clusters': # add cluster hash
    path_out_main = os.path.join(path_out_main, TVpath.split('.')[0].split('_')[-1])
elif target_dataset.split('__')[0] == 'All_State_average': # add cluster hash
    path_out_main = os.path.join(path_out_main, 'All_State_Average')
elif target_dataset in All_states: # add cluster hash
    path_out_main = os.path.join(path_out_main, 'States')

PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)
USBox = (225, 300, 20, 60)

load = 'all'
save = True

list_of_name_path = [(cluster_label, TVpath),
                       ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc')),
                      # ('z500', os.path.join(path_raw, 'z500_1950-2019_1_12_monthly_1.0deg.nc')),
                       ('smi', os.path.join(path_raw, 'SM_ownspi_gamma_2_1950-2019_1_12_monthly_1.0deg.nc'))]


#%% run RGPD


def pipeline(lags, periodnames, use_vars=['sst', 'smi'], load=False):
    #%%
    if int(lags[0][0].split('-')[-2]) > 7: # first month after july
        crossyr = True
    else:
        crossyr = False

    SM_lags = lags.copy()
    for i, l in enumerate(SM_lags):
        orig = '-'.join(l[0].split('-')[:-1])
        repl = '-'.join(l[1].split('-')[:-1])
        SM_lags[i] = [l[0].replace(orig, repl), l[1]]

    list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                                alpha=alpha_corr, FDR_control=True,
                                kwrgs_func={},
                                distance_eps=250, min_area_in_degrees2=3,
                                calc_ts=calc_ts, selbox=GlobalBox,
                                lags=lags, group_split=True,
                                use_coef_wghts=True),
                      BivariateMI(name='smi', func=class_BivariateMI.corr_map,
                                 alpha=alpha_corr, FDR_control=True,
                                 kwrgs_func={},
                                 distance_eps=200, min_area_in_degrees2=3,
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
    rg.figext = '.png'


    subfoldername = target_dataset
                                            #list(np.array(start_end_year, str)))
    subfoldername += append_pathsub


    rg.pp_precursors(detrend=[True, {'tp':False, 'smi':False}],
                     anomaly=[True, {'tp':False, 'smi':False}],
                     auto_detect_mask=[False, {'swvl1':True, 'swvl2':True}])
    if crossyr:
        TV_start_end_year = (start_end_year[0]+1, 2019)
    else:
        TV_start_end_year = (start_end_year[0], 2019)

    kwrgs_core_pp_time = {'start_end_year': TV_start_end_year}
    rg.pp_TV(name_ds=name_ds, detrend={'method':'linear'}, ext_annual_to_mon=False,
             kwrgs_core_pp_time=kwrgs_core_pp_time)
    # if method.split('_')[0]=='leave':
    #     rg.traintest(method, gap_prior=1, gap_after=1, seed=seed,
    #                  subfoldername=subfoldername)
    # else:
    rg.traintest(method, seed=seed, subfoldername=subfoldername)


    #%%
    sst = rg.list_for_MI[0]
    if 'sst' in use_vars:
        load_sst = '{}_a{}_{}_{}_{}'.format(sst._name, sst.alpha,
                                            sst.distance_eps,
                                            sst.min_area_in_degrees2,
                                            periodnames[-1])
        if load == 'maps' or load == 'all':
            loaded = sst.load_files(rg.path_outsub1, load_sst)
        else:
            loaded = False
        if hasattr(sst, 'corr_xr')==False:
            rg.calc_corr_maps('sst')
    #%%
    SM = rg.list_for_MI[1]
    if 'smi' in use_vars:
        load_SM = '{}_a{}_{}_{}_{}'.format(SM._name, SM.alpha,
                                            SM.distance_eps,
                                            SM.min_area_in_degrees2,
                                            periodnames[-1])
        if load == 'maps' or load == 'all':
            loaded = SM.load_files(rg.path_outsub1, load_SM)
        else:
            loaded = False
        if hasattr(SM, 'corr_xr')==False:
            rg.calc_corr_maps('smi')

    #%%

    # sst.distance_eps = 250 ; sst.min_area_in_degrees2 = 4
    if hasattr(sst, 'prec_labels')==False and 'sst' in use_vars:
        rg.cluster_list_MI('sst')

        # check if west-Atlantic is a seperate region, otherwise split region 1
        df_labels = find_precursors.labels_to_df(sst.prec_labels)
        dlat = df_labels['latitude'] - 29
        dlon = df_labels['longitude'] - 290
        zz = pd.concat([dlat.abs(),dlon.abs()], axis=1)
        if zz.query('latitude < 10 & longitude < 10').size==0:
            print('Splitting region west-Atlantic')
            largest_regions = df_labels['n_gridcells'].idxmax()
            split = find_precursors.split_region_by_lonlat
            sst.prec_labels, _ = split(sst.prec_labels.copy(), label=int(largest_regions),
                                    kwrgs_mask_latlon={'upper_right': (263, 16)})

        merge = find_precursors.merge_labels_within_lonlatbox

        # # Ensure that what is in Atlantic is one precursor region
        lonlatbox = [263, 300, 17, 40]
        sst.prec_labels = merge(sst, lonlatbox)
        # Indonesia_oceans = [110, 150, 0, 10]
        # sst.prec_labels = merge(sst, Indonesia_oceans)
        Japanese_sea = [100, 150, 30, 50]
        sst.prec_labels = merge(sst, Japanese_sea)
        Mediterrenean_sea = [0, 45, 30, 50]
        sst.prec_labels = merge(sst, Mediterrenean_sea)
        East_Tropical_Atlantic = [330, 20, -10, 10]
        sst.prec_labels = merge(sst, East_Tropical_Atlantic)



    if 'sst' in use_vars:
        if loaded==False:
            sst.store_netcdf(rg.path_outsub1, load_sst, add_hash=False)
        sst.prec_labels['lag'] = ('lag', periodnames)
        sst.corr_xr['lag'] = ('lag', periodnames)
        rg.quick_view_labels('sst', min_detect_gc=.5, save=save,
                              append_str=periodnames[-1])
        # store forecast month
        months = {'JJ':'August', 'MJ':'July', 'AM':'June', 'MA':'May', 'FM':'April',
                  'SO':'hindcast'}
        last_month = list(sst.corr_xr.lag.values)[-1]
        rg.fc_month = months[last_month]
    #%%
    if hasattr(SM, 'prec_labels')==False and 'smi' in use_vars:
        SM = rg.list_for_MI[1]
        rg.cluster_list_MI('smi')

        lonlatbox = [220, 240, 25, 55] # eastern US
        SM.prec_labels = merge(SM, lonlatbox)
        lonlatbox = [270, 280, 25, 45] # mid-US
        SM.prec_labels = merge(SM, lonlatbox)
    if 'smi' in use_vars:
        if loaded==False:
            SM.store_netcdf(rg.path_outsub1, load_SM, add_hash=False)
        SM.corr_xr['lag'] = ('lag', periodnames)
        SM.prec_labels['lag'] = ('lag', periodnames)
        rg.quick_view_labels('smi', min_detect_gc=.5, save=save,
                              append_str=periodnames[-1])


    #%% Calculate spatial mean timeseries of precursor regions
    filepath_df_output = os.path.join(rg.path_outsub1,
                                      f'df_output_{periodnames[-1]}.h5')
    if load == 'all' and os.path.exists(filepath_df_output):
        df_output = functions_pp.load_hdf5(filepath_df_output)
        rg.df_data  = df_output['df_data']
        rg.df_pvals = df_output['df_pvals']
        rg.df_corr  = df_output['df_corr']
    else:
        rg.get_ts_prec()
        rg.df_data = rg.df_data.rename({rg.df_data.columns[0]:target_dataset},axis=1)
        #%% Causal Inference
        def feature_selection_CondDep(df_data, keys, z_keys=None, alpha_CI=.05, x_lag=0, z_lag=0):

            # Feature selection Cond. Dependence
            keys = list(keys) # must be list
            if z_keys is None:
                z_keys = keys
            corr, pvals = wrapper_PCMCI.df_data_Parcorr(df_data.copy(), keys=keys,
                                                        z_keys=z_keys, z_lag=z_lag)
            # removing all keys that are Cond. Indep. in each trainingset
            keys_dict = dict(zip(range(rg.n_spl), [keys]*rg.n_spl)) # all vars
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
        # select the causal regions from analysys in Causal Inferred Precursors
        print('Start Causal Inference')
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


        rg.df_pvals = pd.concat(list_pvals,axis=0)
        rg.df_corr = pd.concat(list_corr,axis=0)

        df_output = {'df_data': rg.df_data,
                      'df_pvals':rg.df_pvals,
                      'df_corr':rg.df_corr}
        functions_pp.store_hdf_df(df_output, filepath_df_output)
    #%%
    return rg


# pipeline(lags=lags_july, periodnames=periodnames_july)

#%%


if __name__ == '__main__':
    sy = start_end_year[0]
    sy_p1 = start_end_year[0] + 1
    # =============================================================================
    # 4 * bimonthly
    # =============================================================================
    lags_july = np.array([[f'{sy}-12-01', f'{sy_p1}-01-01'],# DJ
                          [f'{sy_p1}-02-01', f'{sy_p1}-03-01'],# FM
                          [f'{sy_p1}-04-01', f'{sy_p1}-05-01'],# AM
                          [f'{sy_p1}-06-01', f'{sy_p1}-07-01'] # JJ
                          ])
    periodnames_july = ['DJ', 'FM', 'AM', 'JJ']

    lags_june = np.array([[f'{sy}-11-01', f'{sy}-12-01'],# FM
                          [f'{sy_p1}-01-01', f'{sy_p1}-02-01'],# FM
                          [f'{sy_p1}-03-01', f'{sy_p1}-04-01'],# AM
                          [f'{sy_p1}-05-01', f'{sy_p1}-06-01'] # JJ
                          ])
    periodnames_june = ['ND', 'JF', 'MA', 'MJ']

    lags_may = np.array([[f'{sy}-10-01', f'{sy}-11-01'],# ON
                          [f'{sy}-12-01', f'{sy_p1}-01-01'],# DJ
                          [f'{sy_p1}-02-01', f'{sy_p1}-03-01'],# FM
                          [f'{sy_p1}-04-01', f'{sy_p1}-05-01'] # AM
                          ])
    periodnames_may = ['ON', 'DJ', 'FM', 'AM']

    lags_april = np.array([[f'{sy}-09-01', f'{sy}-10-01'],# SO
                            [f'{sy}-11-01', f'{sy}-12-01'],# ND
                            [f'{sy_p1}-01-01', f'{sy_p1}-02-01'],# JF
                            [f'{sy_p1}-03-01', f'{sy_p1}-04-01'] # MA
                            ])
    periodnames_april = ['SO', 'ND', 'JF', 'MA']

    lags_march = np.array([[f'{sy}-08-01', f'{sy}-09-01'],# SO
                            [f'{sy}-10-01', f'{sy}-11-01'],# ND
                            [f'{sy}-12-01', f'{sy_p1}-01-01'],# JF
                            [f'{sy_p1}-02-01', f'{sy_p1}-03-01'] # MA
                            ])
    periodnames_march = ['AS', 'ON', 'DJ', 'FM']
    # # =============================================================================
    # # 3 * bimonthly
    # # =============================================================================
    # lags_july = np.array([#['1950-12-01', '1951-01-01'],# DJ
    #                       ['1950-02-01', '1950-03-01'],# FM
    #                       ['1950-04-01', '1950-05-01'],# AM
    #                       ['1950-06-01', '1950-07-01'] # JJ
    #                       ])
    # periodnames_july = ['March', 'May', 'July']


    # lags_june = np.array([#['1950-11-01', '1950-12-01'],# FM
    #                       ['1950-01-01', '1950-02-01'],# FM
    #                       ['1950-03-01', '1950-04-01'],# AM
    #                       ['1950-05-01', '1950-06-01'] # JJ
    #                       ])
    # periodnames_june = ['Feb', 'April', 'June']


    # lags_may = np.array([#['1950-10-01', '1950-11-01'],# ON
    #                       ['1950-12-01', '1951-01-01'],# DJ
    #                       ['1951-02-01', '1951-03-01'],# FM
    #                       ['1951-04-01', '1951-05-01'] # AM
    #                       ])
    # periodnames_may = ['Jan', 'Mar', 'May']


    # lags_april = np.array([#['1950-09-01', '1950-10-01'],# SO
    #                         ['1950-11-01', '1950-12-01'],# ND
    #                         ['1951-01-01', '1951-02-01'],# JF
    #                         ['1951-03-01', '1951-04-01'] # MA
    #                         ])
    # periodnames_april = ['Dec', 'Feb', 'April']



    use_vars_july = ['sst', 'smi']
    use_vars_june = ['sst', 'smi']
    use_vars_may = ['sst', 'smi']
    use_vars_april = ['sst', 'smi']
    use_vars_march = ['sst', 'smi']


    # Run in Parallel
    lag_list = [lags_july, lags_june, lags_may, lags_april, lags_march]
    periodnames_list = [periodnames_july, periodnames_june,
                        periodnames_may, periodnames_april,
                        periodnames_march]
    use_vars_list = [use_vars_july, use_vars_june,
                     use_vars_may, use_vars_april, use_vars_march]

    futures = [] ; rg_list = []
    for lags, periodnames, use_vars in zip(lag_list, periodnames_list, use_vars_list):
        if load == False:
            futures.append(delayed(pipeline)(lags, periodnames, use_vars, load))
        else:
            rg_list.append(pipeline(lags, periodnames, use_vars, load))

    if load == False:
        rg_list = Parallel(n_jobs=n_cpu, backend='loky')(futures)
rg = rg_list[0]





# =============================================================================
# Continuous forecast
# =============================================================================
filepath_df_datas = os.path.join(rg.path_outsub1, 'df_data')
os.makedirs(filepath_df_datas, exist_ok=True)
filepath_verif = os.path.join(rg.path_outsub1, 'verif')
os.makedirs(filepath_verif, exist_ok=True)

#%% Continuous forecast: get Combined Lead time models

from sklearn.linear_model import Ridge
from stat_models_cont import ScikitModel
kwrgs_model1 = {'scoringCV':'neg_mean_squared_error',
                'alpha':list(np.concatenate([np.logspace(-4,0, 5),
                                          np.logspace(.5, 2, num=10)])), # large a, strong regul.
                'normalize':False,
                'fit_intercept':False,
                'kfold':10,
                'n_jobs':n_cpu}
model1_tuple = (ScikitModel(Ridge, verbosity=0),
                kwrgs_model1)


from sklearn.ensemble import RandomForestRegressor
kwrgs_model2={'n_estimators':[400],
              'max_depth':[5,10],
              'scoringCV':'neg_mean_squared_error',
              'criterion':'mse',
              'oob_score':True,
              'random_state':0,
              'min_impurity_decrease':[.02, 0],
              'max_samples':[0.55,.75],
              'kfold':10,
              'n_jobs':n_cpu}

model2_tuple = (ScikitModel(RandomForestRegressor, verbosity=0),
                kwrgs_model2)

# target timeseries, standardize using training data
target_ts = rg.transform_df_data(rg.df_data.iloc[:,[0]].merge(rg.df_splits,
                                                  left_index=True,
                                                  right_index=True),
                                 transformer=fc_utils.standardize_on_train)

for fcmodel, kwrgs_model in [model1_tuple, model2_tuple]:
    kwrgs_model_CL = kwrgs_model.copy() ;
    # kwrgs_model_CL.update({'alpha':kwrgs_model['alpha'][::3]})
    model_name_CL = fcmodel.scikitmodel.__name__
    filepath_dfs = os.path.join(filepath_df_datas,
                                f'CL_models_cont{model_name_CL}.h5')
    df_data_CL = {}
    try:
        df_data_CL = functions_pp.load_hdf5(filepath_dfs)
        loaded = True

    except:
        loaded = False


    for i, rg in enumerate(rg_list):
        print(model_name_CL, i)

        if loaded:
            rg.df_CL_data = df_data_CL[f'{rg.fc_month}_df_data']
            continue
        else:
            pass

        mean_vars=['sst', 'smi']
        # mean_vars=[]
        for i, p in enumerate(rg.list_for_MI):
            if p.name in mean_vars:
                if p.calc_ts == 'pattern cov':
                    mean_vars[i] +='_sp'
        df_data, keys_dict = utils_paper3.get_df_mean_SST(rg,
                                             mean_vars=mean_vars,
                                             alpha_CI=alpha_CI,
                                             n_strongest='all',
                                             weights=True,
                                             fcmodel=fcmodel,
                                             kwrgs_model=kwrgs_model_CL,
                                             target_ts=target_ts,
                                             labels=None)

        rg.df_CL_data = df_data
        df_data_CL[f'{rg.fc_month}_df_data'] = df_data
    # store df_pred_tuple
    functions_pp.store_hdf_df(df_data_CL, filepath_dfs)



#%% Continuous forecast: Make prediction

model_combs = [['Ridge', 'Ridge'],
               ['Ridge', 'RandomForestRegressor'],
               ['RandomForestRegressor', 'RandomForestRegressor']]

fcmodel, kwrgs_model = model1_tuple
for model_name_CL, model_name in model_combs:
    if model_name == 'Ridge':
        fcmodel, kwrgs_model = model1_tuple
    elif model_name == 'RandomForestRegressor':
        fcmodel, kwrgs_model = model2_tuple

    filepath_dfs = os.path.join(filepath_df_datas,
                                f'CL_models_cont{model_name_CL}.h5')
    try:
        df_data_CL = functions_pp.load_hdf5(filepath_dfs)
    except:
        print('loading CL models failed, skipping this model')
        continue
    for nameTarget in ['Target', 'Target*Signal']:
        for i, rg in enumerate(rg_list):
            print(model_name_CL, model_name, i)
            # get CL model of that month
            rg.df_CL_data = df_data_CL[f'{rg.fc_month}_df_data']

        # get forcing per fc_month
        utils_paper3.get_df_forcing_cond_fc(rg_list,
                                    region='only_Pacific',
                                    name_object='df_CL_data')
        for i, rg in enumerate(rg_list):
            keys_dict = {s:rg.df_CL_data.loc[s].dropna(axis=1).columns[:-2] \
                         for s in range(rg.n_spl)}
            # get estimated signal from Pacific

            # adapt target timeseries for fitting
            df_forcing = rg.df_forcing.mean(axis=1) # mean over cols
            # very rare exception that Pac is not causal @ alpha level
            df_forcing = df_forcing.fillna(value=1)

            target_ts = target_ts.rename({target_ts.columns[0]:'Target'},axis=1)
            target_ts_signal = target_ts.multiply(df_forcing.abs(),
                                                  axis=0)
            target_ts_signal= target_ts_signal.rename(\
                                          {rg.df_data.columns[0]:
                                           f'Target * {rg.fc_month} signal'},
                                          axis=1)
            target_ts_signal = rg.transform_df_data(target_ts_signal.merge(rg.df_splits,
                                                                           left_index=True,
                                                                           right_index=True),
                                                    transformer=fc_utils.standardize_on_train)
            if nameTarget == 'Target':
                _target_ts = target_ts
            if nameTarget == 'Target*Signal':
                _target_ts = target_ts_signal


            prediction_tuple = rg.fit_df_data_ridge(df_data=rg.df_CL_data,
                                                    keys=keys_dict,
                                                    target=_target_ts,
                                                    tau_min=0, tau_max=0,
                                                    kwrgs_model=kwrgs_model,
                                                    fcmodel=fcmodel,
                                                    transformer=None)

            predict, weights, models_lags = prediction_tuple

            prediction = predict.rename({predict.columns[0]:nameTarget,
                                         0:rg.fc_month}, axis=1)

            prediction_tuple = (prediction, weights, models_lags)
            rg.prediction_tuple = prediction_tuple

        # store output predictions of each month per model
        model_name = fcmodel.scikitmodel.__name__
        df_predictions, df_w_save = utils_paper3.df_predictions_for_plot(rg_list)
        d_df_preds={'df_predictions':df_predictions, 'df_weights':df_w_save}
        filepath_dfs = os.path.join(filepath_df_datas,
                                    f'predictions_cont_CL{model_name_CL}_'\
                                    f'{model_name}_{nameTarget}.h5')
        functions_pp.store_hdf_df(d_df_preds, filepath_dfs)

#%% Continuous forecast: Verification
verif_combs = [['Target', 'Target'],
               ['Target*Signal', 'Target'],
               ['Target*Signal', 'Target*Signal']]

for model_name_CL, model_name in model_combs:
    for nameTarget_fit, nameTarget in verif_combs:

        f_name = f'predictions_cont_CL{model_name_CL}_{model_name}_'\
                                                    f'{nameTarget_fit}.h5'
        filepath_dfs = os.path.join(filepath_df_datas, f_name)

        try:
            d_dfs = functions_pp.load_hdf5(filepath_dfs)
            df_predictions = d_dfs['df_predictions']
        except:
            print('loading predictions failed, skipping this model')
            continue

        for i, rg in enumerate(rg_list):
            prediction = df_predictions[[nameTarget, rg.fc_month]]

            # metrics
            bench = prediction[nameTarget].mean()
            RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=bench).RMSE
            MAE_SS = fc_utils.ErrorSkillScore(constant_bench=bench).MAE
            score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS,
                               fc_utils.r2_score]
            metric_names = [s.__name__ for s in score_func_list]
            verification_tuple = fc_utils.get_scores(prediction,
                                                     rg.df_data.iloc[:,-2:],
                                                     score_func_list,
                                                     n_boot=n_boot,
                                                     blocksize=1,
                                                     rng_seed=seed)
            df_train_m, df_test_s_m, df_test_m, df_boot = verification_tuple
            rg.verification_tuple = verification_tuple

        df_scores, df_boot, df_tests = utils_paper3.df_scores_for_plot(rg_list,
                                                           'verification_tuple')
        d_dfs={'df_scores':df_scores, 'df_boot':df_boot, 'df_tests':df_tests}
        filepath_dfs = os.path.join(filepath_df_datas,
                        f'scores_cont_CL{model_name_CL}_{model_name}_'\
                            f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5')
        functions_pp.store_hdf_df(d_dfs, filepath_dfs)

#%% Continuous forecast: Verification conditional forecast
import utils_paper3

for model_name_CL, model_name in model_combs:
    for nameTarget_fit in ['Target', 'Target*Signal']:
        print(f'CL: {model_name_CL} -> {model_name} -> {nameTarget_fit}')
        f_name = f'predictions_cont_CL{model_name_CL}_{model_name}_'\
                                                    f'{nameTarget_fit}.h5'
        filepath_dfs = os.path.join(filepath_df_datas, f_name)

        try:
            d_dfs = functions_pp.load_hdf5(filepath_dfs)
            df_predictions = d_dfs['df_predictions']
        except:
            print('loading predictions failed, skipping this model')
            continue

        # metrics
        bench = prediction[nameTarget].mean()
        RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=bench).RMSE
        MAE_SS = fc_utils.ErrorSkillScore(constant_bench=bench).MAE
        score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS,
                           fc_utils.r2_score]

        df_cond = utils_paper3.cond_forecast_table(rg_list, score_func_list,
                                                   df_predictions,
                                                   nameTarget='Target',
                                                   n_boot=n_boot)

        df_cond_b = utils_paper3.cond_forecast_table(rg_list, score_func_list,
                                                   df_predictions,
                                                   nameTarget='Target',
                                                   n_boot=n_boot)
        d_dfs={'df_cond':df_cond, 'df_cond_b':df_cond_b}
        filepath_dfs = os.path.join(filepath_df_datas,
                        f'scores_cont_CL{model_name_CL}_{model_name}_'\
                            f'{nameTarget_fit}_{nameTarget}_{n_boot}_CF.h5')
        functions_pp.store_hdf_df(d_dfs, filepath_dfs)


    # m = models_lags[f'lag_{lag_}'][f'split_{0}']
    # # plt.plot(kwrgs_model['alpha'], m.cv_results_['mean_test_score'])
    # # plt.axvline(m.best_params_['alpha']) ; plt.show() ; plt.close()

    # rg.verification_tuple_c = verification_tuple_c

    # prediction_o = target_ts.merge(prediction.iloc[:,[1]], left_index=True,right_index=True)
    # verification_tuple_c_o = fc_utils.get_scores(prediction_o,
    #                                          rg.df_data.iloc[:,-2:],
    #                                          score_func_list,
    #                                          n_boot=n_boot,
    #                                          blocksize=1,
    #                                          rng_seed=seed)
    # rg.verification_tuple_c_o  = verification_tuple_c_o

#%% Plotting Continuous forecast timeseries
import utils_paper3
metrics_plot = ['corrcoef', 'MAE', 'r2_score']
model_combs_plot = [['RandomForestRegressor', 'RandomForestRegressor']]
fc_month_list = [rg.fc_month for rg in rg_list]
# fig = plt.figure(figsize=(12, 5))
fig, axes = plt.subplots(nrows=len(fc_month_list), ncols=2, figsize=(12,10),
                         gridspec_kw={'width_ratios':[4,1]},
                         sharex=True, sharey=True)
# gs = gridspec.GridSpec(len(fc_month_list), 2, height_ratios=None,

for model_name_CL, model_name in model_combs_plot:
    # for i, (nameTarget_fit, nameTarget) in enumerate(verif_combs):

    # load df_scores
    nameTarget_fit = 'Target*Signal'
    nameTarget = 'Target*Signal'
    df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5'
    filepath_dfs = os.path.join(filepath_df_datas, df_file)
    d_dfscores = functions_pp.load_hdf5(filepath_dfs)

    nameTarget_fit = 'Target*Signal'
    nameTarget = 'Target'
    df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5'
    filepath_dfs = os.path.join(filepath_df_datas, df_file)
    d_dfscores_T = functions_pp.load_hdf5(filepath_dfs)

    f_name = f'predictions_cont_CL{model_name_CL}_{model_name}_'\
                                                f'{nameTarget_fit}.h5'
    filepath_dfs = os.path.join(filepath_df_datas, f_name)
    d_dfpreds = functions_pp.load_hdf5(filepath_dfs)

    print(model_name_CL, model_name, nameTarget_fit, nameTarget)
    if model_name == 'RandomForestRegressor':
        name = 'RF'
    elif model_name == 'Ridge':
        name = 'Ridge'
    for m, fc_month in enumerate(fc_month_list):
        axs = axes[m]
        ax = axs[0]
        df_test_TS = d_dfscores['df_scores'][[fc_month]] # skill TargetSignal
        df_test_T = d_dfscores_T['df_scores'][[fc_month]] # skill Target
        df_test_m = [df_test_TS, df_test_T]
        df_preds  = d_dfpreds['df_predictions']
        df_test   = df_preds[[df_preds.columns[0], fc_month]]
        df_test = functions_pp.get_df_test(df_test,
                                            df_splits=rg_list[m].df_splits)
        df_test[['Target*Signal']] = functions_pp.get_df_train(
                                             df_preds[['Target*Signal']],
                                             df_splits=rg.df_splits)
        target_ts = functions_pp.get_df_test(df_preds[['Target']],
                                             df_splits=rg_list[m].df_splits)
        utils_paper3.plot_forecast_ts(df_test_m, df_test,
                                      target_ts=target_ts,
                                      fig_ax=(fig, axs),
                                      fs=11,
                                      metrics_plot=metrics_plot,
                                      name_model=name)
        ax.set_ylim(-3,3)
        ax.set_yticks(np.arange(-3,3.1,1))
        ax.margins(x=.02)
        if m == 0:
            ax.legend(fontsize=10, loc='lower left')


fig.savefig(os.path.join(filepath_verif,
                         'timeseries_and_skill.pdf'), bbox_inches='tight')
#%% Continuous forecast: plotting skill scores
import utils_paper3
metrics_plot = ['corrcoef', 'MAE', 'RMSE', 'r2_score']
model_combs_plot  = [['Ridge', 'Ridge'],
                     ['Ridge', 'RandomForestRegressor'],
                     ['RandomForestRegressor', 'RandomForestRegressor']]
# model_combs_plot = [['RandomForestRegressor', 'RandomForestRegressor']]
fc_month_list = [rg.fc_month for rg in rg_list]
# fig = plt.figure(figsize=(12, 5))
fig, axes = plt.subplots(nrows=len(model_combs_plot), ncols=len(metrics_plot),
                         figsize=(17,10),
                          # gridspec_kw={'width_ratios':[4,1]},
                          sharex=True, sharey=False)
# gs = gridspec.GridSpec(len(fc_month_list), 2, height_ratios=None,

for j, (model_name_CL, model_name) in enumerate(model_combs_plot):
    # for i, (nameTarget_fit, nameTarget) in enumerate(verif_combs):


    # load df_scores
    nameTarget_fit = 'Target'
    nameTarget = 'Target'
    df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5'
    filepath_dfs = os.path.join(filepath_df_datas, df_file)
    d_dfscores_N = functions_pp.load_hdf5(filepath_dfs)


    # load df_scores Target*Signal + Target*Signal
    nameTarget_fit = 'Target*Signal'
    nameTarget = 'Target*Signal'
    df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5'
    filepath_dfs = os.path.join(filepath_df_datas, df_file)
    d_dfscores = functions_pp.load_hdf5(filepath_dfs)

    nameTarget_fit = 'Target*Signal'
    nameTarget = 'Target'
    df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5'
    filepath_dfs = os.path.join(filepath_df_datas, df_file)
    d_dfscores_T = functions_pp.load_hdf5(filepath_dfs)


    print(model_name_CL, model_name, nameTarget_fit, nameTarget)
    if model_name == 'RandomForestRegressor':
        name = 'RF'
    elif model_name == 'Ridge':
        name = 'Ridge'
    if model_name_CL == 'RandomForestRegressor':
        name_CL = 'RF'
    elif model_name_CL == 'Ridge':
        name_CL = 'Ridge'

    list_verif = [d_dfscores, d_dfscores_T, d_dfscores_N]
    df_scores_list = [d['df_scores'] for d in list_verif]
    df_boot_list = [d['df_boot'] for d in list_verif]

    fig = utils_paper3.plot_scores_wrapper(df_scores_list, df_boot_list,
                                         labels=['Target*Signal','Target', 'Normal'],
                                         metrics_plot=metrics_plot,
                                         fig_ax = (fig, axes[j]))
    axes[j,0].set_ylabel(f'{name_CL} -> {name}', fontsize=18)
    # axes[j].set_xlabel('Forecast month', fontsize=18)
    # title = f'Target fitted: {nameTarget_fit} with CL {model_name_CL} '\
    #         f'model & final {name} model'
    # fig.suptitle(title, y=.95, fontsize=18)
    fig.subplots_adjust(wspace=.2)
    fig.subplots_adjust(hspace=.3)

fig.savefig(os.path.join(filepath_verif,
                       'scores_vs_lags.pdf'), bbox_inches='tight')
#%% Check RF tuning
import scikit_model_analysis as sk_ana
model = models_lags['lag_0'][f'split_{0}']
f = sk_ana.ensemble_error_estimators(model.best_estimator_, kwrgs_model2)
f.savefig(os.path.join(filepath_verif,
                       'RF_tuning.pdf'), bbox_inches='tight')

#%%
df_scores, df_boot, df_tests = utils_paper3.df_scores_for_plot(rg_list, name_object='verification_tuple_c')
d_dfs={'df_scores':df_scores, 'df_boot':df_boot, 'df_tests':df_tests}
filepath_dfs = os.path.join(rg.path_outsub1,
                            f'scores_s{seed}_cont{model_name}.h5')
functions_pp.store_hdf_df(d_dfs, filepath_dfs)

d_dfs = functions_pp.load_hdf5(filepath_dfs)

f = utils_paper3.plot_scores_wrapper(df_scores, df_boot)
f_name = f'{method}_{seed}_cf_PacAtl'
fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
if save:
    f.savefig(fig_path, bbox_inches='tight')

for rg in rg_list: # plotting score per test
    # plot timeseries
    predict = rg.prediction_tuple[0]
    df_test = functions_pp.get_df_test(predict.rename({0:'causal'}, axis=1),
                                        df_splits=rg.df_splits)
    df_test_m = rg.verification_tuple_c[2]
    utils_paper3.plot_forecast_ts(df_test_m, df_test)
    f_name = f'ts_forecast_{method}_{seed}_continuous_{rg.fc_month}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
    if save:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # df_test_s_m = rg.verification_tuple_c[1]
    # fig, ax = plt.subplots(1)
    # df_test_s_m.plot(ax=ax)
    # fig.savefig(os.path.join(rg.path_outsub1, f'CV_scores_{rg.fc_month}.png'),
    #             bbox_inches='tight', dpi=100)

#%% save table conditional forecast (Continuous)
try:

    df_cond_fc = utils_paper3.cond_forecast_table(rg_list, score_func_list,
                                                  n_boot=n_boot)
    # store as .xlsc
    df_cond_fc.to_excel(os.path.join(rg.path_outsub1, f'cond_fc_{method}_s{seed}.xlsx'))
    # Store as .h5
    d_dfs={'df_cond_fc':df_cond_fc}
    filepath_dfs = os.path.join(rg.path_outsub1, f'cond_fc_{method}_s{seed}.h5')
    functions_pp.store_hdf_df(d_dfs, filepath_dfs)

    composites = [30, 50]
    for comp in composites:
        f = utils_paper3.boxplot_cond_fc(df_cond_fc, metrics=None,
                                          forcing_name='Pacific Forcing',
                                          composite=comp)
        filepath = os.path.join(rg.path_outsub1, f'Conditional_forecast_{comp}_cont')
        f.savefig(filepath + rg.figext, bbox_inches='tight')

except:
    print('SST region 1 is not always found in each split')

#%% Collect different splits continuous forecast

orientation = 'horizontal'
alpha = .05
metrics_cols = ['corrcoef', 'MAE', 'RMSE', 'r2_score']
rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
            'MAE':'MAE-SS', 'CRPSS':'CRPSS', 'r2_score':'$R^2$',
            'mean_absolute_percentage_error':'MAPE'}

if orientation=='vertical':
    f, ax = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                      sharex=True) ;
else:
    f, ax = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
                      sharey=False) ;
path = '/'.join(rg.path_outsub1.split('/')[:-1])

cs = ["#a4110f","#f7911d","#fffc33","#9bcd37","#1790c4"]
for s in range(5):

    hash_str = f'scores_s{s}_continuous.h5'
    f_name = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.findall(f'{hash_str}', file):
                print(f'Found file {file}')
                f_name = file
    if f_name is not None:
        d_dfs = functions_pp.load_hdf5(os.path.join(path,
                                                    f's{s}',
                                                    f_name))

        c1, c2 = '#3388BB', '#EE6666'
        for i, m in enumerate(metrics_cols):
            # normal SST

            steps = df_scores.columns.levels[1].size
            labels = [t[0] for t in df_scores.columns][::steps]
            ax[i].plot(labels, d_dfs['df_scores'].reorder_levels((1,0), axis=1).loc[0][m].T,
                    label=f'seed: {s}',
                    color=cs[s],
                    linestyle='solid')
            ax[i].fill_between(labels,
                                d_dfs['df_boot'].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                                d_dfs['df_boot'].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                                edgecolor=cs[s], facecolor=cs[s], alpha=0.3,
                                linestyle='solid', linewidth=2)

            if m == 'corrcoef':
                ax[i].set_ylim(-.2,1)
            else:
                ax[i].set_ylim(-.2,.6)
            ax[i].axhline(y=0, color='black', linewidth=1)
            ax[i].tick_params(labelsize=16, pad=6)
            if i == len(metrics_cols)-1 and orientation=='vertical':
                ax[i].set_xlabel('Forecast month', fontsize=18)
            elif orientation=='horizontal':
                ax[i].set_xlabel('Forecast month', fontsize=18)
            if i == 0:
                ax[i].legend(loc='lower right', fontsize=14)
            ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=-4)


f.subplots_adjust(hspace=.1)
f.subplots_adjust(wspace=.22)
title = 'Verification Soy Yield forecast'
if orientation == 'vertical':
    f.suptitle(title, y=.92, fontsize=18)
else:
    f.suptitle(title, y=.95, fontsize=18)
f_name = f'{method}_{seed}_PacAtl_seeds'
fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
if save:
    plt.savefig(fig_path, bbox_inches='tight')

#%% Low/High yield forecast
from sklearn.linear_model import LogisticRegression

fcmodel = ScikitModel(LogisticRegression, verbosity=0)
kwrgs_model = {'scoringCV':'neg_brier_score',
                'C':list([.1,.5,.8,1,1.2,4,7,10, 20]), # large a, strong regul.
                'random_state':seed,
                'penalty':'l2',
                'solver':'lbfgs',
                'kfold':10,
                'max_iter':200}

thresholds = [0.33, .66]
# thresholds = [.5]
for i, q in enumerate(thresholds):
    for i, rg in enumerate(rg_list):
        # target timeseries, standardize using training data
        target_ts = rg.transform_df_data(rg.df_data.iloc[:,[0]].merge(rg.df_splits,
                                                          left_index=True,
                                                          right_index=True),
                                          transformer=fc_utils.standardize_on_train)
        if q >= 0.5:
            target_ts = (target_ts > target_ts.quantile(q)).astype(int)
        elif q < .5:
            target_ts = (target_ts < target_ts.quantile(q)).astype(int)

        mean_vars=['sst', 'smi']
        for i, p in enumerate(rg.list_for_MI):
            if p.calc_ts == 'pattern cov':
                mean_vars[i] +='_sp'

        df_data, keys_dict = utils_paper3.get_df_mean_SST(rg,
                                              mean_vars=mean_vars,
                                              alpha_CI=alpha_CI,
                                              n_strongest='all',
                                              weights=True,
                                              fcmodel=fcmodel,
                                              kwrgs_model=kwrgs_model,
                                              target_ts=target_ts)
        if q == 0.33:
            df_CL_data_l = df_data
        elif q == 0.66:
            df_CL_data_h = df_data

#%% fit probabilistic forecasts
for i, q in enumerate(thresholds):
    if q == 0.33:
        # find where there is Pacific signal for low yield
        utils_paper3.get_df_forcing_cond_fc(rg_list, target_ts, fcmodel,
                                            kwrgs_model, mean_vars=mean_vars,
                                            region='only_Pacific',
                                            name_object='df_pred_tuple_l')
    elif q == 0.66:
        # find where there is Pacific signal for high yield
        utils_paper3.get_df_forcing_cond_fc(rg_list, target_ts, fcmodel,
                                            kwrgs_model, mean_vars=mean_vars,
                                            region='only_Pacific',
                                            name_object='df_pred_tuple_h')

    for i, rg in enumerate(rg_list):
        # target timeseries, standardize using training data
        target_ts = rg.transform_df_data(rg.df_data.iloc[:,[0]].merge(rg.df_splits,
                                                          left_index=True,
                                                          right_index=True),
                                          transformer=fc_utils.standardize_on_train)
        # target
        target_ts_signal = target_ts.multiply(rg.df_forcing.mean(axis=1).abs(), axis=0)
        target_ts_signal= target_ts_signal.rename({rg.df_data.columns[0]:
                                                    f'Target * {rg.fc_month} signal'},
                                                  axis=1)
        target_ts_signal.loc[0].plot(ax=ax)
        target_ts_signal = rg.transform_df_data(target_ts_signal.merge(rg.df_splits,
                                                                        left_index=True,
                                                                        right_index=True),
                                                transformer=fc_utils.standardize_on_train)

        if q >= 0.5:
            target_ts_signal = (target_ts_signal > float(target_ts.quantile(q))).astype(int)
        elif q < .5:
            target_ts_signal = (target_ts_signal < float(target_ts.quantile(q))).astype(int)

        # metrics
        BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts_signal.mean())).BSS
        score_func_list = [BSS, fc_utils.metrics.roc_auc_score]
        metric_names = [s.__name__ for s in score_func_list]

        lag_ = 0 ;
        prediction_tuple = rg.fit_df_data_ridge(df_data=df_data,
                                                target=target_ts_signal,
                                                tau_min=0, tau_max=0,
                                                kwrgs_model=kwrgs_model,
                                                fcmodel=fcmodel,
                                                transformer=None)

        predict, weights, models_lags = prediction_tuple
        prediction = predict.rename({predict.columns[0]:'Target',
                                      lag_:rg.fc_month}, axis=1)
        prediction_tuple = (prediction, weights, models_lags)




        verification_tuple = fc_utils.get_scores(prediction,
                                                  rg.df_data.iloc[:,-2:],
                                                  score_func_list,
                                                  score_per_test=False,
                                                  n_boot=n_boot,
                                                  blocksize=1,
                                                  rng_seed=seed)
        df_train_m, df_test_s_m, df_test_m, df_boot = verification_tuple


        m = models_lags[f'lag_{lag_}'][f'split_{0}']
        [models_lags[f'lag_{lag_}'][f'split_{s}'].best_params_ for s in range(rg.n_spl)]
        # plt.plot(kwrgs_model['C'], m.cv_results_['mean_test_score'])
        # plt.axvline(m.best_params_['C']) ; plt.show() ; plt.close()

        df_test = functions_pp.get_df_test(predict.rename({lag_:'causal'}, axis=1),
                                            df_splits=rg.df_splits)
        if q == 0.33:
            rg.verification_tuple_l = verification_tuple
            rg.prediction_tuple_l = prediction_tuple
        elif q == 0.66:
            rg.verification_tuple_h = verification_tuple
            rg.prediction_tuple_h = prediction_tuple



#%% Plotting event forecasts


for i, q in enumerate(thresholds):
    if q == 0.33:
        df_scores, df_boot, df_tests = utils_paper3.df_scores_for_plot(rg_list,
                                        name_object='verification_tuple_l')
        df_preds_save, df_w_save = utils_paper3.df_predictions_for_plot(rg_list,
                                                              'prediction_tuple_l')

    elif q == 0.66:
        df_scores, df_boot, df_tests = utils_paper3.df_scores_for_plot(rg_list,
                                        name_object='verification_tuple_h')
        df_preds_save, df_w_save = utils_paper3.df_predictions_for_plot(rg_list,
                                                            'prediction_tuple_h')


    d_dfs={'df_predictions':df_preds_save}
    filepath_dfs = os.path.join(rg.path_outsub1, f'predictions_s{seed}_continuous.h5')
    functions_pp.store_hdf_df(d_dfs, filepath_dfs)
    df_preds_save = functions_pp.load_hdf5(filepath_dfs)['df_predictions']

    d_dfs={'df_scores':df_scores, 'df_boot':df_boot, 'df_tests':df_tests}
    filepath_dfs = os.path.join(rg.path_outsub1, f'scores_s{seed}_q{q}.h5')
    functions_pp.store_hdf_df(d_dfs, filepath_dfs)
    d_dfs = functions_pp.load_hdf5(filepath_dfs)
    df_scores, df_boot = d_dfs['df_scores'], d_dfs['df_boot']

    f = utils_paper3.plot_scores_wrapper(df_scores, df_boot)
    f_name = f'{method}_{seed}_cf_PacAtl_q{q}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
    if save:
        f.savefig(fig_path, bbox_inches='tight')

#%%
for i, q in enumerate(thresholds):
    filepath_dfs = os.path.join(rg.path_outsub1, f'predictions_s{seed}_continuous.h5')
    df_preds_save = functions_pp.load_hdf5(filepath_dfs)['df_predictions']

    # plot timeseries
    utils_paper3.plot_forecast_ts(df_test_m, df_test)
    f_name = f'ts_forecast_{method}_{seed}_{q}_{rg.fc_month}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
    if save:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # df_test_s_m = rg.verification_tuple[1]
    # fig, ax = plt.subplots(1)
    # df_test_s_m.plot(ax=ax)
    # fig.savefig(os.path.join(rg.path_outsub1, f'CV_scores_{q}_{rg.fc_month}.png'),
    #             bbox_inches='tight', dpi=100)

    #%% Old conditional forecast
    # df_cond_fc = utils_paper3.cond_forecast_table(rg_list, score_func_list,
    #                                               n_boot=n_boot)
    # composites = [50, 30]
    # for comp in composites:
    #     f = utils_paper3.boxplot_cond_fc(df_cond_fc, metrics=None,
    #                                      forcing_name='Pacific Forcing',
    #                                      composite=comp)
    #     filepath = os.path.join(rg.path_outsub1, f'Conditional_forecast_{comp}_{q}')
    #     f.savefig(filepath + rg.figext, bbox_inches='tight')
    # # save table conditional forecast (Continuous)

    # # store as .xlsc
    # df_cond_fc.to_excel(os.path.join(rg.path_outsub1, f'cond_fc_{method}_s{seed}_{q}.xlsx'))
    # # Store as .h5
    # d_dfs={'df_cond_fc':df_cond_fc}
    # filepath_dfs = os.path.join(rg.path_outsub1, f'cond_fc_{method}_s{seed}_{q}.h5')
    # functions_pp.store_hdf_df(d_dfs, filepath_dfs)

#%% Collect different splits high/low forecast for plotting

orientation = 'horizontal'
alpha = .05
if 'BSS' in df_scores.columns.levels[1]:
    metrics_cols = ['BSS', 'roc_auc_score']
    rename_m = {'BSS': 'BSS', 'roc_auc_score':'ROC-AUC'}
else:
    metrics_cols = ['corrcoef', 'MAE', 'RMSE', 'r2_score']
    rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                'MAE':'MAE-SS', 'CRPSS':'CRPSS', 'r2_score':'$r^2$',
                'mean_absolute_percentage_error':'MAPE'}


for q in [.33, .5, .66]:
    if orientation=='vertical':
        f, ax = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                          sharex=True) ;
    else:
        f, ax = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
                          sharey=False) ;
    path = '/'.join(rg.path_outsub1.split('/')[:-1])

    cs = ["#a4110f","#f7911d","#fffc33","#9bcd37","#1790c4"]
    for s in range(5):

        hash_str = f'scores_s{s}_q{q}.h5'
        f_name = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if re.findall(f'{hash_str}', file):
                    print(f'Found file {file}')
                    f_name = file
        if f_name is not None:
            d_dfs = functions_pp.load_hdf5(os.path.join(path,
                                                        f's{s}',
                                                        f_name))

            c1, c2 = '#3388BB', '#EE6666'
            for i, m in enumerate(metrics_cols):
                # normal SST

                steps = df_scores.columns.levels[1].size
                labels = [t[0] for t in df_scores.columns][::steps]
                ax[i].plot(labels, d_dfs['df_scores'].reorder_levels((1,0), axis=1).loc[0][m].T,
                            label=f'seed: {s}', color=cs[s], linestyle='solid')
                ax[i].fill_between(labels,
                                    d_dfs['df_boot'].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                                    d_dfs['df_boot'].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                                    edgecolor=cs[s], facecolor=cs[s], alpha=0.3,
                                    linestyle='solid', linewidth=2)

                if m == 'corrcoef':
                    ax[i].set_ylim(-.2,1)
                elif m == 'roc_auc_score':
                    ax[i].set_ylim(0,1)
                else:
                    ax[i].set_ylim(-.2,.6)
                ax[i].axhline(y=0, color='black', linewidth=1)
                ax[i].tick_params(labelsize=16, pad=6)
                if i == len(metrics_cols)-1 and orientation=='vertical':
                    ax[i].set_xlabel('Forecast month', fontsize=18)
                elif orientation=='horizontal':
                    ax[i].set_xlabel('Forecast month', fontsize=18)
                if i == 0:
                    ax[i].legend(loc='lower right', fontsize=14)
                ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=4)


    f.subplots_adjust(hspace=.1)
    f.subplots_adjust(wspace=.22)
    title = f'Verification Soy Yield forecast (q={q})'
    if orientation == 'vertical':
        f.suptitle(title, y=.92, fontsize=18)
    else:
        f.suptitle(title, y=.95, fontsize=18)
    f_name = f'{method}_{seed}_PacAtl_seeds_q{q}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
    if save:
        plt.savefig(fig_path, bbox_inches='tight')


#%%
# =============================================================================
# Plot Causal Links
# =============================================================================
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':0,
                      'wspace':-.15, 'size':3, 'cbar_vert':0.05,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                      'y_ticks':np.arange(-10,61,20), #'x_ticks':np.arange(130, 280, 25),
                      'title':'',
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

kwrgs_plotcorr_SM = kwrgs_plotcorr_sst.copy()
kwrgs_plotcorr_SM.update({'aspect':2, 'hspace':0.2,
                          'wspace':0, 'size':3, 'cbar_vert':0.03})



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



#%%
for rg in rg_list:
    plot_regions(rg, save=save, plot_parcorr=False)
    # kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4, 'hspace':0,
    #               'wspace':-.15, 'size':3, 'cbar_vert':0.05,
    #               'map_proj':ccrs.PlateCarree(central_longitude=220),
    #                'y_ticks':np.arange(-10,61,20), #'x_ticks':np.arange(130, 280, 25),
    #               'title':'',
    #               'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
    # rg.plot_maps_corr('sst', kwrgs_plot=kwrgs_plotcorr_sst, save=save)

#%%
    kwrgs_plotcorr_SM = {'row_dim':'lag', 'col_dim':'split','aspect':2, 'hspace':0.2,
                      'wspace':0, 'size':3, 'cbar_vert':0.04,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                        'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                      'title':'',
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
    rg.plot_maps_corr('smi', kwrgs_plot=kwrgs_plotcorr_SM, save=save)


#%% plot
for rg in rg_list:
    models_lags = rg.prediction_tuple[-1]
    df_wgths, fig = plot_importances(models_lags)
    fig.savefig(os.path.join(rg.path_outsub1, f'weights_{rg.fc_month}.png'),
                bbox_inches='tight', dpi=100)

#%% R2 skill metric proportional with length of dataset
det, trend = core_pp.detrend_wrapper(rg.df_fulltso[['raw_target']], return_trend=True)
trend = trend.rename({'raw_target':'trend'}, axis=1)
target = rg.df_fulltso[['raw_target']].rename({'raw_target':'Soy Yield'})
trend_pred = target.merge(trend, left_index=True, right_index=True)
skill = fc_utils.get_scores(trend_pred,
                            score_func_list=[fc_utils.r2_score])[2]
skill10yrs = fc_utils.get_scores(trend_pred.loc[core_pp.get_subdates(trend_pred.index, None,
                                                                range(2000,2010))],
                            score_func_list=[fc_utils.r2_score])[2]