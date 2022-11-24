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
from sklearn.linear_model import LinearRegression

user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/Vijverberg_et_al_2022_AIES/'))
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
from RGCPD import BivariateMI, class_BivariateMI, functions_pp, find_precursors
from RGCPD import plot_maps, core_pp, wrapper_PCMCI
from RGCPD.forecasting.stat_models_cont import ScikitModel
from RGCPD import climate_indices
from RGCPD import class_EOF
import utils_paper3

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']


target_datasets = ['USDA_Soy_clusters__1']
seeds = [1] # ,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
methods = ['leave_1', 'timeseriessplit_25']#, 'leave_1', timeseriessplit_25', 'timeseriessplit_20', 'timeseriessplit_30']
training_datas = ['all_CD', 'onelag', 'all', 'climind']
combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    yrs,
                                    methods,
                                    training_datas)).T.reshape(-1,5)
i_default = 1
load = 'all'
save = True
fc_types = [0.33, 'continuous']
fc_types = [0.33]
plt.rcParams['savefig.dpi'] = 300

model_combs_cont = [['Ridge', 'Ridge'],
                    ['Ridge', 'RandomForestRegressor'],
                    ['RandomForestRegressor', 'RandomForestRegressor']]
model_combs_bina = [['LogisticRegression', 'LogisticRegression']]
                    # ['LogisticRegression', 'RandomForestClassifier'],
                    # ['RandomForestClassifier', 'RandomForestClassifier']]

# model_combs_bina = [['LogisticRegression', 'LogisticRegression'],
#                     ['RandomForestClassifier', 'RandomForestClassifier']]

# path out main
path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'fc_areaw')
# path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'fc_extra2lags')


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
    root_data = user_dir+'/surfdrive/Scripts/RGCPD/publications/Vijverberg_et_al_2022_AIES/data/'
else:
    root_data = user_dir+'/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/'
raw_filename = os.path.join(root_data, 'masked_rf_gs_county_grids.nc')

if target_dataset.split('__')[0] == 'USDA_Soy_clusters':
    TVpath = os.path.join(main_dir, 'publications/Vijverberg_et_al_2022_AIES/clustering/linkage_ward_nc2_dendo_lindetrendgc_a9943.nc')
    cluster_label = int(target_dataset.split('__')[1]) ; name_ds = 'ts'
elif target_dataset == 'Aggregate_States':
    path =  os.path.join(main_dir, 'publications/Vijverberg_et_al_2022_AIES/data/masked_rf_gs_state_USDA.csv')
    States = ['KENTUCKY', 'TENNESSEE', 'MISSOURI', 'ILLINOIS', 'INDIANA']
    TVpath = read_csv_State(path, State=States, col='obs_yield').mean(1)
    TVpath = pd.DataFrame(TVpath.values, index=TVpath.index, columns=['KENTUCKYTENNESSEEMISSOURIILLINOISINDIANA'])
    name_ds='Soy_Yield' ; cluster_label = ''



alpha_corr = .05
alpha_CI = .05
n_boot = 2000
append_pathsub = f'/{method}/s{seed}'
extra_lag = True
append_main = target_dataset


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
    ds_out = functions_pp.area_weighted(ds_out) # no area weight
    df = ds_out.mean(dim=('latitude', 'longitude')).to_dataframe('1ts')
    return df


#%%
# =============================================================================
# Response-Guided Dimensionality Reduction + Causal selection step
# =============================================================================

def pipeline(lags, periodnames, use_vars=['sst', 'smi'], load=False):
    #%%
    _yrs = [int(l.split('-')[0]) for l in core_pp.flatten(lags)]
    if np.unique(_yrs).size>1: # crossing year
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
                                calc_ts='region mean', selbox=GlobalBox,
                                lags=lags, group_split=True,
                                use_coef_wghts=True),
                      BivariateMI(name='smi', func=class_BivariateMI.corr_map,
                                 alpha=alpha_corr, FDR_control=True,
                                 kwrgs_func={},
                                 distance_eps=250, min_area_in_degrees2=3,
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


    subfoldername = target_dataset + append_pathsub


    # detrend and anomaly (already done for smi)
    rg.pp_precursors(detrend=[True, {'smi':False}],
                     anomaly=[True, {'smi':False}])
    if crossyr:
        TV_start_end_year = (start_end_year[0]+1, 2019)
    else:
        TV_start_end_year = (start_end_year[0], 2019)
    kwrgs_core_pp_time = {'start_end_year': TV_start_end_year}


    # detrending done prior in clustering_soybean
    rg.pp_TV(name_ds=name_ds, detrend=False, ext_annual_to_mon=False,
             kwrgs_core_pp_time=kwrgs_core_pp_time)

    if method.split('_')[0]=='leave':
        subfoldername += 'gp_prior_1_after_1'
        rg.traintest(method, gap_prior=1, gap_after=1, seed=seed,
                     subfoldername=subfoldername)
    else:
        rg.traintest(method, seed=seed, subfoldername=subfoldername)


    if 'timeseries' in method:
        df_splits = rg.df_splits
    else:
        df_splits = None

    dsclust = rg.get_clust()
    path = os.path.join(rg.path_outsub1, 'detrend')
    os.makedirs(path, exist_ok=True)
    if os.path.exists(os.path.join(path, 'target_ts.h5')) and load=='all':
        dfnew = functions_pp.load_hdf5(os.path.join(path,
                                                    'target_ts.h5'))['df_data']
    else:
        dfnew = ds_oos_lindetrend(dsclust, df_splits, path)
        if 'timeseries' not in method:
            dfnew = dfnew.loc[rg.df_splits.index.levels[1]]


        if 'timeseries' in method:
            dfnew = dfnew.loc[df_splits.index]
            # plot difference
            df_test = functions_pp.get_df_test(dfnew, df_splits=rg.df_splits)
            f, ax = plt.subplots(1)
            ax.plot(rg.df_fullts.loc[df_test.index], label='Pre-process all data')
            ax.plot(df_test, label='Pre-process one-step-ahead')
            ax.legend()
            f.savefig(os.path.join(path, 'compared_detrend.jpg'), dpi=250,
                      bbox_inches='tight')
        if load != False: # crashed when done in parallel mode
            functions_pp.store_hdf_df({'df_data':dfnew},
                                      os.path.join(path, 'target_ts.h5'))


    rg.df_fullts = dfnew


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
        sst.group_small_cluster(distance_eps_sc=2000, eps_corr=0.4)

    if 'sst' in use_vars:
        if loaded==False:
            if os.path.exist(os.path.join(rg.path_outsub1, load_sst+'.nc')):
                os.remove(os.path.join(rg.path_outsub1, load_sst+'.nc'))
            sst.store_netcdf(rg.path_outsub1, load_sst, add_hash=False)
        sst.prec_labels['lag'] = ('lag', periodnames)
        sst.corr_xr['lag'] = ('lag', periodnames)
        if loaded == False:
            rg.quick_view_labels('sst', min_detect_gc=.5, save=save,
                                 append_str=periodnames[-1])
        plt.close()



    # # #%% yield vs circulation plots
    # z500 = BivariateMI(name='z500',
    #                     filepath=rg.list_precur_pp[1][1],
    #                     func=class_BivariateMI.corr_map,
    #                     alpha=alpha_corr, FDR_control=True,
    #                     kwrgs_func={},
    #                     distance_eps=250, min_area_in_degrees2=3,
    #                     calc_ts='pattern cov', selbox=(155,355,10,80),
    #                     lags=lags, group_split=True,
    #                     use_coef_wghts=True)

    # z500.load_and_aggregate_precur(rg.kwrgs_load)
    # xrcorr, xrpvals = z500.bivariateMI_map(z500.precur_arr, df_splits,
    #                                       rg.df_fullts)
    # xrcorr, mask = rg._get_sign_splits_masked(xr_in=xrcorr, min_detect=.1, mask=xrcorr['mask'])
    # plot_maps.plot_corr_maps(xrcorr, mask, wspace=0.02, subtitles=[periodnames], y_ticks=False, x_ticks=False, **kwrgs_plot)

    #%%
    if hasattr(SM, 'prec_labels')==False and 'smi' in use_vars:
        SM = rg.list_for_MI[1]
        rg.cluster_list_MI('smi')
        SM.group_small_cluster(distance_eps_sc=1E4, eps_corr=1)
        # lonlatbox = [220, 240, 25, 55] # eastern US
        # SM.prec_labels = merge(SM, lonlatbox)
        # lonlatbox = [270, 280, 25, 45] # mid-US
        # SM.prec_labels = merge(SM, lonlatbox)
    if 'smi' in use_vars:
        if loaded==False:
            if os.path.exist(os.path.join(rg.path_outsub1, load_SM+'.nc')):
                os.remove(os.path.join(rg.path_outsub1, load_SM+'.nc'))
            SM.store_netcdf(rg.path_outsub1, load_SM, add_hash=False)
        SM.corr_xr['lag'] = ('lag', periodnames)
        SM.prec_labels['lag'] = ('lag', periodnames)
        if loaded == False:
            rg.quick_view_labels('smi', min_detect_gc=.5, save=save,
                                 append_str=periodnames[-1])
        plt.close()

    # store forecast month
    months = {'JJ':'August', 'MJ':'July', 'AM':'June', 'MA':'May',
              'FM':'April', 'JF':'March', 'SO':'December', 'DJ':'February'}
    last_month = list(sst.corr_xr.lag.values)[-1]
    rg.fc_month = months[last_month]

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


    #%%
    return rg

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

    lags_feb = np.array([[f'{sy}-07-01', f'{sy}-08-01'],# SO
                         [f'{sy}-09-01', f'{sy}-10-01'],# ND
                         [f'{sy}-11-01', f'{sy}-12-01'],# JF
                         [f'{sy_p1}-01-01', f'{sy_p1}-02-01'] # MA
                         ])
    periodnames_feb = ['JA', 'SO', 'ND', 'JF']

    lags_jan = np.array([[f'{sy}-06-01', f'{sy}-07-01'],# SO
                         [f'{sy}-08-01', f'{sy}-09-01'],# ND
                         [f'{sy}-10-01', f'{sy}-11-01'],# JF
                         [f'{sy}-12-01', f'{sy_p1}-01-01'] # MA
                         ])
    periodnames_jan = ['JJ', 'AS', 'ON', 'DJ']


    use_vars_july = ['sst', 'smi']
    use_vars_june = ['sst', 'smi']
    use_vars_may = ['sst', 'smi']
    use_vars_april = ['sst', 'smi']
    use_vars_march = ['sst', 'smi']
    use_vars_feb = ['sst', 'smi']
    use_vars_jan = ['sst', 'smi']


    # Run in Parallel
    lag_list = [lags_july, lags_june, lags_may, lags_april, lags_march]
    periodnames_list = [periodnames_july, periodnames_june,
                        periodnames_may, periodnames_april,
                        periodnames_march]
    use_vars_list = [use_vars_july, use_vars_june,
                     use_vars_may, use_vars_april, use_vars_march]


    if extra_lag:
        lag_list += [lags_feb, lags_jan]
        periodnames_list += [periodnames_feb, periodnames_jan]
        use_vars_list += [use_vars_feb, use_vars_jan]
        lag_list  = lag_list
        periodnames_list = periodnames_list
        use_vars_list = use_vars_list

    futures = [] ; rg_list = []
    for lags, periodnames, use_vars in zip(lag_list, periodnames_list, use_vars_list):
        if load == False:
            futures.append(delayed(pipeline)(lags, periodnames, use_vars, load))

        else:
            rg_list.append(pipeline(lags, periodnames, use_vars, load))

    if load == False:
        rg_list = Parallel(n_jobs=n_cpu, backend='loky')(futures)

rg = rg_list[0]
#%%
fc_month = 'August'
rg = [rg for rg in rg_list if rg.fc_month == fc_month][0]
sst, SM = rg.list_for_MI

schematicfolder = os.path.join(rg.path_outsub1, 'schematic')
os.makedirs(schematicfolder, exist_ok=True)
split = 15

#%% plot SST corr map
selbox=(180, 250, 20, 60)
corr_map = sst.corr_xr.sel(split=split, lag=['JJ', 'DJ'])
corr_map = core_pp.get_selbox(corr_map, selbox=selbox)
label_map = sst.prec_labels.sel(split=split, lag=['JJ', 'DJ'])
label_map = core_pp.get_selbox(label_map , selbox=selbox)
label_map  = find_precursors.view_or_replace_labels(label_map, regions=[1, 6, 5],
                                                    replacement_labels=[1, 2, 3])

subtitles = np.array([['June-July mean'], ['Dec-Jan mean']])
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':1.5,
                      'hspace':.65, 'wspace':0., 'size': 3, 'cbar_vert':0.035,
                      'map_proj':plot_maps.ccrs.PlateCarree(central_longitude=220),
                      'y_ticks':False, 'x_ticks':False,
                      'subtitle_fontdict':{'fontsize':20},
                      'kwrgs_mask':{'linewidths':1},
                      'units':'Corr. Coef.',
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'},
                      'clevels':np.arange(-0.8, 0.9, .2),
                      'clabels':np.arange(-.8,.9,.4),
                      'cbar_tick_dict':{'labelsize':18},
                      'subtitles':subtitles,
                      'title':'SST correlation vs. Soy yield'}

fg = plot_maps.plot_corr_maps(corr_map.where(~np.isnan(label_map)),
                         mask_xr=np.isnan(label_map), **kwrgs_plotcorr_sst)
fg.fig.savefig(schematicfolder +'/sst_corr.pdf', bbox_inches='tight')
fg.fig.savefig(schematicfolder +'/sst_corr.jpeg', dpi=150, bbox_inches='tight')
#%% plot SST label map
label_map = sst.prec_labels.sel(split=split, lag=['JJ', 'DJ'])
label_map = core_pp.get_selbox(label_map , selbox=selbox)
label_map  = find_precursors.view_or_replace_labels(label_map, regions=[1, 6, 5],
                                                    replacement_labels=[1, 2, 3])

cmp = ["ef476f","00a6ed","06d6a0"]
cmp = plot_maps.get_continuous_cmap(cmp,
                float_list=list(np.linspace(0,1,3)))
kwrgs_plotlabels = kwrgs_plotcorr_sst.copy()
kwrgs_plotlabels.pop('clevels') ; kwrgs_plotlabels.pop('clabels')
kwrgs_plotlabels.update({'units' : 'DBSCAN labels', 'cmap' : cmp,
                         'title':'Precursor region masks'})

d = plot_maps._get_kwrgs_labels(label_map, kwrgs_plot=kwrgs_plotlabels, labelsintext=True)
# d['textinmap'] = [d['textinmap'][0], d['textinmap'][3]]
for l in d['textinmap']:
    [sl[3].update({'fontsize':20, 'color':'black', 'bbox':{'facecolor':'grey', 'alpha':.1}}) for sl in l[1:][0]]


fg = plot_maps.plot_corr_maps(label_map, **d)
fg.fig.savefig(schematicfolder +'/sst_labels.pdf', bbox_inches='tight')
fg.fig.savefig(schematicfolder +'/sst_labels.jpeg', dpi=150, bbox_inches='tight')
#%%
df_t = rg.transform_df_data().loc[split]
#%% Plot SST timeseries
df = df_t[['JJ..1..sst', 'DJ..1..sst', 'JJ..6..sst', 'DJ..6..sst']]##, 'JJ..5..sst', 'DJ..5..sst']]
f, axes = plt.subplots(2, figsize=(10,7))
plt.subplots_adjust(hspace=.65)
cmp = ["#ef476f","#06d6a0"]
for i, lag_name in enumerate(['JJ', 'DJ']):
    ax = axes[i]
    axtitle = 'June-July mean timeseries' if lag_name == 'JJ' else 'Dec-Jan mean timeseries'
    ax.set_title(axtitle, fontsize=20)
    for j, r in enumerate([1, 6]):
        col = f'{lag_name}..{r}..sst'
        ax.plot(df.index, df[col], color=cmp[j], label=False)
f.suptitle('Region mean timeseries', fontsize=18, fontweight='bold')

[ax.plot(df_t.index, df_t.iloc[:,0], color='grey', ls='-.', label='Soy yield') for ax in axes]
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:], labels[-1:], loc = 'lower left')
[ax.axhline(y=0, color='grey', linewidth=.5) for ax in axes]
[ax.tick_params(labelsize=18) for ax in axes]
[ax.set_ylim(-3,3) for ax in axes]
x_length = [-7200,14400]
ax.annotate('train', xy=(np.mean(x_length), -.3), xycoords=ax.get_xaxis_transform(), fontsize=18,
            ha="center", va="top", bbox={'edgecolor':'grey', 'facecolor':'lightgrey'})
ax.plot(x_length,[-.22,-.22], color="k", transform=ax.get_xaxis_transform(), clip_on=False, lw=4)
x_length = [14800,18000]
ax.annotate('test', xy=(np.mean(x_length), -.3), xycoords=ax.get_xaxis_transform(), fontsize=18,
            ha="center", va="top", bbox={'edgecolor':'grey', 'facecolor':'lightgreen'})
ax.plot(x_length,[-.22,-.22], color='lightgreen', transform=ax.get_xaxis_transform(), clip_on=False, lw=4)
f.savefig(schematicfolder + '/sst_timeseries.pdf', bbox_inches='tight')
#%% plot SM corr map
corr_map = SM.corr_xr.sel(split=split, lag=['JJ', 'DJ'])
kwrgs_plotcorr_sst.update({'title': 'SM correlation vs. Soy yield'})
label_map = SM.prec_labels.sel(split=split, lag=['JJ', 'DJ'])
label_map  = find_precursors.view_or_replace_labels(label_map, regions=[1, 6],
                                                    replacement_labels=[1, 1])

fg = plot_maps.plot_corr_maps(corr_map.where(~np.isnan(label_map)),
                         mask_xr=np.isnan(label_map), **kwrgs_plotcorr_sst)
fg.fig.savefig(schematicfolder +'/SM_corr.pdf', bbox_inches='tight')
fg.fig.savefig(schematicfolder +'/SM_corr.jpeg', dpi=150, bbox_inches='tight')
#%% plot SM label map
label_map = SM.prec_labels.sel(split=split, lag=['JJ', 'DJ'])
label_map  = find_precursors.view_or_replace_labels(label_map, regions=[1, 6],
                                                    replacement_labels=[1, 1])
kwrgs_plotlabels = kwrgs_plotcorr_sst.copy()
kwrgs_plotlabels.pop('clevels') ; kwrgs_plotlabels.pop('clabels')
kwrgs_plotlabels.update({'units' : 'DBSCAN mask', 'cmap' : "#064789",
                         'title':'Precursor pattern mask'})


fg = plot_maps.plot_labels(label_map, kwrgs_plot=kwrgs_plotlabels)
fg.fig.savefig(schematicfolder +'/SM_labels.pdf', bbox_inches='tight')
fg.fig.savefig(schematicfolder +'/SM_labels.jpeg', dpi=150, bbox_inches='tight')
#%%
df = df_t[['JJ..0..smi_sp', 'DJ..0..smi_sp']]
f, axes = plt.subplots(2, figsize=(10,7))
plt.subplots_adjust(hspace=.65)

for i, lag_name in enumerate(['JJ', 'DJ']):
    ax = axes[i]
    axtitle = 'June-July mean timeseries' if lag_name == 'JJ' else 'Dec-Jan mean timeseries'
    ax.set_title(axtitle, fontsize=20)
    for j, r in enumerate([0]):
        col = f'{lag_name}..{r}..smi_sp'
        ax.plot(df.index, df[col], color="#064789")
f.suptitle('Spatial covariance of correlation pattern timeseries', fontsize=18, fontweight='bold')
[ax.plot(df_t.index, df_t.iloc[:,0], color='grey', ls='-.', label='Soy yield') for ax in axes]
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:], labels[-1:], loc = 'lower left')
[ax.axhline(y=0, color='grey', linewidth=.5) for ax in axes]
[ax.tick_params(labelsize=18) for ax in axes]
[ax.set_ylim(-3,3) for ax in axes]
x_length = [-7200,14400]
ax.annotate('train', xy=(np.mean(x_length), -.3), xycoords=ax.get_xaxis_transform(), fontsize=18,
            ha="center", va="top", bbox={'edgecolor':'grey', 'facecolor':'lightgrey'})
ax.plot(x_length,[-.22,-.22], color="k", transform=ax.get_xaxis_transform(), clip_on=False, lw=4)
x_length = [14800,18000]
ax.annotate('test', xy=(np.mean(x_length), -.3), xycoords=ax.get_xaxis_transform(), fontsize=18,
            ha="center", va="top", bbox={'edgecolor':'grey', 'facecolor':'lightgreen'})
ax.plot(x_length,[-.22,-.22], color='lightgreen', transform=ax.get_xaxis_transform(), clip_on=False, lw=4)
f.savefig(schematicfolder + '/SM_timeseries.pdf', bbox_inches='tight')