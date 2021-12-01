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
from sklearn.linear_model import LinearRegression

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']


target_datasets = ['USDA_Soy_clusters__1']
seeds = [1] # ,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
methods = ['ranstrat_20', 'timeseriessplit_20', 'timeseriessplit_30', 'timeseriessplit_25', 'leave_1'] # ['ranstrat_20'] timeseriessplit_30

combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    yrs,
                                    methods)).T.reshape(-1,4)
i_default = 0
load = 'all'
save = True
training_data = 'onelag' # or 'all_CD' or 'onelag'
fc_types = [0.33, 'continuous']
fc_types = [0.33]

model_combs_cont = [['Ridge', 'Ridge'],
                    ['Ridge', 'RandomForestRegressor'],
                    ['RandomForestRegressor', 'RandomForestRegressor']]
model_combs_bina = [['LogisticRegression', 'LogisticRegression']]
                    # ['LogisticRegression', 'RandomForestClassifier'],
                    # ['RandomForestClassifier', 'RandomForestClassifier']]

# model_combs_bina = [['LogisticRegression', 'LogisticRegression'],
#                     ['RandomForestClassifier', 'RandomForestClassifier']]


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
    df = ds_out.mean(dim=('latitude', 'longitude')).to_dataframe('1ts')
    return df

#%% run RGPD


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
                                calc_ts=calc_ts, selbox=GlobalBox,
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


    subfoldername = target_dataset
                                            #list(np.array(start_end_year, str)))
    subfoldername += append_pathsub


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
            ax.plot(rg.df_fullts.loc[df_test.index], label='detrend all data')
            ax.plot(df_test, label='detrend one-step-ahead')
            ax.legend()
            f.savefig(os.path.join(path, 'compared_detrend.jpg'), dpi=250,
                      bbox_inches='tight')

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
            sst.store_netcdf(rg.path_outsub1, load_sst, add_hash=False)
        sst.prec_labels['lag'] = ('lag', periodnames)
        sst.corr_xr['lag'] = ('lag', periodnames)
        rg.quick_view_labels('sst', min_detect_gc=.5, save=save,
                              append_str=periodnames[-1])
        plt.close()



    # #%% yield vs circulation plots
    # z500 = BivariateMI(name='z500',
    #                    filepath=rg.list_precur_pp[1][1],
    #                    func=class_BivariateMI.corr_map,
    #                    alpha=alpha_corr, FDR_control=True,
    #                    kwrgs_func={},
    #                    distance_eps=250, min_area_in_degrees2=3,
    #                    calc_ts='pattern cov', selbox=GlobalBox,
    #                    lags=lags, group_split=True,
    #                    use_coef_wghts=True)

    # z500.load_and_aggregate_precur(rg.kwrgs_load)
    # xrcorr, xrpvals = z500.bivariateMI_map(z500.precur_arr, df_splits,
    #                                       rg.df_fullts)
    # plot_maps.plot_corr_maps(xrcorr, xrcorr['mask'])

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
            SM.store_netcdf(rg.path_outsub1, load_SM, add_hash=False)
        SM.corr_xr['lag'] = ('lag', periodnames)
        SM.prec_labels['lag'] = ('lag', periodnames)
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
        regress_SM_same_mon = False
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
                if regress_SM_same_mon==False:
                    z_keys = [k for k in z_keys if f'{mon}..0..smi' not in k]


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



# =============================================================================
#
# =============================================================================

# =============================================================================
#
# =============================================================================

# =============================================================================
# Forecasts
# =============================================================================
if sys.platform == 'linux':
    mpl.use('Agg')
    n_cpu = 10

if 'timeseries' in method:
    btoos = '_T' # if btoos=='_T': binary target out of sample.
    # btoos = '_theor' # binary target based on gaussian quantile
else:
    btoos = ''




# regions for forcing per fc_month
regions_forcing = ['Pacific+SM', 'Pacific+SM', 'only_Pacific',
                   'only_Pacific', 'only_Pacific', 'only_Pacific',
                   'only_Pacific']
if extra_lag:
    regions_forcing = regions_forcing

for fc_type in fc_types:
    #%% forecast: get Combined Lead time models

    pathsub_df = f'df_data_{str(fc_type)}{btoos}'
    pathsub_verif = f'verif_{str(fc_type)}{btoos}'
    if training_data != 'CL':
        pathsub_df  += '_'+training_data
        pathsub_verif += '_'+training_data
    filepath_df_datas = os.path.join(rg.path_outsub1, pathsub_df)
    os.makedirs(filepath_df_datas, exist_ok=True)
    filepath_verif = os.path.join(rg.path_outsub1, pathsub_verif)
    os.makedirs(filepath_verif, exist_ok=True)
    from sklearn.linear_model import Ridge, LogisticRegression

    if fc_type == 'continuous':
        scoringCV = 'neg_mean_squared_error'
        kwrgs_model1 = {'scoringCV':scoringCV,
                        'alpha':list(np.concatenate([np.logspace(-4,0, 5),
                                                  np.logspace(.5, 2, num=10)])), # large a, strong regul.
                        'normalize':False,
                        'fit_intercept':False,
                        'kfold':5,
                        'n_jobs':n_cpu}
        model1_tuple = (ScikitModel(Ridge, verbosity=0),
                        kwrgs_model1)

        from sklearn.ensemble import RandomForestRegressor
        kwrgs_model2={'n_estimators':[450],
                      'max_depth':[3,6],
                      'scoringCV':'neg_mean_squared_error',
                      'oob_score':True,
                      'random_state':0,
                      'min_impurity_decrease':0,
                      'max_samples':[0.4,.7],
                      'max_features':[0.4,0.8],
                      'kfold':5,
                      'n_jobs':n_cpu}
        model2_tuple = (ScikitModel(RandomForestRegressor, verbosity=0),
                        kwrgs_model2)

    else:
        scoringCV = 'neg_brier_score'
        kwrgs_model1 = {'scoringCV':scoringCV,
                        'C':list([.1,.5,.8,1,1.2,4,7,10, 20]), # large a, strong regul.
                        'random_state':seed,
                        'penalty':'l2',
                        'solver':'lbfgs',
                        'kfold':10,
                        'max_iter':200}
        model1_tuple = (ScikitModel(LogisticRegression, verbosity=0),
                        kwrgs_model1)


        from sklearn.ensemble import RandomForestClassifier
        kwrgs_model2={'n_estimators':300,
                      'max_depth':[2,4,6],
                      'scoringCV':scoringCV,
                      # 'criterion':'mse',
                      'oob_score':True,
                      'random_state':0,
                      'min_impurity_decrease':0,
                      'max_features':[0.4,0.8],
                      'max_samples':[0.4, 0.7],
                      'kfold':10,
                      'n_jobs':n_cpu}
        model2_tuple = (ScikitModel(RandomForestClassifier, verbosity=0),
                        kwrgs_model2)

    if fc_type == 'continuous':
        model_combs = model_combs_cont
    else:
        model_combs = model_combs_bina

    # target timeseries, standardize using training data
    target_ts = rg.transform_df_data(rg.df_data.iloc[:,[0]].merge(rg.df_splits,
                                                      left_index=True,
                                                      right_index=True),
                                     transformer=fc_utils.standardize_on_train)
    # quantile has high sample bias, converges to -.44 when using 1E6 dp
    theothreshold = np.quantile(np.random.normal(size=int(1E6)), .33)
    if fc_type != 'continuous':
        if btoos == '_T':
            quantile = functions_pp.get_df_train(target_ts,
                                                 df_splits=rg.df_splits,
                                                 s='extrapolate',
                                                 function='quantile',
                                                 kwrgs={'q':fc_type})
            oos_std = [] ; qs = []
            for s in range(rg.n_spl):
                _std = target_ts.loc[s][(rg.df_splits.loc[s]['TrainIsTrue']!=1).values].std()
                oos_std.append(float(_std))
            tsall = rg.df_fulltso['raw_target']
            tsall = (tsall - tsall.mean()) / tsall.std()
            f, ax = plt.subplots(2)
            ax[1].plot(oos_std, label='oos std')
            ax[0].plot(quantile.mean(axis=0, level=0),
                       label='33th percentile based on training data')
            ax[0].axhline(theothreshold, color='black', lw=1)
            ax[0].text(-0.8, theothreshold+.01,
                       'Theoretical threshold')
            ax[1].set_xlabel('One-step-ahead training sets')
            ax[0].set_ylabel('Emperical threshold')
            ax[1].set_ylabel('std')
            ax[0].legend(loc='upper right') ; ax[1].legend()
            f.savefig(os.path.join(rg.path_outsub1, 'detrend',
                                   'threshold_std'+rg.figext))
            quantile = quantile.values
        elif btoos == '_theor':
            quantile = np.zeros_like(target_ts)
            quantile[:] = theothreshold
        else:
            _target_ts = target_ts.mean(0, level=1)
            _target_ts = (_target_ts - _target_ts.mean()) / _target_ts.std()
            quantile = float(_target_ts.quantile(fc_type))
        if fc_type >= 0.5:
            target_ts = (target_ts > quantile).astype(int)
        elif fc_type < .5:
            target_ts = (target_ts < quantile).astype(int)

    if np.unique(core_pp.flatten(model_combs)).size == 2:
        CL_models = [model1_tuple, model2_tuple]
    else:
        CL_models = [model1_tuple]

    for fcmodel, kwrgs_model in CL_models:
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
            print(fc_type, model_name_CL, i)

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



    #%% Make prediction
    model_name_CL, model_name = model_combs[0]
    for model_name_CL, model_name in model_combs:
        if model_name == 'Ridge' or model_name == 'LogisticRegression':
            fcmodel, kwrgs_model = model1_tuple
        elif 'RandomForest' in model_name:
            fcmodel, kwrgs_model = model2_tuple

        filepath_dfs = os.path.join(filepath_df_datas,
                                    f'CL_models_cont{model_name_CL}.h5')
        try:
            df_data_CL = functions_pp.load_hdf5(filepath_dfs)
        except:
            print('loading CL models failed, skipping this model')
            continue

        for nameTarget in ['Target']:
            for i, rg in enumerate(rg_list):
                # get CL model of that month
                rg.df_CL_data = df_data_CL[f'{rg.fc_month}_df_data']

            filepath_dfs = os.path.join(filepath_df_datas,
                                        f'predictions_cont_CL{model_name_CL}_'\
                                        f'{model_name}_{nameTarget}.h5')
            if os.path.exists(filepath_dfs):
                print('Prediction final model already stored, skipping this model')
                continue

            utils_paper3.get_df_forcing_cond_fc(rg_list,
                                                regions=regions_forcing,
                                                name_object='df_data')
            # loop over forecast months (lags)
            for i, rg in enumerate(rg_list):
                print(model_name_CL, model_name, i)
                keys_dict = {s:rg.df_CL_data.loc[s].dropna(axis=1).columns[:-2] \
                             for s in range(rg.n_spl)}
                # get estimated signal from Pacific

                # adapt target timeseries for fitting
                # mean over cols
                # very rare exception that Pac is not causal @ alpha level

                df_forcing = rg.df_forcing.mean(axis=1)
                df_forcing = df_forcing.fillna(np.nanmean(
                                                rg.df_forcing.values.ravel()))
                # if fc_type == 'continuous':
                df_forcing = pd.DataFrame(df_forcing, columns=[0])

                df_forcing = rg.transform_df_data(df_forcing.merge(rg.df_splits,
                                                               left_index=True,
                                                               right_index=True),
                                      transformer=fc_utils.standardize_on_train)
                df_forcing = df_forcing[0]
                # else:
                #     halfmaxprob = df_forcing.max(axis=0, level=0)/2.
                #     df_forcing  = df_forcing.subtract(halfmaxprob, axis=0, level=0)


                # target timeseries, standardize using training data
                target_ts = rg.transform_df_data(rg.df_data.iloc[:,[0]].merge(
                                                        rg.df_splits,
                                                        left_index=True,
                                                        right_index=True),
                                     transformer=fc_utils.standardize_on_train)

                target_ts = target_ts.rename({target_ts.columns[0]:'Target'},
                                             axis=1)
                target_ts_signal = target_ts.multiply(df_forcing.abs(),
                                                      axis=0)
                target_ts_signal= target_ts_signal.rename(\
                                              {rg.df_data.columns[0]:
                                               f'Target * {rg.fc_month} signal'},
                                              axis=1)
                # if fc_type == 'continuous':
                _t = rg.transform_df_data
                target_ts_signal = _t(target_ts_signal.merge(rg.df_splits,
                                                             left_index=True,
                                                             right_index=True),
                                      transformer=fc_utils.standardize_on_train)
                if nameTarget == 'Target':
                    _target_ts = target_ts
                if nameTarget == 'Target*Signal':
                    _target_ts = target_ts_signal
                if fc_type != 'continuous':
                    if btoos == '_T':
                        # quantile based on only training data, using the
                        # standardized-on-train Target
                        quantile = functions_pp.get_df_train(_target_ts,
                                                             df_splits=rg.df_splits,
                                                             s='extrapolate',
                                                             function='quantile',
                                                             kwrgs={'q':fc_type}).values
                    elif btoos == '_theor':
                        quantile = np.zeros_like(target_ts)
                        quantile[:] = theothreshold
                    else:
                        # using all target data - mean over standardized-on-train
                        _target_ts = _target_ts.mean(0, level=1)
                        _target_ts = (_target_ts - _target_ts.mean()) / _target_ts.std()
                        quantile = float(_target_ts.quantile(fc_type))
                    if fc_type >= 0.5:
                        _target_ts = (_target_ts > quantile).astype(int)
                    elif fc_type < .5:
                        _target_ts = (_target_ts < quantile).astype(int)

                # use combined lead time model for (final) prediction
                if training_data == 'CL':
                    df_input = rg.df_CL_data
                # use all RG-DR timeseries for (final) prediction
                elif training_data == 'all':
                    df_input = rg.df_data
                # use all RG-DR timeseries that are C.D. for (final) prediction
                elif training_data == 'all_CD':
                    df_input = rg.df_data
                    keys_dict = utils_paper3.get_CD_df_data(rg, alpha_CI)
                # use only fist lag of RG-DR timeseries that are C.D.
                elif training_data == 'onelag':
                    df_input = rg.df_data
                    firstlag = str(rg.list_for_MI[0].corr_xr.lag[-1].values)
                    keys_dict = utils_paper3.get_CD_df_data(rg, alpha_CI,
                                                            firstlag)


                prediction_tuple = rg.fit_df_data_ridge(df_data=df_input,
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

                if 'RandomForest' in fcmodel.scikitmodel.__name__:
                    #Check RF tuning
                    try:
                        import scikit_model_analysis as sk_ana
                        model = models_lags['lag_0']['split_0']
                        f = sk_ana.ensemble_error_estimators(model.best_estimator_,
                                                             kwrgs_model,
                                                             min_estimators=1,
                                                             steps=10)
                        f.savefig(os.path.join(filepath_verif,
                                               f'RF_tuning_{rg.fc_month}.pdf'), bbox_inches='tight')
                    except:
                        print('RF tuning plot failed')
                        pass

            # store output predictions of each month per model
            model_name = fcmodel.scikitmodel.__name__
            df_predictions, df_w_save = utils_paper3.df_predictions_for_plot(rg_list)
            d_df_preds={'df_predictions':df_predictions, 'df_weights':df_w_save}
            functions_pp.store_hdf_df(d_df_preds, filepath_dfs)

    #%% Forecast Verification
    verif_combs = [['Target', 'Target']]

    model_name_CL, model_name = model_combs[0] # for testing
    nameTarget_fit, nameTarget = verif_combs[0] # for testing
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

            filepath_dfs = os.path.join(filepath_df_datas,
                            f'scores_cont_CL{model_name_CL}_{model_name}_'\
                                f'{nameTarget_fit}_{nameTarget}_{n_boot}.h5')
            if os.path.exists(filepath_dfs):
                print('Verification of model vs. Targert already stored, skip')
                continue

            for i, rg in enumerate(rg_list):
                prediction = df_predictions[[nameTarget, rg.fc_month]].copy()
                if nameTarget_fit == 'Target*Signal' and nameTarget == 'Target' \
                                                and fc_type != 'continuous':
                    # Unique case, Target is continuous -> convert to binary
                    _target_ts = prediction[['Target']]
                    if btoos == '_T':
                        quantile = functions_pp.get_df_train(_target_ts,
                                                             df_splits=rg.df_splits,
                                                             s='extrapolate',
                                                             function='quantile',
                                                             kwrgs={'q':fc_type}).values
                    elif btoos == '_theor':
                        quantile = np.zeros_like(target_ts)
                        quantile[:] = theothreshold
                    else:
                        quantile = float(_target_ts.quantile(fc_type))
                    if fc_type >= 0.5:
                        _target_ts = (_target_ts > quantile).astype(int)
                    elif fc_type < .5:
                        _target_ts = (_target_ts < quantile).astype(int)

                    prediction[[nameTarget]] = _target_ts

                # # Benchmark step 1: mean training sets extrapolated to test
                # bench = functions_pp.get_df_train(prediction[nameTarget],
                #                                   df_splits=rg.df_splits,
                #                                   s='extrapolate',
                #                                   function='mean')
                # bench = functions_pp.get_df_test(bench, df_splits=rg.df_splits)
                # bench = float(bench.mean())
                if fc_type == 'continuous':
                    # benchmark is climatological mean (after detrending)
                    bench = float(rg.df_data.loc[0].iloc[:,[0]].mean())
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

            functions_pp.store_hdf_df(d_dfs, filepath_dfs)

    #%% Conditional forecast verification
    # import utils_paper3

    for model_name_CL, model_name in model_combs:
        # load CL model to get df_forcing
        filepath_dfs = os.path.join(filepath_df_datas,
                                    f'CL_models_cont{model_name_CL}.h5')
        try:
            df_data_CL = functions_pp.load_hdf5(filepath_dfs)
        except:
            print('loading CL models failed, skipping this model')
            continue

        for i, rg in enumerate(rg_list):
            # get CL model of that month
            rg.df_CL_data = df_data_CL[f'{rg.fc_month}_df_data']
        # get forcing per fc_month
        utils_paper3.get_df_forcing_cond_fc(rg_list,
                                            regions=regions_forcing,
                                            name_object='df_data')

        nameTarget = 'Target'
        for nameTarget_fit in ['Target']:

            f_name = f'predictions_cont_CL{model_name_CL}_{model_name}_'\
                                                        f'{nameTarget_fit}.h5'
            filepath_dfs = os.path.join(filepath_df_datas, f_name)

            try:
                d_dfs = functions_pp.load_hdf5(filepath_dfs)
                df_predictions = d_dfs['df_predictions']
            except:
                print('loading predictions failed, skipping this model')
                continue

            filepath_dfs = os.path.join(filepath_df_datas,
                                             f'scores_cont_CL{model_name_CL}_{model_name}_'\
                                             f'{nameTarget_fit}_{nameTarget}_{n_boot}_CF.h5')
            if os.path.exists(filepath_dfs):
                print('Cond. Verif. of model vs. Target already stored, skip')
                continue
            else:
                print(f'Cond. Verif.: CL-{model_name_CL} -> {model_name} -> {nameTarget_fit}')

            if nameTarget_fit == 'Target*Signal' and fc_type != 'continuous':
                _target_ts = df_predictions[['Target']]
                if btoos == '_T':
                    quantile = functions_pp.get_df_train(target_ts,
                                                         df_splits=rg.df_splits,
                                                         s='extrapolate',
                                                         function='quantile',
                                                         kwrgs={'q':fc_type}).values
                elif btoos == '_theor':
                    quantile = np.zeros_like(target_ts)
                    quantile[:] = theothreshold
                else:
                    __target_ts = _target_ts.mean(0, level=1)
                    __target_ts = (__target_ts - __target_ts.mean()) / __target_ts.std()
                    quantile = float(__target_ts.quantile(fc_type))
                if fc_type >= 0.5:
                    _target_ts = (_target_ts > quantile).astype(int)
                elif fc_type < .5:
                    _target_ts = (_target_ts < quantile).astype(int)


                # drop original continuous Target
                df_predictions = df_predictions.drop(nameTarget, axis=1)
                # replace Target*Signal with binary Target for verification
                df_predictions = df_predictions.rename(
                                        {nameTarget_fit : nameTarget},axis=1)
                df_predictions[[nameTarget]] = _target_ts
                # prediction.drop(nameTarget_fit)
                # prediction = prediction.drop(nameTarget_fit, axis=1)

            # metrics
            if fc_type == 'continuous':
                # benchmark is climatological mean (after detrending)
                bench = float(rg.df_data.loc[0].iloc[:,[0]].mean())
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

            df_cond = utils_paper3.cond_forecast_table(rg_list, score_func_list,
                                                       df_predictions,
                                                       nameTarget='Target',
                                                       n_boot=0)

            df_cond_b = utils_paper3.cond_forecast_table(rg_list, score_func_list,
                                                       df_predictions,
                                                       nameTarget='Target',
                                                       n_boot=n_boot)
            d_dfs={'df_cond':df_cond, 'df_cond_b':df_cond_b}

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

    #%% Plotting forecast timeseries
    import utils_paper3
    if fc_type == 'continuous':
        metrics_plot = ['corrcoef', 'MAE', 'r2_score']
        model_combs_plot  = [['Ridge', 'Ridge'],
                             ['Ridge', 'RandomForestRegressor'],
                             ['RandomForestRegressor', 'RandomForestRegressor']]
        model_combs_plot = [c for c in model_combs_plot if c in model_combs_cont]
    else:
        metrics_plot = ['BSS', 'accuracy', 'precision'] # 'roc_auc_score',
        model_combs_plot  = [['LogisticRegression', 'LogisticRegression'],
                             ['LogisticRegression', 'RandomForestClassifier'],
                             ['RandomForestClassifier', 'RandomForestClassifier']]
        model_combs_plot = [c for c in model_combs_plot if c in model_combs_bina]

    condition = ['strong 50%', 'strong 30%']
    df_forcings = []
    rename_f = {'Pacific+SM':'mean over standardized horseshoe Pacific + Soil Moisture timeseries',
                'only_Pacific':'mean over standardized horseshoe Pacific timeseries'}
    for i, rg in enumerate(rg_list):
        df_forcings.append(pd.DataFrame(rg.df_forcing.mean(axis=1),
                            columns=[f'{rg.fc_month} Signal (S): '+rename_f[regions_forcing[i]]]))
    df_forcings = pd.concat(df_forcings, axis=1)
    # standardize for easy visualization
    df_forcings = rg.transform_df_data(df_forcings.merge(rg.df_splits,
                                             left_index=True,
                                             right_index=True),
                      transformer=fc_utils.standardize_on_train)
    df_forcings.columns.name = int(condition[0][-3:-1])

    # plotting timeseries per 2 months
    fc_month_list = [rg.fc_month for rg in rg_list][::2]
    print('Plotting timeseries')
    model_name_CL, model_name = model_combs_plot[0]
    for model_name_CL, model_name in model_combs_plot:
        print(model_name_CL, model_name)
        if 'RandomForest' in model_name:
            name = 'RF'
        elif model_name == 'Ridge' or model_name=='LogisticRegression':
            name = 'Ridge' if fc_type =='continuous' else 'Logist. Regr.'
        if 'RandomForest' in model_name_CL:
            name_CL = 'RF'
        elif model_name_CL == 'Ridge' or model_name=='LogisticRegression':
            name_CL = 'Ridge' if fc_type =='continuous' else 'Logist. Regr.'

        target_options = [['Target', 'Target | PPS']]

        print('Plotting skill scores')
        for i, target_opt in enumerate(target_options):

            fig, axes = plt.subplots(nrows=len(fc_month_list)*2, ncols=2,
                                     figsize=(17,len(fc_month_list)*3.5),
                                     gridspec_kw={'width_ratios':[3.4,1],
                                          'height_ratios':[3,1] * len(fc_month_list)},
                             sharex=True, sharey=False)
            out = utils_paper3.load_scores(target_opt, model_name_CL, model_name,
                                           n_boot, filepath_df_datas,
                                           condition=condition)

            df_scores, df_boots, df_preds = out
            for m, fc_month in enumerate(fc_month_list):
                rg = [rg for rg in rg_list if rg.fc_month == fc_month][0]
                axs = axes[m*2:m*2+2]
                ax = axs[0]
                df_test_m = [d[fc_month] for d in df_scores]
                df_boots_list = [d[fc_month] for d in df_boots]
                df_test  = df_preds[0][['Target', fc_month]]
                df_test = functions_pp.get_df_test(df_test,
                                           df_splits=rg.df_splits)
                if fc_type != 'continuous' and any(['(fitPPS)' in t for t in target_opt]):
                    if fc_type >= 0.5:
                        df_test[['Target']] = (df_test[['Target']] > \
                                    df_test[['Target']].quantile(fc_type)).astype(int)
                    elif fc_type < .5:
                        df_test[['Target']] = (df_test[['Target']] < \
                                    df_test[['Target']].quantile(fc_type)).astype(int)

                utils_paper3.plot_forecast_ts(df_test_m, df_test,
                                              df_forcings=df_forcings,
                                              df_boots_list=df_boots_list,
                                              fig_ax=(fig, axs),
                                              fs=11,
                                              metrics_plot=metrics_plot,
                                              name_model=f'CL-{name_CL} -> {name}')
                ax[0].margins(x=.02)
                if m == 0:
                    axes[0][0].text(y=0.37, x=1, s='clim. prob.',
                                    horizontalalignment='right',
                                    transform=axes[0][0].transAxes)
                    ax[0].legend(fontsize=10, loc='upper left')
                    ax1b = axs[1][1]
                    qs = [int(c[-3:-1]) for c in condition]
                    sizes = [6, 10] ;
                    colors = [['#e76f51', '#61a5c2'], ['#d00000', '#3f37c9']]
                    handles = []
                    for j, qth in enumerate(qs):
                        handles.append(Line2D([0],[0], color='white', lw=0.1,
                                              marker='o', markersize=sizes[j],
                                              markerfacecolor=colors[j][0],
                            label=f'S>{int((100-qth/2))}p'))
                        handles.append(Line2D([0],[0], color='white', lw=0.1,
                                              marker='o', markersize=sizes[j],
                                              markerfacecolor=colors[j][1],
                            label=f'S<{int((qth/2))}p'))

                    ax1b.legend(handles=handles, loc='center', ncol=2,
                                  bbox_to_anchor = (0,0,0.1,1), fontsize=12,
                                  facecolor='white',
                                  title='Top 50%          Top 30%')
            plt.subplots_adjust(hspace=.4)

            fig.savefig(os.path.join(filepath_verif,
                      f'timeseries_and_skill_{i}_{model_name_CL}_{model_name}.pdf'), bbox_inches='tight')
            # plt.close()
    #%% plotting skill scores as function of lead-time
    # import utils_paper3
    if fc_type == 'continuous':
        metrics_plot = ['corrcoef', 'MAE', 'RMSE', 'r2_score']
        model_combs_plot  = [['Ridge', 'Ridge'],
                             ['Ridge', 'RandomForestRegressor'],
                             ['RandomForestRegressor', 'RandomForestRegressor']]
        model_combs_plot = [c for c in model_combs_plot if c in model_combs_cont]
    else:
        metrics_plot = ['BSS', 'roc_auc_score']
        model_combs_plot  = [['LogisticRegression', 'LogisticRegression'],
                             ['LogisticRegression', 'RandomForestClassifier'],
                             ['RandomForestClassifier', 'RandomForestClassifier']]
        model_combs_plot = [c for c in model_combs_plot if c in model_combs_bina]


    fc_month_list = [rg.fc_month for rg in rg_list]
    target_options = [['Target', 'Target | PPS']]
    print('Plotting skill scores')
    for i, target_opt in enumerate(target_options):
        fig, axes = plt.subplots(nrows=len(model_combs_plot), ncols=len(metrics_plot),
                     figsize=(17,3.33*len(model_combs_plot)),
                      # gridspec_kw={'width_ratios':[4,1]},
                      sharex=True, sharey=False)
        if len(model_combs_plot) == 1: axes = axes.reshape(1, 2)
        for j, (model_name_CL, model_name) in enumerate(model_combs_plot):


            print(model_name_CL, model_name, target_opt)
            if 'RandomForest' in model_name:
                name = 'RF'
            elif model_name == 'Ridge' or model_name=='LogisticRegression':
                name = 'Ridge' if fc_type =='continuous' else 'LR'
            if 'RandomForest' in model_name_CL:
                name_CL = 'RF'
            elif model_name_CL == 'Ridge' or model_name=='LogisticRegression':
                name_CL = 'Ridge' if fc_type =='continuous' else 'LR'

            out = utils_paper3.load_scores(target_opt, model_name_CL, model_name,
                                           n_boot, filepath_df_datas,
                                           condition='strong 50%')[:2]
            df_scores_list, df_boot_list = out


            fig = utils_paper3.plot_scores_wrapper(df_scores_list, df_boot_list,
                                                 labels=target_opt,
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
                                 f'scores_vs_lags_{i}.pdf'), bbox_inches='tight')
        plt.close()



#%% PDO versus trend line
import climate_indices
_df_PDO, PDO_patterns = climate_indices.PDO(rg.list_for_MI[0].filepath,
                                           None)
PDO_plot_kwrgs = {'units':'[-]', 'cbar_vert':-.1,
                  # 'zoomregion':(130,260,20,60),
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'y_ticks':np.array([25,40,50,60]),
                  'x_ticks':np.arange(130, 280, 25),
                  'clevels':np.arange(-.6,.61,.075),
                  'clabels':np.arange(-.6,.61,.3),
                  'subtitles':np.array([['PDO negative loading pattern']])}
fcg = plot_maps.plot_corr_maps(PDO_patterns[0], **PDO_plot_kwrgs)
filepath = os.path.join(rg.path_outsub1, 'PDO_pattern')
fcg.fig.savefig(filepath + '.pdf', bbox_inches='tight')
fcg.fig.savefig(filepath + '.png', bbox_inches='tight')


if 'timeseries' in method:
    df_fullts = functions_pp.get_df_test(rg.df_fullts,
                                         df_splits=rg.df_splits)
else:
    df_fullts  = rg.df_fullts

df_PDO = _df_PDO.loc[0][['PDO']]
df_PDO = df_PDO.groupby(df_PDO.index.year).mean()
df_PDO = df_PDO.iloc[-rg.df_splits.index.levels[1].size:] # lazy way of selecting years
df_PDO.index = rg.df_splits.index.levels[1]
df_Pac = [c for c in rg.df_data.columns if '..1..sst' in c]
df_Pac = functions_pp.get_df_test(rg.df_data[df_Pac], df_splits=rg.df_splits)
df_Pacm = df_Pac.mean(axis=1) ; df_Pacm.name = 'east Pac.'
df_PDO_T = df_fullts.merge(df_PDO, left_index=True, right_index=True)
df_PDO_T_P = df_PDO_T.merge(df_Pacm, left_index=True, right_index=True)
df_PDO_T_P = (df_PDO_T_P - df_PDO_T_P.mean(0)) / df_PDO_T_P.std(0)
#%%
f, ax = plt.subplots(1, figsize=(12,8))
df_PDO_T_P.plot(ax=ax, color=['black', 'grey', 'r'])
ax.axhline(color='black')
ax.legend(['Target', 'PDO(-)', 'Eastern Pacific'])
f.savefig(os.path.join(rg.path_outsub1, 'Target_vs_Pac_ts'+rg.figext),
          bbox_inches='tight')
#%%
_model = fcmodel.scikitmodel.__name__
_month = 'May'
filepath_dfs = os.path.join(filepath_df_datas,
                            f'CL_models_cont{_model}.h5')
df_data_CL = functions_pp.load_hdf5(filepath_dfs)
CL_m = df_data_CL[_month + '_df_data']
CL_Pac = CL_m[[c for c in CL_m.columns if '..1..sst' in c]]
CL_Pac = functions_pp.get_df_train(CL_Pac, df_splits=rg.df_splits)
CL_PacR_mean = CL_Pac.merge(df_Pacm, left_index=True, right_index=True)
f, ax = plt.subplots(1, figsize=(12,8))
CL_PacR_mean.plot(ax=ax, color=['blue', 'r'])
ax.axhline(color='black')
ax.legend([f'{_model} Eastern Pacific', 'Eastern Pacific mean'])
f.savefig(os.path.join(rg.path_outsub1, 'Pacific_model_vs_Pacific_mean'+rg.figext),
          bbox_inches='tight')
#%%
import utils_paper3
utils_paper3.plot_regions(rg_list, save=True, plot_parcorr=False, min_detect=.1,
                           selection='CD')

utils_paper3.plot_regions(rg_list, save=True, plot_parcorr=False, min_detect=.1,
                           selection='CD', min_cd = 0.5)

utils_paper3.plot_regions(rg_list, save=True, plot_parcorr=False, min_detect=.5,
                           selection='CD', min_cd = 0.5)


utils_paper3.plot_regions(rg_list, save=True, plot_parcorr=False, min_detect=.1,
                           selection='CD', min_cd = 0.5, plot_textinmap=False)

utils_paper3.plot_regions(rg_list, save=True, plot_parcorr=False, min_detect=.1,
                          selection='ind')

utils_paper3.plot_regions(rg_list, save=True, plot_parcorr=False, min_detect=.1,
                          selection='all')


#%%

# collecting different train-test splits to plot scores vs lead-time
plot_combs = [[0.33, '50%'],
              ['continuous', '50%'],
              [0.33, '30%'],
              ['continuous', '30%']]
plot_combs = [p for p in plot_combs if p[0] in fc_types]

for fc_type, condition in plot_combs:
    #%% Continuous forecast: get Combined Lead time models
    pathsub_df = f'df_data_{str(fc_type)}{btoos}'
    filepath_df_datas = os.path.join(rg.path_outsub1, pathsub_df)
    os.makedirs(filepath_df_datas, exist_ok=True)
    filepath_verif = os.path.join(rg.path_outsub1, f'verif_{str(fc_type)}{btoos}')
    import re

    if 'timeseries' not in method:
        path = '/'.join(rg.path_outsub1.split('/')[:-1])
        hash_str = 's\d'
    else:
        path = '/'.join(rg.path_outsub1.split('/')[:-2])
        hash_str = 'timeseriessplit_\d\d'



    cs = ["#a4110f","#f7911d","#fffc33","#9bcd37","#1790c4"]
    f_names = []
    for root, dirs, files in os.walk(path):
        for _dir in dirs:
            if re.match(f'{hash_str}', _dir):
                print(f'Found file {_dir}')
                f_names.append(os.path.join(root,_dir))

    if fc_type == 'continuous':
        metrics_plot = ['corrcoef', 'MAE', 'r2_score']
        model_combs_plot  = [['Ridge', 'Ridge'],
                             # ['Ridge', 'RandomForestRegressor'],
                             ['RandomForestRegressor', 'RandomForestRegressor']]
    else:
        metrics_plot = ['BSS', 'accuracy', 'precision'] # 'roc_auc_score',
        model_combs_plot  = [['LogisticRegression', 'LogisticRegression'],
                             # ['LogisticRegression', 'RandomForestClassifier'],
                             ['RandomForestClassifier', 'RandomForestClassifier']]


    f_names = sorted(f_names)
    collectdict = {}
    for j, (model_name_CL, model_name) in enumerate(model_combs_plot):

        for f_name in f_names:
            if 'timeseries' in method:
                _path_df_datas = os.path.join(f_name, f's1/{pathsub_df}')
            else:
                _path_df_datas = os.path.join(f_name, pathsub_df)
            for _target in ['Target', 'Target | PPS']:
                try:
                    out = utils_paper3.load_scores([_target], model_name_CL,
                                                   model_name,
                                                   n_boot, _path_df_datas,
                                                   condition=f'strong {condition}')[:2]
                    df_scores_list, df_boot_list = out
                    collectdict[f_name.split('/')[-1]+str(j)+model_name+_target] = out
                except:
                    # print(f_name)
                    # f_names.remove(f_name)
                    df_scores_list, df_boot_list = None, None
                    continue

    collectdict = dict(sorted(collectdict.items()))
    cvnames = [f.split('/')[-1] for f in f_names]
    if 'timeseries' not in method:
        cvrename = [f.replace('s', '20-fold seed ') for f in cvnames]
        loc = 'best'
    else:
        cvrename = [f.replace('timeseriessplit_', 'one-step-ahead ') for f in cvnames]
        loc = 'lower left'
    cvrename = cvrename[::-1]
    cvrename = [c +f' [Top {condition}, all]' for c in cvrename]

    #%% Plot

    f, axes = plt.subplots(len(metrics_plot),2, figsize=(14,8),
                            sharey=False)
    cols = [model_combs_plot[0][0], model_combs_plot[1][0]]
    rows = metrics_plot
    cs = ['#EE6666', '#3388BB', '#88BB44', '#9988DD', '#EECC55',
          '#FFBBBB']
    rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr.', 'MAE':'MAE-SS',
                  'BSS':'BSS', 'roc_auc_score':'AUC', 'r2_score':'$r^2$',
                  'mean_absolute_percentage_error':'MAPE', 'AUC_SS':'AUC-SS',
                  'precision':'Precision', 'accuracy':'Accuracy'}

    order_labels = [k for k in collectdict.keys() if 'Logist' in k or 'Ridge' in k]
    labels_track = [] ; cvcount = 0 ; lines = {}
    for i, (key, item) in enumerate(collectdict.items()):
        idxcol = [i for i in range(len(cols)) if cols[i] in key][0]
        axrows = axes[:,idxcol]

        idxcv = [i for i in range(len(cvnames)) if cvnames[i] in key][0]
        label = cvrename[idxcv]
        color = cs[idxcv]
        if label in labels_track:
            label = None
        else:
            labels_track.append(label)

        ls = 'dotted' if 'PPS' in key else '-'
        df_sc= item[0][0] ;
        for idxrow, _metric in enumerate(metrics_plot):
            # columns.levels auto-sorts order of labels, to avoid:
            steps = df_sc.columns.levels[1].size
            months = [t[0] for t in df_sc.columns][::steps]
            ax = axrows[idxrow]
            l = ax.plot(months, df_sc.reorder_levels((1,0), axis=1).iloc[0][_metric].T,
                        color=color,
                        linestyle=ls)

            if idxcol == 0:
                lines[key] = l

            if _metric in ['BSS']:
                bench = 0 ; ax.set_ylim(-.1,1)
                ax.set_yticks(np.arange(0,1.01,0.2))
            elif _metric == 'roc_auc_score':
                bench = 0.5 ; ax.set_ylim(0,1)
            elif _metric == 'accuracy':
                bench = 100*((0.33**2) + (0.66**2))
                ax.set_ylim(bench-10,100)
                ax.set_yticks(np.arange(60,101,10))
            elif _metric == 'precision':
                bench = 33 ; ax.set_ylim(bench-10,100)
                ax.set_yticks(np.arange(35,101,15))
            elif _metric in ['corrcoef', 'MAE', 'r2_score']:
                bench = 0 ; ax.set_ylim(-.1,1.)
                ax.set_yticks(np.arange(0,1.01,0.2))
            ax.axhline(bench, color='black', alpha=0.5)
            if idxrow == 0:
                ax.set_title(cols[idxcol])
            if idxcol == 0:
                ax.set_ylabel(rename_met[_metric])

    # Legend stuff
    lines = [lines[ol] for ol in order_labels[::-1]]
    linesu = []
    for i, l in enumerate(lines):
        if i%2 == 0:
            l1 = lines[i][0]
            if len(lines) != i+1:
                l2 = lines[i+1][0]
                linesu.append((l1, l2))
            else:
                linesu.append(l1)
    from matplotlib.legend_handler import HandlerTuple
    axes[1,0].legend(linesu,
                      cvrename,
                      loc=loc,
                      handler_map={tuple: HandlerTuple(ndivide=2)},
                      handlelength=4,
                      fontsize=10)
    f.subplots_adjust(wspace=0.15)
    #%%
    if save:
        f.savefig(os.path.join(filepath_verif, f'different_cvs_{condition[:2]}.jpg'),
                           bbox_inches='tight')
        f.savefig(os.path.join(filepath_verif, f'different_cvs_{condition[:2]}.pdf'),
                           bbox_inches='tight')
















# #%% plot
# for rg in rg_list:
#     models_lags = rg.prediction_tuple[-1]
#     df_wgths, fig = plot_importances(models_lags)
#     fig.savefig(os.path.join(rg.path_outsub1, f'weights_{rg.fc_month}.png'),
#                 bbox_inches='tight', dpi=100)

# #%% R2 skill metric proportional with length of dataset
# det, trend = core_pp.detrend_wrapper(rg.df_fulltso[['raw_target']], return_trend=True)
# trend = trend.rename({'raw_target':'trend'}, axis=1)
# target = rg.df_fulltso[['raw_target']].rename({'raw_target':'Soy Yield'})
# trend_pred = target.merge(trend, left_index=True, right_index=True)
# skill = fc_utils.get_scores(trend_pred,
#                             score_func_list=[fc_utils.r2_score])[2]
# skill10yrs = fc_utils.get_scores(trend_pred.loc[core_pp.get_subdates(trend_pred.index, None,
#                                                                 range(2000,2010))],
#                             score_func_list=[fc_utils.r2_score])[2]