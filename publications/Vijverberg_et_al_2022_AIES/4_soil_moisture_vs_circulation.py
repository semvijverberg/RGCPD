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
import xarray as xr

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
from RGCPD import BivariateMI
from RGCPD import class_BivariateMI
from RGCPD import plot_maps
# import func_models as fc_utils
# import functions_pp, find_precursors
# import wrapper_PCMCI
# import utils_paper3
# from stat_models import plot_importances
# from stat_models_cont import ScikitModel
# from sklearn.linear_model import LinearRegression

All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']


target_datasets = ['smi']
seeds = [1] # ,5]
yrs = ['1950, 2019'] # ['1950, 2019', '1960, 2019', '1950, 2009']
methods = ['leave_1'] # ['ranstrat_20'] timeseriessplit_30
feature_sel = [True]
combinations = np.array(np.meshgrid(target_datasets,
                                    seeds,
                                    yrs,
                                    methods,
                                    feature_sel)).T.reshape(-1,5)
i_default = -1
load = False
save = True


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


init_croptarget = 'a9943/USDA_Soy_clusters__1'
calc_ts= 'region mean' # 'pattern cov'
alpha_corr = .05
alpha_CI = .05
n_boot = 2000
append_pathsub = f'/{method}/s{seed}'
extra_lag = True
append_main = init_croptarget
subfoldername = init_croptarget
subfoldername += append_pathsub
if method.split('_')[0]=='leave':
    subfoldername += 'gp_prior_1_after_1'

path_out_main = os.path.join(user_dir, 'surfdrive', 'output_paper3', 'minor_revision')
pathoutfull = os.path.join(path_out_main, subfoldername)

PacificBox = (130,265,-10,60)
GlobalBox  = (-180,360,-10,60)
USBox = (225, 300, 20, 60)

filename_smi_pp = os.path.join(path_raw, 'preprocessed/SM_ownspi_gamma_2_1950-2019_'
                               'jan_dec_monthly_1.0deg.nc')


#%% run RGPD


def pipeline(lags, periodnames, load=False):
    #%%


    method = False
    SM_lags = lags.copy()
    for i, l in enumerate(SM_lags):
        orig = '-'.join(l[0].split('-')[:-1])
        repl = '-'.join(l[1].split('-')[:-1])
        SM_lags[i] = [l[0].replace(orig, repl), l[1]]

    SM = BivariateMI(name='smi', filepath=filename_smi_pp,
                      func=class_BivariateMI.corr_map,
                      alpha=alpha_corr, FDR_control=True,
                      kwrgs_func={},
                      distance_eps=250, min_area_in_degrees2=3,
                      calc_ts='pattern cov', selbox=USBox,
                      lags=SM_lags, use_coef_wghts=True)

    load_SM = '{}_a{}_{}_{}_{}'.format(SM._name, SM.alpha,
                                        SM.distance_eps,
                                        SM.min_area_in_degrees2,
                                        periodnames[-1])

    loaded = SM.load_files(pathoutfull, load_SM)
    SM.prec_labels['lag'] = ('lag', periodnames)
    SM.corr_xr['lag'] = ('lag', periodnames)
    # SM.get_prec_ts(kwrgs_load={})
    # df_SM = pd.concat(SM.ts_corr, keys=range(len(SM.ts_corr)))


    TVpath = os.path.join(pathoutfull,
                          f'df_output_{periodnames[-1]}.h5')
    z500_maps = []
    for i, periodname in enumerate(periodnames):
        lag = np.array(lags[i])
        _yrs = [int(l.split('-')[0]) for l in lag]
        if np.unique(_yrs).size>1: # crossing year
            crossyr = True
        else:
            crossyr = False
        start_end_TVdate =  ('-'.join(lag[0].split('-')[1:]),
                             '-'.join(lag[1].split('-')[1:]))
        lag = np.array([start_end_TVdate])

        list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=alpha_corr, FDR_control=True,
                                kwrgs_func={},
                                distance_eps=250, min_area_in_degrees2=3,
                                calc_ts='pattern cov', selbox=(155,355,10,80),
                                lags=lag, group_split=True,
                                use_coef_wghts=True)]

        name_ds = f'{periodname}..0..{target_dataset}_sp'
        list_of_name_path = [('', TVpath),
                             ('z500', os.path.join(path_raw,
                              'z500_1950-2019_1_12_monthly_1.0deg.nc'))]


        start_end_year = (1951, 2019)
        if crossyr:
            TV_start_end_year = (start_end_year[0]+1, 2019)
        else:
            TV_start_end_year = (start_end_year[0], 2019)
        kwrgs_core_pp_time = {'start_end_year': TV_start_end_year}

        rg = RGCPD(list_of_name_path=list_of_name_path,
                   list_for_MI=list_for_MI,
                   list_import_ts=None,
                   start_end_TVdate=start_end_TVdate,
                   start_end_date=None,
                   start_end_year=start_end_year,
                   tfreq=None,
                   path_outmain=path_out_main)
        rg.figext = '.png'


        rg.pp_precursors(detrend=[True, {'tp':False, 'smi':False}],
                         anomaly=[True, {'tp':False, 'smi':False}])


        # detrending done prior in clustering_soybean
        rg.pp_TV(name_ds=name_ds, detrend=False, ext_annual_to_mon=False,
                 kwrgs_core_pp_time=kwrgs_core_pp_time)

        # if method.split('_')[0]=='leave':
            # rg.traintest(method, gap_prior=1, gap_after=1, seed=seed,
                         # subfoldername=subfoldername)
        # else:
        rg.traintest(method, seed=seed, subfoldername=subfoldername)


        z500 = rg.list_for_MI[0]
        path_circ = os.path.join(rg.path_outsub1, 'circulation')
        os.makedirs(path_circ , exist_ok=True)
        load_z500 = '{}_a{}_{}_{}_{}'.format(z500._name, z500.alpha,
                                            z500.distance_eps,
                                            z500.min_area_in_degrees2,
                                            periodnames[-1])
        if load == 'maps' or load == 'all':
            loaded = z500.load_files(path_circ, load_z500)
        else:
            loaded = False
        if hasattr(z500, 'corr_xr')==False:
            rg.calc_corr_maps('z500')
        # store forecast month
        months = {'JJ':'August', 'MJ':'July', 'AM':'June', 'MA':'May',
                  'FM':'April', 'JF':'March', 'SO':'December', 'DJ':'February'}
        rg.fc_month = months[periodnames[-1]]

        z500_maps.append(z500.corr_xr)

        if loaded==False:
            z500.store_netcdf(path_circ, load_z500, add_hash=False)

    z500_maps = xr.concat(z500_maps, dim='lag')
    z500_maps['lag'] = ('lag', periodnames)
    #%%
    # merge maps
    xr_merge = xr.concat([SM.corr_xr.mean('split'),
                          z500_maps.drop_vars('split').squeeze()], dim='var')
    xr_merge['var'] = ('var', ['SM', 'z500'])
    xr_merge = xr_merge.sel(lag=periodnames[::-1])
    # get mask
    maskSM = RGCPD._get_sign_splits_masked(SM.corr_xr, min_detect=.1,
                                           mask=SM.corr_xr['mask'])[1]
    xr_mask = xr.concat([maskSM,
                         z500_maps['mask'].drop_vars('split').squeeze()], dim='var')
    xr_mask['var'] = ('var', ['SM', 'z500'])
    xr_mask = xr_mask.sel(lag=periodnames[::-1])

    month_d = {'AS':'Aug-Sep mean', 'JJ':'June-July mean',
               'JA':'July-Aug mean','MJ':'May-June mean',
               'AM':'Apr-May mean',
               'MA':'Mar-Apr mean', 'FM':'Feb-Mar mean',
               'JF':'Jan-Feb mean', 'DJ':'Dec-Jan mean',
               'ND':'Nov-Dec mean', 'ON':'Oct-Nov mean',
               'SO':'Sep-Oct mean'}

    subtitles = np.array([month_d[l] for l in xr_merge.lag.values],
                         dtype='object')[::-1]
    subtitles = np.array([[s + ' SM vs yield' for s in subtitles[::-1]],
                          [s + ' z500 vs SM' for s in subtitles[::-1]]])
    # leadtime = intmon_d[rg.fc_month]
    # subtitles = [subtitles[i-1]+f' ({leadtime+i*2-1}-month lag)' for i in range(1,5)]
    kwrgs_plot = {'zoomregion':(170,355,15,80),
                  'hspace':-.1, 'cbar_vert':.1,
                  'subtitles':subtitles,
                  'subtitle_fontdict':{'fontsize':24},
                  'clevels':np.arange(-0.8, 0.9, .1),
                  'clabels':np.arange(-.8,.9,.2),
                  'units':'Correlation [-]',
                  'y_ticks':False, # np.arange(15,75,15),
                  'x_ticks':False,
                  'kwrgs_cbar' : {'orientation':'horizontal',
                                  'extend':'neither'},
                  'cbar_tick_dict':{'labelsize':30}}
    fg = plot_maps.plot_corr_maps(xr_merge, xr_mask,
                                  col_dim='lag', row_dim='var', **kwrgs_plot)
    facecolorocean = '#caf0f8' ; facecolorland='white'
    for ax in fg.fig.axes[:-1]:
        ax.add_feature(plot_maps.cfeature.__dict__['LAND'],
                       facecolor=facecolorland,
                       zorder=0)
        ax.add_feature(plot_maps.cfeature.__dict__['OCEAN'],
                       facecolor=facecolorocean,
                       zorder=0)

    fg.fig.savefig(os.path.join(path_circ,
                                f'SM_vs_circ_{rg.fc_month}'+rg.figext),
                    bbox_inches='tight')

    # #%%
    # if hasattr(sst, 'prec_labels')==False and 'sst' in use_vars:
    #     rg.cluster_list_MI('sst')
    #     sst.group_small_cluster(distance_eps_sc=2000, eps_corr=0.4)


    #     sst.prec_labels['lag'] = ('lag', periodnames)
    #     sst.corr_xr['lag'] = ('lag', periodnames)
    #     rg.quick_view_labels('sst', min_detect_gc=.5, save=save,
    #                           append_str=periodnames[-1])
    #     plt.close()



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



    # Run in Parallel
    lag_list = [lags_july, lags_june, lags_may, lags_april, lags_march]
    periodnames_list = [periodnames_july, periodnames_june,
                        periodnames_may, periodnames_april,
                        periodnames_march]



    if extra_lag:
        lag_list += [lags_feb, lags_jan]
        periodnames_list += [periodnames_feb, periodnames_jan]
        lag_list  = lag_list[::2] ;
        periodnames_list = periodnames_list[::2]

    futures = [] ; rg_list = []
    for lags, periodnames in zip(lag_list, periodnames_list):
        # if load == False:
        #     futures.append(delayed(pipeline)(lags, periodnames, load))

        # else:
        rg_list.append(pipeline(lags, periodnames, load))

    # if load == False:
        # rg_list = Parallel(n_jobs=n_cpu, backend='loky')(futures)
rg = rg_list[0]


