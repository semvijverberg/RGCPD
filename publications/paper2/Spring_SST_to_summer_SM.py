#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
# from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import matplotlib
# from sklearn import metrics
import pandas as pd
# import xarray as xr
# import sklearn.linear_model as scikitlinear
# import argparse

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
data_dir = os.path.join(main_dir,'publications/paper2/data')
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
import functions_pp; import df_ana
import plot_maps; import core_pp

# Optionally set font to Computer Modern to avoid common missing font errors
# matplotlib.rc('font', family='serif', serif='cm10')

# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

# region = 'eastern'
# targets = ['easternRW']



# expers = np.array(['fixed_corr', 'adapt_corr'])
# seeds = np.array([1,2,3])
# combinations = np.array(np.meshgrid(targets, expers, seeds)).T.reshape(-1,3)

# i_default = 2



# def parseArguments():
#     # Create argument parser
#     parser = argparse.ArgumentParser()

#     # Optional arguments
#     parser.add_argument("-i", "--intexper", help="intexper", type=int,
#                         default=i_default)
#     # Parse arguments
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parseArguments()
#     out = combinations[args.intexper]
#     target = out[0]
#     experiment = out[1]
#     seed = int(out[2])
#     if target[-4:]=='temp':
#         tfreq = 15
#     else:
#         tfreq = 60
#     print(f'arg {args.intexper} - Target {target}, Experiment {experiment}, tfreq {tfreq}')
# else:
#     target = targets[2]
#     tfreq = 60
#
#     experiment = 'adapt_corr'
#     seed = 1




# if target[-4:] == 'temp':
#     TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
#     alpha_corr = .01
#     cluster_label = 2
#     name_ds='ts'
#     if target == 'westerntemp':
#         cluster_label = 1
#     elif target == 'easterntemp':
#         cluster_label = 2
# elif target[-2:] == 'RW':
#     cluster_label = 'z500'
#     name_ds = f'0..0..{cluster_label}_sp'
#     alpha_corr = .05
#     if target == 'easternRW':
#
#     elif target == 'westernRW':
#         TVpath = os.path.join(data_dir, '2020-10-29_10hr_58min_west_RW.h5')

experiment = 'fixed_corr'
calc_ts='pattern cov' #
method     = 'ran_strat10' ; seed=1
n_boot = 5000

append_main = ''



#%% run RGPD
# spring SST correlated with RW
alpha_corr=.05
TVpathERW = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
start_end_TVdate = ('02-01', '05-31')
start_end_date = ('1-1', '12-31')
list_of_name_path = [('z500', TVpathERW),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.parcorr_map_time,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={'precursor':True},
                            distance_eps=1200, min_area_in_degrees2=10,
                            calc_ts=calc_ts, selbox=(160,260,10,60),
                            lags=np.array([0]))]

if calc_ts == 'region mean':
    s = ''
else:
    s = '_' + calc_ts.replace(' ', '')

path_out_main = os.path.join(main_dir, f'publications/paper2/output/easternRW{s}{append_main}/')

rgSST = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=None,
           tfreq=60,
           path_outmain=path_out_main,
           append_pathsub='_' + 'fixed_corr')

title = r'$parcorr(SST_t, mx2t_t\ |\ SST_{t-1},mx2t_{t-1})$'
subtitles = np.array([['']]) #, f'lag 2 (15 day lead)']] )
kwrgs_plotcorr = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', 'zoomregion':(160,260,10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

#%%

rgSST.pp_TV(name_ds= f'0..0..z500_sp', detrend=False)

subfoldername = '_'.join(['easternRW',rgSST.hash, experiment.split('_')[0],
                      str(rgSST.tfreq), str(alpha_corr), method,
                      str(seed)])

rgSST.pp_precursors()
rgSST.traintest(method=method, seed=seed, subfoldername=subfoldername)
rgSST.calc_corr_maps()
rgSST.cluster_list_MI()
rgSST.quick_view_labels(save=True, append_str=experiment)
# plotting corr_map
rgSST.plot_maps_corr(var='sst', save=True,
                  kwrgs_plot=kwrgs_plotcorr,
                  min_detect_gc=1.0,
                  append_str=experiment)
rgSST.get_ts_prec(1)
df_SST = rgSST.df_data.rename({rgSST.TV.name:'E-RW',
                              f'0..0..{rgSST.list_for_MI[0].name}_sp':'SST pattern'},
                              axis=1)
#%% Retrieve Soil Moisture regions connected to eastern U.S. temp
# spring SST correlated with RW
TVpathmx2t = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
calc_ts = 'region mean'
cluster_label = 2
name_ds='ts'
list_of_name_path = [(cluster_label, TVpathmx2t),
                     ('sm', os.path.join(path_raw, 'sm2_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sm', func=class_BivariateMI.parcorr_map_time,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={'precursor':True},
                            distance_eps=1200, min_area_in_degrees2=10,
                            calc_ts=calc_ts, selbox=(200,300,20,73),
                            lags=np.array([0]))]

if calc_ts == 'region mean':
    s = ''
else:
    s = '_' + calc_ts.replace(' ', '')

path_out_main = os.path.join(main_dir, f'publications/paper2/output/easterntemp{s}{append_main}/')

rgT = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=None,
           tfreq=15,
           path_outmain=path_out_main,
           append_pathsub='_' + experiment)

title = r'$parcorr(SM_t, mx2t_t\ |\ SM_{t-1},mx2t_{t-1})$'
subtitles = np.array([['']]) #, f'lag 2 (15 day lead)']] )
kwrgs_plotcorr = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', #'zoomregion':(130,260,-10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}


if experiment == 'fixed_corr':
    rgT.pp_TV(name_ds=name_ds, detrend=False)

    subfoldername = '_'.join(['easterntemp',rgT.hash, experiment.split('_')[0],
                          str(rgT.tfreq), str(alpha_corr), method,
                          str(seed)])

    rgT.pp_precursors()
    rgT.traintest(method=method, seed=seed, subfoldername=subfoldername)
    rgT.calc_corr_maps()
    rgT.cluster_list_MI()
    rgT.quick_view_labels(save=True, append_str=experiment)
    # plotting corr_map
    rgT.plot_maps_corr(var='sm', save=True,
                      kwrgs_plot=kwrgs_plotcorr,
                      min_detect_gc=1.0,
                      append_str=experiment)
rgT.get_ts_prec(precur_aggr=1)
df_T = rgT.df_data.rename({rgT.TV.name:'E-temp',
                              f'0..2..{rgT.list_for_MI[0].name}':'SM'},
                              axis=1)
#%%
df_sm_sst = df_SST[['SST pattern']].merge(df_T.iloc[:,1:],
                                          left_index=True, right_index=True)
df_sm_sst = df_sm_sst[['SST pattern', 'SM']].mean(axis=0, level=1)
df_ = df_sm_sst.resample('1M').mean()
df_['year'] = df_.index.year
df_.index = df_.index.month
quarters = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON',
    12: 'DJF'}
# can be grouped by year and quarters
df_ = df_.groupby(['year',quarters]).mean()

# df_['season'] = df_.index.get_level_values(1)


# df_ = df_sm_sst[['SST pattern', 'SM']].mean(axis=0, level=1).resample('Q-nov', closed='right').mean()
# dtfirst = [s+'-01' for s in df_.index.strftime('%Y-%m').values]
# df_.index = pd.to_datetime(dtfirst)
# df_['SM'] *= 1000
# monthmean = df_.groupby(df_.index.month).mean()
# monthstd  = df_.groupby(df_.index.month).std()
# for m in df_.index.month:
#     df_['SM'][df_.index.month ==m] = (df_['SM'][df_.index.month ==m] - monthmean['SM'].loc[m]) \
#         / monthstd['SM'].loc[m]

df_['year'] = df_.index.get_level_values(0)
df_['season'] = df_.index.get_level_values(1)
# df_ = (df_ - df_.mean())/ df_.std()
interannual_std = df_.groupby(df_['year']).mean().std()

# df_['month'] = df_.index.month_name()
# df_.groupby('month').mean() / df_.groupby('month').std()

# df_[['SST pattern']] * -1
#%%
import matplotlib.dates as mdates
fig, ax = plt.subplots(1,1, figsize=(8,4))

for i, yr in enumerate(np.unique(df_.index.year)):
    legend=False
    singleyeardates = core_pp.get_oneyr(df_.index, yr)
    df_yr = df_.loc[singleyeardates].iloc[:8]
    df_yr.index = df_yr.index.month_name()#core_pp.get_oneyr(df_.index)
    if df_yr['SST pattern'].mean() > 1*interannual_std['SST pattern']:
        alpha = 1
    else:
        alpha = .2
    df_yr['SST pattern'].plot(ax=ax, color='blue', linestyle='--',
                              legend=False, alpha=alpha)
    df_yr['SM'].plot(ax=ax, color='red', legend=False, alpha=alpha)
    ax.set_ylim(-3,3)
    ax.hlines(y=0.5, xmin=0.05, xmax=.95, transform=ax.transAxes)
    # ax.xaxis.set_major_locator(mdates.MonthLocator(1)) # set tick every year
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%M')) # format %Y
    # ax.plot(df_.loc[singleyeardates], alpha=.1, color='purple')

#%%
import matplotlib.dates as mdates
fig, ax = plt.subplots(1,1, figsize=(8,4))

var_winter = df_[['SST pattern']].reorder_levels((1,0), axis=0).loc[['DJF', 'MAM']]
var_winter = var_winter.groupby(var_winter.index.get_level_values(1)).mean()

dominant_yrs = var_winter[var_winter > var_winter.quantile(.8)].dropna().index#.levels[1]

allyrs = []
for i, yr in enumerate(dominant_yrs):
    legend=False
    df_yr = df_.loc[yr]
    df_yr = df_.loc[[yr-1, yr, yr+1]]
    years = df_yr.index.get_level_values(0) - yr
    season = df_yr.index.get_level_values(1)
    tuples = list(zip(*np.array([years, season])))
    df_yr.index = pd.MultiIndex.from_tuples(tuples, names=['year', 'seaon'])
    print(yr)
    alpha = 1
    # df_yr['SST pattern'].plot(ax=ax, color='blue', linestyle='--',
    #                       legend=False, alpha=alpha)
    df_yr['SM'].plot(ax=ax, color='red', legend=False, alpha=alpha)
    ax.set_ylim(-.1,.1)
    ax.hlines(y=0.5, xmin=0.05, xmax=.95, transform=ax.transAxes)
    ax.set_xticks(range(df_yr.index.values.size))
    xticklabels = ['{} {}'.format(*list(item)) for item in df_yr.index.tolist()]
    ax.set_xticklabels(xticklabels, rotation=-45);
    allyrs.append(list(df_yr['SM'].values))

