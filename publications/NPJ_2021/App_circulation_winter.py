#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:44:42 2020

Step for Appendix B to reproduce results of NPJ paper:
"The role of the Pacific Decadal Oscillation and
ocean-atmosphere interactions in driving US temperature variability"

It loads the RW and temperature timeseries that are stored by step 2.
This script generates and plot the results for Appendix B.

@author: semvijverberg
"""
import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
data_dir = os.path.join(main_dir,'publications/NPJ_2021/data')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
from RGCPD import EOF
import class_BivariateMI
import climate_indices
import plot_maps, core_pp, df_ana, functions_pp




west_east = 'west'
adapt_selbox = False
TV = 'USCAnew'

if west_east == 'east':
    path_out_main = os.path.join(main_dir, 'publications/NPJ_2021/output/east/')
    z500_green_bb = (155,300,20,73) #: RW box
elif west_east == 'west':
    path_out_main = os.path.join(main_dir, 'publications/NPJ_2021/output/west/')
    z500_green_bb = (145,325,20,62) # bounding box for western RW

if TV == 'init':
    TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    if west_east == 'east':
        cluster_label = 2
    elif west_east == 'west':
        cluster_label = 1
elif TV == 'USCAnew':
    # mx2t 25N-70N
    TVpath = os.path.join(data_dir, 'tfreq15_nc7_dendo_57db0USCA.nc')
    # TVpath = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf15_nc8_dendo_57db0USCA.nc'
    if west_east == 'east':
        cluster_label = 4
    elif west_east == 'west':
        cluster_label = 1 # 8
    elif west_east == 'northwest':
        cluster_label = 7
elif TV == 'USCA':
    TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc5_dendo_5dbee_USCA.nc'
    # TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc8_dendo_5dbee_USCA.nc'
    if west_east == 'east':
        cluster_label = 1
    elif west_east == 'west':
        cluster_label = 5
        cluster_label = 4

if west_east == 'east':
    # TVpathRW = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
    cluster_label = 4 # 2


name_ds='ts'

# if period == 'summer_center':
start_end_TVdate = ('06-01', '08-31')
start_end_date = None
method='ranstrat_10' ; seed = 1
tfreq = 15
min_detect_gc=.9
z500_selbox = (0,360,-10,90)
path_output = os.path.join(curr_dir, 'output/PDO_and_RW_timeseries/')
os.makedirs(path_output, exist_ok=True)

#%% Circulation vs temperature


list_of_name_path = [(cluster_label, TVpath),
                      ('v300', os.path.join(path_raw, 'v300_1979-2020_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500_1979-2020_1_12_daily_2.5deg.nc'))]


list_import_ts = None #[('W-RW', os.path.join(data_dir, f'westRW_{period}_s{seed}.h5')),
                   # ('E-RW', os.path.join(data_dir, f'eastRW_{period}_s{seed}.h5'))]

list_for_MI   = None # [BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                                # alpha=.05, FDR_control=True,
                                # distance_eps=600, min_area_in_degrees2=1,
                                # calc_ts='pattern cov', selbox=z500_green_bb,
                                # use_sign_pattern=True,
                                # lags=np.array([0]))]


rg_list = []
start_end_TVdates = [('12-01', '02-28'),
                     ('02-01', '05-30'),
                     ('06-01', '08-31')]
for start_end_TVdate in start_end_TVdates:

    list_for_EOFS = [EOF(name='z500', neofs=1, selbox=z500_green_bb,
                        n_cpu=1, start_end_date=start_end_TVdate)]

    list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=z500_selbox,
                                use_sign_pattern=True,
                                lags = np.array([0]), n_cpu=2)]

    rg = RGCPD(list_of_name_path=list_of_name_path,
                list_for_MI=list_for_MI,
                list_for_EOFS=list_for_EOFS,
                list_import_ts=list_import_ts,
                start_end_TVdate=start_end_TVdate,
                start_end_date=start_end_date,
                start_end_year=None,
                tfreq=tfreq,
                path_outmain=path_out_main,
                append_pathsub='_' + name_ds)


    rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=True)

    rg.pp_precursors()

    rg.traintest(method, seed=seed,
                 subfoldername='comparison_winter_summer_circulation')

    rg.calc_corr_maps()
    rg_list.append(rg)

#%%
dfs = []
for rg in rg_list:
    precur = rg.list_for_MI[0]
    precur.tfreq = 15
    precur.selbox = z500_green_bb
    rg.kwrgs_load['selbox'] = z500_green_bb
    rg.pp_precursors()
    rg.calc_corr_maps()
    rg.cluster_list_MI()

    precur.get_prec_ts(1)
    dfs.append(pd.concat(precur.ts_corr, keys=range(rg.n_spl)))

season_names = ['winter', 'spring', 'summer']

dfs = [dfs[i].rename({'0..0..z500_sp':season_names[i]},axis=1) for i in range(3)]
#%%
corr = xr.concat([rg.list_for_MI[0].corr_xr.copy() for rg in rg_list],
                 dim='lag')
mask = xr.concat([rg.list_for_MI[0].corr_xr['mask'].copy() for rg in rg_list],
                 dim='lag')
corr['lag'] = ('lag', np.arange(0,3))
mask['lag'] = ('lag', np.arange(0,3))

corr, mask = RGCPD._get_sign_splits_masked(corr, min_detect=0.9, mask=mask)


import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

# Adjusted box for Reviewer 1, z500_green_bb = (155,255,20,73) #: RW box
subtitles = [['winter (DJF)'], ['spring (MAM)'], ['summer (JJA)']]
drawbox = ['all', z500_green_bb]
# drawbox = [[(0,i), z500_green_bb] for i in range(len(subtitles))]
title = f'$corr(z500_t, T^{west_east.capitalize()[0]}_t)$'

kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':3.8, 'size':2.5,
              'hspace':0.2, 'cbar_vert':.01, 'units':'Corr. Coeff. [-]',
              'zoomregion':(-180,360,0,80), 'drawbox':drawbox,
              'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.array([10,30,50,70,90]),
              'clim':(-.6,.6), 'title':title, 'subtitles':subtitles,
              'title_fontdict':{'y':0.96, 'fontsize':18}}

g = plot_maps.plot_corr_maps(corr, mask, **kwrgs_plot)

g.fig.savefig(os.path.join(path_output, 'z500_vs_T_seasonal_dependence.jpg'), dpi=300,
          bbox_inches='tight')
g.fig.savefig(os.path.join(path_output, 'z500_vs_T_seasonal_dependence.pdf'),
          bbox_inches='tight')
#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
for rg in rg_list:
    rg.get_EOFs()
    rg.plot_EOFs()
#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

load = functions_pp.load_hdf5
df_PDO = load(os.path.join(data_dir, 'df_PDOs_monthly.h5'))['df_data'][['PDO']]
df_RWE = load(os.path.join(data_dir, 'eastRW_spring_center_s1_RepeatedKFold_10_7_tf1.h5'))
df_RWE = df_RWE['df_data'][['0..0..z500_sp']].mean(axis=0,level=1)


# annual mean
df_PDO_am = df_PDO.groupby(df_PDO.index.year).mean()
df_PDO_am = df_PDO_am.rename({'PDO':'$PDO_{annual}$'}, axis=1)

#%%%


start_end_TVdates = [None,
                     ('12-01', '02-28'),
                     ('02-01', '05-30'),
                     ('06-01', '08-31'),
                     ('09-01', '11-30')]

seasons = ['{annual}', '{DJF}', '{MAM}', '{JJA}', '{OND}']

start_end_TVdates = [None,
                     ('12-01', '02-28'),
                     ('02-01', '05-30'),
                     ('06-01', '08-31')]

seasons = ['{annual}', '{DJF}', '{MAM}', '{JJA}']

f, ax = plt.subplots(len(seasons), figsize=(10,18), sharex=True)
only_summer = True

for p, startenddate in enumerate(start_end_TVdates):




    if only_summer:
        seldates = core_pp.get_subdates(df_RWE.index,
                                        start_end_date=startenddate)
        df_RWE_am = df_RWE.loc[seldates].groupby(seldates.year).mean()
    else:
        idx = max(0, p-1)
        _d = dfs[idx].mean(axis=0, level=1)
        seldates = core_pp.get_subdates(_d.index,
                                        start_end_date=startenddate)
        df_RWE_am = _d.loc[seldates].groupby(seldates.year).mean()


    seas = seasons[p]
    RWcolname = df_RWE_am.columns[0]
    df_RWE_am  = df_RWE_am.rename({RWcolname: f'$RW_{seas}^E$'}, axis=1)
    df_merge = df_PDO_am.merge(df_RWE_am, left_index=True, right_index=True)
    df_merge  = (df_merge - df_merge.mean(0) ) / df_merge.std(0)




    # colors = ["fb5607", "ffbe0b"]

    cmp = plot_maps.get_continuous_cmap(["3a86ff", "fb5607"],
                              float_list=list(np.linspace(0,1,2)))


    for i, col in enumerate(df_merge.columns):
        ax[p].plot(df_merge.index, df_merge[col], color=cmp(i*256), label=col)
        ax[p].scatter(df_merge.index, df_merge[col], color=cmp(i*256))

    corrval = round(df_merge.corr().values[0,1],2)
    ax[p].text(0.95,0.05, f'Correlation = {corrval}',
               horizontalalignment='right',
               transform=ax[p].transAxes, fontsize=14)
    ax[p].axhline(color='black')
    ax[p].legend(loc='lower left', fontsize=14)
    if only_summer:
        sub = f'Annual mean PDO and {seas[1:-1]} mean $RW^E$ (based upon summer pattern)'
    else:
        if p == 0 and only_summer==False: seas='{DJF}'
        sub = f'Annual mean PDO and {seas[1:-1]} mean $RW^E$ (based upon {seas[1:-1]} pattern)'
    ax[p].set_title(sub, fontsize=14)
    ax[p].tick_params(labelsize=14)

f.savefig(os.path.join(path_output, f'PDO_vs_RW_only_summer{only_summer}.jpg'), dpi=300,
          bbox_inches='tight')
f.savefig(os.path.join(path_output, f'PDO_vs_RW_only_summer{only_summer}.pdf'), bbox_inches='tight')

