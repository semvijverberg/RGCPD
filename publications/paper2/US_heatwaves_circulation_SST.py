#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
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
from RGCPD import EOF
import class_BivariateMI
import climate_indices
import plot_maps, core_pp, df_ana

west_east = 'west'
TV = 'US'
if TV == 'init':
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    if west_east == 'east':
        cluster_label = 2
    elif west_east == 'west':
        cluster_label = 1
elif TV == 'US':
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf15_nc6_dendo_0cbf8_US.nc'
    if west_east == 'east':
        cluster_label = 3
    elif west_east == 'west':
        cluster_label = 1
elif TV == 'USCA':
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc5_dendo_5dbee_USCA.nc'
    if west_east == 'east':
        cluster_label = 1
    elif west_east == 'west':
        cluster_label = 5
        cluster_label = 5

if west_east == 'east':
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/east/')
elif west_east == 'west':
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')





name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = None
method='ranstrat_10' ; seed = 1
tfreq = 15
min_detect_gc=.9
start_end_year = (1979, 2018)

# z500_green_bb = (140,260,20,73) #: Pacific box
if west_east == 'east':
    z500_green_bb = (155,300,20,73) #: RW box
    v300_green_bb = (170,359,23,73)
elif west_east == 'west':
    z500_green_bb = (145,325,20,62)
    v300_green_bb = (100,330,24,70)

#%% Circulation vs temperature
list_of_name_path = [(cluster_label, TVpath),
                      ('v300', os.path.join(path_raw, 'v300hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

lags = np.array([0])

list_for_MI   = [BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                              alpha=.05, FDR_control=True, lags=lags,
                              distance_eps=600, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=(0,360,-10,90),
                              use_sign_pattern=True),
                 BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=(0,360,-10,90),
                                use_sign_pattern=True)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=start_end_year,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors()

rg.traintest(method=method, seed=seed,
             subfoldername='US_heatwave_circulation_v300_z500_SST')



#%%

rg.calc_corr_maps()

#%% Plot corr(z500, mx2t)
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']



title = f'$corr(z500_t, T^{west_east.capitalize()[0]}_t)$'
subtitles = np.array([['']] )
kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':3.8, 'size':2.5,
              'hspace':0.0, 'cbar_vert':-.08, 'units':'Corr. Coeff. [-]',
              'zoomregion':(-180,360,0,80), 'drawbox':[(0,0), z500_green_bb],
              'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6,
              'clim':(-.6,.6), 'title':title, 'subtitles':subtitles}
save = True
rg.plot_maps_corr(var='z500', save=save,
                  append_str=''.join(map(str, z500_green_bb)),
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)

#%% Plot corr(v300, mx2t)


kwrgs_plot['title'] = f'$corr(v300_t, T^{west_east.capitalize()[0]}_t)$'
kwrgs_plot['drawbox'] = [(0,0), v300_green_bb]
rg.plot_maps_corr(var='v300', save=save,
                  kwrgs_plot=kwrgs_plot,
                  min_detect_gc=min_detect_gc)

#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#%% Determine Rossby wave within green rectangle, become target variable for feedback

rg.list_for_MI = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=z500_green_bb,
                                use_sign_pattern=True, lags = np.array([0]))]
rg.list_for_EOFS = None
rg.calc_corr_maps(['z500'])
rg.cluster_list_MI(['z500'])
rg.get_ts_prec(precur_aggr=1)
rg.store_df(append_str='z500_'+'-'.join(map(str, z500_green_bb))+TV)

#%% SST vs T
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

lags = np.array([0,2])

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(120,260,-10,90))]


rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=None,
            start_end_year=start_end_year,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(name_ds=name_ds, detrend=False)
rg.traintest(method, seed=seed,
             subfoldername='US_heatwave_circulation_v300_z500_SST')
rg.pp_precursors()
rg.calc_corr_maps()


#%% Plot corr(SST, T)
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

save=True
SST_green_bb = (140,235,20,59)#(170,255,11,60)
# subtitles = np.array([[f'lag {l}: SST vs E-U.S. mx2t' for l in rg.lags]])
title = r'$corr(SST_{t-lag},$'+f'$T^{west_east.capitalize()[0]}_t)$'
subtitles = np.array([['lag 0', f'lag 2 (15-day gap)']] )
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.6,.6), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'n_yticks':6, 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title}
rg.plot_maps_corr(var='sst', save=save,
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)

#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)