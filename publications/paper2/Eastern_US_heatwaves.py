#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

@author: semvijverberg
"""


#%%
import os, inspect, sys
import numpy as np

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
# from RGCPD import EOF



TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/east/')
cluster_label = 2
name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
tfreq = 15
#%%
list_of_name_path = [(cluster_label, TVpath),
                      ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]



list_for_MI   = [BivariateMI(name='v200', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.01, 'FDR_control':True},
                              distance_eps=600, min_area_in_degrees2=1,
                              calc_ts='pattern cov'),
                   BivariateMI(name='z500', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.01, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov')]

list_for_EOFS = None #[EOF(name='v200', neofs=2, selbox=[-180, 360, 10, 90],
                     # n_cpu=1)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            list_for_EOFS=list_for_EOFS,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0]),
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors(selbox=(0,360,10,90))

rg.traintest('no_train_test_split')

import cartopy.crs as ccrs
rg.calc_corr_maps()
# rg.get_EOFs()



subtitles = np.array([['Eastern U.S. one-point correlation map v-wind 200hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='v200', aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  clim=(-.6,.6))

z500_green_bb = (170,250,23,73)
subtitles = np.array([['Eastern U.S. one-point correlation map Z 500hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='z500', aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', z500_green_bb],
                  clim=(-.6,.6))

#%% Determine Rossby wave within green rectangle, become target variable for feedback

list_of_name_path = [(cluster_label, TVpath),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov')]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)


rg.pp_precursors(selbox=z500_green_bb, anomaly=True)
rg.pp_TV(name_ds=name_ds)

rg.traintest(method='no_train_test_split')

rg.calc_corr_maps()
subtitles = np.array([['E-U.S. Temp. correlation map Z 500hpa green box']])
rg.plot_maps_corr(var='z500', cbar_vert=-.05, subtitles=subtitles, save=False)
rg.cluster_list_MI(var='z500')
# rg.get_ts_prec(precur_aggr=None)
rg.get_ts_prec(precur_aggr=1)
rg.store_df()
#%% Determine Rossby wave within green rectangle, become target variable for HM

list_of_name_path = [(cluster_label, TVpath),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov')]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)


rg.pp_precursors(selbox=(170,300,15,73), anomaly=True)
rg.pp_TV(name_ds=name_ds)

rg.traintest(method='no_train_test_split')

rg.calc_corr_maps()
subtitles = np.array([['E-U.S. Temp. correlation map Z 500hpa green box']])
rg.plot_maps_corr(var='z500', cbar_vert=-.05, subtitles=subtitles, save=False)
rg.cluster_list_MI(var='z500')
# rg.get_ts_prec(precur_aggr=None)
rg.get_ts_prec(precur_aggr=1)
rg.store_df()


#%%
import class_RV
RV_ts = rg.fulltso.sel(time=rg.TV.aggr_to_daily_dates(rg.dates_TV))
threshold = class_RV.Ev_threshold(RV_ts, event_percentile=85)
RV_bin, np_dur = class_RV.Ev_timeseries(RV_ts, threshold=threshold, grouped=True)
plt.hist(np_dur[np_dur!=0])
