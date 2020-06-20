#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:21:41 2020

@author: semvijverberg
"""
import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs

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
import functions_pp



path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')

start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 15

# =============================================================================
#%% Hovmoller diagram
# =============================================================================

# the RW
# TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/1ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-06-11_15hr_16min_df_data_z500_dt1_0ff31.h5'
# name_or_cluster_label = 'z500'
# name_ds='0..0..z500_sp'

# Temperature
TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
name_or_cluster_label = 1
name_ds='ts'

list_of_name_path = [(name_or_cluster_label, TVpath),
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

rg.pp_TV(name_ds=name_ds)
greenrectangle_WestUS_bb = (140,325,24,62)
wide_WestUS_bb = (0,360,0,73)
rg.pp_precursors(selbox=wide_WestUS_bb, anomaly=True)


kwrgs_events = {'event_percentile':85, 'window':'mean'}#,
                # 'min_dur':7,'max_break':3, 'grouped':True,'reference_group':'center'}
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

rg.calc_corr_maps()


subtitles = np.array([['Western U.S. one-point correlation map Z 500hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='z500', aspect=2, size=5, cbar_vert=.19, save=False,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', greenrectangle_WestUS_bb],
                  clim=(-.6,.6))



event_dates = rg.TV.RV_bin[rg.TV.RV_bin.astype(bool).values].index
one_ev_peryear = functions_pp.remove_duplicates_list(list(event_dates.year))[1]
event_dates = event_dates[one_ev_peryear]

#%%
from class_hovmoller import Hovmoller
kwrgs_load = rg.kwrgs_load.copy()
kwrgs_load['tfreq'] = 1
HM = Hovmoller(kwrgs_load=kwrgs_load, event_dates=event_dates,
               seldates=rg.TV.aggr_to_daily_dates(rg.dates_TV), standardize=True, lags_prior=35,
               lags_posterior=35, rollingmeanwindow=5, zoomdim=(15,55))
self = HM
HM.get_HM_data(rg.list_precur_pp[0][1])


fname1 = f'HM_{self.name}'+'_'.join(['{}_{}'.format(*ki) for ki in kwrgs_events.items()])
fname2 = '_'.join(np.array(HM.kwrgs_load['selbox']).astype(str)) + \
                    f'_w{self.rollingmeanwindow}_std{self.standardize}.pdf'
fig_path = os.path.join(rg.path_outsub1, '_'.join([fname1, fname2]))
HM.plot_HM(clevels=np.arange(-.5, .51, .1), height_ratios=[1.5,6],
           fig_path=fig_path)

# HM.quick_HM_plot()