#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:19:10 2020

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
import pandas as pd



# =============================================================================
#%% Hovmoller diagram
# =============================================================================
tfreq = 15
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
west_or_east = 'eastern'

path_out_main = os.path.join(main_dir, f'publications/paper2/output/{west_or_east}_HM/')


if west_or_east == 'western':
    TVpathHM = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/west/1ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-20_15hr_40min_df_data_z500_v200_dt1_0ff31_z500_145-325-20-62.h5'
elif west_or_east == 'eastern':
    TVpathHM = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-20_15hr_22min_df_data_z500_v200_dt1_0ff31_z500_140-300-20-73.h5'


name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'

list_of_name_path = [(name_or_cluster_label, TVpathHM),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('v300',os.path.join(path_raw, 'v300hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov', use_sign_pattern=True),
                 BivariateMI(name='v300', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov', use_sign_pattern=True),
                 BivariateMI(name='sst', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov', selbox=(0,360,0,73))]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)

rg.pp_TV(name_ds=name_ds)


rg.pp_precursors(anomaly=True)


kwrgs_events = {'event_percentile':85, 'window':'mean',
                'min_dur':1,'max_break':1, 'grouped':True}#,
# kwrgs_events = {'event_percentile':66, 'window':'mean'}
                # 'min_dur':7,'max_break':3, 'grouped':True,'reference_group':'center'}
rg.traintest(method='no_train_test_split', kwrgs_events=kwrgs_events)

rg.calc_corr_maps()

greenrectangle_EastUS_bb = (170,300,15,73)
subtitles = np.array([[f'z-500 hPa vs {west_or_east} U.S. RW ({name_or_cluster_label})']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='z500', aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', greenrectangle_EastUS_bb],
                  clim=(-.6,.6))

subtitles = np.array([[f'SST vs {west_or_east} U.S. RW ({name_or_cluster_label})']])
rg.plot_maps_corr(var='sst', aspect=2, size=5, cbar_vert=.17, save=False,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  clim=(-.6,.6))


greenrectangle_EastUS_bb = (170,300,15,73)
subtitles = np.array([[f'v-300 hPa vs {west_or_east} U.S. RW ({name_or_cluster_label})']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='v300', aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', greenrectangle_EastUS_bb],
                  clim=(-.6,.6))
# rg.cluster_list_MI()

# rg.get_ts_prec()


event_dates = rg.TV.RV_bin[rg.TV.RV_bin.astype(bool).values].index
one_ev_peryear = functions_pp.remove_duplicates_list(list(event_dates.year))[1]
event_dates = event_dates[one_ev_peryear]
event_vals = rg.TV.RV_ts.loc[event_dates]
event_dates = event_vals.sort_values(by=event_vals.columns[0], ascending=False)[:21].index

#%%
from class_hovmoller import Hovmoller
var, filepath = rg.list_precur_pp[1];
if var != 'sst':
    var = var[0]+'-'+var[1:] + ' hPa'
rollingmeanwindow = 10
if name_or_cluster_label == 'z500':
    name = f'Hovmoller plot {var} (using {rollingmeanwindow}-day rolling mean)'
if west_or_east == 'western':
    zoomdim=(25,55)
    lag_composite = 0
elif west_or_east == 'eastern':
    zoomdim=(25,60)
    lag_composite = 0
kwrgs_load = rg.kwrgs_load.copy()
kwrgs_load['selbox'] = rg.list_for_MI[-1].selbox # selbox of SST
kwrgs_load['tfreq'] = 1
HM = Hovmoller(name=var, kwrgs_load=kwrgs_load, event_dates=event_dates,
               seldates=rg.TV.aggr_to_daily_dates(rg.dates_TV, tfreq=tfreq),
               standardize=True, lags_prior=35, lags_posterior=35,
               rollingmeanwindow=rollingmeanwindow,
               zoomdim=zoomdim, ignore_overlap_events=False)
self = HM
HM.get_HM_data(filepath, dim='latitude')
# HM.quick_HM_plot()

fname1 = f'HM_{self.name}'+'_'.join(['{}_{}'.format(*ki) for ki in kwrgs_events.items()])
fname2 = '_'.join(np.array(HM.kwrgs_load['selbox']).astype(str)) + \
                    f'_w{self.rollingmeanwindow}_std{self.standardize}_' + \
                    f'lag{lag_composite}_Evtfreq{rg.tfreq}'
fig_path = os.path.join(rg.path_outsub1, '_'.join([fname1, fname2]))
HM.plot_HM(clevels=np.arange(-.5, .51, .1), height_ratios=[1.5,6],
           fig_path=fig_path, lag_composite=lag_composite)

#%% Snap shots composite means
import plot_maps
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
ds_sm = self.ds_seldates.rolling(time=5).mean()
ds_raw_e = ds_sm.sel(time=np.concatenate(self.event_lagged))

xarray = ds_raw_e.copy().rename({'time':'lag'})
xarray = xarray.assign_coords(lag=np.concatenate(self.lag_axes))
xarray = xarray / self.ds_seldates.std(dim='time')

xr_snap = xarray.groupby('lag').mean().sel(lag=np.arange(-20,21,5))
kwrgs_plot = {'y_ticks':np.arange(0,61, 20),
              'map_proj':ccrs.PlateCarree(central_longitude=180),
              'hspace':.2, 'cbar_vert':.05,
              'clevels':np.arange(-.5, .51, .1)}
plot_maps.plot_corr_maps(xr_snap, row_dim='lag', col_dim='split',
                         **kwrgs_plot)
#%% Correlation PNA-like RW with Wavenumber 6 phase 2 # only for eastern
import core_pp, find_precursors
values = []
if west_or_east == 'eastern':
    lags_list = range(-10,10)
    for lag in lags_list:
        selbox = (0,360,25,60)
        # selbox = (140,300,20,73)
        tfreq = 1
        # lag = 0
        dates_RV = core_pp.get_subdates(pd.to_datetime(rg.fulltso.time.values),
                                       start_end_date=rg.start_end_TVdate)
        RV_ts = rg.fulltso.sel(time=dates_RV)
        ds_v300 = core_pp.import_ds_lazy(rg.list_precur_pp[1][1])
        dslocal = core_pp.get_selbox(ds_v300, selbox=selbox)



        datesRW = core_pp.get_subdates(pd.to_datetime(dslocal.time.values),
                                       start_end_date=rg.start_end_TVdate)
        datesRW = datesRW + pd.Timedelta(f'{lag}d')
        dslocal = dslocal.sel(time=datesRW)

        wv6local = core_pp.get_selbox(xarray.sel(lag=5), selbox=selbox)
        patternlocal = wv6local.mean(dim='lag')
        ts = find_precursors.calc_spatcov(dslocal, patternlocal)
        ts_15, d = functions_pp.time_mean_bins(ts, tfreq, start_end_date=start_end_TVdate,
                                                   closed_on_date=start_end_TVdate[-1])
        RV_15, d = functions_pp.time_mean_bins(RV_ts, tfreq, start_end_date=start_end_TVdate,
                                                   closed_on_date=start_end_TVdate[-1])
        corr_value = np.corrcoef(ts_15.values.squeeze(), RV_15.values.squeeze())[0][1]
        print('corr: {:.2f}'.format(corr_value))
        values.append(corr_value)
    plt.plot(range(-10,10), values)
    # df_wv6 = ts_15.to_dataframe(name='wv6p2')
#%%
sst = rg.list_for_MI[2]

dates_years = functions_pp.get_oneyr(sst.df_splits.loc[0].index, *event_dates.year)
sst.precur_arr.sel(time=dates_years).mean(dim='time').plot(vmin=-.3, vmax=.3,
                                                           cmap=plt.cm.RdBu_r)
