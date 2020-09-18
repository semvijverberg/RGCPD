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

# =============================================================================
#%% Hovmoller diagram
# =============================================================================
tfreq = 1
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
west_or_east = 'western'

path_out_main = os.path.join(main_dir, f'publications/paper2/output/{west_or_east}_HM/')



TVpathHMw = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/west/1ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-20_15hr_40min_df_data_z500_v200_dt1_0ff31_z500_145-325-20-62.h5'
TVpathHMe = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-20_15hr_22min_df_data_z500_v200_dt1_0ff31_z500_140-300-20-73.h5'

TVs = []
for TVpathHM in [TVpathHMw, TVpathHMe]:
    name_or_cluster_label = 'z500'
    name_ds = f'0..0..{name_or_cluster_label}_sp'

    list_of_name_path = [(name_or_cluster_label, TVpathHM)]

    rg = RGCPD(list_of_name_path=list_of_name_path,
               start_end_TVdate=start_end_TVdate,
               start_end_date=start_end_date,
               tfreq=tfreq, lags_i=np.array([0]),
               path_outmain=path_out_main,
               append_pathsub='_' + name_ds)

    rg.pp_TV(name_ds=name_ds)


    # rg.pp_precursors(anomaly=True)


    kwrgs_events = {'event_percentile':95, 'window':'mean',
                    'min_dur':1,'max_break':1, 'grouped':True}#,
    # kwrgs_events = {'event_percentile':66, 'window':'mean'}
                    # 'min_dur':7,'max_break':3, 'grouped':True,'reference_group':'center'}
    rg.traintest(method='no_train_test_split', kwrgs_events=kwrgs_events)

    TVs.append(rg.TV)

# event_dates = rg.TV.RV_bin[rg.TV.RV_bin.astype(bool).values].index
# one_ev_peryear = functions_pp.remove_duplicates_list(list(event_dates.year))[1]
# event_dates = event_dates[one_ev_peryear]
# event_vals = rg.TV.RV_ts.loc[event_dates]
# event_dates = event_vals.sort_values(by=event_vals.columns[0], ascending=False)[:21].index

#%%
durw = TVs[0].RV_dur[TVs[0].RV_dur!=0]
dure = TVs[1].RV_dur[TVs[1].RV_dur!=0]

fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.hist(durw, alpha=.6, label='western RW', density=True)
ax.hist(dure, alpha=.6, label='eastern RW', density=True)
plt.legend()

