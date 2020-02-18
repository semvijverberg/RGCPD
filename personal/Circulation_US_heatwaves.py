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
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'

from RGCPD import RGCPD

TVpath = '/Users/semvijverberg/surfdrive/cluster/surfdrive/xrclustered_c66a4.nc'
list_of_name_path = [(2, TVpath), 
                     ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]


start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')
kwrgs_corr = {'alpha':1E-2}

rg = RGCPD(list_of_name_path=list_of_name_path, 
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([0,1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW')

rg.pp_TV()
rg.pp_precursors(selbox=(-180, 360, -10, 90))

rg.traintest('no_train_test_split')

rg.calc_corr_maps(**kwrgs_corr) 

rg.plot_maps_corr(save=True)


#%%
from RGCPD import RGCPD
TVpath = '/Users/semvijverberg/surfdrive/cluster/surfdrive/xrclustered_c66a4.nc'
list_of_name_path = [(2, TVpath), 
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm2', os.path.join(path_raw, 'sm2_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm3', os.path.join(path_raw, 'sm3_1979-2018_1_12_daily_1.0deg.nc'))]


start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')
kwrgs_corr = {'alpha':1E-3}

rg = RGCPD(list_of_name_path=list_of_name_path, 
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW')

rg.pp_TV()
selbox = [None, {'sst':[-180,360,-10,90]}]
rg.pp_precursors(selbox=selbox)

rg.traintest(method='random10')

rg.calc_corr_maps(alpha=1E-3) 
rg.cluster_regions(distance_eps=700, min_area_in_degrees2=5)
rg.quick_view_labels() 
rg.get_ts_prec(precur_aggr=1)

rg.df_data

rg.store_df()