#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:25:11 2019

@author: semvijverberg
"""
import sys, os, io


import functions_RGCPD as rgcpd
import itertools
import numpy as np
import xarray as xr
import datetime
import cartopy.crs as ccrs
import pandas as pd
import functions_pp
import core_pp

kwrgs_ENSO = {'tfreq' : 30,
              'method' : 'no_train_test_split'
              }
# get indices
def ENSO_34(file_path, ex, df_splits=None):
    #%%
    file_path = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/sst_1979-2018_1_12_daily_2.5deg.nc'
    
    
    kwrgs_pp = {'selbox' :  {'la_min':-10, # select domain in degrees east
                             'la_max':10,
                             'lo_min':-10,
                             'lo_max':-60}}
    
    ds = core_pp.import_ds_lazy(file_path, **kwrgs_pp)
    
    to_freq = ex['tfreq']
    if to_freq != 1:        
        ds, dates = functions_pp.time_mean_bins(ds, ex, to_freq=to_freq, seldays='all')
        ds['time'] = dates
    
    if df_splits is None:
        RV = ex[ex['RV_name']]
        df_splits, ex = functions_pp.rand_traintest_years(RV, ex)
            
    splits = df_splits.index.levels[0]
    
    list_splits = []
    for s in splits:
        
        progress = 100 * (s+1) / splits.size

        dates_RV  = pd.to_datetime(RV.RV_ts.time.values)
        n = dates_RV.size ; r = int(100*n/RV.dates_RV.size )
        print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")
        
        data = ds.mean(dim=('latitude', 'longitude'))
        list_splits.append(pd.DataFrame(data=data.values, 
                                     index=ds['time'], columns=['ENSO_34']))
    
    df_ENSO = pd.concat(list_splits, axis=0, keys=splits)
    #%%
    return df_ENSO

    #%%
    
