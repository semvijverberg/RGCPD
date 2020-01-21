#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:25:11 2019

@author: semvijverberg
"""
import sys, os, io


import find_precursors
from time import time
import numpy as np
import xarray as xr
#import datetime
#import cartopy.crs as ccrs
import pandas as pd
import functions_pp
from concurrent.futures import ProcessPoolExecutor
#import func_CPPA # temp
import core_pp 

kwrgs_ENSO = {'tfreq' : 30,
              'method' : 'no_train_test_split'
              }
# get indices
def ENSO_34(filepath, df_splits=None):
    #%%
#    file_path = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/sst_1979-2018_1_12_daily_2.5deg.nc'
    '''
    See http://www.cgd.ucar.edu/staff/cdeser/docs/deser.sstvariability.annrevmarsci10.pdf    
    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    '''
#    if df_splits is None:
#        RV = ex[ex['RV_name']]
#        df_splits, ex = functions_pp.rand_traintest_years(RV, ex)
#        seldates = None
#    else:
    seldates = df_splits.loc[0].index

#    {'la_min':-5, # select domain in degrees east
#     'la_max':5,
#     'lo_min':-170,
#     'lo_max':-120},
    kwrgs_pp = {'selbox' :  (190, 240, -5, 5),
                'format_lon': 'only_east',
                'seldates': seldates}

    
    ds = core_pp.import_ds_lazy(filepath, **kwrgs_pp)
    
#    to_freq = ex['tfreq']
#    if to_freq != 1:        
#        ds, dates = functions_pp.time_mean_bins(ds, ex, to_freq=to_freq, seldays='all')
#        ds['time'] = dates
    
    dates = pd.to_datetime(ds.time.values)            
    splits = df_splits.index.levels[0]
    
    list_splits = []
    for s in splits:
        
        progress = 100 * (s+1) / splits.size
        print(f"\rProgress ENSO traintest set {progress}%)", end="")

        
        data = functions_pp.area_weighted(ds).mean(dim=('latitude', 'longitude'))
        
        list_splits.append(pd.DataFrame(data=data.values, 
                                     index=dates, columns=['ENSO34']))
    
    df_ENSO = pd.concat(list_splits, axis=0, keys=splits)
    #%%
    return df_ENSO

    #%%

def PDO_single_split(s, ds_monthly, ds, df_splits):
    
    splits = df_splits.index.levels[0]
    progress = 100 * (s+1) / splits.size
    dates_train_origtime = df_splits.loc[s]['TrainIsTrue'][df_splits.loc[s]['TrainIsTrue']].index
    dates_test_origtime  = df_splits.loc[s]['TrainIsTrue'][~df_splits.loc[s]['TrainIsTrue']].index

    n = dates_train_origtime.size ; r = int(100*n/df_splits.loc[s].index.size )
    print(f"\rProgress PDO traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")
    
    # convert Train test year from original time to monthly 
    train_yrs = np.unique(dates_train_origtime.year)
    dates_monthly = pd.to_datetime(ds_monthly.time.values)
    dates_all_train = pd.to_datetime([d for d in dates_monthly if d.year in train_yrs])
    
    
    
    PDO_pattern, solver, adjust_sign = get_PDO(ds_monthly.sel(time=dates_all_train))
    data_train = find_precursors.calc_spatcov(ds.sel(time=dates_train_origtime), PDO_pattern)
    df_train = pd.DataFrame(data=data_train.values, index=dates_train_origtime, columns=['PDO'])   
    
    data_test = find_precursors.calc_spatcov(ds.sel(time=dates_test_origtime), PDO_pattern)
    df_test = pd.DataFrame(data=data_test.values, index=dates_test_origtime, columns=['PDO'])
         
    
    df = pd.concat([df_test, df_train]).sort_index()
    return (df, PDO_pattern)

def PDO(filepath, df_splits):
    #%%
    '''
    PDO is calculated based upon all data points in the training years,
    Subsequently, the PDO pattern is projection on the sst.sel(time=dates_train)
    to enable retrieving the PDO timeseries on a subset on the year.
    It is similarly also projected on the dates_test
    From https://climatedataguide.ucar.edu/climate-data/pacific-decadal-oscillation-pdo-definition-and-indices
    See http://www.cgd.ucar.edu/staff/cdeser/docs/deser.sstvariability.annrevmarsci10.pdf
    
    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    '''
    t0 = time()


#    old format selbox
#    {'la_min':20, # select domain in degrees east
#     'la_max':65,
#     'lo_min':115,
#     'lo_max':250},        
        
    kwrgs_pp = {'selbox' :  (115, 250, 20, 65),
                'format_lon': 'only_east'}    



    ds = core_pp.import_ds_lazy(filepath, **kwrgs_pp)
    # ensure latitude is in increasing order
    if np.where(ds.latitude == ds.latitude.min()) > np.where(ds.latitude == ds.latitude.max()):
        ds = ds.sortby('latitude')
    ds_monthly = ds.resample(time='M',restore_coord_dims=False).mean(dim='time', skipna=True)
    
    

        
    splits = df_splits.index.levels[0]
    data = np.zeros( (splits.size, ds.latitude.size, ds.longitude.size) )
    PDO_patterns = xr.DataArray(data, 
                                coords=[splits, ds.latitude.values, ds.longitude.values],
                                dims = ['split', 'latitude', 'longitude'])    

    
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            futures = [pool.submit(PDO_single_split, s, ds_monthly, ds, df_splits) for s in range(splits.size)]
            results = [future.result() for future in futures]
    except:
        results = [PDO_single_split(s, ds_monthly, ds, df_splits) for s in range(splits.size)]
    
    
    list_PDO_ts = [r[0] for r in results]
    
    time_ = time() - t0
    print(time_/60)   
    
    for s in splits:
        PDO_patterns[s] = results[s][1]
    
 
    
    df_PDO = pd.concat(list_PDO_ts, axis=0, keys=splits)
    if splits.size > 1:
        # train test splits should not be equal
        assert float((df_PDO.loc[1] - df_PDO.loc[0]).mean()) != 0, ('something '
                'went wrong with train test splits')
    #%%
    return df_PDO, PDO_patterns



#def PDO_temp(filename, ex, df_splits=None):
#    #%%
#    '''
#    PDO is calculated based upon all data points in the training years,
#    Subsequently, the PDO pattern is projection on the sst.sel(time=dates_train)
#    to enable retrieving the PDO timeseries on a subset on the year.
#    It is similarly also projected on the dates_test.
#    From https://climatedataguide.ucar.edu/climate-data/pacific-decadal-oscillation-pdo-definition-and-indices
#    '''
#    
#
#    if df_splits is None:
#        RV = ex[ex['RV_name']]
#        df_splits, ex = functions_pp.rand_traintest_years(RV, ex)
#
#
#        
#        
#    kwrgs_pp = {'selbox' :  {'la_min':20, # select domain in degrees east
#                             'la_max':65,
#                             'lo_min':115,
#                             'lo_max':250},
#                'format_lon': 'only_east'}        
#    ds = core_pp.import_ds_lazy(filename, **kwrgs_pp)
#    
#    to_freq = ex['tfreq']
#    if to_freq != 1:        
#        ds, dates = functions_pp.time_mean_bins(ds, ex, to_freq=to_freq, seldays='all')
#        ds['time'] = dates
#    
#    dates = pd.to_datetime(ds.time.values)
#
#        
#    splits = df_splits.index.levels[0]
#    data = np.zeros( (splits.size, ds.latitude.size, ds.longitude.size) )
#    PDO_patterns = xr.DataArray(data, 
#                                coords=[splits, ds.latitude.values, ds.longitude.values],
#                                dims = ['split', 'latitude', 'longitude'])    
#    list_splits = []
#    for s in splits:
#        
#        progress = 100 * (s+1) / splits.size
#        dates_train = df_splits.loc[s]['TrainIsTrue'][df_splits.loc[s]['TrainIsTrue']].index
#        train_yrs = np.unique(dates_train.year)
#        dates_all_train = pd.to_datetime([d for d in dates if d.year in train_yrs])
#        dates_test  = df_splits.loc[s]['TrainIsTrue'][~df_splits.loc[s]['TrainIsTrue']].index
#        n = dates_train.size ; r = int(100*n/df_splits.loc[s].index.size )
#        print(f"\rProgress PDO traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")
#        
#        PDO_patterns[s], solver, adjust_sign = get_PDO(ds.sel(time=dates_all_train))
#        
#        PDO_patterns[s] = PDO_patterns[s].interpolate_na(dim='longitude')
#        data_train = find_precursors.calc_spatcov(ds.sel(time=dates_train), PDO_patterns[s])
#        data_test = find_precursors.calc_spatcov(ds.sel(time=dates_test), PDO_patterns[s])
#        
#        df_test = pd.DataFrame(data=data_test.values, index=dates_test, columns=['0_901_PDO'])
#        df_train = pd.DataFrame(data=data_train.values, index=dates_train, columns=['0_901_PDO'])  
#        
#        df = pd.concat([df_test, df_train]).sort_index()
#        list_splits.append(df)
#    
#    df_PDO = pd.concat(list_splits, axis=0, keys=splits)
#    #%%
#    return df_PDO


def get_PDO(sst_Pacific):
    #%%
    from eofs.xarray import Eof
#    PDO   = functions_pp.find_region(sst, region='PDO')[0]
    coslat = np.cos(np.deg2rad(sst_Pacific.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(coslat[..., np.newaxis],(1,sst_Pacific.longitude.size))
    area_weights = area_weights / area_weights.mean()
    solver = Eof(sst_Pacific, area_weights)
    # Retrieve the leading EOF, expressed as the correlation between the leading
    # PC time series and the input SST anomalies at each grid point, and the
    # leading PC time series itself.
    eof1 = solver.eofsAsCovariance(neofs=1).squeeze()
    PDO_warmblob = eof1.sel(latitude=slice(30,40)).sel(longitude=slice(180,200)).mean() # flip sign oef pattern and ts
    assert ~np.isnan(PDO_warmblob), 'Could not verify sign of PDO, check lat lon axis'
    init_sign = np.sign(PDO_warmblob)
    if init_sign != -1.:
        adjust_sign = -1
    else:
        adjust_sign = 1
    eof1 *= adjust_sign
    return eof1, solver, adjust_sign

#def project_PDO(sst_Pacific)
