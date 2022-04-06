#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:33:50 2020

@author: semvijverberg
"""
import inspect
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
cluster_func = os.path.join(main_dir, 'clustering/')
RGCPD_func = os.path.join(main_dir, 'RGCPD')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(cluster_func)
if RGCPD_func not in sys.path:
    sys.path.append(RGCPD_func)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import clustering_spatial as cl
import core_pp
import func_fc
import functions_pp
import numpy as np
import plot_maps
import validation as valid
from class_RV import aggr_to_daily_dates

max_cpu = multiprocessing.cpu_count()


def spatial_valid(var_filename, mask, y_pred_all, y_pred_c, lags_i=None,
                  seldates=None, clusters=None, kwrgs_events=None, alpha=0.05, n_boot=0, 
                  blocksize=10, threshold_pred='upper_clim'):
                  
    '''
    var_filename must be 3d netcdf file with only one variable
    mask can be nc file containing only a mask, or a latlon box in format
    [west_lon, east_lon, south_lat, north_lat] in format in common west-east degrees 
    '''
    var_filename = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/preprocessed/t2mmax_US_1979-2018_1jan_31dec_daily_0.25deg.nc'
    mask = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/preprocessed/cluster_output.nc'

    if lags_i is None:
        lags_i = list(y_pred_all.columns)
        
    # load in daily xarray and mask
    xarray = core_pp.import_ds_lazy(var_filename)    
    npmask = cl.get_spatial_ma(var_filename, mask)
    
    
    # process temporal infor
    freq = (y_pred_c.index[1] - y_pred_c.index[0]).days
    if seldates is None:
        seldates = aggr_to_daily_dates(y_pred_c.index)
        start  = f'{seldates[0].month}-{seldates[0].day}'
        end    = f'{seldates[-1].month}-{seldates[-1].day}'
        start_end_date = (start, end)
    xarray, dates = functions_pp.time_mean_bins(xarray, to_freq=freq, 
                                                start_end_date=start_end_date)
    
    # if switching to event timeseries:
    if kwrgs_events is None:
        kwrgs_events = {'event_percentile':66}
    # unpack other optional arguments for defining event timeseries 
    kwrgs = {key:item for key, item in kwrgs_events.items() if key != 'event_percentile'}
    
    if clusters is None:
        clusters = list(np.unique(npmask[~np.isnan(npmask)]))
    elif type(clusters) is int:
        clusters = [clusters]
    elif clusters is not None:
        clusters = clusters
        
    dict_allclus = {}
    for clus in clusters:
        
        latloni = np.where(npmask==clus)
        latloni = [(latloni[0][i], latloni[1][i]) for i in range(latloni[0].size)]
        
 
        futures = {}
        with ProcessPoolExecutor(max_workers=max_cpu) as pool:
                     
            for ll in latloni:
                latloni = latloni
                xr_gridcell = xarray.isel(latitude=ll[0]).isel(longitude=ll[1])
                threshold = func_fc.Ev_threshold(xr_gridcell, kwrgs_events['event_percentile'])
                y_i = func_fc.Ev_timeseries(xr_gridcell, threshold, **kwrgs)[0]
                
            
                futures[ll]  = pool.submit(valid.get_metrics_sklearn, 
                                                y_i.values, y_pred_all[lags_i], y_pred_c, 
                                                alpha=alpha,
                                                n_boot=n_boot,
                                                blocksize=blocksize,
                                                threshold_pred=threshold_pred)
        results = {key:future.result() for key, future in futures.items()}
        dict_allclus[clus] = results
        
    df_valid = dict_allclus[clus][ll][0]
    metrics = np.unique(df_valid.index.get_level_values(0))
    lags_tf = [l*freq for l in lags_i]
    if freq != 1:
        # the last day of the time mean bin is tfreq/2 later then the centerered day
        lags_tf = [l_tf- int(freq/2) if l_tf!=0 else 0 for l_tf in lags_tf] 
        
    for clus in clusters:
        results = dict_allclus[clus]
        xroutput = xarray.isel(time=lags_i).rename({'time':'lag'})
        xroutput['lag'] = lags_tf
        xroutput = xroutput.expand_dims({'metric':metrics}, 0)
        npdata = np.array(np.zeros_like(xroutput), dtype='float32')
        for ll in latloni:
            df_valid = dict_allclus[clus][ll][0]
            for i, met in enumerate(metrics):
                lat_i = ll[0] ; lon_i = ll[1]
                npdata[i,:,lat_i,lon_i] = df_valid.loc[met].loc[met]
        xroutput.values = npdata
        
    plot_maps.plot_corr_maps(xroutput.where(npmask==clus), row_dim='metric', size=4, clevels=np.arange(-1,1.1, 0.2))
    BSS = xroutput.where(npmask==clus).sel(metric='BSS')
    plot_maps.plot_corr_maps(BSS, row_dim='metric', size=4, clevels=np.arange(-0.25,0.251, 0.05), cbar_vert=-0.1)
        # output xarray
        
        
#            valid.get_metrics_sklearn(y_i.values, y_pred_all, y_pred_c, 
#                                                alpha=alpha,
#                                                n_boot=n_boot,
#                                                blocksize=blocksize,
#                                                threshold_pred=threshold_pred)
        
#        # split into number of CPU chunks
#        old_index = range(0,len(latloni),1)
#        n_bl = int(len(latloni) / max_cpu)
#        chunks = [old_index[n_bl*i:n_bl*(i+1)] for i in range(int(len(old_index)/n_bl))]
#        # avoid missing the last due indices due to rounding of int:
#        chunks[-1] = range(chunks[-1][0], len(latloni))
#        sublatloni = [np.array(latloni)[c] for c in chunks]
#        for ll in latloni:
#            xr_gridcell = xarray.isel(latitude=ll[0]).isel(longitude=ll[1])
#            threshold = func_fc.Ev_threshold(xr_gridcell, kwrgs_events['event_percentile'])
#    
#            y_i = func_fc.Ev_timeseries(xr_gridcell, threshold, **kwrgs)[0]
#            df_valid, metrics_dict = valid.get_metrics_sklearn(y_i.values, y_pred_all, y_pred_c)
#            dict_clus[str(ll)] = df_valid 

