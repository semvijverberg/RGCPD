#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 1 2020

@author: semvijverberg
"""

import numpy as np
import os
import sklearn.cluster as cluster
import core_pp

def labels_to_latlon(time_space_3d, labels, output_space_time, indices_mask, mask2d):
    xrspace = time_space_3d[0].copy()
    output_space_time[indices_mask] = labels
    output_space_time = output_space_time.reshape( (time_space_3d.latitude.size, time_space_3d.longitude.size)  )
    xrspace.values = output_space_time
    xrspace = xrspace.where(mask2d==True)
    return xrspace

def skclustering(time_space_3d, mask2d=None, clustermethodkey='AgglomerativeClustering', 
                 kwrgs={'n_clusters':4}):
    '''
    Is build upon sklean clustering. Techniques available are listed in cluster.__dict__,
    e.g. KMeans, or AgglomerativeClustering, kwrgs are techinque dependend.
    '''
    algorithm = cluster.__dict__[clustermethodkey]

    cluster_method = algorithm(**kwrgs)
    space_time_vec, output_space_time, indices_mask = create_vector(time_space_3d, mask2d)
    results = cluster_method.fit(space_time_vec)
    labels = results.labels_ + 1
    xrclustered = labels_to_latlon(time_space_3d, labels, output_space_time, indices_mask, mask2d)
    return xrclustered, results

def create_vector(time_space_3d, mask2d):
    time_space_3d = time_space_3d.where(mask2d == True)
    # create mask for to-be-clustered time_space_3d
    n_space = time_space_3d.longitude.size*time_space_3d.latitude.size
    mask_1d = np.reshape( mask2d, (1, n_space))
    mask_1d = np.swapaxes(mask_1d, 1,0 )
    mask_space_time = np.array(np.tile(mask_1d, (1,time_space_3d.time.size)), dtype=int)
    # track location of mask to store output
    output_space_time = np.array(mask_space_time[:,0].copy(), dtype=int)
    indices_mask = np.argwhere(mask_space_time[:,0] == 1)[:,0]
    # convert all space_time_3d gridcells to time_space_2d_all
    time_space_2d_all = np.reshape( time_space_3d.values, 
                                   (time_space_3d.time.size, n_space) )
    space_time_2d_all = np.swapaxes(time_space_2d_all, 1,0)
    # # only keep the mask gridcells for clustering
    space_time_2d = space_time_2d_all[mask_space_time == 1]
    space_time_vec = space_time_2d.reshape( (indices_mask.size, time_space_3d.time.size)  )
    return space_time_vec, output_space_time, indices_mask

def binary_occurences_quantile(xarray, q=95):
    '''
    creates binary occuences of 'extreme' events defined as exceeding the qth percentile
    '''
    
    import numpy as np
    np.warnings.filterwarnings('ignore')
    perc = xarray.reduce(np.percentile, dim='time', keep_attrs=True, q=q)
    rep_perc = np.tile(perc, (xarray.time.size,1,1))
    indic = xarray.where(np.squeeze(xarray.values) > rep_perc)
    indic.values = np.nan_to_num(indic)
    indic.values[indic.values > 0 ] = 1
    return indic

def get_spatial_ma(var_filename, mask=None):
    '''
    var_filename must be 3d netcdf file with only one variable
    mask can be nc file containing only a mask, or a latlon box in format
    [west_lon, east_lon, south_lat, north_lat] in format in common west-east degrees 
    Is build upon sklean clustering. Techniques available are listed in sklearn.cluster.__dict__,
    e.g. KMeans, or AgglomerativeClustering, kwrgs are techinque dependend, see sklearn docs.
    '''
    if mask is None:
        xarray = core_pp.import_ds_lazy(var_filename)
        lons = xarray.longitude.values
        lats = xarray.latitude.values
        mask = [min(lons), max(lons), min(lats), max(lats)]
        print(f'no mask given, entire array of box {mask} will be clustered')
    if type(mask) is str:
        xrmask = core_pp.import_ds_lazy(mask)
        variables = list(xrmask.variables.keys())
        strvars = [' {} '.format(var) for var in variables]
        common_fields = ' time time_bnds longitude latitude lev lon lat level '
        var = [var for var in strvars if var not in common_fields]
        if len(var) != 0:
            var = var[0].replace(' ', '')
            npmask = xrmask[var].values
        else:
            npmask = xrmask.values
    elif type(mask) is list:
        selregion = core_pp.import_ds_lazy(var_filename, selbox=mask)
        lons_mask = list(selregion.longitude.values)
        lon_mask  = [True if l in lons_mask else False for l in xarray.longitude]
        lats_mask = list(selregion.latitude.values)
        lat_mask  = [True if l in lats_mask else False for l in xarray.latitude]
        npmask = np.meshgrid(lon_mask, lat_mask)[0]
    
    return npmask