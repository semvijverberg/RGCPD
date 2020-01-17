#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 1 2020

@author: semvijverberg
"""
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD/')
if RGCPD_func not in sys.path:
    sys.path.append(RGCPD_func)
import numpy as np
import sklearn.cluster as cluster
import core_pp
import find_precursors
import xarray as xr
import plot_maps
import uuid

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


def dendogram_clustering(var_filename, mask=None, kwrgs_load={}, q=70, 
                         clustermethodkey='AgglomerativeClustering', 
                         kwrgs_clust={'n_clusters':3}):
    
    xarray = core_pp.import_ds_lazy(var_filename, **kwrgs_load)        
    npmask = get_spatial_ma(var_filename, mask)
    xarray = binary_occurences_quantile(xarray, q=q)
    xrclustered, results = skclustering(xarray, npmask, 
                                        clustermethodkey=clustermethodkey, 
                                       kwrgs=kwrgs_clust)
    xrclustered.attrs['method'] = clustermethodkey
    xrclustered.attrs['kwrgs'] = str(kwrgs_clust)
    xrclustered.attrs['target'] = f'{xarray.name}_exceedances_of_{q}th_percentile'
    if 'hash' not in xrclustered.attrs.keys():
        xrclustered.attrs['hash']   = uuid.uuid4().hex[:5]
    return xrclustered, results

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
        if xrmask.attrs['is_DataArray'] == False:
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
        xarray = core_pp.import_ds_lazy(var_filename)
        selregion = core_pp.import_ds_lazy(var_filename, selbox=mask)
        lons_mask = list(selregion.longitude.values)
        lon_mask  = [True if l in lons_mask else False for l in xarray.longitude]
        lats_mask = list(selregion.latitude.values)
        lat_mask  = [True if l in lats_mask else False for l in xarray.latitude]
        npmask = np.meshgrid(lon_mask, lat_mask)[0]
    
    return npmask

def store_netcdf(xarray, filepath=None, append_hash=None):
    if 'is_DataArray' in xarray.attrs.keys():
        del xarray.attrs['is_DataArray']
    if filepath is None:
        path = get_download_path()
        if hasattr(xarray, 'name'):
            name = xarray.name
            filename = name + '.nc'
        elif type(xr.Dataset()) == type(xarray):
            # guessing I need to fillvalue for 'xrclustered'
            name = 'xrclustered'
            filename = name + '.nc'
        else:
            name = 'no_name'
            filename = name + '.nc'
        filepath = os.path.join(path, filename)
    if append_hash is not None:
        filepath = filepath.split('.')[0] +'_'+ append_hash + '.nc'
    # ensure mask
    xarray = xarray.where(xarray.values != 0.).fillna(-9999)
    encoding = ( {name : {'_FillValue': -9999}} )
    print(f'to file:\n{filepath}')
    # save netcdf
    xarray.to_netcdf(filepath, mode='w', encoding=encoding)    
    return 


def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'Downloads')
    
def spatial_mean_clusters(var_filename, xrclustered):
    #%%
    xarray = core_pp.import_ds_lazy(var_filename)
    labels = xrclustered.values
    nparray = xarray.values
    track_names = []
    area_grid = find_precursors.get_area(xarray)
    regions_for_ts = list(np.unique(labels[~np.isnan(labels)]))
    a_wghts = area_grid / area_grid.mean()

    # this array will be the time series for each feature
    ts_clusters = np.zeros((xarray.shape[0], len(regions_for_ts)))


    # calculate area-weighted mean over labels
    for r in regions_for_ts:
        track_names.append(int(r))
        idx = regions_for_ts.index(r)
        # start with empty lonlat array
        B = np.zeros(xrclustered.shape)
        # Mask everything except region of interest
        B[labels == r] = 1
        # Calculates how values inside region vary over time
        ts_clusters[:,idx] = np.nanmean(nparray[:,B==1] * a_wghts[B==1], axis =1)
    xrts = xr.DataArray(ts_clusters.T, 
                        coords={'cluster':track_names, 'time':xarray.time}, 
                        dims=['cluster', 'time'])
    ds = xr.Dataset({'xrclustered':xrclustered, 'ts':xrts})
    #%%
    return ds

def regrid_array(xr_or_filestr, to_grid, periodic=False):
    import functions_pp
    
    if type(xr_or_filestr) == str:
        xarray = core_pp.import_ds_lazy(xr_or_filestr)
        plot_maps.plot_corr_maps(xarray[0])
        xr_regrid = functions_pp.regrid_xarray(xarray, to_grid, periodic=periodic)
        plot_maps.plot_corr_maps(xr_regrid[0])
    else:
        plot_maps.plot_labels(xr_or_filestr)
        xr_regrid = functions_pp.regrid_xarray(xr_or_filestr, to_grid, periodic=periodic)
        plot_maps.plot_labels(xr_regrid)
        plot_maps.plot_labels(xr_regrid.where(xr_regrid.values==3))
    return xr_regrid