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
import functions_pp
import find_precursors
import xarray as xr
import plot_maps
import uuid
from itertools import product

try:
    from joblib import Parallel, delayed
    import multiprocessing
    n_cpu = multiprocessing.cpu_count() - 1
except:
    n_cpu = 1
    print('Could not import joblib, no parallel processing')


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
    Is build upon sklean clustering. Algorithms available are listed in cluster.__dict__,
    e.g. KMeans, or AgglomerativeClustering, kwrgs are algorithms dependend.
    '''
    algorithm = cluster.__dict__[clustermethodkey]

    cluster_method = algorithm(**kwrgs)
    space_time_vec, output_space_time, indices_mask = create_vector(time_space_3d, mask2d)
    results = cluster_method.fit(space_time_vec)
    labels = results.labels_ + 1
    xrclustered = labels_to_latlon(time_space_3d, labels, output_space_time, indices_mask, mask2d)
    return xrclustered.values, results

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

def adjust_kwrgs(kwrgs_o, new_coords, v1, v2):
    if new_coords[0] != 'fake':
        if new_coords[0] in list(kwrgs_o.keys()):
            kwrgs_o[new_coords[0]] = v1 # overwrite params
    if new_coords[1] in list(kwrgs_o.keys()):
        kwrgs_o[new_coords[1]] = v2
    return kwrgs_o

def correlation_clustering(var_filename, mask=None, kwrgs_load={},
                           clustermethodkey='DBSCAN',
                           kwrgs_clust={'eps':600}):

    if 'selbox' in kwrgs_load.keys():
        kwrgs_l = dict(selbox=kwrgs_load['selbox'])
    else:
        kwrgs_l = {}
    xarray = core_pp.import_ds_lazy(var_filename, **kwrgs_l)

    if 'selbox' in kwrgs_l.keys() and mask is None:
        npmask = np.ones_like(xarray[0].values, dtype=bool)
    else:
        npmask = get_spatial_ma(var_filename, mask)


    kwrgs_loop = {k:i for k, i in kwrgs_clust.items() if type(i) == list}
    [kwrgs_loop.update({k:i}) for k, i in kwrgs_load.items() if type(i) == list]

    if len(kwrgs_loop) == 1:
        # insert fake axes
        kwrgs_loop['fake'] = [0]
    if len(kwrgs_loop) >= 1:
        new_coords = []
        xrclustered = xarray[0].drop('time')
        for k, list_v in kwrgs_loop.items(): # in alphabetical order
            new_coords.append(k)
            dim_coords = {str(k):list_v}
            xrclustered = xrclustered.expand_dims(dim_coords).copy()
        new_coords = [d for d in xrclustered.dims if d not in ['latitude', 'longitude']]
        results = []
        first_loop = kwrgs_loop[new_coords[0]]
        second_loop = kwrgs_loop[new_coords[1]]
        for i, v1 in enumerate(first_loop):
            for j, v2 in enumerate(second_loop):
                kwrgs = adjust_kwrgs(kwrgs_clust.copy(), new_coords, v1, v2)
                kwrgs_l = adjust_kwrgs(kwrgs_load.copy(), new_coords, v1, v2)
                print(f"\rclustering {new_coords[0]}: {v1}, {new_coords[1]}: {v2} ", end="")
                xarray = functions_pp.import_ds_timemeanbins(var_filename, **kwrgs_l)


                xrclustered[i,j], result = skclustering(xarray, npmask,
                                                   clustermethodkey=clustermethodkey,
                                                   kwrgs=kwrgs)
                results.append(result)
        if 'fake' in new_coords:
            xrclustered = xrclustered.squeeze().drop('fake').copy()
    else:
        xrclustered, results = skclustering(xarray, npmask,
                                            clustermethodkey=clustermethodkey,
                                            kwrgs=kwrgs_clust)
    xrclustered.attrs['method'] = clustermethodkey
    xrclustered.attrs['kwrgs'] = str(kwrgs_clust)
    xrclustered.attrs['target'] = f'{xarray.name}'
    if 'hash' not in xrclustered.attrs.keys():
        xrclustered.attrs['hash']   = uuid.uuid4().hex[:5]
    return xrclustered, results

def dendogram_clustering(var_filename, mask=None, kwrgs_load={},
                         clustermethodkey='AgglomerativeClustering',
                         kwrgs_clust={'q':70, 'n_clusters':3}):

    if 'selbox' in kwrgs_load.keys():
        kwrgs_l = dict(selbox=kwrgs_load['selbox'])

    else:
        kwrgs_l = {}
    xarray = core_pp.import_ds_lazy(var_filename, **kwrgs_l)

    if 'selbox' in kwrgs_l.keys() and mask is None:
        npmask = np.ones_like(xarray[0].values, dtype=bool)
    else:
        npmask = get_spatial_ma(var_filename, mask)


    kwrgs_loop = {k:i for k, i in kwrgs_clust.items() if type(i) == list}
    kwrgs_loop_load = {k:i for k, i in kwrgs_load.items() if type(i) == list}
    [kwrgs_loop.update({k:i}) for k, i in kwrgs_loop_load.items()]
    q = kwrgs_clust['q']

    # always reload / reaggregate xarray. Joblib cannot pickle xr.DataArray as input
    # if len(kwrgs_loop_load) == 0:
    #     reload = False # xarray will always be the same
    #     xarray_ts = functions_pp.import_ds_timemeanbins(var_filename, **kwrgs_load)
    #     if type(q) is int:
    #         xarray = binary_occurences_quantile(xarray_ts, q=q)
    # else:
    #     reload = True

    if len(kwrgs_loop) == 1:
        # insert fake axes
        kwrgs_loop['fake'] = [0]
    if len(kwrgs_loop) >= 1:

        new_coords = []
        xrclustered = xarray[0].drop('time')
        for k, list_v in kwrgs_loop.items(): # in alphabetical order
            new_coords.append(k)
            dim_coords = {str(k):list_v}
            xrclustered = xrclustered.expand_dims(dim_coords).copy()
        new_coords = [d for d in xrclustered.dims if d not in ['latitude', 'longitude']]
        results = []
        first_loop = kwrgs_loop[new_coords[0]]
        second_loop = kwrgs_loop[new_coords[1]]
        comb = [[v1, v2] for v1, v2 in product(first_loop, second_loop)]

        def generator(var_filename, xarray, comb, new_coords, kwrgs_clust, kwrgs_load, q):
            for v1,v2 in comb:
                kwrgs = adjust_kwrgs(kwrgs_clust.copy(), new_coords, v1, v2)
                kwrgs_l = adjust_kwrgs(kwrgs_load.copy(), new_coords, v1, v2)
                del kwrgs['q']
                print(f"clustering {new_coords[0]}: {v1}, {new_coords[1]}: {v2}")
                yield kwrgs, kwrgs_l, v1, v2

        def execute_to_dict(var_filename, npmask, v1, v2, q, clustermethodkey, kwrgs, kwrgs_l):

            # if reload: # some param has been adjusted
            xarray_ts = functions_pp.import_ds_timemeanbins(var_filename, **kwrgs_l)
            if type(q) is int:
                xarray = binary_occurences_quantile(xarray_ts, q=q)
            if type(q) is list:
                xarray = binary_occurences_quantile(xarray_ts, q=v2)

            xrclusteredij, result = skclustering(xarray, npmask,
                                                 clustermethodkey=clustermethodkey,
                                                 kwrgs=kwrgs)
            return {f'{v1}..{v2}': (xrclusteredij, result)}

        if n_cpu > 1:
            futures = []
            for kwrgs, kwrgs_l, v1, v2 in generator(var_filename, xarray, comb, new_coords, kwrgs_clust, kwrgs_load, q):
                futures.append(delayed(execute_to_dict)(var_filename, npmask, v1, v2, q, clustermethodkey, kwrgs, kwrgs_l))
            output = Parallel(n_jobs=n_cpu, backend='loky')(futures)
        else:
            output = []
            for kwrgs, kwrgs_l, v1, v2 in generator(var_filename, xarray, comb, new_coords, kwrgs_clust, kwrgs_load, q):
                output.append(execute_to_dict(var_filename, npmask, v1, v2, q, clustermethodkey, kwrgs, kwrgs_l))

        # unpack output when done loop over parameters
        for run in output:
            for key, out in run.items():
                v1, v2 = float(key.split('..')[0]), float(key.split('..')[1])
                i, j = first_loop.index(v1), second_loop.index(v2)
                xrclustered[i,j] = xr.DataArray(out[0],
                                                dims=['latitude', 'longitude'],
                                                coords=[xrclustered.latitude,
                                                        xrclustered.longitude])

        if 'fake' in new_coords:
            xrclustered = xrclustered.squeeze().drop('fake').copy()
    else:
        del kwrgs_clust['q']
        xrclustered, results = skclustering(xarray, npmask,
                                            clustermethodkey=clustermethodkey,
                                            kwrgs=kwrgs_clust)
    print('\n')
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
    elif type(mask) is list or type(mask) is tuple:
        xarray = core_pp.import_ds_lazy(var_filename)
        selregion = core_pp.import_ds_lazy(var_filename, selbox=mask)
        lons_mask = list(selregion.longitude.values)
        lon_mask  = [True if l in lons_mask else False for l in xarray.longitude]
        lats_mask = list(selregion.latitude.values)
        lat_mask  = [True if l in lats_mask else False for l in xarray.latitude]
        npmasklon = np.meshgrid(lon_mask, lat_mask)[0]
        npmasklat = np.meshgrid(lon_mask, lat_mask)[1]
        npmask = np.logical_and(npmasklon, npmasklat)
    elif type(mask) is type(xr.DataArray([0])):
        # lo_min = float(mask.longitude.min()); lo_max = float(mask.longitude.max())
        # la_min = float(mask.latitude.min()); la_max = float(mask.latitude.max())
        # selbox = (lo_min, lo_max, la_min, la_max)
        # selregion = core_pp.import_ds_lazy(var_filename, selbox=selbox)
        # selregion = selregion.where(mask)
        npmask = mask.values
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
    elif type(xarray) == xr.Dataset:
        if 'xrclustered' in list(xarray.variables.keys()):
            name = xarray['xrclustered'].name
    elif type(xarray) == xr.DataArray:
        name = xarray.name
    if append_hash is not None:
        filepath = filepath.split('.')[0] +'_'+ append_hash + '.nc'
    else:
        filepath = filepath.split('.')[0] + '.nc'
    # ensure mask
    # xarray = xarray.where(xarray.values != 0.).fillna(-9999)
    # encoding = ( {name : {'_FillValue': -9999}} )
    print(f'to file:\n{filepath}')
    # save netcdf
    xarray.to_netcdf(filepath)
    return filepath


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

def spatial_mean_clusters(var_filename, xrclust, selbox=None):
    #%%
    xarray = core_pp.import_ds_lazy(var_filename, selbox=selbox)
    labels = xrclust.values
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
        B = np.zeros(xrclust.shape)
        # Mask everything except region of interest
        B[labels == r] = 1
        # Calculates how values inside region vary over time
        ts_clusters[:,idx] = np.nanmean(nparray[:,B==1] * a_wghts[B==1], axis =1)
    xrts = xr.DataArray(ts_clusters.T,
                        coords={'cluster':track_names, 'time':xarray.time},
                        dims=['cluster', 'time'])

    # extract selected setting for ts
    dims = list(xrclust.coords.keys())
    standard_dim = ['latitude', 'longitude', 'time', 'mask', 'cluster']
    dims = [d for d in dims if d not in standard_dim]
    if 'n_clusters' in dims:
        idx = dims.index('n_clusters')
        dims[idx] = 'ncl'
        xrclust = xrclust.rename({'n_clusters':dims[idx]}).copy()
    var1 = int(xrclust[dims[0]])
    var2 = int(xrclust[dims[1]])
    dim1 = dims[0]
    dim2 = dims[1]
    xrts.attrs[dim1] = var1 ; xrclust.attrs[dim1] = var1
    xrts.attrs[dim2] = var2 ; xrclust.attrs[dim2] = var2
    ds = xr.Dataset({'xrclustered':xrclust.drop(dim1).drop(dim2), 'ts':xrts})
    #%%
    return ds

def percentile_cluster(var_filename, xrclust, q=75, tailmean=True, selbox=None):
    xarray = core_pp.import_ds_lazy(var_filename, selbox=selbox)
    labels = xrclust.values
    nparray = xarray.values
    n_t = xarray.time.size
    track_names = []
    area_grid = find_precursors.get_area(xarray)
    regions_for_ts = list(np.unique(labels[~np.isnan(labels)]))

    if tailmean:
        tmp_wgts = (area_grid / area_grid.mean())[:,:]
        a_wghts = np.tile(tmp_wgts[None,:], (n_t,1,1))
    else:
        a_wghts = area_grid / area_grid.mean()
    # this array will be the time series for each feature
    ts_clusters = np.zeros((xarray.shape[0], len(regions_for_ts)))

    # calculate area-weighted mean over labels
    for r in regions_for_ts:
        track_names.append(int(r))
        idx = regions_for_ts.index(r)
        # start with empty lonlat array
        B = np.zeros(xrclust.shape)
        # Mask everything except region of interest
        B[labels == r] = 1
        # Calculates how values inside region vary over time
        if tailmean == False:
            ts_clusters[:,idx] = np.nanpercentile(nparray[:,B==1] * a_wghts[B==1], q=q,
                                                  axis =1)
        elif tailmean:
            # calc percentile of space for each timestep, not we will
            # have a timevarying spatial mask.
            ts_clusters[:,idx] = np.nanpercentile(nparray[:,B==1], q=q,
                                                  axis =1)
            # take a mean over all gridpoints that pass the percentile instead
            # of taking the single percentile value of a spatial region
            mask_B_perc = nparray[:,B==1] > ts_clusters[:,idx, None]
            # if unlucky, the amount of gridcells that pass the percentile
            # value, were not always equal in each timestep. When this happens,
            # we can no longer reshape the array to (time, space) axis, and thus
            # we cannot take the mean over time.
            # check if have same size over time
            cs_ = [int(mask_B_perc[t][mask_B_perc[t]].shape[0]) for t in range(n_t)]
            if np.unique(cs_).size != 1:
                # what is the most common size:
                common_shape = cs_[np.argmax([cs_.count(v) for v in np.unique(cs_)])]


                # convert all masks to most common size by randomly
                # adding/removing a True
                for t in range(n_t):
                    while mask_B_perc[t][mask_B_perc[t]].shape[0] < common_shape:
                        mask_B_perc[t][np.argwhere(mask_B_perc[t]==False)[0][0]] = True
                    while mask_B_perc[t][mask_B_perc[t]].shape[0] > common_shape:
                        mask_B_perc[t][np.argwhere(mask_B_perc[t]==True)[0][0]] = False

            nptimespacefull = nparray[:,B==1].reshape(nparray.shape[0],-1)
            npuppertail = nptimespacefull[mask_B_perc]
            wghtsuppertail = a_wghts[:,B==1][mask_B_perc]

            y = np.nanmean(npuppertail.reshape(n_t,-1) * \
                            wghtsuppertail.reshape(n_t,-1), axis =1)

            ts_clusters[:,idx] = y
    xrts = xr.DataArray(ts_clusters.T,
                        coords={'cluster':track_names, 'time':xarray.time},
                        dims=['cluster', 'time'])
    return xrts

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


