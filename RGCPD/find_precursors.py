#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:17:25 2019

@author: semvijverberg
"""

import itertools
from math import radians as _r

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from netCDF4 import num2date
from sklearn import cluster
from sklearn.metrics.pairwise import haversine_distances

from . import core_pp, functions_pp, plot_maps

flatten = lambda l: list(itertools.chain.from_iterable(l))
from typing import List, Tuple, Union

#%%

def add_info_precur(precur, corr_xr, pval_xr):
    precur.corr_xr = corr_xr
    precur.pval_xr = pval_xr
    precur.lat_grid = precur.precur_arr.latitude.values
    precur.lon_grid = precur.precur_arr.longitude.values
    precur.grid_res = abs(precur.lon_grid[1] - precur.lon_grid[0])



def calculate_region_maps(precur, df_RVfull, df_splits, kwrgs_load):
    '''
    tfreq : aggregate precursors with bins of window size = tfreq
    selbox : selbox is tuple of:
            (degrees_east, degrees_west, degrees_south, degrees_north)
    loadleap : if leap day should loaded yes/no
    seldates : if a selection of dates should be loaded

    '''
    #%%
    # precur = rg.list_for_MI[0] ; df_RVfull = rg.df_fullts; df_splits = rg.df_splits ; kwrgs_load = rg.kwrgs_load

    precur.load_and_aggregate_precur(kwrgs_load.copy())
    # =============================================================================
    # Load external timeseries for partial_corr_z
    # =============================================================================
    precur.load_and_aggregate_ts(df_splits)
    # =============================================================================
    # Calculate BivariateMI (correlation) map
    # =============================================================================
    corr_xr, pval_xr = precur.bivariateMI_map(precur.precur_arr, df_splits, df_RVfull)
    # =============================================================================
    # update class precur
    # =============================================================================
    add_info_precur(precur, corr_xr, pval_xr)

    return precur

def get_area(ds):
    longitude = ds.longitude
    latitude = ds.latitude

    Erad = 6.371e6 # [m] Earth radius
    #    global_surface = 510064471909788
    # Semiconstants
    gridcell = np.abs(longitude[1] - longitude[0]).values # [degrees] grid cell size

    # new area size calculation:
    lat_n_bound = np.minimum(90.0 , latitude + 0.5*gridcell)
    lat_s_bound = np.maximum(-90.0 , latitude - 0.5*gridcell)

    A_gridcell = np.zeros([len(latitude),1])
    A_gridcell[:,0] = (np.pi/180.0)*Erad**2 * abs( np.sin(lat_s_bound*np.pi/180.0) - np.sin(lat_n_bound*np.pi/180.0) ) * gridcell
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
    #    A_mean = np.mean(A_gridcell2D)
    return A_gridcell2D


def mask_sig_to_cluster(mask_and_data_s, wght_area, distance_eps, min_area_samples,
                        n_jobs=-1):


    mask_sig_1d = mask_and_data_s.mask.astype('bool').values == False
    data = mask_and_data_s.data
    lons = mask_and_data_s.longitude.values #!!! Ensure west_east format?
    lats = mask_and_data_s.latitude.values
    n_lags = mask_and_data_s.lag.size

    np_dbregs   = np.zeros( (n_lags, lats.size, lons.size), dtype=int )
    labels_sign_lag = []
    label_start = 0

    for sign in [-1, 1]:
        mask = mask_sig_1d.copy()
        mask[np.sign(data) != sign] = False
        n_gc_sig_sign = mask[mask==True].size
        labels_for_lag = np.zeros( (n_lags, n_gc_sig_sign), dtype=bool)
        meshgrid = np.meshgrid(lons.data, lats.data)
        mask_sig = np.reshape(mask, (n_lags, lats.size, lons.size))
        sign_coords = [] ; count=0
        weights_core_samples = []
        for l in range(n_lags):
            sign_c = meshgrid[0][ mask_sig[l,:,:] ], meshgrid[1][ mask_sig[l,:,:] ]
            n_sign_c_lag = len(sign_c[0])
            labels_for_lag[l][count:count+n_sign_c_lag] = True
            count += n_sign_c_lag
            # shape sign_coords = [(lats, lons)]
            sign_coords.append( [[_r(sign_c[1][i]), _r(sign_c[0][i]-180)] for i in range(sign_c[0].size)] )
            weights_core_samples.append(wght_area[mask_sig[l,:,:]].reshape(-1))

        sign_coords = flatten(sign_coords)
        if len(sign_coords) != 0:
            weights_core_samples = flatten(weights_core_samples)
            # calculate distance between sign coords accross all lags to keep labels
            # more consistent when clustering
            distance = haversine_distances(sign_coords) * 6371000/1000 # multiply by Earth radius to get kilometers
            dbresult = cluster.DBSCAN(eps=distance_eps, min_samples=int(min_area_samples),
                                      metric='precomputed', n_jobs=n_jobs).fit(distance,
                                      sample_weight=weights_core_samples)
            labels = dbresult.labels_ + 1
            # all labels == -1 (now 0) are seen as noise:
            labels[labels==0] = -label_start
            individual_labels = labels + label_start
            [labels_sign_lag.append((l, sign)) for l in np.unique(individual_labels) if l != 0]

            for l in range(n_lags):
                mask_sig_lag = mask[l,:,:]==True
                np_dbregs[l,:,:][mask_sig_lag] = individual_labels[labels_for_lag[l]]
            label_start = int(np_dbregs[mask].max())
        else:
            pass
        np_regs = np.array(np_dbregs, dtype='int')
    return np_regs, labels_sign_lag

def close_small_clusters(prec_labels, corr_xr, distance_eps_sc,
                         min_gridcells_sc=1, n_jobs=1):

    df = labels_to_df(prec_labels.mean(dim='split'),
                      corr_xr.mean(dim='split'), return_mean_latlon=True)
    merge = []
    for sign in [-1, 1]:
        npmask = np.logical_and(df['sign']==sign,
                                df['n_gridcells'] >= min_gridcells_sc)
        labels_sign = df[npmask].copy()
        init_labels = labels_sign.index

        labels_sign.loc[:,'longitude'] = ((labels_sign['longitude'] + 180) % 360) - 180

        sign_coords = labels_sign[['latitude', 'longitude']].values
        # convert to radians
        sign_coords = [[_r(c[0]), _r(c[1])] for c in sign_coords]
        if len(sign_coords) != 0:
            # calculate distance between sign coords accross all lags to keep labels
            # more consistent when clustering
            distance = haversine_distances(sign_coords) * 6371000/1000 # multiply by Earth radius to get kilometers
            dbresult = cluster.DBSCAN(eps=distance_eps_sc,
                                      min_samples=1,
                                      metric='precomputed',
                                      n_jobs=n_jobs).fit(distance)
            labels = dbresult.labels_

            # creates an array of indices, sorted by unique element
            idx_sort = np.argsort(labels)
            # sorts records array so all unique elements are together
            sorted_records_array = labels[idx_sort]
            # returns the unique values, the index of the first occurrence of a value, and the count for each element
            vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

            # splits the indices into separate arrays
            res = np.split(idx_sort, idx_start[1:])

            dupl = [list(r) for r in res if r.size>1]
            merge += [list(init_labels[d]) for d in dupl]
    return merge




def calc_spatcov(full_timeserie: np.ndarray, pattern: np.ndarray, area_wght=None):
    #%%
    '''
    Calculate spatial covariance over time-axis, which should be first index.
    '''
    mask = np.ma.make_mask(np.isnan(pattern)==False)
    n_time = full_timeserie.shape[0]
    n_space = pattern.size

    if area_wght is not None:
        pattern *= area_wght
    # select only gridcells where there is not a nan
    full_ts = np.nan_to_num(np.reshape( full_timeserie, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern, (n_space) ))

    mask_pattern = np.reshape( mask, (n_space) )
    full_ts = full_ts[:,mask_pattern]
    pattern = pattern[mask_pattern]

    spatcov   = np.zeros( (n_time) )
    for t in range(n_time):
        # Corr(X,Y) = cov(X,Y) / ( std(X)*std(Y) )
        # cov(X,Y) = E( (x_i - mu_x) * (y_i - mu_y) )
        # covself[t] = np.mean( (full_ts[t] - np.mean(full_ts[t])) * (pattern - np.mean(pattern)) )
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,1] #/ (np.sqrt(np.cov(M)[0,0]) * np.sqrt(np.cov(M)[1,1]))

    #%%
    return spatcov

def get_spatcovs(dict_ds, df_splits, s, outdic_actors, normalize=True): #, df_split, outdic_actors #TODO is df_split same as df_splits
    #%%

    lag = 0
    TrainIsTrue = df_splits['TrainIsTrue']
    times = df_splits.index
    options = ['_spatcov', '_spatcov_caus']
    columns = []
    for var in outdic_actors.keys():
        for select in options:
            columns.append(var+select)


    data = np.zeros( (len(columns), times.size) )
    df_sp_s = pd.DataFrame(data.T, index=times, columns=columns)
    dates_train = TrainIsTrue[TrainIsTrue.values].index
    #    dates_test  = TrainIsTrue[TrainIsTrue.values==False].index
    for var, pos_prec in outdic_actors.items():
        ds = dict_ds[var]
        for i, select in enumerate(['_labels', '_labels_tigr']):
            # spat_cov over test years using corr fields from training

            full_timeserie = pos_prec.precur_arr
            corr_vals = ds[var + "_corr"].sel(split=s).isel(lag=lag)
            mask = ds[var + select].sel(split=s).isel(lag=lag)
            pattern = corr_vals.where(~np.isnan(mask))
            if np.isnan(pattern.values).all():
                # no regions of this variable and split
                pass
            else:
                if normalize == True:
                    spatcov_full = calc_spatcov(full_timeserie, pattern)
                    mean = spatcov_full.sel(time=dates_train).mean(dim='time')
                    std = spatcov_full.sel(time=dates_train).std(dim='time')
                    spatcov_test = ((spatcov_full - mean) / std)
                elif normalize == False:
                    spatcov_test = calc_spatcov(full_timeserie, pattern)
                pd_sp = pd.Series(spatcov_test.values, index=times)
                col = options[i]
                df_sp_s[var + col] = pd_sp
    for i, bckgrnd in enumerate(['tcov', 'caus']):
        cols = [col for col in df_sp_s.columns if col[-4:] == bckgrnd]
        key = options[i]
        df_sp_s['all'+key] = df_sp_s[cols].mean(axis=1)
    #%%
    return df_sp_s



def cluster_DBSCAN_regions(precur):
    #%%

    """
    Clusters regions together of same sign using DBSCAN
    """

    if hasattr(precur, 'corr_xr_'): # already clustered with group_lab before
        corr_xr  = precur.corr_xr_.copy() # restore original corr. maps
    else:
        corr_xr  = precur.corr_xr.copy()

    var = precur.name

    lags = corr_xr.lag.values
    n_spl  = corr_xr.coords['split'].size
    precur.area_grid = get_area(precur.precur_arr)
    n_jobs_clust = precur.n_jobs_clust



    if precur.group_lag: # group over regions found in range of lags
        if hasattr(precur, 'corr_xr_'): # already clustered before
            corr_xr  = precur.corr_xr_.copy() # restore original corr. maps
        else:
            print('Corr map now mean over lags, original is stored in corr_xr_')
            precur.corr_xr_ = corr_xr.copy()
        lags = corr_xr.lag.values
        sign_mask = (corr_xr['mask'].sum(dim='lag')==lags.size).astype(int)
        corr_xr = corr_xr.mean(dim='lag') # mean over lags
        sign_mask = sign_mask.expand_dims('lag', axis=1)
        corr_xr = corr_xr.expand_dims('lag', axis=1)
        lags = np.array(['-'.join(np.array(lags, str))])
        corr_xr['lag'] = ('lag', lags) ; sign_mask['mask'] = ('lag', lags)
        corr_xr['mask'] = sign_mask
        precur.corr_xr = corr_xr # corr_xr is updated to mean over lags

    n_lags              = lags.size
    lats                = corr_xr.latitude
    lons                = corr_xr.longitude
    area_grid           = precur.area_grid/ 1E6 # in km2
    distance_eps        = precur.distance_eps
    min_area_in_degrees2= precur.min_area_in_degrees2
    group_split         = precur.group_split


    aver_area_km2 = 7939     # np.mean(precur.area_grid) with latitude 0-90 / 1E6
    wght_area = area_grid / aver_area_km2
    min_area_km2 = min_area_in_degrees2 * 111.131 * min_area_in_degrees2 * 78.85
    min_area_samples = min_area_km2 / aver_area_km2



    prec_labels_np = np.zeros( (n_spl, n_lags, lats.size, lons.size), dtype=int )
    labels_sign_lag = np.zeros( (n_spl), dtype=list)
    mask_and_data = corr_xr.copy()

    # group regions per split (no information leak train test)
    if group_split == False:
        for s in range(n_spl):
            progress = int(100 * (s+1) / n_spl)
            print(f"\rProgress traintest set {progress}%", end="")
            mask_and_data_s = corr_xr.sel(split=s)
            grouping_split = mask_sig_to_cluster(mask_and_data_s, wght_area,
                                                distance_eps, min_area_samples,
                                                n_jobs=n_jobs_clust)
            prec_labels_np[s] = grouping_split[0]
            labels_sign_lag[s] = grouping_split[1]
    # group regions regions the same accross splits
    elif group_split:

        mask = abs(mask_and_data.mask -1)
        mask_all = mask.sum(dim='split') / mask.sum(dim='split')
        m_d_together = mask_and_data.isel(split=0).copy()
        m_d_together['mask'] = abs(mask_all.fillna(0) -1)
        m_d_together.values = np.sign(mask_and_data).mean(dim='split')
        grouping_split = mask_sig_to_cluster(m_d_together, wght_area,
                                            distance_eps, min_area_samples)

        for s in range(n_spl):
            m_all = grouping_split[0].copy()
            mask_split = corr_xr.sel(split=s).mask
            m_all[mask_split.astype('bool').values] = 0
            labs = np.unique(mask_split==0)[1:]
            l_sign = grouping_split[1]
            labs_s = [t for t in l_sign if t[0] in labs]
            prec_labels_np[s] = m_all
            labels_sign_lag[s] = labs_s


    if np.nansum(prec_labels_np) == 0. and mask_and_data.mask.all()==False:
        print('\nSome significantly correlating gridcells found, but too randomly located and '
            'interpreted as noise by DBSCAN, make distance_eps lower '
            'to relax constrain.\n')
        prec_labels_ord = prec_labels_np
    if mask_and_data.mask.all()==True:
        print(f'\nNo significantly correlating gridcells found for {var}.\n')
        prec_labels_ord = prec_labels_np
    else:
        prec_labels_ord = np.zeros_like(prec_labels_np)
        if group_split == False:
            for s in range(n_spl):
                prec_labels_s = prec_labels_np[s]
                corr_vals     = corr_xr.sel(split=s).values
                reassign = reorder_strength(prec_labels_s, corr_vals, area_grid,
                                            min_area_samples)
                prec_labels_ord[s] = relabel(prec_labels_s, reassign)
        elif group_split:
            # order based on mean corr_value:
            corr_vals = corr_xr.mean(dim='split').values
            prec_label_s = grouping_split[0].copy()
            prec_label_s[mask_split.astype('bool').values] = 0
            reassign = reorder_strength(prec_label_s, corr_vals, area_grid,
                                        min_area_samples)
            for s in range(n_spl):
                prec_labels_s = prec_labels_np[s]
                prec_labels_ord[s] = relabel(prec_labels_s, reassign)



    prec_labels = xr.DataArray(data=prec_labels_ord, coords=[range(n_spl), lags, lats, lons],
                        dims=['split', 'lag','latitude','longitude'],
                        name='{}_labels_init'.format(precur.name),
                        attrs={'units':'Precursor regions [ordered for Corr strength]'})
    prec_labels = prec_labels.where(prec_labels_ord!=0.)
    prec_labels.attrs['title'] = prec_labels.name
    precur.prec_labels = prec_labels
    #%%
    return precur

def reorder_strength(prec_labels_s, corr_vals, area_grid, min_area_km2):
    #%%
    # order regions on corr strength
    # based on median of upper 25 percentile

    n_lags = prec_labels_s.shape[0]
    Number_regions_per_lag = np.zeros(n_lags)
    corr_strength = {}
    totalsize_lag0 = area_grid[prec_labels_s[0]!=0].mean() / 1E5
    for l_idx in range(n_lags):
        # check if region is higher lag is actually too small to be a cluster:
        prec_field = prec_labels_s[l_idx,:,:]

        for i, reg in enumerate(np.unique(prec_field)[1:]):
            are = area_grid.copy()
            are[prec_field!=reg]=0
            area_prec_reg = are.sum()/1E5
            if area_prec_reg < min_area_km2/1E5:
                print(reg, area_prec_reg, min_area_km2/1E5, 'not exceeding area')
                prec_field[prec_field==reg] = 0
            if area_prec_reg >= min_area_km2/1E5:
                Corr_value = corr_vals[l_idx, prec_field==reg]
                weight_by_size = (area_prec_reg / totalsize_lag0)**0.1
                Corr_value.sort()
                strength   = abs(np.median(Corr_value[int(0.75*Corr_value.size):]))
                Corr_strength = np.round(strength * weight_by_size, 10)
                corr_strength[Corr_strength + l_idx*1E-5] = '{}_{}'.format(l_idx,reg)
        Number_regions_per_lag[l_idx] = np.unique(prec_field)[1:].size
        prec_labels_s[l_idx,:,:] = prec_field


    # Reorder - strongest correlation region is number 1, etc... ,
    strongest = sorted(corr_strength.keys())[::-1]
    #    actor.corr_strength = corr_strength
    reassign = {} ; key_dupl = [] ; new_reg = 0
    order_str_actor = {}
    for i, key in enumerate(strongest):
        old_lag_reg = corr_strength[key]
        old_reg = int(old_lag_reg.split('_')[-1])
        if old_reg not in key_dupl:
            new_reg += 1
            reassign[old_reg] = new_reg
        key_dupl.append( old_reg )
        new_lag_reg = old_lag_reg.split('_')[0] +'_'+ str(reassign[old_reg])
        order_str_actor[new_lag_reg] = i+1
    return reassign

def relabel(prec_labels_s, reassign):
    prec_labels_ord = np.zeros(prec_labels_s.shape, dtype=int)
    for i, reg in enumerate(reassign.keys()):
        prec_labels_ord[prec_labels_s == reg] = reassign[reg]
    return prec_labels_ord


def xrmask_by_latlon(xarray,
                     upper_right: Tuple[float, float]=None,
                     bottom_right: Tuple[float, float]=None,
                     upper_left: Tuple[float, float]=None,
                     bottom_left: Tuple[float, float]=None,
                     latmax: float=None, lonmax: float=None,
                     latmin: float=None, lonmin: float=None):

    '''
    Applies mask to lat-lon xarray defined by lat lon coordinates.
    xarray.where returns values where mask==True.

    Consensus: everything above latmax/lonmax is masked, or everything below
    latmin/lonmin is masked.


    Parameters
    ----------
    xarray : xr.DataArray
        DESCRIPTION.
    upper_right : Tuple[float, float], optional
        upper right masking, defined by tuple of (lonmax, latmax).
        The default is None.
    bottom_right : Tuple[float, float], optional
        upper left masking, defined by tuple of (lonmax, latmin).
        The default is None.
    upper_left : Tuple[float, float], optional
        bottom left masking, defined by tuple of (lonmin, latmin).
        The default is None.
    bottom_left : Tuple[float, float], optional
        bottom left masking, defined by tuple of (lonmin, latmin).
        The default is None.

    latmax : float, optional
        north of latmax is masked. The default is None.
    lonmax : float, optional
        east of lonmax is masked. The default is None.
    latmin : float, optional
        DESCRIPTION. The default is None.
    lonmin : float, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    xarray : xr.DataArray
        DESCRIPTION.

    '''
    ll = np.meshgrid(xarray.longitude, xarray.latitude)
    # north of latmax is masked
    if latmax is not None and lonmax is None:
        xarray = xarray.where(ll[1] < latmax)
    # east of lonmax is masked
    if lonmax is not None and latmax is None:
        xarray = xarray.where(ll[0]<lonmax)
    if latmin is not None and lonmin is None:
        xarray = xarray.where(ll[1] > latmin)
    if lonmin is not None and latmin is None:
        xarray = xarray.where(ll[0] > lonmin)
    # upper right masking
    if upper_right is not None:
        lonmax, latmax = upper_right
        npmask = np.logical_or(ll[1] < latmax, ll[0]<lonmax)
        xarray = xarray.where(npmask)
    # bottom right masking
    if bottom_right is not None:
        lonmax, latmin = bottom_right
        npmask = np.logical_or(ll[1] > latmin, ll[0]<lonmax)
        xarray = xarray.where(npmask)
    # bottom left masking
    if bottom_left is not None:
        lonmin, latmin  = bottom_left
        npmask = np.logical_or(ll[1] > latmin, ll[0] > lonmin)
        xarray = xarray.where(npmask)
    # upper left masking
    if upper_left is not None:
        lonmin, latmax = upper_left
        npmask = np.logical_or(ll[1] < latmax, ll[0] > lonmin)
        xarray = xarray.where(npmask)
    return xarray

def split_region_by_lonlat(prec_labels, label=int, plot_s=0,
                           plot_l=0, kwrgs_mask_latlon={} ):

    # before:
    plot_maps.plot_labels(prec_labels.isel(split=plot_s, lag=plot_l),
                          kwrgs_plot={'size':1.5,
                                      'subtitles':np.array([['old']])})
    splits = list(prec_labels.split.values)
    lags   = list(prec_labels.lag.values)
    copy_labels = prec_labels.copy()
    np_labels = copy_labels.values
    orig_labels = np.unique(prec_labels.values[~np.isnan(prec_labels.values)])
    print(f'\nNew label will become {max(orig_labels) + 1}')
    if max(orig_labels) >= 20:
        print('\nwarning, more then (or equal to) 20 regions')
    from itertools import product
    for s, l in product(splits, lags):
        i_s = splits.index(s)
        i_l = lags.index(l)
        single = copy_labels.sel(split=s, lag=l)
        orig_mask_label = ~np.isnan(single.where(single.values==label))
        for key, mask_latlon in kwrgs_mask_latlon.items():
#            print(key, mask_latlon)
            mask_label = xrmask_by_latlon(orig_mask_label,
                                          **{str(key):mask_latlon})
        # mask_label = np.logical_and(~np.isnan(mask_label), mask_label!=0)
        mask_label = np.logical_and(np.isnan(mask_label), orig_mask_label)
        # assign new label
        single.values[mask_label.values] = max(orig_labels) + 1
        np_labels[i_s, i_l] = single.values
    copy_labels.values = np_labels
    # after
    plot_maps.plot_labels(copy_labels.isel(split=plot_s, lag=plot_l),
                          kwrgs_plot={'size':1.5,
                                      'subtitles':np.array([['new']])})
    return copy_labels, max(orig_labels) + 1

def manual_relabel(prec_labels, replace_label: int=None, with_label: int=None):
    '''
    Can Automatically relabel based on prevailence.

    If replace_label and with_label are not given:

    Smallest prevailence of first 10 labels is replaced with maximum prevailence
    label of last 15 labels

    '''

    copy_labels = prec_labels.copy()
    all_labels = prec_labels.values[~np.isnan(prec_labels.values)]
    uniq_labels = np.unique(all_labels)
    prevail = {l:list(all_labels).count(l) for l in uniq_labels}
    prevail = functions_pp.sort_d_by_vals(prevail)
    l_keys = list(prevail.keys())
    # smallest 10
    if with_label is None:
        half_len = int(.5*len(l_keys))
        with_label = min(l_keys[:min(10,half_len)])
    if replace_label is None:
        half_len = int(.5*len(l_keys))
        replace_label = max(list(prevail.keys())[half_len:])

    reassign = {replace_label:with_label}
    np_labels = copy_labels.values
    for i, reg in enumerate(reassign.keys()):
        np_labels[np_labels == reg] = reassign[reg]
    copy_labels.values = np_labels

    return copy_labels

def merge_labels_within_lonlatbox(precur, lonlatbox=list):
    prec_labels = precur.prec_labels.copy()
    corr_xr = precur.corr_xr
    new = core_pp.get_selbox(prec_labels, lonlatbox)
    regions = np.unique(new)[~np.isnan(np.unique(new))]
    pregs = [] ; nregs = [] # positive negative regions
    for r in regions:
        m = view_or_replace_labels(prec_labels, regions=[r])
        s = np.sign(np.mean(corr_xr.values[~np.isnan(m).values]))
        if s == 1:
            pregs.append(r)
        elif s == -1:
            nregs.append(r)
    for regs in [pregs, nregs]:
        if len(regs) != 0:
            maskregions = view_or_replace_labels(prec_labels, regions=regs)
            prec_labels.values[~np.isnan(maskregions).values] = min(regs)
    return prec_labels

def view_or_replace_labels(xarr: xr.DataArray, regions: Union[int,list],
           replacement_labels: Union[int,list]=None):
    '''
    View or replace a subset of labels.

    Parameters
    ----------
    xarr : xr.DataArray
        xarray with precursor region labels.
    regions : Union[int,list]
        region labels to select (for replacement).
    replacement_labels : Union[int,list], optional
        If replacement_labels given, should be same length as regions.
        The default is that no labels are replaced.

    Returns
    -------
    xarr : xr.DataArray
        xarray with precursor labels defined by argument regions, if
        replacement_labels are given; region labels are replaced by values
        in replacement_labels.

    '''
    if replacement_labels is None:
        replacement_labels = regions
    if type(regions) is int:
        regions = [regions]
    if type(replacement_labels) is int:
        replacement_labels = [replacement_labels]
    xarr = xarr.copy() # avoid replacement of init prec_labels xarray
    shape = xarr.shape
    df = pd.Series(np.round(xarr.values.flatten(), 0), dtype=float)
    d = dict(zip(regions, replacement_labels))
    out = df.map(d).values
    xarr.values = out.reshape(shape)
    return xarr

def labels_to_df(prec_labels, corr_xr=None, return_mean_latlon=True):
    '''
    Returns pd.DataFrame with labels and the # of gridcells in each label and
    optional extra stuff.

    Parameters
    ----------
    prec_labels : xr.DataArray
        xr.DataArray with precursor labels from DBSCAN clustering.
    corr_xr : xr.DataArray, optional
        Corresponding correlation map, if given returns sign of each region.
        The default is None.
    return_mean_latlon : bool, optional
        If return_mean_latlon==True, returns mean latitude longitude coordinate.
        The default is True.

    Returns
    -------
    pd.DataFrame

    '''

    dims = [d for d in prec_labels.dims if d not in ['latitude', 'longitude']]
    df = prec_labels.mean(dim=tuple(dims)).to_dataframe().dropna()

    if corr_xr is not None:
        spatdim = ['latitude', 'longitude', 'lat', 'lon', 'mask']
        dims = [d for d in prec_labels.dims if d not in spatdim]
        xrlabels = prec_labels.mean(tuple(dims)).values
        xrcorr = corr_xr.mean(tuple(dims)).values

    if return_mean_latlon or corr_xr is not None:
        labels = np.unique(prec_labels)[~np.isnan(np.unique(prec_labels))]
        if return_mean_latlon and corr_xr is not None:
            data = np.zeros( (len(labels), 4))
        elif return_mean_latlon and corr_xr is None:
            data = np.zeros( (len(labels), 3))
        else:
            data = np.zeros( (len(labels), 1))

        for i,l in enumerate(labels):
            if return_mean_latlon:
                columns_crds =['latitude', 'longitude', 'n_gridcells']
                latlon = np.array(df[(df==l).values].index)
                latlon = np.array([list(l) for l in latlon])

                if latlon.size != 0:
                    data[i][:2] = np.median(latlon,axis=0)
                    # data[i][:2] = latlon.mean(0) # mean coords
                    data[i][2] = latlon.shape[0] # size
            else:
                columns_crds = []

            if corr_xr is not None:
                data[i][-1] = np.sign(xrcorr[xrlabels==l].mean())
                columns_s = ['sign']
            else:
                columns_s = []

        columns = columns_crds + columns_s

        df = pd.DataFrame(data, index=labels,
                     columns=columns)

    return df

def df_data_prec_regs(list_MI, df_splits): #, outdic_precur, df_splits, TV #TODO
    '''
    Be aware: the amount of precursor vary over train test splits,
    each split will contain a column if a precursor was present in
    only one of the splits. These columns should be dropna'ed when
    extracting the actual timeseries belonging to a single split.
    If a precursor timeseries does not belong to a particular split
    the columns will be filled with nans.
    '''
    #%%
    splits = df_splits.index.levels[0]
    n_regions_list = []
    df_data_s   = np.zeros( (splits.size) , dtype='object')
    for s in range(splits.size):

        # create list with all actors, these will be merged into the fulldata array
        # allvar = list(self.outdic_precur.keys())
        var_names_corr = [] ; precur_list = [] ; cols = []

        for var_idx, precur in enumerate(list_MI):
            if hasattr(precur, 'ts_corr'):
                if precur.ts_corr[s].size != 0:
                    ts_train = precur.ts_corr[s].values
                    precur_list.append(ts_train)
                    # create array which numbers the regions
                    n_regions = precur.ts_corr[s].shape[1]
                    precur.var_info = [[i+1, precur.ts_corr[s].columns[i], var_idx] for i in range(n_regions)]
                    # Array of corresponing regions with var_names_corr (first entry is RV)
                    var_names_corr = var_names_corr + precur.var_info
                    cols.append(list(precur.ts_corr[s].columns))
                    index_dates = precur.ts_corr[s].index
                else:
                    print(f'No timeseries retrieved for {precur.name} on split {s}')

        # stack actor time-series together:
        if len(precur_list) > 0:
            fulldata = np.concatenate(tuple(precur_list), axis = 1)
            n_regions_list.append(fulldata.shape[1])
            df_data_s[s] = pd.DataFrame(fulldata, columns=flatten(cols), index=index_dates)
        else:
            df_data_s[s] = pd.DataFrame(index=df_splits.loc[s].index) # no ts for this split


    print(f'There are {n_regions_list} regions in total (list of different splits)')
    df_data  = pd.concat(list(df_data_s), keys= range(splits.size), sort=False)
    #%%
    return df_data


def import_precur_ts(list_import_ts : List[tuple],
                     df_splits: pd.DataFrame,
                     start_end_date: Tuple[str, str],
                     start_end_year: Tuple[int, int],
                     start_end_TVdate: Tuple[str, str],
                     # cols: list=None,
                     precur_aggr: int=1):
    '''
    list_import_ts has format List[tuples],
    [(columns, path_data)]

    columns : int or list
    columns to load from pd.DataFrame called df_data
    '''
    #%%
    # list_import_ts = self.list_import_ts
    # start_end_year = kwrgs_load['start_end_year']; start_end_TVdate=kwrgs_load['start_end_TVdate']

    splits = df_splits.index.levels[0]
    orig_traintest = functions_pp.get_testyrs(df_splits)
    df_data_ext_s   = np.zeros( (splits.size) , dtype='object')
    counter = 0
    for i, (cols, path_data) in enumerate(list_import_ts):

        df_data_e_all = functions_pp.load_hdf5(path_data)['df_data']
        if type(df_data_e_all) is pd.Series:
            df_data_e_all = pd.DataFrame(df_data_e_all)

        if cols is None:
            cols = list(df_data_e_all.columns[(df_data_e_all.dtypes != bool).values])
        elif type(cols) is str:
            cols = [cols]

        if hasattr(df_data_e_all.index, 'levels'):
            dates_subset = core_pp.get_subdates(df_data_e_all.loc[0].index, start_end_date,
                                                start_end_year)
            dates_subset = pd.MultiIndex.from_product([df_data_e_all.index.levels[0],
                                                       dates_subset])
            df_data_e_all = df_data_e_all.loc[dates_subset]
        else:
            dates_subset = core_pp.get_subdates(df_data_e_all.index, start_end_date,
                                                start_end_year)
            df_data_e_all = df_data_e_all.loc[dates_subset]

        if 'TrainIsTrue' in df_data_e_all.columns:
            _c = [k for k in df_splits.columns if k in ['TrainIsTrue', 'RV_mask']]
            # check if traintest split is correct
            ext_traintest = functions_pp.get_testyrs(df_data_e_all[_c])
            _check_traintest = all(np.equal(core_pp.flatten(ext_traintest), core_pp.flatten(orig_traintest)))
            assert _check_traintest, ('Train test years of df_splits are not the '
                                      'same as imported timeseries')

        for s in range(splits.size):
            if 'TrainIsTrue' in df_data_e_all.columns:
                df_data_e = df_data_e_all.loc[s]
            else:
                df_data_e = df_data_e_all


            df_data_ext_s[s] = df_data_e[cols]
            tfreq_date_e = int(np.median((df_data_e.index[1:]-df_data_e.index[:-1]).days))

            if precur_aggr != tfreq_date_e:
                try:
                    df_data_ext_s[s] = functions_pp.time_mean_bins(df_data_ext_s[s],
                                                         precur_aggr,
                                                        start_end_date,
                                                        start_end_year,
                                                        start_end_TVdate=start_end_TVdate)[0]
                except KeyError as e:
                    print('KeyError captured, likely the requested dates '
                          'given by start_end_date and start_end_year are not'
                          'found in external pandas timeseries.\n{}'.format(str(e)))
        print(f'loaded in exterinal timeseres: {cols}')

        if counter == 0:
            df_data_ext = pd.concat(list(df_data_ext_s), keys=range(splits.size))
        else:
            df_add = pd.concat(list(df_data_ext_s), keys=range(splits.size))
            df_data_ext = df_data_ext.merge(df_add, left_index=True, right_index=True)
        counter += 1 ; cols = None
    #%%
    return df_data_ext