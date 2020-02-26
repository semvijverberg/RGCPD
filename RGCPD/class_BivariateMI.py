#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:17:25 2019

@author: semvijverberg
"""

import itertools
import numpy as np
import xarray as xr
#import datetime
import scipy
import pandas as pd
from statsmodels.sandbox.stats import multicomp
import functions_pp
from class_RV import RV_class
#import plot_maps
flatten = lambda l: list(itertools.chain.from_iterable(l))


class act:
    def __init__(self, name, filepath, corr_xr, precur_arr):
        self.name = name
        self.filepath = filepath
        self.corr_xr = corr_xr
        self.precur_arr = precur_arr
        self.lat_grid = precur_arr.latitude.values
        self.lon_grid = precur_arr.longitude.values
        self.area_grid = get_area(precur_arr)
        self.grid_res = abs(self.lon_grid[1] - self.lon_grid[0])

class BivariateMI:

    def __init__(self, name, func=None, kwrgs_func={}, lags=np.array([1]), 
                 distance_eps=400, min_area_in_degrees2=3, group_split='together', 
                 verbosity=1):
        # calculate_corr_maps
        # self.TV = TV
        # self.df_splits = df_splits
        # self.kwrgs_load = kwrgs_load
        # self.list_precur_pp = list_precur_pp
        self.name = name
        self.outdic_precur = None
        
        if func is None:
            self.func = self.corr_map
            self.kwrgs_func = {'alpha':.05, 'FDF_control':True}
        # if func is None
        # calculate_corr_maps & corr_map
        # self.lags = lags
        # self.alpha = alpha
        # self.FDR_control = FDR_control

        #get_prec_ts & spatial_mean_regions
        # self.kwrgs_load = kwrgs_load
        # self.precur_aggr = precur_aggr

        # cluster_DBSCAN_regions
        self.distance_eps = distance_eps
        self.min_area_in_degrees2 = min_area_in_degrees2
        self.group_split = group_split

        #RV_and_traintest
        # self.method = method
        # self.kwrgs_events = kwrgs_events
        # self.precursor_ts = precursor_ts
        # self.seed = seed
        self.verbosity = verbosity

        return
    
    
      
    def corr_map(self, precur_arr, df_splits, RV): #, lags=np.array([0]), alpha=0.05, FDR_control=True #TODO
        #%%
        #    v = ncdf ; V = array ; RV.RV_ts = ts of RV, time_range_all = index range of whole ts
        """
        This function calculates the correlation maps for precur_arr for different lags.
        Field significance is applied to test for correltion.
        This function uses the following variables (in the ex dictionary)
        prec_arr: array
        time_range_all: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
        lag_steps: number of lags
        time_cycle: time cycyle of dataset, =12 for monthly data...
        RV_period: indices that matches the response variable time series
        alpha: significance level

        """
        RV = self.TV
        n_lags = len(self.lags)
        lags = self.lags
        assert n_lags >= 0, ('Maximum lag is larger then minimum lag, not allowed')


        df_splits = self.df_splits
        n_spl = df_splits.index.levels[0].size
        # make new xarray to store results
        xrcorr = precur_arr.isel(time=0).drop('time').copy()
        # add lags
        list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(n_lags)]
        xrcorr = xr.concat(list_xr, dim = 'lag')
        xrcorr['lag'] = ('lag', lags)
        # add train test split
        list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(n_spl)]
        xrcorr = xr.concat(list_xr, dim = 'split')
        xrcorr['split'] = ('split', range(n_spl))

        print('\n{} - calculating correlation maps'.format(precur_arr.name))
        np_data = np.zeros_like(xrcorr.values)
        np_mask = np.zeros_like(xrcorr.values)
        def corr_single_split(RV_ts, precur): #, lags, alpha, FDR_control

            lat = precur_arr.latitude.values
            lon = precur_arr.longitude.values

            z = np.zeros((lat.size*lon.size,len(lags) ) )
            Corr_Coeff = np.ma.array(z, mask=z)


            dates_RV = RV_ts.index
            for i, lag in enumerate(lags):

                dates_lag = functions_pp.func_dates_min_lag(dates_RV, lag)[1]
                prec_lag = precur_arr.sel(time=dates_lag)
                prec_lag = np.reshape(prec_lag.values, (prec_lag.shape[0],-1))


                # correlation map and pvalue at each grid-point:
                corr_val, pval = corr_new(prec_lag, RV_ts.values.squeeze())

                if self.FDR_control == True:
                    # test for Field significance and mask unsignificant values
                    # FDR control:
                    adjusted_pvalues = multicomp.multipletests(pval, method='fdr_bh')
                    ad_p = adjusted_pvalues[1]

                    corr_val.mask[ad_p > self.alpha] = True

                else:
                    corr_val.mask[pval > self.alpha] = True


                Corr_Coeff[:,i] = corr_val[:]

            Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
            Corr_Coeff = Corr_Coeff.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
            return Corr_Coeff

        RV_mask = df_splits.loc[0]['RV_mask']
        for s in xrcorr.split.values:
            progress = 100 * (s+1) / n_spl
            # =============================================================================
            # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']
            # =============================================================================
            RV_train_mask = np.logical_and(RV_mask, df_splits.loc[s]['TrainIsTrue'])
            RV_ts = RV.fullts[RV_train_mask.values]
            precur = precur_arr[df_splits.loc[s]['TrainIsTrue'].values]

        #        dates_RV  = pd.to_datetime(RV_ts.time.values)
            dates_RV = RV_ts.index
            n = dates_RV.size ; r = int(100*n/RV.dates_RV.size )
            print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")

            ma_data = corr_single_split(RV_ts, precur)
            np_data[s] = ma_data.data
            np_mask[s] = ma_data.mask
        print("\n")
        xrcorr.values = np_data
        mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
        xrcorr.coords['mask'] = mask
        #%%
        return xrcorr
    
    def get_prec_ts(self, precur_aggr=None, kwrgs_load=None): #, outdic_precur #TODO
        # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
        # stacked on top of each other (from lag_min to lag_max)

        n_tot_regs = 0
        allvar = list(self.outdic_precur.keys()) # list of all variable names
        for var in allvar[:]: # loop over all variables
            precur = self.outdic_precur[var]
            splits = precur.corr_xr.split
            if np.isnan(precur.prec_labels.values).all():
                precur.ts_corr = np.array(splits.size*[[]])
            else:
                precur.ts_corr = self.spatial_mean_regions(precur, 
                                                  precur_aggr=precur_aggr, 
                                                  kwrgs_load=kwrgs_load)
                self.outdic_precur[var] = precur
                n_tot_regs += max([precur.ts_corr[s].shape[1] for s in range(splits.size)])
        return self.outdic_precur
    
    def df_data_prec_regs(self): #, outdic_precur, df_splits, TV #TODO
        '''
        Be aware: the amount of precursor vary over train test splits,
        each split will contain a column if a precursor was present in 
        only one of the splits. These columns should be dropna'ed when 
        extracting the actual timeseries belonging to a single split. 
        If a precursor timeseries does not belong to a particular split
        the columns will be filled with nans.
        '''
        #%%
        splits = self.df_splits.index.levels[0]
        n_regions_list = []
        df_data_s   = np.zeros( (splits.size) , dtype=object)
        for s in range(splits.size):

            # create list with all actors, these will be merged into the fulldata array
            allvar = list(self.outdic_precur.keys())
            var_names_corr = [] ; pos_prec_list = [] ; cols = [[self.TV.name]]
        
            for var in allvar[:]:
                pos_prec = self.outdic_precur[var]
                if pos_prec.ts_corr[s].size != 0:
                    ts_train = pos_prec.ts_corr[s].values
                    pos_prec_list.append(ts_train)
                    # create array which numbers the regions
                    var_idx = allvar.index(var) 
                    n_regions = pos_prec.ts_corr[s].shape[1]
                    pos_prec.var_info = [[i+1, pos_prec.ts_corr[s].columns[i], var_idx] for i in range(n_regions)]
                    # Array of corresponing regions with var_names_corr (first entry is RV)
                    var_names_corr = var_names_corr + pos_prec.var_info
                    cols.append(list(pos_prec.ts_corr[s].columns))
                    index_dates = pos_prec.ts_corr[s].index
            var_names_corr.insert(0, self.TV.name)
            # stack actor time-series together:
            fulldata = np.concatenate(tuple(pos_prec_list), axis = 1)
            n_regions_list.append(fulldata.shape[1])
            # add the full 1D time series of interest as first entry:
            fulldata = np.column_stack((self.TV.fullts, fulldata))
            df_data_s[s] = pd.DataFrame(fulldata, columns=flatten(cols), index=index_dates)
        print(f'There are {n_regions_list} regions for {var} (list of different splits)')
        self.df_data  = pd.concat(list(df_data_s), keys= range(splits.size))
        #%%
        return self.df_data

    def import_precur_ts(self, import_prec_ts, to_freq, start_end_date,
                     start_end_year): #df_splits
        '''
        import_prec_ts has format tuple (name, path_data)
        '''
        splits = self.df_splits.index.levels[0]
        df_data_ext_s   = np.zeros( (splits.size) , dtype=object)
        counter = 0
        for i, (name, path_data) in enumerate(import_prec_ts):
            for s in range(splits.size):
                # skip first col because it is the RV ts
                df_data_e = functions_pp.load_hdf5(path_data)['df_data'].iloc[:,1:].loc[s]
                cols_ts = np.logical_or(df_data_e.dtypes == 'float64', df_data_e.dtypes == 'float32')
                cols_ext = list(df_data_e.columns[cols_ts])
                # cols_ext must be of format '{}_{int}_{}'
                lab_int = 100
                for i, c in enumerate(cols_ext):
                    char = c.split('_')[1]
                    if char.isdigit():
                        pass
                    else:
                        cols_ext[i] = c.replace(char, str(lab_int)) + char
                        lab_int += 1
                        
                df_data_ext_s[s] = df_data_e[cols_ext]
                tfreq_date_e = (df_data_e.index[1] - df_data_e.index[0]).days
                
                if to_freq != tfreq_date_e:
                    try:
                        df_data_ext_s[s] = functions_pp.time_mean_bins(df_data_ext_s[s], 
                                                            to_freq,
                                                            start_end_date,
                                                            start_end_year)[0]
                    except KeyError as e:
                        print('KeyError captured, likely the requested dates '
                            'given by start_end_date and start_end_year are not' 
                            'found in external pandas timeseries.\n{}'.format(str(e)))
                                                            
            if counter == 0:
                df_data_ext = pd.concat(list(df_data_ext_s), keys=range(splits.size))
            else:
                df_data_ext.merge(df_data_ext, left_index=True, right_index=True)
        return df_data_ext

        #            df_data_ext = functions_pp.time_mean_bins(df_data_ext,
        #                                                     ex, ex['tfreq'], 
        #                                                     seldays='part')[0]
        #    # Expand var_names_corr
        #    n = var_names_full[-1][0] + 1 ; add_n = n + len(cols_ext)
        #    n_var_idx = var_names_full[-1][-1] + 1
        #    for i in range(n, add_n):
        #        var_names_full.append([i, cols_ext[i-n], n_var_idx])


    
    def RV_and_traintest(self, fullts, TV_ts): #, method=str, kwrgs_events=None, precursor_ts=None, seed=int, verbosity=1

        # Define traintest:
        df_fullts = pd.DataFrame(fullts.values, 
                                index=pd.to_datetime(fullts.time.values),
                                columns=[fullts.name])
        df_RV_ts = pd.DataFrame(TV_ts.values,
                                index=pd.to_datetime(TV_ts.time.values),
                                columns=['RV'+fullts.name])

        if self.method[:9] == 'ran_strat' and self.kwrgs_events is None and self.precursor_ts is not None:
                # events need to be defined to enable stratified traintest.
                self.kwrgs_events = {'event_percentile': 66,
                                'min_dur' : 1,
                                'max_break' : 0,
                                'grouped' : False}
                if self.verbosity == 1:
                    print("kwrgs_events not given, creating stratified traintest split "
                        "based on events defined as exceeding the {}th percentile".format(
                            self.kwrgs_events['event_percentile']))

        self.TV = RV_class(df_fullts, df_RV_ts, self.kwrgs_events)
        
        
        if self.precursor_ts is not None:
            print('Retrieve same train test split as imported ts')
            path_data = ''.join(self.precursor_ts[0][1])
            self.df_splits = functions_pp.load_hdf5(path_data)['df_data'].loc[:,['TrainIsTrue', 'RV_mask']]
            test_yrs_imp  = functions_pp.get_testyrs(self.df_splits)
            self.df_splits = functions_pp.rand_traintest_years(self.TV, test_yrs=test_yrs_imp,
                                                            method=self.method,
                                                            seed=self.seed, 
                                                            kwrgs_events=self.kwrgs_events, 
                                                            verb=self.verbosity)
        #        df_splits = functions_pp.rand_traintest_years(TV, method=self.method,
        #                                                          seed=self.seed, 
        #                                                          kwrgs_events=self.kwrgs_events, 
        #                                                          verb=self.verbosity)
            test_yrs_set  = functions_pp.get_testyrs(self.df_splits)
            assert (np.equal(test_yrs_imp, test_yrs_set)).all(), "Train test split not equal"
        else:
            self.df_splits = functions_pp.rand_traintest_years(self.TV, method=self.method,
                                                            seed=self.seed, 
                                                            kwrgs_events=self.kwrgs_events, 
                                                            verb=self.verbosity)
        return self.TV, self.df_splits


def calculate_region_maps(precur, func, TV, df_splits, kwrgs_load, list_precur_pp=list): #, lags=np.array([1]), alpha=0.05, FDR_control=True #TODO
    '''
    tfreq : aggregate precursors with bins of window size = tfreq 
    selbox : selbox is tuple of:
            (degrees_east, degrees_west, degrees_south, degrees_north)
    loadleap : if leap day should loaded yes/no
    seldates : if a selection of dates should be loaded
    
    '''  
    #%%

    precur.outdic_precur = dict()

    for name, filepath in list_precur_pp: # loop over all variables
        # =============================================================================
        # Unpack non-default arguments
        # =============================================================================
        kwrgs = {}
        for key, value in kwrgs_load.items():
            if type(value) is list and name in value[1].keys():
                kwrgs[key] = value[1][name]
            elif type(value) is list and name not in value[1].keys():
                kwrgs[key] = value[0] # plugging in default value
            else:
                kwrgs[key] = value
        #===========================================
        # 3c) Precursor field
        #===========================================
        precur_arr = functions_pp.import_ds_timemeanbins(filepath, **kwrgs)
        # =============================================================================
        # Calculate correlation
        # =============================================================================
        corr_xr = func(precur_arr, df_splits, kwrgs_load) #precur.corr_map

        # =============================================================================
        # Cluster into precursor regions
        # =============================================================================
        pos_prec = act(name, filepath, corr_xr, precur_arr)

        precur.outdic_precur[pos_prec.name] = pos_prec
    
    return precur.outdic_precur

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

def mask_sig_to_cluster(mask_and_data_s, wght_area, distance_eps, min_area_samples):
    from sklearn import cluster
    from sklearn import metrics
    from haversine import haversine
    mask_sig_1d = mask_and_data_s.mask.astype('bool').values == False
    data = mask_and_data_s.data
    lons = mask_and_data_s.longitude.values
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
            sign_coords.append( [(sign_c[1][i], sign_c[0][i]-180) for i in range(sign_c[0].size)] )
            weights_core_samples.append(wght_area[mask_sig[l,:,:]].reshape(-1))

        sign_coords = flatten(sign_coords)
        if len(sign_coords) != 0:
            weights_core_samples = flatten(weights_core_samples)
            # calculate distance between sign coords accross all lags to keep labels
            # more consistent when clustering
            distance = metrics.pairwise_distances(sign_coords, metric=haversine)
            dbresult = cluster.DBSCAN(eps=distance_eps, min_samples=min_area_samples,
                                      metric='precomputed').fit(distance,
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

def calc_spatcov(full_timeserie, pattern):
    #%%
    mask = np.ma.make_mask(np.isnan(pattern.values)==False)
    n_time = full_timeserie.time.size
    n_space = pattern.size

    # select only gridcells where there is not a nan
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))

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

    dates_test = full_timeserie.time
    # cov xarray
    spatcov = xr.DataArray(spatcov, coords=[dates_test.values], dims=['time'])
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

def corr_new(field, ts):
    """
    This function calculates the correlation coefficent r and 
    the pvalue p for each grid-point of field vs response-variable ts
    """
    x = np.ma.zeros(field.shape[1])
    corr_vals = np.ma.array(data = x, mask =False)
    pvals = np.array(x)
    fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]
    
    for i in nonans_gc:
        corr_vals[i], pvals[i] = scipy.stats.pearsonr(ts,field[:,i])
        
    return corr_vals, pvals

def cluster_DBSCAN_regions(pos_prec, distance_eps=400, min_area_in_degrees2=3, group_split='together'): 
    #%%
    """
    Calculates the time-series of the possible precursors based on the correlation coefficients and plots the according regions.
    Only caluclates regions with significant correlation coefficients
    """
    

    #    var = 'sst'
    #    pos_prec = outdic_actors[var]
    corr_xr  = pos_prec.corr_xr
    n_spl  = corr_xr.coords['split'].size
    lags = pos_prec.corr_xr.lag.values
    n_lags = lags.size
    lats    = corr_xr.latitude
    lons    = corr_xr.longitude
    area_grid   = pos_prec.area_grid/ 1E6 # in km2

    aver_area_km2 = 7939     # np.mean(pos_prec.area_grid) with latitude 0-90 / 1E6
    wght_area = area_grid / aver_area_km2
    min_area_km2 = min_area_in_degrees2 * 111.131 * min_area_in_degrees2 * 78.85
    min_area_samples = min_area_km2 / aver_area_km2

    

    prec_labels_np = np.zeros( (n_spl, n_lags, lats.size, lons.size), dtype=int )
    labels_sign_lag = np.zeros( (n_spl), dtype=list)
    mask_and_data = corr_xr.copy()

    # group regions per split (no information leak train test)
    if group_split == 'seperate':
        for s in range(n_spl):
            progress = 100 * (s+1) / n_spl
            print(f"\rProgress traintest set {progress}%", end="")
            mask_and_data_s = corr_xr.sel(split=s)
            grouping_split = mask_sig_to_cluster(mask_and_data_s, wght_area, 
                                                distance_eps, min_area_samples)
            prec_labels_np[s] = grouping_split[0]
            labels_sign_lag[s] = grouping_split[1]
    # group regions regions the same accross splits
    elif group_split == 'together':

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
            'to relax contrain.\n')
        prec_labels_ord = prec_labels_np
    if mask_and_data.mask.all()==True:
        print('\nNo significantly correlating gridcells found.\n')
        prec_labels_ord = prec_labels_np
    else:
        prec_labels_ord = np.zeros_like(prec_labels_np)
        if group_split == 'seperate':
            for s in range(n_spl):
                prec_labels_s = prec_labels_np[s]
                corr_vals     = corr_xr.sel(split=s).values
                reassign = reorder_strength(prec_labels_s, corr_vals, area_grid, 
                                            min_area_samples)
                prec_labels_ord[s] = relabel(prec_labels_s, reassign)
        elif group_split == 'together':
            # order based on mean corr_value:
            corr_vals = corr_xr.mean(dim='split').values
            prec_label_s = grouping_split[0].copy()
            reassign = reorder_strength(prec_label_s, corr_vals, area_grid, 
                                        min_area_samples)
            for s in range(n_spl):
                prec_labels_s = prec_labels_np[s]
                prec_labels_ord[s] = relabel(prec_labels_s, reassign)



    prec_labels = xr.DataArray(data=prec_labels_ord, coords=[range(n_spl), lags, lats, lons],
                        dims=['split', 'lag','latitude','longitude'],
                        name='{}_labels_init'.format(pos_prec.name),
                        attrs={'units':'Precursor regions [ordered for Corr strength]'})
    prec_labels = prec_labels.where(prec_labels_ord!=0.)
    prec_labels.attrs['title'] = prec_labels.name
    pos_prec.prec_labels = prec_labels
    #%%
    return pos_prec

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

def spatial_mean_regions(precur, precur_aggr=None, kwrgs_load=None):
    #%%

    name            = precur.name
    corr_xr         = precur.corr_xr
    prec_labels     = precur.prec_labels
    n_spl           = corr_xr.split.size
    lags            = precur.corr_xr.lag.values
    
    
    
    if precur_aggr is None:
        # use precursor array with temporal aggregation that was used to create 
        # correlation map
        precur_arr = precur.precur_arr
    else:
        # =============================================================================
        # Unpack kwrgs for loading 
        # =============================================================================
        filepath = precur.filepath
        kwrgs = {}
        for key, value in kwrgs_load.items():
            if type(value) is list and name in value[1].keys():
                kwrgs[key] = value[1][name]
            elif type(value) is list and name not in value[1].keys():
                kwrgs[key] = value[0] # plugging in default value
            else:
                kwrgs[key] = value
        kwrgs['tfreq'] = precur_aggr
        precur_arr = functions_pp.import_ds_timemeanbins(filepath, **kwrgs)
        
    dates = pd.to_datetime(precur_arr.time.values)
    actbox = precur_arr.values
    
    ts_corr = np.zeros( (n_spl), dtype=object)

    for s in range(n_spl):
        corr = corr_xr.isel(split=s) # changed this from 0 to s 13-12-19
        labels = prec_labels.isel(split=s) # changed this from 0 to s 13-12-19

        ts_list = np.zeros( (lags.size), dtype=list )
        track_names = []
        for l_idx, lag in enumerate(lags):
            labels_lag = labels.isel(lag=l_idx).values

            regions_for_ts = list(np.unique(labels_lag[~np.isnan(labels_lag)]))
            a_wghts = precur.area_grid / precur.area_grid.mean()

            # this array will be the time series for each feature
            ts_regions_lag_i = np.zeros((actbox.shape[0], len(regions_for_ts)))
            # ts_regions_lag_i = []
            
            # track sign of eacht region
            sign_ts_regions = np.zeros( len(regions_for_ts) )


            # calculate area-weighted mean over features
            for r in regions_for_ts:
                
                idx = regions_for_ts.index(r)
                # start with empty lonlat array
                B = np.zeros(labels_lag.shape)
                # Mask everything except region of interest
                B[labels_lag == r] = 1
        #        # Calculates how values inside region vary over time, wgts vs anomaly
        #        wgts_ano = meanbox[B==1] / meanbox[B==1].max()
        #        ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * cos_box_array[:,B==1] * wgts_ano, axis =1)
                # Calculates how values inside region vary over time
                ts = np.nanmean(actbox[:,B==1] * a_wghts[B==1], axis =1)

                # check for nans
                if ts[np.isnan(ts)].size !=0:
                    print(ts)
                    perc_nans = ts[np.isnan(ts)].size / ts.size
                    if perc_nans == 1:
                        # all NaNs
                        print(f'All timesteps were NaNs split {s}'
                            f' for region {r} at lag {lag}')
                        
                    else:
                        print(f'{perc_nans} NaNs split {s}'
                            f' for region {r} at lag {lag}')
                    
    
                track_names.append(f'{lag}..{int(r)}..{name}')

                ts_regions_lag_i[:,idx] = ts
                # get sign of region
                sign_ts_regions[idx] = np.sign(np.mean(corr.isel(lag=l_idx).values[B==1]))                
                
            ts_list[l_idx] = ts_regions_lag_i

        tsCorr = np.concatenate(tuple(ts_list), axis = 1)
        df_tscorr = pd.DataFrame(tsCorr, index=dates,
                                columns=track_names)
        df_tscorr.name = str(s)
        ts_corr[s] = df_tscorr
    if any(df_tscorr.isna().values.flatten()):
        print('Warnning: nans detected')
    #%%
    return ts_corr