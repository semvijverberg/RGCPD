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
import find_precursors
#import plot_maps
flatten = lambda l: list(itertools.chain.from_iterable(l))




class BivariateMI:

    def __init__(self, name, func=None, kwrgs_func={}, lags=np.array([1]), 
                 distance_eps=400, min_area_in_degrees2=3, group_split='together', 
                 calc_ts='region_mean', verbosity=1):

        self.name = name
        if func is None:
            self.func = self.corr_map
            
        else:
            self.func = self.corr_map
        if kwrgs_func is None:
            self.kwrgs_func = {'alpha':.05, 'FDR_control':True}
        else:
            self.kwrgs_func = kwrgs_func


        #get_prec_ts & spatial_mean_regions
        self.calc_ts = calc_ts
        # cluster_DBSCAN_regions
        self.distance_eps = distance_eps
        self.min_area_in_degrees2 = min_area_in_degrees2
        self.group_split = group_split
        
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
        
        A land sea mask is assumed from settin all the nan value to True (masked).
        For xrcorr['mask'], all gridcell which are significant are not masked,
        i.e. bool == False
        """

        n_lags = len(self.lags)
        lags = self.lags
        assert n_lags >= 0, ('Maximum lag is larger then minimum lag, not allowed')

        self.df_splits = df_splits # add df_splits to self
        n_spl = df_splits.index.levels[0].size
        # make new xarray to store results
        xrcorr = precur_arr.isel(time=0).drop('time').copy()
        orig_mask = np.isnan(precur_arr[0])
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
        def corr_single_split(RV_ts, precur, alpha, FDR_control): #, lags, alpha, FDR_control

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

                if FDR_control == True:
                    # test for Field significance and mask unsignificant values
                    # FDR control:
                    adjusted_pvalues = multicomp.multipletests(pval, method='fdr_bh')
                    ad_p = adjusted_pvalues[1]

                    corr_val.mask[ad_p > alpha] = True

                else:
                    corr_val.mask[pval > alpha] = True


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

            ma_data = corr_single_split(RV_ts, precur, **self.kwrgs_func)
            np_data[s] = ma_data.data
            np_mask[s] = ma_data.mask
        print("\n")
        xrcorr.values = np_data
        mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
        xrcorr.coords['mask'] = mask
        # fill nans with mask = True
        xrcorr['mask'] = xrcorr['mask'].where(orig_mask==False, other=orig_mask) 
        #%%
        return xrcorr
  
    # def corr_map(self, precur_arr, df_splits, RV): #, lags=np.array([0]), alpha=0.05, FDR_control=True #TODO
    #     #%%
    #     #    v = ncdf ; V = array ; RV.RV_ts = ts of RV, time_range_all = index range of whole ts
    #     """
    #     This function calculates the correlation maps for precur_arr for different lags.
    #     Field significance is applied to test for correltion.
    #     This function uses the following variables (in the ex dictionary)
    #     prec_arr: array
    #     time_range_all: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
    #     lag_steps: number of lags
    #     time_cycle: time cycyle of dataset, =12 for monthly data...
    #     RV_period: indices that matches the response variable time series
    #     alpha: significance level

    #     """
        
    #     self.df_splits = df_splits # add df_splits to self
    #     n_lags = len(self.lags)
    #     lags = self.lags
    #     assert n_lags >= 0, ('Maximum lag is larger then minimum lag, not allowed')



    #     n_spl = df_splits.index.levels[0].size
    #     # make new xarray to store results
    #     xrcorr = precur_arr.isel(time=0).drop('time').copy()
    #     # add lags
    #     list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(n_lags)]
    #     xrcorr = xr.concat(list_xr, dim = 'lag')
    #     xrcorr['lag'] = ('lag', lags)
    #     # add train test split
    #     list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(n_spl)]
    #     xrcorr = xr.concat(list_xr, dim = 'split')
    #     xrcorr['split'] = ('split', range(n_spl))

    #     print('\n{} - calculating correlation maps'.format(precur_arr.name))
    #     np_data = np.zeros_like(xrcorr.values)
    #     np_mask = np.zeros_like(xrcorr.values)
        
    #     def corr_single_split(RV_ts, precur_RV, alpha, FDR_control): #, lags, alpha, FDR_control

    #         lat = precur_RV.latitude.values
    #         lon = precur_RV.longitude.values

    #         z = np.ones((lat.size*lon.size,len(lags) ) )
    #         Corr_Coeff = np.ma.array(z, mask=z)


    #         dates_RV = RV_ts.index
    #         for i, lag in enumerate(lags):

    #             dates_lag = functions_pp.func_dates_min_lag(dates_RV, lag)[1]
    #             prec_lag = precur_RV.sel(time=dates_lag)
    #             prec_lag = np.reshape(prec_lag.values, (prec_lag.shape[0],-1))


    #             # correlation map and pvalue at each grid-point:
    #             corr_val, pval = corr_new(prec_lag, RV_ts.values.squeeze())

    #             if FDR_control == True:
    #                 # test for Field significance and mask unsignificant values
    #                 # FDR control:
    #                 adjusted_pvalues = multicomp.multipletests(pval, method='fdr_bh')
    #                 ad_p = adjusted_pvalues[1]

    #                 corr_val.mask[ad_p <= alpha] = False

    #             else:
    #                 corr_val.mask[pval <= alpha] = False


    #             Corr_Coeff[:,i] = corr_val[:]

    #         Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
    #         Corr_Coeff = Corr_Coeff.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
    #         return Corr_Coeff

    #     RV_mask = df_splits.loc[0]['RV_mask']
    #     for s in xrcorr.split.values:
    #         progress = 100 * (s+1) / n_spl
    #         # =============================================================================
    #         # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']
    #         # =============================================================================
    #         RV_train_mask = np.logical_and(RV_mask, df_splits.loc[s]['TrainIsTrue'])
    #         RV_ts = RV.fullts[RV_train_mask.values]
    #         precur_RV = precur_arr[df_splits.loc[s]['TrainIsTrue'].values]

    #     #        dates_RV  = pd.to_datetime(RV_ts.time.values)
    #         dates_RV = RV_ts.index
    #         n = dates_RV.size ; r = int(100*n/RV.dates_RV.size )
    #         print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")

    #         ma_data = corr_single_split(RV_ts, precur_RV, **self.kwrgs_func)
    #         np_data[s] = ma_data.data
    #         np_mask[s] = ma_data.mask
    #     print("\n")
    #     xrcorr.values = np_data
    #     mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
    #     xrcorr.coords['mask'] = mask
    #     #%%
    #     return xrcorr
    
    def get_prec_ts(self, precur_aggr=None, kwrgs_load=None): #, outdic_precur #TODO
        # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
        # stacked on top of each other (from lag_min to lag_max)

        n_tot_regs = 0
        # allvar = list(self.outdic_precur.keys()) # list of all variable names
        # for var in allvar[:]: # loop over all variables
        # precur = self.outdic_precur[var]
        splits = self.corr_xr.split
        if np.isnan(self.prec_labels.values).all():
            self.ts_corr = np.array(splits.size*[[]])
        else:
            if self.calc_ts == 'region_mean':
                self.ts_corr = find_precursors.spatial_mean_regions(self, 
                                              precur_aggr=precur_aggr, 
                                              kwrgs_load=kwrgs_load)
            elif self.calc_ts == 'spatcov':
                pass
            # self.outdic_precur[var] = precur
            n_tot_regs += max([self.ts_corr[s].shape[1] for s in range(splits.size)])
        return 

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

# def corr_new(field, ts):
#     """
#     This function calculates the correlation coefficent r and 
#     the pvalue p for each grid-point of field vs response-variable ts
#     Note, mask is True, == insignificant
#     """
#     x = np.ma.ones(field.shape[1])
#     corr_vals = np.ma.array(data = x, mask=True)
#     pvals = np.array(x)
#     fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    
#     nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]
    
#     for i in nonans_gc:
#         corr_vals[i], pvals[i] = scipy.stats.pearsonr(ts,field[:,i])
        
#     return corr_vals, pvals

# def loop_get_spatcov(precur):
    
#     df_splits = precur.df_splits
#     splits = df_splits.index.levels[0]
#     for s in splits:
#         lag = 0
#         TrainIsTrue = df_splits.loc[s]['TrainIsTrue']
#         times = df_splits.index
#         data = np.zeros( (len(columns), times.size) )
#         df_sp_s = pd.DataFrame(data.T, index=times, columns=columns)
#         dates_train = TrainIsTrue[TrainIsTrue.values].index
    
#         full_timeserie = precur.precur_arr
#         corr_vals = precur.corr_xr.sel(split=s).isel(lag=lag)
#         mask = precur.prec_labels.sel(split=s).isel(lag=lag)
#         pattern = corr_vals.where(~np.isnan(mask))
#         if np.isnan(pattern.values).all():
#             # no regions of this variable and split
#             pass
#         else:
#             if normalize == True:
#                 spatcov_full = calc_spatcov(full_timeserie, pattern)
#                 mean = spatcov_full.sel(time=dates_train).mean(dim='time')
#                 std = spatcov_full.sel(time=dates_train).std(dim='time')
#                 spatcov_test = ((spatcov_full - mean) / std)
#             elif normalize == False:
#                 spatcov_test = calc_spatcov(full_timeserie, pattern)
#             pd_sp = pd.Series(spatcov_test.values, index=times)
#             col = options[i]
#             df_sp_s[var + col] = pd_sp
