#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:17:25 2019

@author: semvijverberg
"""
import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator

import func_models as fc_utils
import itertools, os, re
import numpy as np
import xarray as xr
import scipy
import pandas as pd
from statsmodels.sandbox.stats import multicomp
import functions_pp, core_pp
import find_precursors
from func_models import apply_shift_lag
from class_RV import RV_class
from typing import Union
import uuid
flatten = lambda l: list(itertools.chain.from_iterable(l))
from joblib import Parallel, delayed

try:
    from tigramite.independence_tests import ParCorr
except:
    pass


class BivariateMI:

    def __init__(self, name, filepath: str=None,func=None, kwrgs_func={},
                 alpha: float=0.05, FDR_control: bool=True, lags=np.array([1]),
                 lag_as_gap: bool=False, distance_eps: int=400,
                 min_area_in_degrees2=3, group_lag: bool=False,
                 group_split : bool=True, calc_ts='region mean',
                 selbox: tuple=None, use_sign_pattern: bool=False,
                 use_coef_wghts: bool=True, path_hashfile: str=None,
                 hash_str: str=None, dailytomonths: bool=False, n_cpu=1,
                 n_jobs_clust=-1, verbosity=1):
        '''

        Parameters
        ----------
        name : str
            Name that links to a filepath pointing to a Netcdf4 file.
        filepath : str
            Optionally give manual filepath to Netcdf4. Normally this is
            automatically done by the RGCPD framework.
        func : function to apply to calculate the bivariate
            Mutual Informaiton (MI), optional
            The default is applying a correlation map.
        kwrgs_func : dict, optional
            Arguments for func. The default is {}.
        alpha : float, optional
            significance threshold
        FDR_control: bool, optional
            Control for multiple hypothesis testing
        lags : np.ndarray, optional
            lag w.r.t. the the target variable at which to calculate the MI.
            The default is np.array([1]).
        lag_as_gap : bool, optional
            Interpret the lag as days in between last day of precursor
            aggregation window and first day of target window.
        distance_eps : int, optional
            The maximum distance between two gridcells for one to be considered
            as in the neighborhood of the other, only gridcells with the same
            sign are grouped together.
            The default is 400.
        min_area_in_degrees2 : TYPE, optional
            The number of samples gridcells in a neighborhood for a
            region to be considered as a core point. The parameter is
            propotional to the average size of 1 by 1 degree gridcell.
            The default is 400.
        group_split : str, optional
            If True, then region labels will be equal between different train test splits.
            If False, splits will clustered separately.
            The default is 'together'.
        calc_ts : str, optional
            Choose 'region mean' or 'pattern cov'. If 'region mean', a
            timeseries is calculated for each label. If 'pattern cov', the
            spatial covariance of the whole pattern is calculated.
            The default is 'region_mean'.
        selbox : tuple, optional
            has format of (lon_min, lon_max, lat_min, lat_max)
        use_sign_pattern : bool, optional
            When calculating spatial covariance, do not use original pattern
            but focus on the sign of each region. Used for quantifying Rossby
            waves.
        use_coef_wghts : bool, optional
            When True, using (corr) coefficient values as weights when calculating
            spatial mean. (will always be area weighted).
        dailytomonths : bool, optional
            When True, the daily input data will be aggregated to monthly data,
            subsequently, the pre-processing steps are performed (detrend/anomaly).
        n_cpu : int, optional
            Calculate different train-test splits in parallel using Joblib.
        n_jobs_clust : int, optional
            Perform DBSCAN clustering calculation in parallel. Beware that for
            large memory precursor fields with many precursor regions, DBSCAN
            can become memory demanding. If all cpu's are used, there may not be
            sufficient working memory for each cpu left.
        verbosity : int, optional
            Not used atm. The default is 1.

        Returns
        -------
        Initialization of the BivariateMI class

        '''
        self.name = name
        if func is None:
            self.func = corr_map
        else:
            self.func = func
        self._name = name + '_'+ self.func.__name__

        self.kwrgs_func = kwrgs_func

        self.alpha = alpha
        self.FDR_control = FDR_control

        #get_prec_ts & spatial_mean_regions
        self.filepath = filepath
        self.calc_ts = calc_ts
        self.selbox = selbox
        self.use_sign_pattern = use_sign_pattern
        self.use_coef_wghts = use_coef_wghts
        self.lags = lags
        self.lag_as_gap = lag_as_gap


        # cluster_DBSCAN_regions
        self.distance_eps = distance_eps
        self.min_area_in_degrees2 = min_area_in_degrees2
        self.group_split = group_split
        self.group_lag = group_lag

        # other parameters
        self.dailytomonths = dailytomonths
        self.verbosity = verbosity
        self.n_cpu = n_cpu
        self.n_jobs_clust = n_jobs_clust
        if hash_str is not None:
            assert path_hashfile is not None, 'Give path to search hashfile'
            self.load_files(self, path_hashfile=str, hash_str=str)

        return


    def bivariateMI_map(self, precur_arr, df_splits, df_RVfull): #
        #%%
        # self=rg.list_for_MI[0] ; precur_arr=self.precur_arr ;
        # df_splits = rg.df_splits ; df_RVfull = rg.df_fullts
        """
        This function calculates the correlation maps for precur_arr for different lags.
        Field significance is applied to test for correltion.
        RV_period: indices that matches the response variable time series
        alpha: significance level

        A land sea mask is assumed from settin all the nan value to True (masked).
        For xrcorr['mask'], all gridcell which are significant are not masked,
        i.e. bool == False
        """

        if type(self.lags) is np.ndarray and type(self.lags[0]) is not np.ndarray:
            self.lags = np.array(self.lags, dtype=np.int16) # fix dtype
            self.lag_coordname = self.lags
        else:
            self.lag_coordname = np.arange(len(self.lags)) # for period_means
        n_lags = len(self.lags)
        lags = self.lags
        self.df_splits = df_splits # add df_splits to self
        dates = self.df_splits.loc[0].index

        TrainIsTrue = df_splits.loc[0]['TrainIsTrue']
        RV_train_mask = np.logical_and(df_splits.loc[0]['RV_mask'], TrainIsTrue)
        if hasattr(df_RVfull.index, 'levels'):
            RV_ts = df_RVfull.loc[0][RV_train_mask.values]
        else:
            RV_ts = df_RVfull[RV_train_mask.values]
        targetstepsoneyr = functions_pp.get_oneyr(RV_ts)
        if type(self.lags[0]) == np.ndarray and targetstepsoneyr.size>1:
            raise ValueError('Precursor and Target do not align.\n'
                             'One aggregated value taken for months '
                             f'{self.lags[0]}, while target timeseries has '
                             f'multiple timesteps per year:\n{targetstepsoneyr}')
        yrs_precur_arr = np.unique(precur_arr.time.dt.year)
        if np.unique(dates.year).size != yrs_precur_arr.size:
            raise ValueError('Numer of years between precursor and Target '
                             'not match. Check if precursor period is crossyr, '
                             'while target period is not. '
                             'Mannually ensure start_end_year is aligned.')

        oneyr = functions_pp.get_oneyr(dates)
        if oneyr.size == 1: # single val per year precursor
            self._tfreq = 365
        else:
            self._tfreq = (oneyr[1] - oneyr[0]).days

        n_spl = df_splits.index.levels[0].size
        # make new xarray to store results
        xrcorr = precur_arr.isel(time=0).drop('time').copy()
        orig_mask = np.isnan(precur_arr[1])
        if 'lag' not in xrcorr.dims:
            # add lags
            list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(n_lags)]
            xrcorr = xr.concat(list_xr, dim = 'lag')
            xrcorr['lag'] = ('lag', self.lag_coordname)
        # add train test split
        list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(n_spl)]
        xrcorr = xr.concat(list_xr, dim = 'split')
        xrcorr['split'] = ('split', range(n_spl))
        xrpvals = xrcorr.copy()


        def MI_single_split(df_RVfull, precur_train, s, alpha=.05, FDR_control=True):


            lat = precur_train.latitude.values
            lon = precur_train.longitude.values

            z = np.zeros((lat.size*lon.size,len(lags) ) )
            Corr_Coeff = np.ma.array(z, mask=z)
            pvals = np.ones((lat.size*lon.size,len(lags) ) )

            RV_mask = df_splits.loc[s]['RV_mask']
            TrainIsTrue = df_splits.loc[s]['TrainIsTrue'].values==True
            RV_train_mask = np.logical_and(RV_mask, TrainIsTrue)
            if hasattr(df_RVfull.index, 'levels'):
                df_RVfull_s = df_RVfull.loc[s]
            else:
                df_RVfull_s = df_RVfull
            RV_ts = df_RVfull_s[RV_train_mask.values]
            dates_RV = RV_ts.index
            for i, lag in enumerate(lags):
                if type(lag) is np.int16 and self.lag_as_gap==False:
                    if self.func.__name__=='parcorr_map':
                        corr_val, pval = self.func(precur_train, df_RVfull_s,
                                                   self.df_splits.loc[s].copy(),
                                                   lag=lag,
                                                   **self.kwrgs_func)
                    else:
                        m = apply_shift_lag(self.df_splits.loc[s], lag)
                        dates_lag = m[np.logical_and(m['TrainIsTrue']==1, m['x_fit'])].index
                        corr_val, pval = self.func(precur_train.sel(time=dates_lag),
                                                   RV_ts.values.squeeze(),
                                                   **self.kwrgs_func)
                elif type(lag) == np.int16 and self.lag_as_gap==True:
                    # if only shift tfreq, then gap=0
                    datesdaily = RV_class.aggr_to_daily_dates(dates_RV, tfreq=self._tfreq)
                    dates_lag = functions_pp.func_dates_min_lag(datesdaily,
                                                                self._tfreq+lag)[1]

                    tmb = functions_pp.time_mean_bins
                    corr_val, pval = self.func(tmb(precur_train.sel(time=dates_lag),
                                                           to_freq=self._tfreq)[0],
                                               RV_ts.values.squeeze(),
                                               **self.kwrgs_func)
                elif type(lag) == np.ndarray:
                    corr_val, pval = self.func(precur_train.sel(lag=i),
                                               RV_ts.values.squeeze(),
                                               **self.kwrgs_func)



                mask = np.ones(corr_val.size, dtype=bool)
                if FDR_control == True:
                    # test for Field significance and mask unsignificant values
                    # FDR control:
                    adjusted_pvalues = multicomp.multipletests(pval, method='fdr_bh')
                    ad_p = adjusted_pvalues[1]
                    pvals[:,i] = ad_p
                    mask[ad_p <= alpha] = False

                else:
                    pvals[:,i] = pval
                    mask[pval <= alpha] = False

                Corr_Coeff[:,i] = corr_val[:]
                Corr_Coeff[:,i].mask = mask

            Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
            Corr_Coeff = Corr_Coeff.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
            pvals = pvals.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
            return Corr_Coeff, pvals

        print('\n{} - calculating correlation maps'.format(precur_arr.name))
        np_data = np.zeros_like(xrcorr.values)
        np_mask = np.zeros_like(xrcorr.values)
        np_pvals = np.zeros_like(xrcorr.values)


        #%%
        # start_time = time()

        def calc_corr_for_splits(self, splits, df_RVfull, np_precur_arr, df_splits, output):
            '''
            Wrapper to divide calculating a number of splits per core, instead of
            assigning each split to a seperate worker.
            '''
            n_spl = df_splits.index.levels[0].size
            # reload numpy array to xarray (xarray not always picklable by joblib)
            precur_arr = core_pp.back_to_input_dtype(np_precur_arr[0], np_precur_arr[1],
                                                     np_precur_arr[2])
            RV_mask = df_splits.loc[0]['RV_mask']
            for s in splits:
                progress = int(100 * (s+1) / n_spl)
                # =============================================================================
                # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']
                # =============================================================================
                TrainIsTrue = df_splits.loc[s]['TrainIsTrue'].values==True
                if self.lag_as_gap: # no clue why selecting all datapoints, changed 26-01-2021
                    train_dates = df_splits.loc[s]['TrainIsTrue'][TrainIsTrue].index
                    precur_train = precur_arr.sel(time=train_dates)
                else:
                    precur_train = precur_arr[TrainIsTrue] # only train data

                n = RV_ts.size ; r = int(100*n/RV_mask[RV_mask].size)
                print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")

                output[s] = MI_single_split(df_RVfull, precur_train, s,
                                            alpha=self.alpha,
                                            FDR_control=self.FDR_control)
            return output

        output = {}
        np_precur_arr = core_pp.to_np(precur_arr)
        if self.n_cpu == 1:
            splits = df_splits.index.levels[0]
            output = calc_corr_for_splits(self, splits, df_RVfull, np_precur_arr,
                                          df_splits, output)
        elif self.n_cpu > 1:
            splits = df_splits.index.levels[0]
            futures = []
            for _s in np.array_split(splits, self.n_cpu):
                futures.append(delayed(calc_corr_for_splits)(self, _s, df_RVfull,
                                                             np_precur_arr, df_splits,
                                                             output))
            futures = Parallel(n_jobs=self.n_cpu, backend='loky')(futures)
            [output.update(d) for d in futures]

        # unpack results
        for s in xrcorr.split.values:
            ma_data, pvals = output[s]
            np_data[s] = ma_data.data
            np_mask[s] = ma_data.mask
            np_pvals[s]= pvals
        print("\n")
        # print(f'Time: {(time()-start_time)}')
        #%%


        xrcorr.values = np_data
        xrpvals.values = np_pvals
        mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
        xrcorr.coords['mask'] = mask
        # fill nans with mask = True
        xrcorr['mask'] = xrcorr['mask'].where(orig_mask==False, other=orig_mask).drop('time')
        #%%
        return xrcorr, xrpvals

    def group_small_cluster(self, distance_eps_sc=1750, eps_corr=0.4,
                            precur_aggr=None, kwrgs_load=None,
                            min_gridcells_sc=1):

        #%%
        if hasattr(self, 'prec_labels_'):
            self.prec_labels_ = self.prec_labels_.copy()
        else:
            print()
            self.prec_labels_ = self.prec_labels.copy()

        close = find_precursors.close_small_clusters
        merge = close(self.prec_labels_,
                      self.corr_xr.copy(), distance_eps_sc,
                      min_gridcells_sc)
        print('Near clusters', merge)

        f_replace = find_precursors.view_or_replace_labels
        if eps_corr is not None:
            merge_upd = []
            suggested_group = f_replace(self.prec_labels_,
                                        regions=core_pp.flatten(merge))

            self.prec_labels = suggested_group.copy()
            calc_ts_o = self.calc_ts
            self.calc_ts = 'region mean'
            self.get_prec_ts(precur_aggr, kwrgs_load)
            df = pd.concat(self.ts_corr,
                           keys=range(len(self.ts_corr))).mean(0,level=1)
            for l in core_pp.flatten(merge):
                cols = [c for c in df.columns if f'..{int(l)}..' in c]
                df = df.rename({c:str(l) for c in cols}, axis=1)
            # groupby lags
            df_grlabs = df.groupby(by=df.columns, axis=1).mean()
            df_grlabs.columns = df_grlabs.columns.astype(float)
            for m in merge:
                DBSCAN = find_precursors.cluster.DBSCAN
                cl = DBSCAN(eps=eps_corr,
                             min_samples=1,
                             metric='correlation',
                             n_jobs=1).fit(df_grlabs.loc[:,m].T)
                for grl in np.unique(cl.labels_):
                    _grl = np.array(m)[cl.labels_==grl]
                    if _grl.size > 1:
                        merge_upd.append(list(_grl))
        # restore calc ts
        self.calc_ts = calc_ts_o
        print('Near clusters correlating', merge_upd)


        keys = [min(cl) for cl in merge_upd]
        [m.remove(keys[i]) for i,m in enumerate(merge_upd)]
        relabeldict = {im:keys[merge_upd.index(m)] for m in merge_upd for im in m}


        regions = list(relabeldict.keys())
        replacement_labels = list(relabeldict.values())

        all_regions = np.unique(self.prec_labels_.mean(dim='split'))
        all_regions = all_regions[~np.isnan(all_regions)]
        keepregions = [r for r in all_regions if r not in regions]
        regions += keepregions  ; replacement_labels += keepregions
        self.prec_labels = f_replace(self.prec_labels_.copy(), regions=regions,
                                    replacement_labels=replacement_labels)
        return


    def adjust_significance_threshold(self, alpha):
        self.alpha = alpha
        self.corr_xr.mask.values = (self.pval_xr > self.alpha).values

    def load_and_aggregate_precur(self, kwrgs_load):
        '''
        Wrapper to load in Netcdf and aggregated to n-mean bins or a period
        mean, e.g. DJF mean (see seasonal_mode.ipynb).

        Parameters
        ----------
        kwrgs_load : TYPE
            dictionary passed to functions_pp.import_ds_timemeanbins or
            to functions_pp.time_mean_periods.
        df_splits : pd.DataFrame, optional
            See class_RGCPD. The default is using the df_splits that was used
            for calculating the correlation map.

        Returns
        -------
        None.

        '''
        # self = rg.list_for_MI[0] ; df_splits = rg.df_splits ; kwrgs_load = rg.kwrgs_load
        name = self.name
        filepath = self.filepath
        # =============================================================================
        # Unpack non-default arguments
        # =============================================================================
        kwrgs = {'selbox':self.selbox, 'dailytomonths':self.dailytomonths}
        for key, value in kwrgs_load.items():
            if type(value) is list and name in value[1].keys():
                kwrgs[key] = value[1][name]
            elif type(value) is list and name not in value[1].keys():
                kwrgs[key] = value[0] # plugging in default value
            elif hasattr(self, key):
                # Overwrite RGCPD parameters with MI specific parameters
                kwrgs[key] = self.__dict__[key]
            else:
                kwrgs[key] = value
        if self.lag_as_gap: kwrgs['tfreq'] = 1
        self.kwrgs_load = kwrgs.copy()
        #===========================================
        # Precursor field
        #===========================================
        self.precur_arr = functions_pp.import_ds_timemeanbins(filepath, **kwrgs)

        if type(self.lags[0]) == np.ndarray:
            tmp = functions_pp.time_mean_periods
            self.precur_arr = tmp(self.precur_arr, self.lags,
                                  kwrgs_load['start_end_year'])
        return

    def load_and_aggregate_ts(self, df_splits: pd.DataFrame=None):
        if df_splits is None:
            df_splits = self.df_splits
        # =============================================================================
        # Load external timeseries for partial_corr_z
        # =============================================================================
        kwrgs = self.kwrgs_load
        if hasattr(self, 'kwrgs_z') == False: # copy so info remains stored
            self.kwrgs_z = self.kwrgs_func.copy() # first time copy
        if 'filepath' in self.kwrgs_z.keys():
            if type(self.kwrgs_z['filepath']) is str:
                print('Loading and aggregating {}'.format(self.kwrgs_z['keys_ext']))
                f = find_precursors.import_precur_ts
                self.df_z = f([('z', self.kwrgs_z['filepath'])],
                              df_splits,
                              start_end_date=kwrgs['start_end_date'],
                              start_end_year=kwrgs['start_end_year'],
                              start_end_TVdate=kwrgs['start_end_TVdate'],
                              cols=self.kwrgs_z['keys_ext'],
                              precur_aggr=kwrgs['tfreq'])

                if hasattr(self.df_z.index, 'levels'): # has train-test splits
                    f = functions_pp
                    self.df_z = f.get_df_test(self.df_z.merge(df_splits,
                                                              left_index=True,
                                                              right_index=True)).iloc[:,:1]
                k = list(self.kwrgs_func.keys())
                [self.kwrgs_func.pop(k) for k in k if k in ['filepath','keys_ext']]
                self.kwrgs_func.update({'df_z':self.df_z}) # overwrite kwrgs_func
                k = [k for k in list(self.kwrgs_z.keys()) if k not in ['filepath','keys_ext']]

                equal_dates = all(np.equal(self.df_z.index,
                                           pd.to_datetime(self.precur_arr.time.values)))
                if equal_dates==False:
                    raise ValueError('Dates of timeseries z not equal to dates of field')
            elif type(self.kwrgs_z['filepath']) is pd.DataFrame:
                self.df_z = self.kwrgs_z['filepath']
                k = list(self.kwrgs_func.keys())
                [self.kwrgs_func.pop(k) for k in k if k in ['filepath','keys_ext']]
                self.kwrgs_func.update({'df_z':self.df_z}) # overwrite kwrgs_func
        return


    def get_prec_ts(self, precur_aggr=None, kwrgs_load=None):
        # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
        # stacked on top of each other (from lag_min to lag_max)

        n_tot_regs = 0
        splits = self.corr_xr.split
        if hasattr(self, 'prec_labels') == False:
            print(f'{self.name} is not clustered yet')
        else:
            if np.isnan(self.prec_labels.values).all():
                self.ts_corr = np.array(splits.size*[[]])
            else:
                self.ts_corr = calc_ts_wrapper(self,
                                               precur_aggr=precur_aggr,
                                               kwrgs_load=kwrgs_load)
                n_tot_regs += max([self.ts_corr[s].shape[1] for s in range(splits.size)])

        return

    def store_netcdf(self, path: str=None, f_name: str=None, add_hash=True):
        assert hasattr(self, 'corr_xr'), 'No MI map calculated'
        if path is None:
            path = functions_pp.get_download_path()
        hash_str  = uuid.uuid4().hex[:6]
        if f_name is None:
            f_name = '{}_a{}'.format(self._name, self.alpha)
        self.corr_xr.attrs['alpha'] = self.alpha
        self.corr_xr.attrs['FDR_control'] = int(self.FDR_control)
        self.corr_xr['lag'] = ('lag', range(self.lags.shape[0]))
        if 'mask' in self.precur_arr.coords:
                self.precur_arr = self.precur_arr.drop('mask')
        # self.precur_arr.attrs['_tfreq'] = int(self._tfreq)
        if hasattr(self, 'prec_labels'):
            self.prec_labels['lag'] = self.corr_xr['lag'] # must be same
            self.prec_labels.attrs['distance_eps'] = self.distance_eps
            self.prec_labels.attrs['min_area_in_degrees2'] = self.min_area_in_degrees2
            self.prec_labels.attrs['group_lag'] = int(self.group_lag)
            self.prec_labels.attrs['group_split'] = int(self.group_split)
            if f_name is None:
                f_name += '_{}_{}'.format(self.distance_eps,
                                          self.min_area_in_degrees2)

            ds = xr.Dataset({'corr_xr':self.corr_xr,
                             'prec_labels':self.prec_labels,
                             'precur_arr':self.precur_arr})
        else:
            ds = xr.Dataset({'corr_xr':self.corr_xr,
                             'precur_arr':self.precur_arr})
        if add_hash:
            f_name += f'_{hash_str}'
        self.filepath_experiment = os.path.join(path, f_name+ '.nc')
        ds.to_netcdf(self.filepath_experiment)
        print(f'Dataset stored with hash: {hash_str}')

    def load_files(self, path_hashfile=str, hash_str: str=None):
        #%%
        if hash_str is None:
            hash_str = '{}_a{}_{}_{}'.format(self._name, self.alpha,
                                           self.distance_eps,
                                           self.min_area_in_degrees2)
        if path_hashfile is None:
            path_hashfile = functions_pp.get_download_path()
        f_name = None
        for root, dirs, files in os.walk(path_hashfile):
            for file in files:
                if re.findall(f'{hash_str}', file):
                    print(f'Found file {file}')
                    f_name = file
        if f_name is not None:
            filepath = os.path.join(path_hashfile, f_name)
            self.corr_xr = core_pp.import_ds_lazy(filepath, var='corr_xr')
            self.alpha = self.corr_xr.attrs['alpha']
            self.FDR_control = bool(self.corr_xr.attrs['FDR_control'])
            self.precur_arr = core_pp.import_ds_lazy(filepath,
                                                      var='precur_arr')
            # self._tfreq = self.precur_arr.attrs['_tfreq']
            try:
                self.prec_labels = core_pp.import_ds_lazy(filepath,
                                                          var='prec_labels')
                self.distance_eps = self.prec_labels.attrs['distance_eps']
                self.min_area_in_degrees2 = self.prec_labels.attrs['min_area_in_degrees2']
                self.group_lag = bool(self.prec_labels.attrs['group_lag'])
                self.group_split = bool(self.prec_labels.attrs['group_split'])
            except:
                print('prec_labels not found. Precursor regions not yet clustered')
            loaded = True
        else:
            print('No file that matches the hash_str or instance settings in '
                  f'folder {path_hashfile}')
            loaded = False
        return loaded


        #%%
def check_NaNs(field, ts):
    '''
    Return shortened timeseries of both field and ts if a few NaNs are detected
    at boundary due to large lag. At boundary time-axis, large lags
    often result in NaNs due to missing data.
    Removing timesteps from timeseries if
    1. Entire field is filled with NaNs
    2. Number of timesteps are less than a single year
       of datapoints.
    '''
    t = functions_pp.get_oneyr(field).size # threshold NaNs allowed.
    field = np.reshape(field.values, (field.shape[0],-1))
    i = 0 ; # check NaNs in first year
    if bool(np.isnan(field[i]).all()):
        i+=1
        while bool(np.isnan(field[i]).all()):
            i+=1
            if i > t:
                raise ValueError('More NaNs detected then # of datapoints in '
                                 'single year')
    j = -1 ; # check NaNs in last year
    if bool(np.isnan(field[j]).all()):
        j-=1
        while bool(np.isnan(field[j]).all()):
            j-=1
            if j < t:
                raise ValueError('More NaNs detected then # of datapoints in '
                                 'single year')
    else:
        j = field.shape[0]
    return field[i:j], ts[i:j]


def corr_map(field, ts):
    """
    This function calculates the correlation coefficent r and
    the pvalue p for each grid-point of field vs response-variable ts
    If more then a single year of NaNs is detected, a NaN will
    be returned, otherwise corr is calculated over non-NaN values.

    """
    # if more then one year is filled with NaNs -> no corr value calculated.
    field, ts = check_NaNs(field, ts)
    x = np.ma.zeros(field.shape[1])
    corr_vals = np.array(x)
    pvals = np.array(x)

    fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]

    for i in nonans_gc:
        corr_vals[i], pvals[i] = scipy.stats.pearsonr(ts,field[:,i])
    # restore original nans
    corr_vals[fieldnans] = np.nan
    # correlation map and pvalue at each grid-point:

    return corr_vals, pvals

def parcorr_map(field: xr.DataArray, ts: pd.DataFrame, df_splits_s, lag,
                lag_y: Union[int,list]=None, lag_x: Union[int,list]=None,
                df_z: pd.DataFrame=None, lag_z: Union[int,list]=None,
                lagzxrelative=True):

    '''
    Only works for subseasonal data (more then 1 datapoint per year).

    Parameters
    ----------
    field : xr.DataArray
        (time, lat, lon) field.
    ts : np.ndarray
        Target timeseries.
    df_splits_s : pd.DataFrame
        DF with TrainIsTrue and RV_mask to handle lag shifting
    lag : int
        lag used to select precursor dates. The default is 1.
    lag_x : int or list, optional
        lags of precursor lag to regress out, default is None.
    lag_y : int or list, optional
        lags w.r.t. target RV period to regress out, default is None.
    df_z : pd.DataFrame, optional
        1-d timeseries to regress out, default is None.
    lag_z : int or list, optional
        lags of precursor lag to regress out, default is None.
    lagzxrelative : bool, optional
        Whether to define lag_z and lag_x w.r.t. the lag of the precursor.
        Default is True.


    Returns
    -------
    corr_vals : np.ndarray
    pvals : np.ndarray

    '''
    #%%
    # field = precur_train; ts = df_RVfull_s ; df_splits_s=self.df_splits.loc[s]
    # lagzxrelative=True; lag_y=None; lag_z=None ; lag_x=None
    # df_z = self.kwrgs_func['df_z']
    if type(lag_y) is int:
        lag_y = [lag_y]
    if type(lag_x) is int:
        lag_x = [lag_x]
    if type(lag_z) is int:
        lag_z = [lag_z]

    # get lagged precursor
    m = apply_shift_lag(df_splits_s, lag)
    dates_lag = m[np.logical_and(m['TrainIsTrue']==1, m['x_fit'])].index
    RV_mask = df_splits_s.sum(axis=1) == 2 # both train and RV dates of target
    # if more then one year is filled with NaNs -> no corr value calculated.
    field_lag, ts_target = check_NaNs(field.sel(time=dates_lag), ts.values.squeeze()[RV_mask])
    x = np.ma.zeros(field_lag.shape[1])
    corr_vals = np.array(x)
    pvals = np.array(x)

    fieldnans = np.array([np.isnan(field_lag[:,i]).any() for i in range(x.size)])
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]

    # =============================================================================
    # checking if data is not available at edges due to lag shifting
    # =============================================================================
    missing_edge_data = 0
    if lag_y is not None:
        m = apply_shift_lag(df_splits_s, max(lag_y))
        m = np.logical_and(m['TrainIsTrue']==1, m['x_fit'])
        missing_edge_data = max(ts_target.shape[0] - ts[m].values.shape[0],
                                missing_edge_data)
    if lag_x is not None:
        _lag = max(lag_x) + lag if lagzxrelative else max(lag_x)
        m = apply_shift_lag(df_splits_s, _lag)
        dates_lag = m[np.logical_and(m['TrainIsTrue']==1, m['x_fit'])].index
        missing_edge_data = max(ts_target.shape[0] - dates_lag.size, missing_edge_data)
    if lag_z is not None:
        _lag = max(lag_z) + lag if lagzxrelative else max(lag_z)
        m = apply_shift_lag(df_splits_s, _lag)
        m = np.logical_and(m['TrainIsTrue']==1, m['x_fit'])
        missing_edge_data = max(ts_target.shape[0] - df_z.loc[m[m].index].values.shape[0],
                                missing_edge_data)

    # =============================================================================
    # Get (lag shifted) zy
    # =============================================================================
    if lag_y is not None:
        zy = []
        for _lag in lag_y:
            m = apply_shift_lag(df_splits_s, _lag)
            m = np.logical_and(m['TrainIsTrue']==1, m['x_fit'])
            missing_edge = ts_target.shape[0] - ts[m].values.shape[0]
            zy.append(ts[m].values[missing_edge_data-missing_edge:])
        zy = np.concatenate(zy, axis=1)
    else:
        zy = None

    # =============================================================================
    # Get (lag shifted) zz
    # =============================================================================
    if lag_z is not None:
        zz = []
        for _lag in lag_z:
            _lag = _lag + lag if lagzxrelative else _lag
            m = apply_shift_lag(df_splits_s, _lag)
            m = np.logical_and(m['TrainIsTrue']==1, m['x_fit'])
            missing_edge = ts_target.shape[0] - df_z.loc[m[m].index].values.shape[0]
            zz.append(df_z.loc[m[m].index].values[missing_edge_data-missing_edge:])
        zz = np.concatenate(zz, axis=1)
    else:
        zz = None

    # =============================================================================
    # Get (lag shifted) zx_field
    # =============================================================================
    if lag_x is not None:
        # this workflow might a bit too memory intensive since copying precursor
        # len(lag_x) times. But otherwise need to compute this in loop over gridcells.
        zx_field = []
        for _lag in lag_x:
            _lag_ = _lag + lag if lagzxrelative else _lag
            m = apply_shift_lag(df_splits_s, _lag_)
            dates_lag = m[np.logical_and(m['TrainIsTrue']==1, m['x_fit'])].index
            missing_edge = missing_edge_data - (ts_target.shape[0]  - dates_lag.size)
            _d = dates_lag[missing_edge:]
            zx_field.append(field.sel(time=_d).values.reshape(_d.size,1,-1))
        zx_field = np.concatenate(zx_field, axis=1)
    else:
        zx = None


    y = np.expand_dims(ts_target[missing_edge_data:], axis=1)
    # dict_ana = {}
    for i in nonans_gc:
        cond_ind_test = ParCorr()
        if lag_x is not None:
            zx = zx_field[:,:,i]
        z = None
        for _z in [zx, zy, zz]:
            if _z is not None and z is None:
                z = np.concatenate([_z], axis=1) # create first z
            elif _z is not None and z is not None:
                z = np.append(z, _z, axis=1) # append other z's
        field_i = np.expand_dims(field_lag[missing_edge_data:,i], axis=1)
        a, p = cond_ind_test.run_test_raw(field_i, y, z)
        # acorr, pcorr = cond_ind_test.run_test_raw(field_i, y, z=None)
        # if abs(a) - abs(acorr) > .15 and p < 0.01 and a < 0:
        #     dict_ana[i] = [field_i, y, z]
        #     #
        #     print('corr increased due to regressing out z', a, acorr)
        #     plot_maps.show_field_point(field, i=i)
        #     break
        #     field_i = core_pp.detrend_wrapper(field_i)
        #     y       = core_pp.detrend_wrapper(y)
        #     for zaxis in range(z.shape[1]):
        #         z[:,zaxis] = core_pp.detrend_wrapper(z[:,zaxis])
        #     avald, pvald = cond_ind_test.run_test_raw(field_i, y, z)


        corr_vals[i] = a
        pvals[i] = p

    # i_gridcells = np.array(list(dict_ana.keys()))
    # dict_ana1 = {i:dict_ana[i] for i in i_gridcells[i_gridcells<5000]}
    # dict_ana2 = {i:dict_ana[i] for i in i_gridcells[i_gridcells>5000]}
    # dic = dict_ana1
    # if len(dic) != 0:
    #     arr = np.array([np.concatenate(dic[i], axis=1) for n,i in enumerate(dic.keys())])

    #     z_label='$x_{t-2}$'
    #     # for n,i in enumerate(dic.keys()):
    #     #     field_i = arr[n][:,0] ; y = arr[n][:,1] ; z = arr[n][:,2:]
    #     #     check_parcorr_plots(field, field_i, y, z, i, dates_lag, z_label)
    #     arr = arr.mean(0) # mean over timeseries
    #     field_i = arr[:,0] ; y = arr[:,1] ; z = arr[:,2:]
    #     m = apply_shift_lag(df_splits_s, 1)
    #     dates_lag = m[np.logical_and(m['TrainIsTrue']==1, m['x_fit'])].index
    #     dates_lag = dates_lag[missing_edge_data:]
    #     check_parcorr_plots(field, field_i, y, z, 'red_mean', dates_lag, z_label)
    # restore original nans
    corr_vals[fieldnans] = np.nan
    #%%
    return corr_vals, pvals

#%% Tigramite check of parcorr, datapoint: i = 3455

# df_x = pd.DataFrame(field.values.reshape(field.shape[0],-1)[:,i], index=pd.to_datetime(field.time.values),
#                     columns=['x'])
# df_y = df_RVfull_s[df_splits_s['TrainIsTrue']]
# masks = df_splits_s[df_splits_s['TrainIsTrue']]
# df_tig = pd.concat([df_y.merge(df_x, left_index=True,
#                     right_index=True).merge(masks,
#                                             left_index=True,right_index=True)], keys=[0])
# rg.df_data = df_tig
# kwrgs_tigr = {'tau_min':0, 'tau_max':1, 'max_conds_dim':10,
#                   'pc_alpha':0.05, 'max_combinations':10, 'max_conds_px':1}
# rg.precur_aggr = 2
# rg.PCMCI_df_data(kwrgs_tigr=kwrgs_tigr)
#%% Some timeseries plots
def check_parcorr_plots(field, field_i, y, z, i, dates_lag, z_label='$x_{t-2}$'):
    #%%
    import matplotlib.pyplot as plt
    fs = 20
    # z = zx
    # z_label = '$x_{t-2}$'
    # z_label = '$x^{pattern}_{t-2}$'
    # z = zz
    array = np.vstack((field_i.T, y.T, z.T))
    array = (array - np.mean(array, 1)[:,None]) / np.std(array, 1)[:,None]

    # Plot x_t-1 and z
    path_out = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper2_Sem/Third_submission/figures/parcorr_figs/gridcells'
    path_df_z = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/NPJ_2021/data/df_SST_lag1.h5'

    # plot corr(y,x) vs corr(y,res_x)

    df_z = functions_pp.load_hdf5(path_df_z)['df_data'].mean(axis=0, level=1)
    df_z = df_z.loc[dates_lag].values.squeeze()
    res_x = ParCorr()._get_single_residuals(np.copy(array), target_var=0)
    res_y = ParCorr()._get_single_residuals(np.copy(array), target_var=1)

    df_z = ((df_z - df_z.mean()) / df_z.std()) * -1
    res_x = (res_x - res_x.mean()) / res_x.std()
    res_y = (res_y - res_y.mean()) / res_y.std()
    xlag = ((array[0]-np.mean(array[0],0))/np.std(array[0],0)).squeeze()

    corr_function_patt = ((array[1]-np.mean(array[1])) * (df_z - df_z.mean()))
    corr_function_xlag = ((array[1]-np.mean(array[1])) * (xlag-xlag.mean()))
    corr_function_resx = ((res_y-np.mean(res_y)) * (res_x - res_x.mean()))
    diff = abs(corr_function_xlag  - corr_function_resx)
    remove_idx = 6
    idx_skip = np.argwhere(diff >= sorted(diff)[-remove_idx]).squeeze()
    # index_argmin = np.arange(0,73)[diff<sorted(diff)[-remove_idx]]


    f, axes = plt.subplots(5,1, figsize=(15,25))
    # bbox_to_anchor=(0.5, 1.42)
    ax1 = axes[0]
    ax1.plot(corr_function_xlag, c='#0096c7', label=r'$(y_t-\overline{y_t})\cdot (x_{t-1}-\overline{x_{t-1}})$')
    ax1.plot(corr_function_resx, c='purple',
            label=r'$(res_y-\overline{res_y})\cdot (res_x-\overline{res_x})$')
    ax1.legend(fontsize=fs, loc='lower left')#, bbox_to_anchor=bbox_to_anchor, loc='upper center')
    # ax1.scatter(idx_skip, corr_function_resx[idx_skip], c='r')
    corr_n = scipy.stats.pearsonr(array[1].squeeze(), array[0].squeeze())
    corr_r = scipy.stats.pearsonr(res_y,res_x)
    # corr_r_excl = scipy.stats.pearsonr(res_y[index_argmin],res_x[index_argmin])
    ax1.set_ylim(corr_function_resx.min()-2,
                max(corr_function_resx.max(), corr_function_xlag.max())+.5)
    textstr = '$corr(y_t,x_{t-1})=$'+f'${round(corr_n[0],2)}\ ({round(corr_n[1],3)})$\n'
    textstr += '$corr(res_y,res_x)=$'+f'${round(corr_r[0],2)}\ ({round(corr_r[1],3)})$'
    # textstr += '$corr(res_y,res_x)=$'+f'{round(corr_r_excl,2)} (excl red datapoint)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=fs,
            verticalalignment='bottom',
            horizontalalignment='right', bbox=props)


    ax2 = axes[1]
    # y_t and x_t-1
    ax2.plot(array[1], c='red', label='$y_t$', alpha=.8, ls='--')
    ax2.scatter(range(array[1].size), array[1],
                color='red', alpha=.5) ;
    ax2.plot(array[0], c='#0096c7', label='$[x_{t-1}]$') ;
    ax2.scatter(range(array[0].size), array[0],
                color='#0096c7', alpha=.8) ;
    ax2.set_ylim(min(array[:2].ravel())-2,
                max(array[:2].ravel())+.5)
    ax2.legend(fontsize=fs, loc='lower left')#, bbox_to_anchor=bbox_to_anchor, loc='upper center')

    # res_y and res_x
    ax3 = axes[2]
    ax3.plot(res_y, c='red', label='$res_y\ [y_{t} | $'+f'{z_label}$]$', alpha=.8, ls='--')
    ax3.scatter(range(array[1].size), res_y,
                color='red', alpha=.5) ;
    ax3.plot(res_x, c = 'purple', label='$res_x\ [x_{t-1} | $'+f'{z_label}$]$')
    ax3.scatter(range(array[0].size), res_x,
                color='purple', alpha=.8) ;
    ax3.set_ylim(min(min(res_y), min(res_x))-4,
                 max(max(res_y), max(res_x))+.5)
    ax3.legend(fontsize=fs, loc='lower left')#, bbox_to_anchor=bbox_to_anchor, loc='upper center')

    # Gradient vs Residual x
    ax4 = axes[3]
    xlag = ((field_i - np.mean(field_i)) / np.std(field_i)).squeeze()
    gradient = (xlag) - (((z-np.mean(z,0))/np.std(z,0)).squeeze())
    gradient = (gradient - np.mean(gradient)) / np.std(gradient)
    ax4.plot(gradient, c='orange', label='Gradient $[x_{t-1} - $'+'$x_{t-2}$'+'$]$')
    ax4.plot(res_x, c='purple', label='Residual $[x_{t-1} | $'+'$x_{t-2}$'+'$]$') ;
    ax4.set_ylim(min(min(xlag), min(gradient))-3,
                 max(max(xlag), max(gradient))+.5)
    ax4.legend(fontsize=fs, loc='lower left')#, bbox_to_anchor=bbox_to_anchor, loc='upper center')

    ax5 = axes[4]
    ax5.plot(corr_function_patt, c='green', alpha=.5,
             label=r'$(y_t-\overline{y_t})\cdot (x^{pattern}_{t-1}-\overline{x^{pattern}_{t-1}})$')
    ax5.plot(corr_function_xlag, c='#0096c7', label=r'$(y_t-\overline{y_t})\cdot (x_{t-1}-\overline{x_{t-1}})$')
    ax5.legend(fontsize=fs, loc='lower left')#, bbox_to_anchor=bbox_to_anchor, loc='upper center')
    corr_n = scipy.stats.pearsonr(array[1].squeeze(), array[0].squeeze())
    corr_r = scipy.stats.pearsonr(array[1].squeeze(),df_z)
    ax5.set_ylim(corr_function_resx.min()-1.5,
                max(corr_function_resx.max(), corr_function_xlag.max())+.5)
    textstr = '$corr(y_t,x_{t-1})=$'+f'${round(corr_n[0],2)}\ ({round(corr_n[1],3)})$\n'
    textstr += '$corr(y_t,x^{pattern}_{t-1})=$'+f'${round(corr_r[0],2)}\ ({round(corr_r[1],3)})$'
    # textstr += '$corr(res_y,res_x)=$'+f'{round(corr_r_excl,2)} (excl red datapoint)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax5.text(0.98, 0.05, textstr, transform=ax5.transAxes, fontsize=fs,
            verticalalignment='bottom',
            horizontalalignment='right', bbox=props)


    plt.subplots_adjust(hspace=.2) ;
    if 'purple' in i:
        plot_labels = ['A', 'B', 'C', 'D', 'E']
    elif 'red' in i:
        plot_labels = ['F', 'G', 'H', 'I', 'J']
    for i_ax, ax in enumerate(axes):
        ax.minorticks_on() ; ax.axhline(y=0, color='black')
        ax.grid(b=True, which='minor', color='grey', linestyle='--')
        [ax.axvline(x=x, c='grey', alpha=.8) for x in idx_skip]
        ax.tick_params(labelsize=16)
        ax.text(0.01, 0.96, plot_labels[i_ax], transform=ax.transAxes, fontsize=fs+4,
        verticalalignment='top',
        horizontalalignment='left', bbox={'facecolor':'white'})
    f.savefig(path_out + f'/{i}_correlation_function_{z_label}.jpeg', bbox_inches='tight')
    #%%
    # plt.savefig(path_out + f'/{i}_residual_parcorr(xt-1_{z_label}).jpeg')


    # Original y and residual y
    plt.figure(figsize=(10,5))
    plt.plot(array[1], label='Original $[y_t]$')
    plt.plot(res_y, label='Residual $[y_{1} | $'+f'{z_label}$]$') ;
    plt.legend(fontsize=fs, loc='upper center')
    plt.savefig(path_out + f'/{i}_residual_parcorr(yt_{z_label}).jpeg')
    dim, T = array.shape
    xyz = np.array([0 for i in range(field_i.shape[-1])] +
                   [1 for i in range(y.shape[-1])] +
                   [2 for i in range(z.shape[-1])])
    print('\n',scipy.stats.pearsonr(res_y,res_x)[0],
            ParCorr().get_significance(scipy.stats.pearsonr(res_y,res_x)[0], array, xyz, T, dim))



    # print('\n',scipy.stats.pearsonr(array[1].squeeze(),gradient))
    # print('\n',scipy.stats.pearsonr(res_x,gradient))
    # print('\n',scipy.stats.pearsonr(array[1].squeeze(),res_x))
    # print('\n',scipy.stats.pearsonr(array[0].squeeze(),res_x))

    # Relationship between x_t-1 and gradient
    plt.figure(figsize=(10,5))
    xlag = array[0]
    gradient = xlag - ((z-np.mean(z,0))/np.std(z,0)).squeeze()
    gradient = (gradient - np.mean(gradient) / np.std(gradient))
    plt.style.use('bmh') ;
    plt.plot(gradient, label='Gradient $[x_{t-1} - $'+'$x_{t-2}$'+'$]$')
    plt.plot(xlag, label='Original $[x_{t-1}]$') ;
    plt.legend(fontsize=fs)
    plt.savefig(path_out + f'/{i}_xt-1_vs_gradient_{z_label}.jpeg')

    plt.figure(figsize=(10,5)) ; plt.style.use('bmh')
    plt.plot(array[0], label='$x_{t-1}$') ;
    plt.plot(array[2], c='b', label=z_label) ;
    plt.legend(fontsize=fs, loc='upper center')
    plt.savefig(path_out + f'/{i}_timeseries_xt-1_{z_label}.jpeg')
    #%%
    return
#%%


#%%




# #%% Relationship between x_t-1 and gradient
# plt.figure(figsize=(10,5))
# xlag = ((field_i - np.mean(field_i)) / np.std(field_i)).squeeze()
# gradient = xlag - ((zx-np.mean(zx,0))/np.std(zx,0)).squeeze()
# gradient = (gradient - np.mean(gradient) / np.std(gradient))
# plt.style.use('bmh') ;
# plt.plot(gradient, label='Gradient $[x_{t-1} - $'+'$x_{t-2}$'+'$]$')
# plt.plot(xlag, label='Original $[x_{t-1}]$') ;
# plt.legend(fontsize=fs)
# plt.savefig(path_out + f'/xt-1_vs_gradient_{z_label}.jpeg')

# #%% Residual with x_t-2 versus residual with x_t-2 pattern
# array = np.vstack((field_i.T, y.T, zx.T))
# array = (array - np.mean(array, 1)[:,None]) / np.std(array, 1)[:,None]
# res_xx = ParCorr()._get_single_residuals(np.copy(array), target_var=0)
# array = np.vstack((field_i.T, y.T, zz.T))
# array = (array - np.mean(array, 1)[:,None]) / np.std(array, 1)[:,None]
# res_xz = ParCorr()._get_single_residuals(np.copy(array), target_var=0)

# plt.figure(figsize=(10,5))
# plt.plot(res_xx, label='Residual $[x_{t-1} | x_{t-2}]$') ;
# plt.plot(res_xz, label='Residual $[x_{t-1} | x^{pattern}_{t-2}]$')
# plt.legend(fontsize=fs, loc='upper center')
# plt.ylim(-2.5,3)
# #%%
# plt.figure(figsize=(10,5))
# plt.plot(abs(array[1] - ((zz-np.mean(zz,0))/np.std(zz,0)).squeeze()), label='$y_t - x_{t-1} | x^{pattern}_{t-2}$')
# plt.plot(abs(array[1] - ((zx-np.mean(zx,0))/np.std(zx,0)).squeeze()), label='$y_t - x_{t-1} | x_{t-2}$')
# plt.legend(fontsize=fs, loc='upper center')
# plt.ylim(0,4)
# #%%
# plt.figure(figsize=(10,5))
# plt.plot(abs(array[1] - xlag), label='$y_t - x_{t-1}$')
# plt.plot(abs(array[1] - ((zx-np.mean(zx,0))/np.std(zx,0)).squeeze()), label='$y_t - x_{t-1} | x_{t-2}$')
# plt.legend(fontsize=fs, loc='upper center')
# plt.ylim(0,4)

# #%%


# #%%
# plt.figure(figsize=(10,5))
# plt.plot(abs(array[1] - array[0]), label='$y_t - x_{t-1}$') ;
# plt.plot(abs(array[1] - res_xx), label='$y_t - parcorr(x_{t-1} | x_{t-2})$') ;
# plt.legend(fontsize=fs, loc='upper center')
# plt.ylim(0,5)
# # print('\n',scipy.stats.pearsonr(res_xx,res_xz))

# #%% Plot x_t-2 and x^pattern_t-2
# plt.figure(figsize=(10,5)) ; plt.style.use('bmh')
# plt.plot(((zx-np.mean(zx,0))/np.std(zx,0)).squeeze(), label='$x_{t-2}$') ;
# plt.plot(((zz-np.mean(zz,0))/np.std(zz,0)).squeeze(), c='b', label='$x^{pattern}_{t-2}$') ;
# plt.legend(fontsize=fs, loc='upper center')
# plt.savefig(path_out + '/timeseries_xt-2_xpattern_t-2.jpeg')
# #%% autocorrelation plot
# import df_ana
# df = pd.DataFrame(field.values.reshape(field.shape[0],-1), index=pd.to_datetime(field.time.values)).iloc[:,i]
# f, ax = plt.subplots(1) ;
# df_ana.plot_ac(df_z, ax=ax, s = 6, color='#EE6666') ;
# df_ana.plot_ac(df, ax=ax, s = 6, color='#3388BB') ;
# ax.set_xticklabels(labels=np.arange(0, 12,2))
# ax.set_xlabel('Months') ; ax.set_ylabel('Autocorrelation')
#%%

def parcorr_z(field: xr.DataArray, ts: np.ndarray, z: pd.DataFrame, lag_z: int=0):
    '''
    Regress out influence of 1-d timeseries z. if lag_z==0, dates of z will match
    dates of field. Note, lag_z >= 1 probably only makes sense when using
    subseasonal data (more then 1 value per year).

    Parameters
    ----------
    field : xr.DataArray
        (time, lat, lon) field.
    ts : np.ndarray
        Target timeseries.
    z : pd.DataFrame
        1-d timeseries.

    Returns
    -------
    corr_vals : np.ndarray
    pvals : np.ndarray

    '''
    if type(lag_z) is int:
        lag_z = [lag_z]

    max_lag = max(lag_z)
    # if more then one year is filled with NaNs -> no corr value calculated.
    dates = pd.to_datetime(field.time.values)
    field, ts = check_NaNs(field, ts)
    x = np.ma.zeros(field.shape[1])
    corr_vals = np.array(x)
    pvals = np.array(x)
    fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]

    # ts = np.expand_dims(ts[:], axis=1)
    # adjust to shape (samples, dimension) and remove first datapoints if
    # lag_z != 0.
    y = np.expand_dims(ts[max_lag:], axis=1)
    if len(z.values.squeeze().shape)==1:
        z = np.expand_dims(z.loc[dates].values.squeeze(), axis=1)
    else:
        z = z.loc[dates].values.squeeze() # lag_z shifted wrt precursor dates

    list_z = []
    if 0 in lag_z:
        list_z = [z[max_lag:]]

    if max(lag_z) > 0:
        [list_z.append(z[max_lag-l:-l]) for l in lag_z if l != 0]
        z = np.concatenate(list_z, axis=1)

    # if lag_z >= 1:
        # z = z[:-lag_z] # last values are 'removed'
    for i in nonans_gc:
        cond_ind_test = ParCorr()
        field_i = np.expand_dims(field[max_lag:,i], axis=1)
        a, b = cond_ind_test.run_test_raw(field_i, y, z)
        corr_vals[i] = a
        pvals[i] = b
    # restore original nans
    corr_vals[fieldnans] = np.nan
    return corr_vals, pvals

def pp_calc_ts(precur, precur_aggr=None, kwrgs_load: dict=None,
                      force_reload: bool=False, lags: list=None):
    '''
    Pre-process for calculating timeseries of precursor regions or pattern.
    '''
    #%%
    if hasattr(precur, 'corr_xr_'):
        corr_xr         = precur.corr_xr_
    else:
        corr_xr         = precur.corr_xr
    prec_labels     = precur.prec_labels

    if kwrgs_load is None:
        kwrgs_load = precur.kwrgs_load

    if lags is not None:
        lags        = np.array(lags) # ensure lag is np.ndarray
        corr_xr     = corr_xr.sel(lag=lags).copy()
        prec_labels = prec_labels.sel(lag=lags).copy()
    else:
        lags        = prec_labels.lag.values

    if precur_aggr is None and force_reload==False:
        precur_arr = precur.precur_arr
    else:
        if precur_aggr is not None:
            precur.tfreq = precur_aggr
        precur.load_and_aggregate_precur(kwrgs_load.copy())
        precur_arr = precur.precur_arr

    if type(precur.lags[0]) is np.ndarray and precur_aggr is None:
        precur.period_means_array = True
    else:
        precur.period_means_array = False

    if precur_arr.shape[-2:] != corr_xr.shape[-2:]:
        print('shape loaded precur_arr != corr map, matching coords')
        corr_xr, prec_labels = functions_pp.match_coords_xarrays(precur_arr,
                                          *[corr_xr, prec_labels])
    #%%
    return precur_arr, corr_xr, prec_labels

# def loop_get_spatcov(precur, precur_aggr=None, kwrgs_load: dict=None,
#                      force_reload: bool=False, lags: list=None):
#     '''
#     Calculate spatial covariance between significantly correlating gridcells
#     and observed (time, lat, lon) data.
#     '''
#     #%%

#     precur_arr, corr_xr, prec_labels = pp_calc_ts(precur, precur_aggr,
#                                                   kwrgs_load,
#                                                   force_reload, lags)

#     lags        = prec_labels.lag.values
#     precur.area_grid = find_precursors.get_area(precur_arr)
#     splits          = corr_xr.split
#     use_sign_pattern = precur.use_sign_pattern



#     ts_sp = np.zeros( (splits.size), dtype=object)
#     for s in splits:
#         ts_list = np.zeros( (lags.size), dtype=list )
#         track_names = []
#         for il,lag in enumerate(lags):

#             # if lag represents aggregation period:
#             if type(precur.lags[il]) is np.ndarray and precur_aggr is None:
#                 precur_arr = precur.precur_arr.sel(lag=il)

#             corr_vals = corr_xr.sel(split=s).isel(lag=il)
#             mask = prec_labels.sel(split=s).isel(lag=il)
#             pattern = corr_vals.where(~np.isnan(mask))
#             if use_sign_pattern == True:
#                 pattern = np.sign(pattern)
#             if np.isnan(pattern.values).all():
#                 # no regions of this variable and split
#                 nants = np.zeros( (precur_arr.time.size, 1) )
#                 nants[:] = np.nan
#                 ts_list[il] = nants
#                 pass
#             else:
#                 xrts = find_precursors.calc_spatcov(precur_arr, pattern,
#                                                     area_wght=True)
#                 ts_list[il] = xrts.values[:,None]
#             track_names.append(f'{lag}..0..{precur.name}' + '_sp')

#         # concatenate timeseries all of lags
#         tsCorr = np.concatenate(tuple(ts_list), axis = 1)

#         dates = pd.to_datetime(precur_arr.time.values)
#         ts_sp[s] = pd.DataFrame(tsCorr,
#                                 index=dates,
#                                 columns=track_names)
#     # df_sp = pd.concat(list(ts_sp), keys=range(splits.size))
#     #%%
#     return ts_sp

def single_split_calc_spatcov(precur, precur_arr: np.ndarray, corr: np.ndarray,
                              labels: np.ndarray, a_wghts: np.ndarray,
                              lags: np.ndarray, use_sign_pattern: bool):
    ts_list = np.zeros( (lags.size), dtype=list )
    track_names = []
    for il,lag in enumerate(lags):

        # if lag represents aggregation period:
        if precur.period_means_array == True:
            precur_arr = precur.precur_arr.sel(lag=il).values

        pattern = np.copy(corr[il]) # copy to fix ValueError: assignment destination is read-only
        mask = labels[il]
        pattern[np.isnan(mask)] = np.nan
        if use_sign_pattern == True:
            pattern = np.sign(pattern)
        if np.isnan(pattern).all():
            # no regions of this variable and split
            nants = np.zeros( (precur_arr.shape[0], 1) )
            nants[:] = np.nan
            ts_list[il] = nants
            pass
        else:
            xrts = find_precursors.calc_spatcov(precur_arr, pattern,
                                                area_wght=a_wghts)
            ts_list[il] = xrts[:,None]
        track_names.append(f'{lag}..0..{precur.name}' + '_sp')
    return ts_list, track_names


def single_split_spatial_mean_regions(precur, precur_arr: np.ndarray,
                                      corr: np.ndarray,
                                      labels: np.ndarray, a_wghts: np.ndarray,
                                      lags: np.ndarray,
                                      use_coef_wghts: bool):
    '''
    precur : class_BivariateMI

    precur_arr : np.ndarray
        of shape (time, lat, lon). If lags define period_means;
        of shape (lag, time, lat, lon).
    corr : np.ndarray
        if shape (lag, lat, lon).
    labels : np.ndarray
        of shape (lag, lat, lon).
    a_wghts : np.ndarray
        if shape (lat, lon).
    use_coef_wghts : bool
        Use correlation coefficient as weights for spatial mean.

    Returns
    -------
    ts_list : list of splits with numpy timeseries
    '''
    ts_list = np.zeros( (lags.size), dtype=list )
    track_names = []
    for l_idx, lag in enumerate(lags):
        labels_lag = labels[l_idx]

        # if lag represents aggregation period:
        if precur.period_means_array == True:
            precur_arr = precur.precur_arr[:,l_idx].values

        regions_for_ts = list(np.unique(labels_lag[~np.isnan(labels_lag)]))

        if use_coef_wghts:
            coef_wghts = abs(corr[l_idx]) / abs(np.nanmax(corr[l_idx]))
            wghts = a_wghts * coef_wghts # area & corr. value weighted
        else:
            wghts = a_wghts

        # this array will be the time series for each feature
        ts_regions_lag_i = np.zeros((precur_arr.shape[0], len(regions_for_ts)))

        # track sign of eacht region

        # calculate area-weighted mean over features
        for r in regions_for_ts:
            idx = regions_for_ts.index(r)
            # start with empty lonlat array
            B = np.zeros(labels_lag.shape)
            # Mask everything except region of interest
            B[labels_lag == r] = 1
            # Calculates how values inside region vary over time
            ts = np.nanmean(precur_arr[:,B==1] * wghts[B==1], axis =1)

            # check for nans
            if ts[np.isnan(ts)].size !=0:
                print(ts)
                perc_nans = ts[np.isnan(ts)].size / ts.size
                if perc_nans == 1:
                    # all NaNs
                    print(f'All timesteps were NaNs for split'
                        f' for region {r} at lag {lag}')

                else:
                    print(f'{perc_nans} NaNs for split'
                        f' for region {r} at lag {lag}')

            track_names.append(f'{lag}..{int(r)}..{precur.name}')

            ts_regions_lag_i[:,idx] = ts
            # get sign of region
            # sign_ts_regions[idx] = np.sign(np.mean(corr.isel(lag=l_idx).values[B==1]))

        ts_list[l_idx] = ts_regions_lag_i

    return ts_list, track_names

def calc_ts_wrapper(precur, precur_aggr=None, kwrgs_load: dict=None,
                    force_reload: bool=False, lags: list=None):
    '''
    Wrapper for calculating 1-d spatial mean timeseries per precursor region
    or a timeseries of the spatial pattern (only significantly corr. gridcells).

    Parameters
    ----------
    precur : class_BivariateMI instance
    precur_aggr : int, optional
        If None, same precur_arr is used as for the correlation maps.
    kwrgs_load : dict, optional
        kwrgs to load in timeseries. See functions_pp.import_ds_timemeanbins or
        functions_pp.time_mean_period. The default is None.
    force_reload : bool, optional
        Force reload a different precursor array (precur_arr). The default is
        False.

    Returns
    -------
    ts_corr : list
        list of DataFrames.

    '''
    #%%
    # precur=rg.list_for_MI[0];precur_aggr=None;kwrgs_load=None;force_reload=False;lags=None
    # start_time  = time()
    precur_arr, corr_xr, prec_labels = pp_calc_ts(precur, precur_aggr,
                                                  kwrgs_load,
                                                  force_reload, lags)
    lags        = prec_labels.lag.values
    use_coef_wghts  = precur.use_coef_wghts
    if hasattr(precur, 'area_grid')==False:
        precur.area_grid = find_precursors.get_area(precur_arr)
    a_wghts         = precur.area_grid / precur.area_grid.mean()
    splits          = corr_xr.split.values
    dates = pd.to_datetime(precur_arr.time.values)



    if precur.calc_ts == 'pattern cov':
        kwrgs = {'use_sign_pattern':precur.use_sign_pattern}
        _f = single_split_calc_spatcov
    elif precur.calc_ts == 'region mean':
        kwrgs = {'use_coef_wghts':precur.use_coef_wghts}
        _f = single_split_spatial_mean_regions


    def splits_spatial_mean_regions(_f,
                                    splits: np.ndarray,
                                    precur, precur_arr: np.ndarray,
                                    corr_np: np.ndarray,
                                    labels_np: np.ndarray, a_wghts: np.ndarray,
                                    lags: np.ndarray,
                                    use_coef_wghts: bool):
        '''
        Wrapper to divide calculating a 'list of splits' per core, instead of
        assigning each split to a seperate worker (high overhead).
        '''
        for s in splits:
            corr = corr_np[s]
            labels = labels_np[s]
            output[s] = _f(precur, precur_arr, corr, labels, a_wghts, lags,
                           **kwrgs)
        return output


    precur_arr = precur_arr.values
    corr_np = corr_xr.values
    labels_np = prec_labels.values
    output = {}
    if precur.n_cpu == 1:
        output = splits_spatial_mean_regions(_f, splits, precur, precur_arr,
                                             corr_np, labels_np,
                                             a_wghts, lags, use_coef_wghts)

    elif precur.n_cpu > 1:
        futures = []
        for _s in np.array_split(splits, precur.n_cpu):
            futures.append(delayed(splits_spatial_mean_regions)(_f, _s,
                                                                precur,
                                                                precur_arr,
                                                                corr_np,
                                                                labels_np,
                                                                a_wghts, lags,
                                                                use_coef_wghts))
        futures = Parallel(n_jobs=precur.n_cpu, backend='loky')(futures)
        [output.update(d) for d in futures]


    ts_corr = np.zeros( (splits.size), dtype=object)
    for s in range(splits.size):
        ts_list, track_names = output[s] # list of ts at different lags
        tsCorr = np.concatenate(tuple(ts_list), axis = 1)
        df_tscorr = pd.DataFrame(tsCorr,
                                 index=dates,
                                 columns=track_names)
        df_tscorr.name = str(s)
        if any(df_tscorr.isna().values.flatten()):
            print('Warnning: nans detected')
        ts_corr[s] = df_tscorr

    # print(f'End time: {time() - start_time} seconds')

    #%%
    return ts_corr