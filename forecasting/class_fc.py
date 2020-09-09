#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:54:45 2019

@author: semvijverberg
"""
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_dir = os.path.join(main_dir, 'RGCPD')
df_ana_path = os.path.join(main_dir, 'df_analysis/df_analysis/')
if df_ana_path not in sys.path:
    sys.path.append(df_ana_path)
    sys.path.append(RGCPD_dir)
import pandas as pd
import numpy as np
import xarray as xr
import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
max_cpu = multiprocessing.cpu_count()
# print(f'{max_cpu} cpu\'s detected')
from itertools import chain
flatten = lambda l: list(chain.from_iterable(l))
from typing import List, Tuple, Union

import stat_models
import class_RV
import validation as valid
import functions_pp
import df_ana
import exp_fc
import core_pp
import func_models as util
# from RGCPD import RGCPD



class fcev():

    number_of_times_called = 0
    def __init__(self, path_data, precur_aggr: int=None, TV_aggr: int=None,
                   use_fold: List[Union[int]]=None,
                   start_end_TVdate: Tuple[str, str]=None,
                   start_end_PRdate: Tuple[str,str]=None,
                   stat_model: Tuple[str, dict]=('logit', None),
                   kwrgs_pp: dict={}, causal: bool=False,
                   dataset: str=None,
                   keys_d: Union[dict,str]=None,
                   n_cpu: int=None,
                   verbosity: int=0):
        '''


        Parameters
        ----------
        path_data : str
            path pointing towards a .h5 data object.
        precur_aggr : int, optional
            Aggregate precursor (and target if TV_aggr is not given) to n-day means.
            The default is 1.
        TV_aggr : int, optional
            If TV_aggr is given, TV is aggregated over different range than
            the precursor (latter defined by precur_aggr). The default is None.
        use_fold : List[Union[int]], optional
            select a single fold training of a range of training. The default is None.
        start_end_TVdate : Tuple[str, str], optional
            tuple of start- and enddate for target variable in
            format ('mm-dd', 'mm-dd'). Default is the RV_mask present in the
            .h5 dataframe 'df_data'.
        start_end_PRdate : Tuple[str,str], optional
            tuple of start- and enddate for precursor in
            format ('mm-dd', 'mm-dd'). This can be use to do annual mean
            forecasts of the target (TV period), overwrites lags_i.
        stat_model : Tuple[str, dict], optional
            tuple of name user gives to model function written in stat_models.py
            and a dictionairy (kwrgs) of the arguments that are needed for that
            function. The default is ('logit', sklearn.logit()).
        kwrgs_pp : dict, optional
            dictionary with arguments for preprocessing of input data,
            see class_fc.prepare_data() for options.
        causal : bool, optional
            DESCRIPTION. The default is False.
        dataset : str, optional
            user-given name of dataset that is used. The default is None.
        keys_d : Union[dict,str], optional
            Different options possible:
            (1) Either selecting all keys in the pd.DataFrame or
            (2) Expecting dict with traintest number as key and associated
            list of keys.
            (3) Expecting tuple with str name and dict with traintest number
            as key and associated list of keys.
            (4) Give string that links to exp_fc.py to extract manual subset
            of keys.
            The default (None) links to option 1.
        n_cpu : int, optional
            Parallel computation of different train-test splits.
            Default is using all -1 cpu's available.
        verbosity : int, optional
            Higher number determines (to some extent) how much feedback in the
            form of printed text is given to the user. The default is 0.


        Returns
        -------
        Initialized instance of class_fc.

        '''



        fcev.number_of_times_called += 1
        self.path_data = path_data
        if dataset is None:
            self.dataset = f'exper_{fcev.number_of_times_called}'
        else:
            self.dataset = dataset

        self.df_data_orig = df_ana.load_hdf5(self.path_data)['df_data']
        self.fold = use_fold
        if self.fold is not None:
            if type(self.fold) is int:
                if np.sign(self.fold) == 1:
                    self._select_fold()
                if np.sign(self.fold) == -1:
                    print(f'removing fold {self.fold}')
                    self.df_data =self._remove_folds()
            if type(self.fold) is not int:
                if np.sign(self.fold[0]) == 1:
                    self.df_data = self.df_data_orig.loc[[0,1]]

                if np.sign(self.fold[0]) == -1:
                    print(f'removing folds {self.fold}')
                    self.df_data =self._remove_folds()


        # if self.fold is not None and np.sign(self.fold) == -1:
        #     # remove all data from test years
        #     print(f'removing fold {self.fold}')
        #     self.df_data =self._remove_folds()
        if self.fold is None:
            self.df_data = self.df_data_orig.copy()

        self.dates_df  = self.df_data.loc[0].index.copy()
        self.precur_aggr = precur_aggr
        self.TV_aggr = TV_aggr
        self.splits  = self.df_data.index.levels[0]
        self.tfreq = (self.df_data.loc[0].index[1] - self.df_data.loc[0].index[0]).days
        self.start_end_PRdate = start_end_PRdate


        if start_end_TVdate is not None:
            fcev._redefine_RV_mask(self, start_end_TVdate=start_end_TVdate)


        self.RV_mask = self.df_data['RV_mask']
        self.TrainIsTrue = self.df_data['TrainIsTrue']
        self.test_years = valid.get_testyrs(self.df_data)
        # assuming hash is the last piece of string before the format
        self.hash = self.path_data.split('.h5')[0].split('_')[-1]

        # Model related:
        self.stat_model = stat_model
        self.kwrgs_pp = kwrgs_pp
        self.causal = causal
        if keys_d is None:
            print('keys is None: Using all keys in training sets')
            self.experiment = 'all'
            self.keys_d = None
        elif isinstance(keys_d, dict):
            self.experiment = 'manual'
            # expecting dict with traintest number as key and associated list of keys
            self.keys_d = keys_d
        elif isinstance(keys_d, tuple):
            self.experiment = keys_d[0]
            # expecting dict with traintest number as key and associated list of keys
            self.keys_d = keys_d[1]
        if isinstance(keys_d, str):
            print(f'getting keys associated with name {keys_d}')
            self.experiment = keys_d
            if self.causal:
                self.experiment += '_causal'
            self.keys_d = exp_fc.normal_precursor_regions(self.path_data,
                                                          keys_options=[keys_d],
                                                          causal=self.causal)[keys_d]
            if self.kwrgs_pp['add_autocorr']:
                self.experiment += '+AR'
        columns = self.df_data.columns[self.df_data.dtypes!=bool]
        self.df_precurset = self.df_data[columns].count(axis=0, level=1).iloc[0]

        if n_cpu is None:
            self.n_cpu = max_cpu - 1
        else:
            self.n_cpu = n_cpu
        self.verbosity = verbosity
        return



    def load_TV_from_cluster(self, path_cluster=str, label: int=1, name_ds='ts'):
        list_of_name_path = [(label, path_cluster)]
        f = functions_pp
        self.fulltso, self.hash = f.load_TV(list_of_name_path, name_ds=name_ds)
        # overwriting first column of df_data

        df_TV_split = self.df_data.iloc[:,0].loc[0]
        self.fullts = f.detrend1D(self.fulltso.sel(time=df_TV_split.index))
        new_vals = self.fullts.sel(time=df_TV_split.index)
        new_df = pd.DataFrame(new_vals.values, index=df_TV_split.index,
                              columns=[f'{label}_{name_ds}'])
        new_df = pd.concat([new_df]*self.splits.size, keys=self.splits)
        self.df_data = pd.merge(new_df,
                                self.df_data.drop(columns=self.df_data.columns[0]),
                                left_index=True, right_index=True)


    def get_TV(self, kwrgs_events=None, detrend=False, RV_anomaly=False,
               RV_name: str=None):

        if hasattr(self, 'df_data') == False:
            print("df_data not loaded, initialize fcev class with path to df_data")

        self.detrend = detrend
        self.RV_anomaly = RV_anomaly
        self.df_TV = self.df_data.iloc[:,[0,-2,-1]].copy()
        if RV_name is None:
            self.RV_name = self.df_TV.columns[0]
        else:
            self.RV_name = RV_name

        df_fold = self.df_TV.loc[0]
        self.fulltso = df_fold.iloc[:,0].to_xarray().squeeze()
        if self.detrend:
            self.fullts = functions_pp.detrend1D(self.fulltso, anomaly=self.RV_anomaly)
            n_spl = self.df_data.index.levels[0].size
            df_RV_s   = np.zeros( (n_spl) , dtype=object)
            for s in range(n_spl):
                df_RV_s[s] = pd.DataFrame(self.fullts.values,
                                        columns=[self.RV_name],
                                        index=self.df_TV.loc[0].index)
            self.df_TV[self.RV_name] = pd.concat(list(df_RV_s), keys= range(n_spl))
            # replacing TV in df_data dataframe
            self.df_data[self.RV_name] = self.df_TV[self.RV_name]


        # aggregation from daily to n-day means
        if self.start_end_PRdate is None:
            if self.TV_aggr is None and self.precur_aggr is not None:
                self.TV_aggr = self.precur_aggr
            if self.TV_aggr is not None:
                self.df_TV, dates_tobin = _daily_to_aggr(self.df_TV, self.TV_aggr)
            else:
                dates_tobin = None
        else:
            # annual mean forecast of TV, converting start_end_TVdate to 1 value
            # per year.
            self.df_TV = start_end_date_mean(self.df_TV,
                                             start_end_date=self.start_end_TVdate)
            dates_tobin = self.start_end_PRdate # will no longer aggerated to bins,



        # target events
        if kwrgs_events is None:
            self.kwrgs_events = {'event_percentile': 66,
                        'min_dur' : 1,
                        'max_break' : 0,
                        'grouped' : False,
                        'window' : 'mean'}
            _kwrgs_events = self.kwrgs_events
        else:

            self.kwrgs_events = kwrgs_events
            if 'window' not in kwrgs_events.keys():
                self.kwrgs_events['window'] = 'mean'
            _kwrgs_events = self.kwrgs_events.copy()


        if _kwrgs_events['window'] == 'single_event' and self.tfreq==1:
            _kwrgs_events['window'] = self.df_data.iloc[:,[0]].loc[0].copy()

        TV = df_data_to_RV(self.df_TV, kwrgs_events=_kwrgs_events,
                           fit_model_dates=None)
        TV.TrainIsTrue = self.df_TV['TrainIsTrue']
        TV.RV_mask = self.df_TV['RV_mask']
        TV.name = TV.RV_ts.columns[0]
        TV.get_obs_clim()
        TV.dates_tobin = dates_tobin
        if self.start_end_PRdate is None:
            TV.dates_tobin_TV = TV.aggr_to_daily_dates(TV.dates_RV)
        self.TV = TV
        return

    def fit_models(self, lead_max=np.array([1]),
                   verbosity=None):
        '''
        stat_model_l:   list of with model string and kwrgs
        keys_d      :   dict, with keys : list of variables to fit, if None
                        all keys in each training set will be used to fit.
                        If string is given, exp_py will follow some rules to
                        keep only keys you want to fit.
        precur_aggr:  int: convert daily data to aggregated {int} day mean
        '''

        # still need to get rid of list statmodels
        self.stat_model_l = [self.stat_model]
        model_names = [n[0] for n in self.stat_model_l]
        model_count = {n:model_names.count(n) for n in np.unique(model_names)}
        new = {m+f'--{i+1}':m for i,m in enumerate(model_names) if model_count[m]>1}

        if verbosity is None:
            verbosity = self.verbosity



        if isinstance(lead_max, int):
            if self.tfreq == 1:
                self.lags_i = np.arange(0, lead_max+1E-9, max(10,self.tfreq), dtype=int)
            else:
                self.lags_i = np.array(np.arange(0, lead_max+self.tfreq/2+1E-9,
                                            max(10,self.tfreq))/max(10,self.tfreq),
                                            dtype=int)
        elif type(lead_max) == np.ndarray:
            self.lags_i = lead_max
        else:
            print('lead_max should be integer or np.ndarray')

        if self.precur_aggr is None:
            if self.tfreq == 1:
                self.lags_t = np.array([l * self.tfreq for l in self.lags_i])
            else:
                if self.lags_i[0] == 0:
                    self.lags_t = [0]
                    for l in self.lags_i[1:]:
                        self.lags_t.append(int((l-1) * self.tfreq + self.tfreq/2))
                else:
                    self.lags_t = np.array([(l-1) * self.tfreq + self.tfreq/2 for l in self.lags_i])
                self.lags_t = np.array(self.lags_t)
            print(f'tfreq: {self.tfreq}, max lag: {self.lags_i[-1]}, i.e. {self.lags_t[-1]} days')
        else:
            self.lags_t = np.array(self.lags_i)
            print(f'precur_aggr: {self.precur_aggr}, max lag: {self.lags_t[-1]} days')

        self.dict_preds = {}
        self.dict_models = {}
        c = 0
        for i, stat_model in enumerate(self.stat_model_l):
            if stat_model[0] in list(new.values()):
                self.stat_model_l[i] = (list(new.keys())[c], stat_model[1])
                c += 1

            y_pred_all, y_pred_c, models = self._fit_model(stat_model=stat_model,
                                                           verbosity=verbosity)

            uniqname = self.stat_model_l[i][0]
            self.dict_preds[uniqname] = (y_pred_all, y_pred_c)
            self.dict_models[uniqname] = models
        return

    @staticmethod
    def _create_new_traintest_split(df_data, method='random9', seed=1, kwrgs_events=None):

        # insert fake train test split to make RV
        df_data = pd.concat([df_data], axis=0, keys=[0])
        RV = df_data_to_RV(df_data, kwrgs_events=kwrgs_events)
        df_data = df_data.loc[0][df_data.loc[0]['TrainIsTrue'].values]
        df_data = df_data.drop(['TrainIsTrue', 'RV_mask'], axis=1)
        # create CV inside training set
        df_splits = functions_pp.rand_traintest_years(RV, method=method,
                                                      seed=seed,
                                                      kwrgs_events=kwrgs_events)
        # add Train test info
        splits = df_splits.index.levels[0]
        df_data_s   = np.zeros( (splits.size) , dtype=object)
        for s in splits:
            df_data_s[s] = pd.merge(df_data, df_splits.loc[s],
                                    left_index=True, right_index=True)

        df_data  = pd.concat(list(df_data_s), keys= range(splits.size))
        return df_data

    def _select_fold(self):
        # overwriting self.df_data, selecting single fold
        self.test_years_orig = valid.get_testyrs(self.df_data_orig)
        df_data = self.df_data_orig.loc[self.fold][self.df_data_orig.loc[self.fold]['TrainIsTrue'].values]
        df_data = self._create_new_traintest_split(df_data.copy())
        return df_data

    def _remove_folds(self):
        if type(self.fold) is int:
            remove_folds = [abs(self.fold)]
        else:
            remove_folds = [abs(f) for f in self.fold]

        orig_fold = np.unique(self.df_data_orig.index.get_level_values(0))
        rem_yrs = valid.get_testyrs(self.df_data_orig.loc[remove_folds])
        keep_folds = range(len(orig_fold) - len(remove_folds))
        df_data_s   = np.zeros( (len(keep_folds)) , dtype=object)
        for s in keep_folds:
            df_keep = self.df_data_orig.loc[s]
            rm_yrs_mask = np.sum([df_keep.index.year != yr for yr in rem_yrs.flatten()],axis=0)
            rm_yrs_mask = rm_yrs_mask == rm_yrs_mask.max()
            df_data_s[s] = df_keep[rm_yrs_mask]
            yrs = np.unique([yr for yr in df_data_s[s].index.year if yr not in rem_yrs])
            assert (len([y for y in yrs if y in rem_yrs.flatten()]))==0, \
                        'check rem yrs'
        df_data  = pd.concat(list(df_data_s), keys=range(len(keep_folds)))
        self.rem_yrs = rem_yrs
        return df_data

    def _get_precursor_used(self):
        '''
        Retrieving keys used to train the model(s)
        If same keys are used, keys are stored as 'same_keys_used_by_models'
        '''
        models = [m[0] for m in self.stat_model_l]
        each_model = {}
        flat_arrays = []
        for m in models:
            flat_array = []
            each_lag = {}
            model_splits = self.dict_models[m]
            for lag_key, m_splits in model_splits.items():
                each_split = {}
                for split_key, model in m_splits.items():
                    m_class = model_splits[lag_key][split_key]
                    each_split[split_key] = m_class.X_pred.columns
                    flat_array.append( np.array(each_split[split_key]))
                each_lag[lag_key] = each_split
            each_model[m] = each_lag
            flat_arrays.append(np.array(flatten(flat_array)).flatten())
        # hacky, need to be improved
        try:
            if all( all(flat_arrays[1]==arr) for arr in flat_arrays[1:]):
                # each model used same variables:
                self.keys_used = dict(same_keys_used_by_models=each_model[models[0]])
        except:
            self.keys_used = each_model
        return self.keys_used

    def _get_statmodelobject(self, model=None, lag=None, split=0):
        if model is None:
            model = list(self.dict_models.keys())[0]
        if lag is None:
            lag = int(list(self.dict_models[model].keys())[0].split('_')[1])
        if split == 'all':
            m = self.dict_models[model][f'lag_{lag}']
        else:
            m = self.dict_models[model][f'lag_{lag}'][f'split_{split}']
        return m

    def _get_outpaths(self,  list_of_fc=None, subfoldername: str=None, f_name: str=None,
                      pathexper: str=None):
        if list_of_fc is None:
            list_of_fc = [self]
        if subfoldername is None:
            subfoldername = 'forecasts'
        if pathexper is None:
            working_folder = '/'.join(self.path_data.split('/')[:-1])
            working_folder = os.path.join(working_folder, subfoldername)
            self.working_folder = working_folder
            if os.path.isdir(working_folder) != True : os.makedirs(working_folder)
            today = datetime.datetime.today().strftime('%Hhr_%Mmin_%d-%m-%Y')
        if f_name is None and pathexper is None:
            # define pathexper
            if type(self.kwrgs_events) is tuple:
                percentile = self.kwrgs_events[1]['event_percentile']
            else:
                percentile = self.kwrgs_events['event_percentile']
            folds = list(np.unique([str(f.fold) for f in list_of_fc]))
            folds_used = str(folds).replace('[\'',
                            '').replace(', ','_').replace('\']','')
            f_name = f'{self.TV.name}_{self.precur_aggr}d_{percentile}p_fold{folds_used}_{today}'
            pathexper = os.path.join(working_folder, f_name)
        if f_name is not None and pathexper is None:
            today_str = f'_{today}'
            pathexper = os.path.join(working_folder, f_name+today_str)
        self.pathexper = pathexper
        if os.path.isdir(self.pathexper) != True : os.makedirs(self.pathexper)
        self.working_folder = working_folder

    def _print_sett(self, list_of_fc=None, subfoldername=None, f_name=None,
                    pathexper=None):

        self._get_outpaths(list_of_fc=None, subfoldername=subfoldername,
                           f_name=f_name,
                           pathexper=pathexper)



        file= open(os.path.join(self.pathexper,"exper.txt"),"w+")
        lines = []
        lines.append("\nEvent settings:")
        e = 1
        for i, fc_i in enumerate(list_of_fc):

            lines.append(f'\n\n***Experiment {e}***\n\n')
            lines.append(f'Title \t : {fc_i.dataset}')
            lines.append(f'file \t : {fc_i.path_data}')
            lines.append(f'kwrgs_events \t : {fc_i.kwrgs_events}')
            lines.append(f'kwrgs_pp \t : {fc_i.kwrgs_pp}')
            lines.append(f'TV dates: \t {fc_i._get_start_end_TVdate()}')
            lines.append(f'tfreq: \t {fc_i.tfreq}')
            lines.append(f'precur_aggr: \t {fc_i.precur_aggr}')
            lines.append(f'TV_aggr: \t {fc_i.TV_aggr}')
            lines.append(f'alpha \t : {fc_i.alpha}')
            lines.append(f'nboot: {fc_i.n_boot}')
            lines.append(f'stat_models:')
            lines.append('\n'.join(str(m) for m in fc_i.stat_model_l))
            lines.append(f'fold: {fc_i.fold}')
            lines.append(f'fc threshold: {fc_i.threshold_pred}')
            lines.append(f'keys_d: \n{fc_i.keys_d}')
            lines.append(f'keys_used: \n{fc_i._get_precursor_used()}')


            e+=1

        [print(n, file=file) for n in lines]
        file.close()
        [print(n) for n in lines[:-2]]
        return self.working_folder, self.pathexper

    def perform_validation(self, n_boot=2000, blocksize='auto',
                           threshold_pred='upper_clim', alpha=0.05):
        self.n_boot = n_boot
        self.threshold_pred = threshold_pred
        # self.dict_sum = {}
        self.alpha = alpha
        for stat_model in self.stat_model_l:
            name = stat_model[0]
            y_pred_all, y_pred_c = self.dict_preds[name]

            if blocksize == 'auto':
                self.blocksize = valid.get_bstrap_size(self.TV.fullts)
            else:
                self.blocksize = blocksize
            y = self.TV.RV_bin.squeeze().values
            out = valid.get_metrics_sklearn(y, y_pred_all, y_pred_c,
                                            n_boot=n_boot,
                                            alpha=self.alpha,
                                            blocksize=self.blocksize,
                                            threshold_pred=threshold_pred)
            df_valid, metrics_dict = out
            df_TV = self.TV.prob_clim.merge(self.TV.RV_bin,
                                       left_index=True, right_index=True)
            self.dict_sum = (df_valid, df_TV, y_pred_all)
            self.metrics_dict = metrics_dict
        return

    def _redefine_RV_mask(self, start_end_TVdate):
        self.df_data = self.df_data.copy()
        self.start_end_TVdate_orig = fcev._get_start_end_TVdate(self)
        self.start_end_TVdate = start_end_TVdate
        RV_mask_orig   = self.df_data['RV_mask'].copy()
        dates_RV = core_pp.get_subdates(self.dates_df , start_end_TVdate,
                                        start_end_year=None)
        new_RVmask = RV_mask_orig.loc[0].copy()
        new_RVmask.loc[:] = False
        new_RVmask.loc[dates_RV] = True
        self.df_data['RV_mask'] = pd.concat([new_RVmask] * self.splits.size,
                                            keys=self.splits)
    def _get_start_end_TVdate(self):
        RV_mask_orig   = self.df_data['RV_mask'].loc[0].copy()
        dates_RV = RV_mask_orig[RV_mask_orig].index
        return (f'{dates_RV[0].month}-{dates_RV[0].day}',
                                 f'{dates_RV[-1].month}-{dates_RV[-1].day}')

    def plot_freq_year(self, df: pd.DataFrame=None):
        import valid_plots as df_plots
        if df is None:
            df = self.TV.RV_bin
        df_plots.plot_freq_per_yr(df)

    # @classmethod
    def plot_scatter(self, keys=None, colwrap=3, sharex='none', s=0, mask='RV_mask', aggr=None,
                     title=None):
        import df_ana
        df_d = self.df_data.mean(axis=0, level=1)
        if mask is None:
            tv = self.df_data.loc[0].iloc[:,0]
            df_d = df_d
        elif mask == 'RV_mask':
            tv = self.df_data.loc[0].iloc[:,0][self.RV_mask.loc[s]]
            df_d = df_d[self.RV_mask.loc[s]]
        else:
            tv = self.df_data.loc[0].iloc[:,0][mask]
            df_d = df_d[mask]
        kwrgs = {'tv':tv,
                'aggr':aggr,
                 'title':title}
        df_ana.loop_df(df_d, df_ana.plot_scatter, keys=keys, colwrap=colwrap,
                            sharex=sharex, kwrgs=kwrgs)
        return


    def plot_feature_importances(self, model=None, lag=None, keys=None,
                                 cutoff=10):

        if model is None:
            model = [n[0] for n in self.stat_model_l][0]
        models_splits_lags = self.dict_models[model]
        if lag is None:
            lag = self.lags_i
        self.df_importance, fig = stat_models.plot_importances(models_splits_lags,
                                                               lag=lag,
                                                         keys=keys, cutoff=cutoff)
        return self.df_importance, fig


    def plot_oneway_partial_dependence(self, keys=None, lags=None):
        GBR_models_split_lags = self.dict_models['GBC']
        df_all, fig = stat_models.plot_oneway_partial_dependence(
                                        GBR_models_split_lags,
                                        keys=keys, lags=lags)
        return fig


    def plot_logit_regularization(self, lag_i=0):
        models = [m for m in self.dict_models.keys() if 'logitCV' in m]
        models_splits_lags = self.dict_models[models[0]]
        fig = stat_models.plot_regularization(models_splits_lags, lag_i=lag_i)
        return fig

    def apply_df_ana_plot(self, df=None, func=None,
                          hspace=.4, sharex=False, sharey=False,
                          kwrgs_func={}):
        if df is None:
            df = self.df_data
        if func is None:
            func = df_ana.plot_ac ; kwrgs_func = {'AUC_cutoff':(14,30),'s':60}
        return df_ana.loop_df(df, function=func, sharex=sharex, sharey=sharey,
                             colwrap=2, hspace=hspace, kwrgs=kwrgs_func)

    def _fit_model(self, stat_model=tuple, verbosity=0):

        #%%

        RV = self.TV
        kwrgs_pp = self.kwrgs_pp
        keys_d = self.keys_d
        df_data = self.df_data
        precur_aggr = self.precur_aggr
        dates_tobin = RV.dates_tobin
        if precur_aggr is not None:
            lags_i = self.lags_t
        else:
            lags_i = self.lags_i

        # lags_i = [1]
        # stat_model = self.stat_model_l[0]

        # do forecasting accros lags
        splits  = df_data.index.levels[0]
        y_pred_all = []
        y_pred_c = []
        models = []

        # store target variable (continuous and binary in y_ts dict)
        if hasattr(RV, 'RV_bin_fit'):
            y_ts = {'cont':RV.RV_ts_fit, 'bin':RV.RV_bin_fit}
        else:
            y_ts = {'cont':RV.RV_ts_fit}

        print(f'{stat_model}')
        from time import time
        try:
            t0 = time()
            futures = {}
            with ProcessPoolExecutor(max_workers=self.n_cpu) as pool:
                for lag in lags_i:
                    for split in splits:
                        fitkey = f'{lag}_{split}'
                        futures[fitkey] = pool.submit(fit,
                                                      y_ts=y_ts,
                                                      df_data=df_data,
                                                      lag=lag,
                                                      split=split,
                                                      stat_model=stat_model,
                                                      keys_d=keys_d,
                                                      dates_tobin=dates_tobin,
                                                      precur_aggr=precur_aggr,
                                                      kwrgs_pp=kwrgs_pp,
                                                      verbosity=verbosity)
                results = {key:future.result() for key, future in futures.items()}
            print(time() - t0)
        except:
            print('parallel failed')
            # if on cluster, stop
            assert sys.platform != 'linux', ('Parallel Failed on cluster')

            t0 = time()
            results = {}
            for lag in lags_i:
                for split in splits:
                    fitkey = f'{lag}_{split}'
                    results[fitkey] = fit(y_ts=y_ts, df_data=df_data, lag=lag,
                                          split=split, stat_model=stat_model,
                                          keys_d=keys_d, dates_tobin=RV.dates_tobin,
                                          precur_aggr=precur_aggr, kwrgs_pp=kwrgs_pp,
                                          verbosity=verbosity)
            print('in {:.0f} seconds'.format(time() - t0))
        # unpack results
        models = dict()
        for lag in lags_i:
            y_pred_l = []
            model_lag = dict()
            for split in splits:
                prediction, model = results[f'{lag}_{split}']
                # store model
                model_lag[f'split_{split}'] = model

                # retrieve original input data
                df_norm = model.df_norm
                TestRV  = (df_norm['TrainIsTrue']==False)[df_norm['y_pred']]
                y_pred_l.append(prediction[TestRV.values])

                if lag == lags_i[0]:
                    # ensure that RV timeseries matches y_pred
                    TrainRV = (df_norm['TrainIsTrue'])[df_norm['y_pred']]
                    RV_bin_train = RV.RV_bin[TrainRV.values] # index no longer align 20-2-10

                    # predicting RV might not be possible
                    # determining climatological prevailance in training data
                    y_c_mask = RV_bin_train==1
                    y_clim_val = RV_bin_train[y_c_mask.values].size / RV_bin_train.size
                    # filling test years with clim of training data
                    y_clim = RV.RV_bin[TestRV.values==True].copy()
                    y_clim[:] = y_clim_val
                    y_pred_c.append(y_clim)

            models[f'lag_{lag}'] = model_lag

            y_pred_l = pd.concat(y_pred_l)
            y_pred_l = y_pred_l.sort_index()

            if lag == lags_i[0]:
                y_pred_c = pd.concat(y_pred_c)
                y_pred_c = y_pred_c.sort_index()


            y_pred_all.append(y_pred_l)
        y_pred_all = pd.concat(y_pred_all, axis=1)
        print("\n")
    #%%
        return y_pred_all, y_pred_c, models

def df_data_to_RV(df_data=pd.DataFrame, kwrgs_events=dict, only_RV_events=True,
                  fit_model_dates=None):
    '''
    input df_data according to RGCPD format
    '''

    RVfullts = pd.DataFrame(df_data[df_data.columns[0]][0])
    RV_ts    = pd.DataFrame(df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] )
    RV = class_RV.RV_class(fullts=RVfullts, RV_ts=RV_ts, kwrgs_events=kwrgs_events,
                          only_RV_events=only_RV_events, fit_model_dates=fit_model_dates)
    return RV


def fit(y_ts, df_data, lag, split=int, stat_model=str,
        keys_d=None, dates_tobin=None, precur_aggr=None, kwrgs_pp={},
        verbosity=0):
    #%%

    if keys_d is not None:
        keys = keys_d[split].copy()
    else:
        keys = None

    model_name, kwrgs = stat_model
    df_split = df_data.loc[split].copy()
    df_split = df_split.dropna(axis=1, how='all')
    df_norm, keys = prepare_data(y_ts, df_split, lag_i=int(lag),
                                 dates_tobin=dates_tobin,
                                 precur_aggr=precur_aggr,
                                 keys=keys,
                                 **kwrgs_pp)
#             if s == 0 and lag ==1:
#                 x_fit_mask, y_fit_mask, x_pred_mask, y_pred_mask = stat_models.get_masks(df_norm)
#                 print(x_fit_mask)
#                 print(y_fit_mask)

#                print(keys, f'\n lag {lag}\n')
#                print(df_norm[x_fit_mask]['RV_ac'])
#                print(RV.RV_bin)
    # forecasting models
    if model_name == 'logit':
        prediction, model = stat_models.logit(y_ts, df_norm, keys=keys)
    if model_name == 'logitCV':
        kwrgs_logit = kwrgs
        prediction, model = stat_models.logit_skl(y_ts, df_norm, keys,
                                                  kwrgs_logit=kwrgs_logit)
    if model_name == 'GBC':
        kwrgs_GBC = kwrgs
        prediction, model = stat_models.GBC(y_ts, df_norm, keys,
                                                    kwrgs_GBM=kwrgs_GBC,
                                                    verbosity=verbosity)

    # store original data used for fit into model
    model.df_norm = df_norm

    if precur_aggr is None:
        tfreq = (y_ts['cont'].index[1] - y_ts['cont'].index[0]).days
        lags_tf = [l*tfreq for l in [lag]]
        if tfreq != 1:
            # the last day of the time mean bin is tfreq/2 later then the centerered day
            lags_tf = [l_tf- int(tfreq/2) if l_tf!=0 else 0 for l_tf in lags_tf]
    else:
        lags_tf = [lag]
    prediction = pd.DataFrame(prediction.values, index=prediction.index,
                              columns=lags_tf)
    #%%
    return (prediction, model)




def prepare_data(y_ts, df_split, lag_i=int, dates_tobin=None,
                     precur_aggr=None, normalize='datesRV', remove_RV=True,
                     keys=None, add_autocorr=True, EOF=False, expl_var=None):



    '''
    TrainisTrue     : Specifies train and test dates, col of df_split.
    RV_mask         : Specifies what data will be predicted, col of df_split.
    fit_model_dates : Deprecated... It can be desirable to train on
                      more dates than what you want to predict, col of df_split.
    remove_RV       : First column is the RV, and is removed.
    lag_i           : Mask for fitting and predicting will be shifted with
                      {lag_i} periods

    returns:
        df_norm     : Dataframe
        x_keys      : updated set of keys to fit model
    '''
    #%%
    # lag_i=fc.lags_i[0]
    # normalize='datesRV'
    # remove_RV=True
    # keys=None
    # add_autocorr=True
    # EOF=False
    # expl_var=None
    # =============================================================================
    # Select features / variables
    # =============================================================================
    if keys is None:
        keys = np.array(df_split.dtypes.index[df_split.dtypes != bool], dtype='object')

    RV_name = df_split.columns[0]
    df_RV = df_split[RV_name]
    if remove_RV is True:
        # completely remove RV timeseries
        df_prec = df_split.drop([RV_name], axis=1).copy()
        keys = np.array([k for k in keys if k != RV_name], dtype='object')
    else:
        keys = np.array(keys, dtype='object')
        df_prec = df_split.copy()
    # not all keys are present in each split:
    keys = [k for k in keys if k in list(df_split.columns)]
    x_keys = np.array(keys, dtype='object')

    dates_TV = y_ts['cont'].index
    tfreq_TV = (dates_TV[1] - dates_TV[0]).days


    # if type(add_autocorr) is int: # not really a clue what this does
    #     adding_ac_mlag = lag_i <= 2
    # else:
    #     adding_ac_mlag = True

    if add_autocorr:

        if lag_i == 0 and precur_aggr is None:
            # minimal shift of lag 1 or it will follow shift with x_fit mask
            RV_ac = df_RV.shift(periods=-1).copy()
        elif precur_aggr is not None and lag_i < int(tfreq_TV/2):

            # df_RV is daily and should be shifted more tfreq/2 otherwise just
            # predicting with the (part) of the observed ts.
            # I am selecting dates_min_lag, thus adding RV that is also shifted
            # min_lag days will result in that I am selecting the actual
            # observed ts.
            # lag  < tfreq_TV
            shift = tfreq_TV - lag_i
            RV_ac = df_RV.shift(periods=-shift).copy()
            # for lag_i = 0, tfreq_TV=10
            # 1979-06-15    7.549415 is now:
            # 1979-06-20    7.549415
            # when selecting value of 06-15, I am actually selecting val of 6-10
            # minimal shift of 10 days backward in time is realized
        else:
            RV_ac = df_RV.copy() # RV will shifted according fit_masks, lag will be > 1

        # plugging in the mean value for the last date if no data
        # is available to shift backward
        RV_ac.loc[RV_ac.isna()] = RV_ac.mean()

        df_prec.insert(0, 'autocorr', RV_ac)
        # add key to keys
        if 'autocorr' not in keys:
            x_keys = np.array(np.insert(x_keys, 0, 'autocorr'), dtype='object')


    # =============================================================================
    # Shifting data w.r.t. index dates
    # =============================================================================
    if dates_tobin is None:
        # we can only make lag steps of size tfreq, e.g. if df_data contains
        # 10 day means, we can make a lag_step of 1, 2, etc, resulting in lag
        # in days of 5, 15, 25.
        fit_masks = df_split.loc[:,['TrainIsTrue', 'RV_mask']].copy()
        fit_masks = util.apply_shift_lag(fit_masks, lag_i)
    elif type(dates_tobin) == pd.DatetimeIndex:
        # df_data contain daily data, we can shift dates_tobin to allow any
        # lag in days w.r.t. target variable
        dates_TV = y_ts['cont'].index
        tfreq_TV = (dates_TV[1] - dates_TV[0]).days
        if lag_i == 0:
            base_lag = 0
        else:
            base_lag = int(tfreq_TV / 2) # minimal shift to get lag vs onset
        last_centerdate = dates_TV[-1]
        fit_masks = df_split.loc[:,['TrainIsTrue', 'RV_mask']].copy()
        fit_masks = util.apply_shift_lag(fit_masks, lag_i+base_lag)
        df_prec = df_prec[x_keys].merge(fit_masks, left_index=True, right_index=True)
        dates_bin_shift = functions_pp.func_dates_min_lag(dates_tobin, lag_i+base_lag)[1]

        df_prec, dates_tobin_p = _daily_to_aggr(df_prec.loc[dates_bin_shift], precur_aggr)
        fit_masks = df_prec[df_prec.columns[df_prec.dtypes==bool]]
        # check y_fit mask
        fit_masks = util._check_y_fitm(fit_masks, lag_i, base_lag)
        lag_v = (last_centerdate - df_prec[df_prec['x_fit']].index[-1]).days
        if tfreq_TV == precur_aggr:
            assert lag_v == lag_i+base_lag, (
                f'lag center precur vs center TV is {lag_v} days, with '
                f'lag_i {lag_i} and base_lag {base_lag}')
    elif type(dates_tobin) is tuple:
        df_prec = start_end_date_mean(df_prec,
                            start_end_date=dates_tobin)
        fit_masks = util.apply_shift_lag(df_prec[['TrainIsTrue', 'RV_mask']], 0)


    df_prec = df_prec[x_keys]
    # =============================================================================
    # Normalize data using datesRV or all training data in dataframe
    # =============================================================================
    if normalize=='all':
        # Normalize using all training dates
        TrainIsTrue = fit_masks['TrainIsTrue']
        df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainIsTrue].mean(0)) \
                / df_prec[x_keys][TrainIsTrue].std(0)
    elif normalize=='datesRV' or normalize==True:
        # Normalize only using the RV dates
        TrainRV = np.logical_and(fit_masks['TrainIsTrue'],fit_masks['y_pred']).values
        df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainRV].mean(0)) \
                / df_prec[x_keys][TrainRV].std(0)
    elif normalize=='x_fit':
        # Normalize only using the RV dates
        TrainRV = np.logical_and(fit_masks['TrainIsTrue'],fit_masks['x_fit']).values
        df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainRV].mean(0)) \
                / df_prec[x_keys][TrainRV].std(0)
    elif normalize==False:
        pass


    if EOF:
        if expl_var is None:
            expl_var = 0.75
        else:
            expl_var = expl_var
        df_prec = transform_EOF(df_prec, fit_masks['TrainIsTrue'],
                                fit_masks['x_fit'], expl_var=expl_var)
        df_prec.columns = df_prec.columns.astype(str)
        upd_keys = np.array(df_prec.columns.values.ravel(), dtype=str)
    else:
        upd_keys = x_keys

    # =============================================================================
    # Replace masks
    # =============================================================================
    df_prec = df_prec.merge(fit_masks, left_index=True, right_index=True)
    #%%
    return df_prec, upd_keys


def transform_EOF(df_prec, TrainIsTrue, RV_mask, expl_var=0.8):
    '''
    EOF is based upon all Training data.
    '''
    #%%
    import eofs
    dates_train = df_prec[TrainIsTrue].index
    dates_test  = df_prec[TrainIsTrue==False].index

    to_xr = df_prec.to_xarray().to_array().rename({'index':'time'}).transpose()
    xr_train = to_xr.sel(time=dates_train)
    xr_test = to_xr.sel(time=dates_test)
    eof = eofs.xarray.Eof(xr_train)
    for n in range(df_prec.columns.size):
        frac = eof.varianceFraction(n).sum().values
        if frac >= expl_var:
            break
    xr_train = eof.pcs(npcs=n)
    xr_proj = eof.projectField(xr_test, n)
    xr_proj = xr_proj.rename({'pseudo_pcs', 'pcs'})
    xr_eof  = xr.concat([xr_train, xr_proj], dim='time').sortby('time')
    df_eof  = xr_eof.T.to_dataframe().reset_index(level=0)
    df_eof  = df_eof.pivot(columns='mode', values='pcs' )
    #%%
    return df_eof


def _daily_to_aggr(df_data, daily_to_aggr=int):
    import functions_pp
    if hasattr(df_data.index, 'levels'):
        splits = df_data.index.levels[0]
        df_data_s   = np.zeros( (splits.size) , dtype=object)
        for s in splits:
            df_data_s[s], dates_tobin = functions_pp.time_mean_bins(
                                                        df_data.loc[s],
                                                        to_freq=daily_to_aggr,
                                                        start_end_date=None,
                                                        start_end_year=None,
                                                        verbosity=0)
        df_data_resample  = pd.concat(list(df_data_s), keys= range(splits.size))
    else:
        df_data_resample, dates_tobin = functions_pp.time_mean_bins(df_data,
                                                       to_freq=daily_to_aggr,
                                                       start_end_date=None,
                                                       start_end_year=None,
                                                       verbosity=0)
    return df_data_resample, dates_tobin

def start_end_date_mean(df_data, start_end_date):

    # create mask to aggregate
    if hasattr(df_data.index, 'levels'):
        pd_dates = df_data.loc[0].index
    else:
        pd_dates = df_data.index
    subset_dates = core_pp.get_subdates(pd_dates , start_end_date)
    dates_to_aggr_mask = pd.Series(np.repeat(False, pd_dates.size), index=pd_dates)
    dates_to_aggr_mask.loc[subset_dates] = True
    if hasattr(df_data.index, 'levels'):
        years = df_data.loc[0][dates_to_aggr_mask].index.year
    else:
        years = df_data[dates_to_aggr_mask].index.year
    index = [functions_pp.get_oneyr(subset_dates, yr).mean() for yr in np.unique(years)]

    if hasattr(df_data.index, 'levels'):
        splits = df_data.index.levels[0]
        df_data_s   = np.zeros( (splits.size) , dtype=object)
        for s in splits:
            df_s = df_data.loc[s]
            df_s = df_s[dates_to_aggr_mask].groupby(years).mean()
            df_s.index = pd.to_datetime(index)
            df_data_s[s] = df_s
        df_data_resample  = pd.concat(list(df_data_s), keys= range(splits.size))
    else:
         df_data_resample = df_data[dates_to_aggr_mask].groupby(years).mean()
         df_data_resample.index = pd.to_datetime(index)
    return df_data_resample
