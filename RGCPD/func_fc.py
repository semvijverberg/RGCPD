#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:54:45 2019

@author: semvijverberg
"""
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
import h5py
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import stat_models
import class_RV
import validation as valid
df_ana_path = os.path.join(main_dir, 'df_analysis/df_analysis/')
if df_ana_path not in sys.path:
    sys.path.append(df_ana_path)
import df_ana
import exp_fc
import multiprocessing
max_cpu = multiprocessing.cpu_count()
print(f'{max_cpu} cpu\'s detected')


class fcev():
    
    number_of_times_called = 0
    def __init__(self, path_data, name=None):
        '''
        Instance for certain dataset with keys and list of stat models
        
        n_boot      :   times to bootstrap
        '''
        
        self.path_data = path_data

        if name is None:
            self.name = 'exper1'
        else:    
            self.name = name
            
        if fcev.number_of_times_called == 0:
            fcev.df_data = load_hdf5(self.path_data)['df_data']
            fcev.splits  = fcev.df_data.index.levels[0]
            fcev.tfreq = (fcev.df_data.loc[0].index[1] - fcev.df_data.loc[0].index[0]).days
            fcev.RV_mask = fcev.df_data['RV_mask']
            fcev.TrainIsTrue = fcev.df_data['TrainIsTrue']
            fcev.test_years = valid.get_testyrs(fcev.df_data)

        fcev.number_of_times_called += 1
        return 

    @classmethod
    def get_test_data(cls, stat_model_l=None, keys_d=None, causal=False, n_boot=100):
        path_py   = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        name = 'E-US_temp_test'
        test_fname = 'test_TV-US-temp_X_sst-z500-sm.h5'
        path_data = os.path.join('/'.join(path_py.split('/')[:-1]), 'data', test_fname)
        return cls(path_data, name=name)
         
    def get_TV(self, kwrgs_events=None, fit_model_dates=None):
        
        if hasattr(self, 'df_data') == False:
            print("df_data not loaded, initialize fcev class with path to df_data") 
        
        # target events
        if kwrgs_events is None:
            self.kwrgs_events = {'event_percentile': 66,
                        'min_dur' : 1,
                        'max_break' : 0,
                        'grouped' : False}
        else:
            self.kwrgs_events = kwrgs_events
            
        
        TV = df_data_to_RV(self.df_data, kwrgs_events=self.kwrgs_events, 
                           fit_model_dates=fit_model_dates)
        TV.TrainIsTrue = self.df_data['TrainIsTrue']
        TV.RV_mask = self.df_data['RV_mask']
    
        splits  = self.df_data.index.levels[0]
        fit_model_mask = pd.concat([TV.fit_model_mask] * splits.size, keys=splits)
        self.df_data = self.df_data.merge(fit_model_mask, left_index=True, right_index=True)
        TV.prob_clim = get_obs_clim(TV)
        self.TV = TV
        return 
    
    def fit_models(self, stat_model_l=[('logit', None)], lead_max=np.array([1]), 
                   keys_d=None, causal=False, kwrgs_pp=None):
        '''
        stat_model_l:   list of with model string and kwrgs 
        keys_d      :   dict, with keys : list of variables to fit, if None
                        all keys in each training set will be used to fit.
                        If string is given, exp_py will follow some rules to 
                        keep only keys you want to fit.
        '''
        
        self.stat_model_l = stat_model_l
        self.causal = causal
            
        if keys_d is None:
            print('keys is None: Using all keys in training sets')
            self.experiment = 'all'
            self.keys_d = exp_fc.normal_precursor_regions(self.path_data, causal=False)['all']
        if isinstance(keys_d, dict):
            self.experiment = 'manual'
            # expecting dict with traintest number as key and associated list of keys
            self.keys_d = keys_d
        if isinstance(keys_d, str):
            print(f'getting keys associated with name {keys_d}')
            self.experiment = keys_d
            self.keys_d = exp_fc.normal_precursor_regions(self.path_data, 
                                                          keys_options=[keys_d], 
                                                          causal=self.causal)[keys_d]
        if isinstance(lead_max, int): 
            if fcev.tfreq == 1:
                self.lags_i = np.arange(0, lead_max+1E-9, max(10,fcev.tfreq), dtype=int)
            else:
                self.lags_i = np.array(np.arange(0, lead_max+fcev.tfreq/2+1E-9, 
                                            max(10,fcev.tfreq))/max(10,fcev.tfreq), 
                                            dtype=int)
        elif type(lead_max) == np.ndarray:
            self.lags_i = lead_max
        else:
            print('lead_max should be integer or np.ndarray')

        if fcev.tfreq == 1: 
            self.lags_t = np.array([l * fcev.tfreq for l in self.lags_i])
        else:
            if self.lags_i[0] == 0:
                self.lags_t = [0]
                for l in self.lags_i[1:]:
                    self.lags_t.append(int((l-1) * fcev.tfreq + fcev.tfreq/2))
            else:
                self.lags_t = np.array([(l-1) * fcev.tfreq + fcev.tfreq/2 for l in self.lags_i])
            self.lags_t = np.array(self.lags_t)
        print(f'tfreq: {fcev.tfreq}, max lag: {self.lags_i[-1]}, i.e. {self.lags_t[-1]} days')

        if kwrgs_pp is None:
            self.kwrgs_pp = {'EOF':False, 
                    'expl_var':0.5,
                    'fit_model_dates' : None}
        else:
            self.kwrgs_pp = kwrgs_pp

        self.dict_preds = {}
        self.dict_models = {}
        for stat_model in stat_model_l:
            name = stat_model[0]
            y_pred_all, y_pred_c, models = _fit_model(self.TV, 
                                                      df_data=self.df_data, 
                                                      keys_d=self.keys_d, 
                                                      kwrgs_pp=kwrgs_pp, 
                                                      stat_model=stat_model, 
                                                      lags_i=self.lags_i)
            self.dict_preds[name] = (y_pred_all, y_pred_c)
            self.dict_models[name] = models
            
                     
        return 

    def perform_validation(self, n_boot=2000, blocksize='auto', 
                           threshold_pred='upper_clim'):
        self.n_boot = n_boot
        self.threshold_pred = threshold_pred
        self.dict_sum = {}
        for stat_model in self.stat_model_l:
            name = stat_model[0]
            y_pred_all, y_pred_c = self.dict_preds[name]
            
            if blocksize == 'auto':    
                self.blocksize = valid.get_bstrap_size(self.TV.fullts, plot=False)
            else:
                self.blocksize = blocksize
            out = valid.get_metrics_sklearn(self.TV, y_pred_all, y_pred_c, 
                                            n_boot=n_boot,
                                            blocksize=self.blocksize, 
                                            threshold_pred=threshold_pred)
            df_valid, metrics_dict = out
            self.dict_sum[name] = (df_valid, self.TV, y_pred_all)
            self.metrics_dict = metrics_dict
        return 

    @classmethod
    def plot_scatter(cls, keys=None, colwrap=3, sharex='none', s=0, mask='RV_mask', aggr=None, 
                     title=None):
        df_d = cls.df_data.loc[s]
        if mask is None:
            tv = cls.df_data.loc[0].iloc[:,0]
            df_d = df_d
        elif mask == 'RV_mask':
            tv = cls.df_data.loc[0].iloc[:,0][cls.RV_mask.loc[s]]
            df_d = df_d[fcev.RV_mask.loc[s]]
        else:
            tv = cls.df_data.loc[0].iloc[:,0][mask]
            df_d = df_d[mask]
        kwrgs = {'tv':tv,
                'aggr':aggr,
                 'title':title}
        df_ana.loop_df(df_d, df_ana.plot_scatter, keys=keys, colwrap=colwrap, 
                            sharex=sharex, kwrgs=kwrgs)
        return    

    
    def plot_freq_year(self):
        import valid_plots as df_plots
        df_plots.plot_freq_per_yr(self.TV)
    
    def plot_GBR_feature_importances(self, lag=None, cutoff=6):
        GBR_models_split_lags = self.dict_models['GBR-logitCV']
        if lag is None:
            lag = self.lags_i
        self.df_importance = stat_models.plot_importances(GBR_models_split_lags, lag=lag, 
                                     cutoff=cutoff)


        
        
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


def fit(y_ts, df_data, lag, split, stat_model=str, keys_d=None, 
        kwrgs_pp={}, verbosity=0):
    #%%
    
    if keys_d is not None:
        keys = keys_d[split].copy()
    else:
        keys = None
        
    model_name, kwrgs = stat_model
    df_split = df_data.loc[split].copy()
    df_norm, keys = prepare_data(df_split, lag_i=int(lag),
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
    if model_name == 'logit-CV':
        kwrgs_logit = kwrgs
        prediction, model = stat_models.logit_skl(y_ts, df_norm, keys,
                                                  kwrgs_logit=kwrgs_logit)
    if model_name == 'GBR-logitCV':
        kwrgs_GBR = kwrgs
        prediction, model = stat_models.GBR_logitCV(y_ts, df_norm, keys,
                                                    kwrgs_GBR=kwrgs_GBR,
                                                    verbosity=verbosity)
    
    # store original data used for fit into model
    model.X = df_norm 
        
    prediction = pd.DataFrame(prediction.values, index=prediction.index,
                              columns=[lag])
    #%%
    return (prediction, model)
            

def _fit_model(RV, df_data, keys_d=None, kwrgs_pp={}, stat_model=tuple, lags_i=list,
              verbosity=0):
    #%%
#    stat_model = fc.stat_model_l[0]
#    RV = fc.TV
#    lags_i = [1]
#    kwrgs_pp={}
#    keys_d=None
#    df_data = fc.df_data
#    verbosity=0
   
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
        
    from time import time
    t0 = time()
    futures = {}
    with ProcessPoolExecutor(max_workers=max_cpu) as pool:
        for lag in lags_i:

            for split in splits:
                fitkey = f'{lag}_{split}'

                futures[fitkey] = pool.submit(fit, y_ts, df_data, lag, split, 
                                           stat_model=stat_model, keys_d=keys_d, 
                                           kwrgs_pp=kwrgs_pp, verbosity=verbosity)
        results = {key:future.result() for key, future in futures.items()}
    
    
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
            df_norm = model.X            
            TestRV  = (df_norm['TrainIsTrue']==False)[df_norm['y_pred']]
            y_pred_l.append(prediction[TestRV.values])

            if lag == lags_i[0]:
                # ensure that RV timeseries matches y_pred
                TrainRV = (df_norm['TrainIsTrue'])[df_norm['y_pred']]
                RV_bin = RV.RV_bin.loc[TrainRV.index]

                # predicting RV might not be possible 
                # determining climatological prevailance in training data                
                y_c_mask = np.logical_and(TrainRV, RV_bin.squeeze()==1)
                y_clim_val = RV_bin[y_c_mask].size / RV_bin.size
                # filling test years with clim of training data
                y_clim = RV_bin[TestRV==True].copy()
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
    print(time() - t0)
    print(f'{stat_model} ')
    #%%
    return y_pred_all, y_pred_c, models

def prepare_data(df_split, lag_i=int, normalize='datesRV', remove_RV=True, 
                 keys=None, add_autocorr=True, EOF=False, 
                 expl_var=None):
                 
    #%%
    '''
    TrainisTrue     : Specifies train and test dates, col of df_split.
    RV_mask         : Specifies what data will be predicted, col of df_split.
    fit_model_dates : Optional: It can be desirable to train on
                      more dates than what you want to predict, col of df_split.
    remove_RV       : First column is the RV, and is removed.
    lag_i           : Mask for fitting and predicting will be shifted with 
                      {lag_i} periods
     
    returns:
        df_norm     : Dataframe
        x_keys      : updated set of keys to fit model
    '''
# lag_i=1
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
    x_keys = np.array(keys, dtype='object')
    
    if type(add_autocorr) is int:
        adding_ac_mlag = lag_i <= 2
    else:
        adding_ac_mlag = True
        
    if add_autocorr and adding_ac_mlag:
        # minimal shift of lag 1 or it will follow shift with x_fit mask
        if lag_i == 0:
            RV_ac = df_RV.shift(periods=-1).copy()
        else: 
            RV_ac = df_RV.copy() # RV will shifted according fit_masks, lag will be > 1
            
        # plugging in the mean value for the last date if no data 
        # is available to shift backward
        RV_ac.loc[RV_ac.isna()] = RV_ac.mean()

        df_prec.insert(0, 'autocorr', RV_ac)
        # add key to keys
        if 'autocorr' not in keys:
            x_keys = np.array(np.insert(x_keys, 0, 'autocorr'), dtype='object')

    df_prec = df_prec[x_keys]

    # =============================================================================
    # Shifting data w.r.t. index dates
    # =============================================================================
    fit_masks = df_split.loc[:,['TrainIsTrue', 'RV_mask', 'fit_model_mask']].copy()
    fit_masks = apply_shift_lag(fit_masks, lag_i)
#    if 
    # =============================================================================
    # Normalize data using datesRV or all training data in dataframe
    # =============================================================================
    if normalize=='all':
        # Normalize using all training dates
        TrainIsTrue = fit_masks['TrainIsTrue']
        df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainIsTrue].mean(0)) \
                / df_prec[x_keys][TrainIsTrue].std(0)
    elif normalize=='datesRV':
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
                                fit_masks['x_fit'], expl_var=0.8)
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

def apply_shift_lag(fit_masks, lag_i):
    '''
    only shifting the boolean masks, Traintest split info is contained
    in the TrainIsTrue mask.
    '''
    RV_mask = fit_masks['RV_mask'].copy()
    y_fit = fit_masks['fit_model_mask'].copy()
    x_fit = y_fit.shift(periods=-int(lag_i))
    n_nans = x_fit[~x_fit.notna()].size
    # set last x_fit date to False if x_fit caused nan
    if n_nans > 0:
        # take into account that last x_fit_train should be False to have
        # equal length y_train & x_fit and to avoid Train-test mix-up due to lag
        x_fit[~x_fit.notna()] = False
        x_date = x_fit[fit_masks['TrainIsTrue']].index[-n_nans:]
        x_fit.loc[x_date] = False
    
    x_pred = RV_mask.shift(periods=-int(lag_i))
    x_pred[~x_pred.notna()] = False
    # first indices of RV_mask cannot be predicted at lag > lag_i
    if lag_i > 0:
        # y_date cannot be predicted elsewise mixing
        # Train test dates
        y_date = RV_mask[fit_masks['TrainIsTrue']].index[:int(lag_i)]
        RV_mask.loc[y_date] = False
        y_fit.loc[y_date] = False # 

    fit_masks['x_fit'] = x_fit
    fit_masks['y_fit'] = y_fit
    fit_masks['x_pred'] = x_pred
    fit_masks['y_pred'] = RV_mask
    fit_masks = fit_masks.drop(['RV_mask'], axis=1)
    fit_masks = fit_masks.drop(['fit_model_mask'], axis=1)
    return fit_masks.astype(bool)

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


def get_freq_years(RV_bin):
    all_years = np.unique(RV_bin.index.year)
    binary = RV_bin.values
    freq = []
    for y in all_years:
        n_ev = int(binary[RV_bin.index.year==y].sum())
        freq.append(n_ev)
    return pd.Series(freq, index=all_years)

def get_obs_clim(RV):
    splits = RV.TrainIsTrue.index.levels[0]
    RV_mask_s = RV.RV_mask
    TrainIsTrue = RV.TrainIsTrue
    y_prob_clim = RV.RV_bin.copy()
    y_prob_clim = y_prob_clim.rename(columns={'RV_binary':'prob_clim'})
    for s in splits:
        RV_train_mask = TrainIsTrue[s][RV_mask_s[s]]
        y_b_train = RV.RV_bin[RV_train_mask]
        y_b_test  = RV.RV_bin[RV_train_mask==False]

        clim_prevail = y_b_train.sum() / y_b_train.size
        clim_arr = np.repeat(clim_prevail, y_b_test.size).values
        pdseries = pd.Series(clim_arr, index=y_b_test.index)
        y_prob_clim.loc[y_b_test.index, 'prob_clim'] = pdseries
    return y_prob_clim

def Ev_threshold(xarray, event_percentile):
    if event_percentile == 'std':
        # binary time serie when T95 exceeds 1 std
        threshold = xarray.mean() + xarray.std()
    else:
        percentile = event_percentile

        threshold = np.percentile(xarray.values, percentile)
    return float(threshold)

def Ev_timeseries(xr_or_df, threshold, min_dur=1, max_break=0, grouped=False,
                  high_ano_events=True):
    #%%
    '''
    Binary events timeseries is created according to parameters:
    threshold   : if ts exceeds threshold hold, timestep is 1, else 0
    min_dur     : minimal duration of exceeding a threshold, else 0
    max_break   : break in minimal duration e.g. ts=[1,0,1], is still kept
                  with min_dur = 2 and max_break = 1.
    grouped     : boolean. 
                  If consecutive events (with possible max_break) are grouped
                  the centered date is set is to 1. 
    high_ano_events : boolean. 
                      if True: all timesteps above threshold is 1,
                      if False, all timesteps below threshold is 1.
    '''
    types = [type(xr.Dataset()), type(xr.DataArray([0])), type(pd.DataFrame([0]))]

    assert (type(xr_or_df) in types), ('{} given, should be in {}'.format(type(xr_or_df), types) )


    if type(xr_or_df) == types[-1]:
        xarray = xr_or_df.to_xarray().to_array()
        give_df_back = True
        try:
            old_name = xarray.index.name
            xarray = xarray.rename({old_name:'time'})
        except:
            pass
        xarray = xarray.squeeze()
    if type(xr_or_df) in types[:-1]:
        xarray = xr_or_df
        give_df_back = False


    tfreq_RVts = pd.Timedelta((xarray.time[1]-xarray.time[0]).values)
    min_dur = min_dur ; max_break = max_break  + 1
    min_dur = pd.Timedelta(min_dur, 'd') / tfreq_RVts
    max_break = pd.Timedelta(max_break, 'd') / tfreq_RVts

    if high_ano_events:
        Ev_ts = xarray.where( xarray.values > threshold)
    else:
        Ev_ts = xarray.where( xarray.values < threshold)

    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    events_idx = [list(xarray.time.values).index(E) for E in Ev_dates.values]
    n_timesteps = Ev_ts.size

    peak_o_thresh = Ev_binary(events_idx, n_timesteps, min_dur, max_break, grouped)
    event_binary_np  = np.array(peak_o_thresh != 0, dtype=int)

    dur = np.zeros( (peak_o_thresh.size) )
    for i in np.arange(1, max(peak_o_thresh)+1):
        size = peak_o_thresh[peak_o_thresh==i].size
        dur[peak_o_thresh==i] = size

    if np.sum(peak_o_thresh) < 1:
        Events = Ev_ts.where(peak_o_thresh > 0 ).dropna(how='all', dim='time').time
        pass
    else:
        peak_o_thresh[peak_o_thresh == 0] = np.nan
        Ev_labels = xr.DataArray(peak_o_thresh, coords=[Ev_ts.coords['time']])
        Ev_dates = Ev_labels.dropna(how='all', dim='time').time

#        Ev_dates = Ev_ts.time.copy()
#        Ev_dates['Ev_label'] = Ev_labels
#        Ev_dates = Ev_dates.groupby('Ev_label').max().values
#        Ev_dates.sort()
        Events = xarray.sel(time=Ev_dates)
    if give_df_back:
        event_binary = pd.DataFrame(event_binary_np, index=pd.to_datetime(xarray.time.values),
                                   columns=['RV_binary'])
        Events = Events.to_dataframe(name='events')
    else:
        event_binary = xarray.copy()
        event_binary.values = event_binary_np
    #%%
    return event_binary, Events, dur

def Ev_binary(events_idx, n_timesteps, min_dur, max_break, grouped=False):

    max_break = max_break + 1
    peak_o_thresh = np.zeros((n_timesteps))
    ev_num = 1
    # group events inter event time less than max_break
    for i in range(len(events_idx)):
        if i < len(events_idx)-1:
            curr_ev = events_idx[i]
            next_ev = events_idx[i+1]
        elif i == len(events_idx)-1:
            curr_ev = events_idx[i]
            next_ev = events_idx[i-1]

        if abs(next_ev - curr_ev) <= max_break:
            peak_o_thresh[curr_ev] = ev_num
        elif abs(next_ev - curr_ev) > max_break:
            peak_o_thresh[curr_ev] = ev_num
            ev_num += 1

    # remove events which are too short
    for i in np.arange(1, max(peak_o_thresh)+1):
        No_ev_ind = np.where(peak_o_thresh==i)[0]
        # if shorter then min_dur, then not counted as event
        if No_ev_ind.size < min_dur:
            peak_o_thresh[No_ev_ind] = 0

    if grouped == True:
        data = np.concatenate([peak_o_thresh[:,None],
                               np.arange(len(peak_o_thresh))[:,None]],
                                axis=1)
        df = pd.DataFrame(data, index = range(len(peak_o_thresh)),
                                  columns=['values', 'idx'], dtype=int)
        grouped = df.groupby(df['values']).mean().values.squeeze()[1:]
        peak_o_thresh[:] = 0
        peak_o_thresh[np.array(grouped, dtype=int)] = 1
    else:
        pass

    return peak_o_thresh

def func_dates_min_lag(dates, lag, indays = True):
    if indays == True:
        dates_min_lag = pd.to_datetime(dates.values) - pd.Timedelta(int(lag), unit='d')
    else:
        timedelta = (dates[1]-dates[0]) * lag
        dates_min_lag = pd.to_datetime(dates.values) - timedelta
    ### exlude leap days from dates_train_min_lag ###


    # ensure that everything before the leap day is shifted one day back in time
    # years with leapdays now have a day less, thus everything before
    # the leapday should be extended back in time by 1 day.
    mask_lpyrfeb = np.logical_and(dates_min_lag.month == 2,
                                         dates_min_lag.is_leap_year
                                         )
    mask_lpyrjan = np.logical_and(dates_min_lag.month == 1,
                                         dates_min_lag.is_leap_year
                                         )
    mask_ = np.logical_or(mask_lpyrfeb, mask_lpyrjan)
    new_dates = np.array(dates_min_lag)
    new_dates[mask_] = dates_min_lag[mask_] - pd.Timedelta(1, unit='d')
    dates_min_lag = pd.to_datetime(new_dates)
    # to be able to select date in pandas dataframe
    dates_min_lag_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_min_lag]
    return dates_min_lag_str, dates_min_lag

def load_hdf5(path_data):
    hdf = h5py.File(path_data,'r+')
    dict_of_dfs = {}
    for k in hdf.keys():
        dict_of_dfs[k] = pd.read_hdf(path_data, k)
    hdf.close()
    return dict_of_dfs
