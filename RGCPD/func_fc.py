#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:54:45 2019

@author: semvijverberg
"""
import h5py
import pandas as pd
import numpy as np
import stat_models
import validation as valid


def forecast_and_valid(RV, df_data, keys_d, stat_model=tuple, lags=list, 
                       n_boot=0, verbosity=0):
    #%%
    # do forecasting accros lags
    splits  = df_data.index.levels[0]
    y_pred_all = []
    y_pred_c = []
    c = 0
    
    for lag in lags:
        
        y_pred_l = []
    
        for s in splits:
            c += 1
            progress = int(100 * (c) / (len(splits) * len(lags)))
            print(f"\rProgress {progress}%", end="")
            
            df_norm, RV_mask, TrainIsTrue = prepare_data(df_data.loc[s])
        
            keys = keys_d[s].copy()
            model_name, kwrgs = stat_model
            
    #        dates_all = pd.to_datetime(RV.index.values)
            
            dates_l  = func_dates_min_lag(RV.dates_RV, lag, indays=False)[1]
            df_norm = df_norm.loc[dates_l]
            if lag != 0:
                df_norm['RV_autocorr'] = RV.RVfullts.loc[dates_l]
                keys = np.insert(keys, 0, 'RV_autocorr')
            
            # forecasting models
            if model_name == 'logit':
                prediction, model = stat_models.logit(RV, df_norm, keys=keys)
            if model_name == 'GBR':
                kwrgs_GBR = kwrgs
                prediction, model = stat_models.GBR(RV, df_norm, keys, 
                                                    kwrgs_GBR=kwrgs_GBR, 
                                                    verbosity=verbosity)
            if model_name == 'logit_skl':
                kwrgs_logit = kwrgs
                prediction, model = stat_models.logit_skl(RV, df_norm, keys, 
                                                          kwrgs_logit=kwrgs_logit)
            if model_name == 'GBR_logitCV':
                kwrgs_GBR = kwrgs
                prediction, model = stat_models.GBR_logitCV(RV, df_norm, keys, 
                                                            kwrgs_GBR=kwrgs_GBR, 
                                                            verbosity=verbosity)
            if model_name == 'GBR_classes':
                kwrgs_GBR = kwrgs
                prediction, model = stat_models.GBR_classes(RV, df_norm, keys, 
                                                            kwrgs_GBR=kwrgs_GBR, 
                                                            verbosity=verbosity)
                
            prediction = pd.DataFrame(prediction.values, index=RV.dates_RV,
                                      columns=[lag])
            y_pred_l.append(prediction[(df_norm['TrainIsTrue']==False).values])  
            
            if lag == lags[0]:
                # determining climatological prevailance in training data
                y_c_mask = np.logical_and(df_norm['TrainIsTrue'].values, RV.RV_bin.squeeze().values==1)
                y_clim_val = RV.RV_bin[y_c_mask].size / RV.RV_bin[df_norm['TrainIsTrue'].values].size
                y_clim = RV.RV_bin[df_norm['TrainIsTrue'].values==False].copy()
                y_clim[:] = y_clim_val
                y_pred_c.append(y_clim)
            
        y_pred_l = pd.concat(y_pred_l) 
        y_pred_l = y_pred_l.sort_index()
        
        if lag == lags[0]:
            y_pred_c = pd.concat(y_pred_c) 
            y_pred_c = y_pred_c.sort_index()
    
    
        y_pred_all.append(y_pred_l)
    y_pred_all = pd.concat(y_pred_all, axis=1) 
    print("\n")
    # do validation

    print(f'{stat_model} ')

    
    blocksize = valid.get_bstrap_size(RV.RVfullts, plot=False)
    out = valid.get_metrics_sklearn(RV, y_pred_all, y_pred_c, n_boot=n_boot,
                                    blocksize=blocksize)
    df_valid, metrics_dict = out
    #%%

    y_pred_all[1][y_pred_all[1]==1].size / RV.RV_bin[RV.RV_bin==1].size
    return df_valid, RV, y_pred_all

def load_hdf5(path_data):
    hdf = h5py.File(path_data,'r+')   
    dict_of_dfs = {}
    for k in hdf.keys(): 
        dict_of_dfs[k] = pd.read_hdf(path_data, k)
    hdf.close()
    return dict_of_dfs

def prepare_data(df_split):
    '''
    TrainisTrue : specifies train and test dates
    RV_mask     : specifies what data will be predicted
    RV          : Response Variable (raw data)
    df_norm     : Normalized precursor data    
    '''
    TrainisTrue = df_split['TrainIsTrue']
    RV_mask = df_split['RV_mask']
    RV_name = df_split.columns[0]
    df_prec = df_split.drop([RV_name], axis=1)
    x_keys  = df_prec.columns[df_prec.dtypes == np.float64]
    df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainisTrue].mean(0)) \
                / df_prec[x_keys][TrainisTrue].std(0)

    return df_prec, RV_mask, TrainisTrue



def Ev_threshold(xarray, event_percentile):
    if event_percentile == 'std':
        # binary time serie when T95 exceeds 1 std
        threshold = xarray.mean() + xarray.std()
    else:
        percentile = event_percentile
        
        threshold = np.percentile(xarray.values, percentile)
    return float(threshold)

def Ev_timeseries(xarray, threshold, min_dur=1, max_break=0, grouped=False, 
                  high_ano_events=True):  
    #%%
    import xarray as xr 
    if type(xarray) != type(xr.DataArray([0])):
        give_df_back = True
        xarray = xarray.to_xarray().rename({'index':'time'})
    else:
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
        Events = Events.to_dataframe()
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