#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:54:45 2019

@author: semvijverberg
"""
import h5py
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer


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

def logit_model(y, df_norm, keys):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    if keys[0] == None:
        no_data_col = ['TrainIsTrue', 'RV_mask']
        keys = df_norm.columns
        keys = [k for k in keys if k not in no_data_col]
        
    X = df_norm[keys]
    X = add_constant(X)
    TrainIsTrue = df_norm['TrainIsTrue']
    
    
    X_train = X[TrainIsTrue]
    y_train = y[TrainIsTrue.values] 
    model_set = sm.Logit(y_train.values, X_train, disp=0)
    try:
        model = model_set.fit( disp=0, maxfun=35 )
        prediction = model.predict(X)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            model = model_set.fit(method='bfgs', disp=0 )
            prediction = model.predict(X)
        else:
            raise
    except Exception as e:
        print(e)
        model = model_set.fit(method='bfgs', disp=0 )
        prediction = model.predict(X)
    #%%
    return prediction, model

def GBR(RV, df_norm, keys, kwrgs_GBR=None):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':1,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_ts = RV.RV_ts
    TrainIsTrue = df_norm['TrainIsTrue']
  
    X_train = X[TrainIsTrue]
    y_train = RV_ts[TrainIsTrue.values] 
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    regressor = GradientBoostingRegressor(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
        scoring   = 'r2'
#        scoring   = 'neg_mean_squared_error'
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=5, refit=scoring, 
                  return_train_score=False)
        regressor.fit(X_train, y_train.values.ravel())
        results = regressor.cv_results_
        scores = results['mean_test_score'] / results['std_test_score']
        improv = int(100* (min(scores)-max(scores)) / max(scores))
        print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
              "best params {}".format(
                improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    

    prediction = pd.DataFrame(regressor.predict(X), index=X.index, columns=[0])
    prediction['TrainIsTrue'] = pd.Series(TrainIsTrue.values, index=X.index)
    
    logit_pred, model_logit = logit_model(RV.RV_bin, prediction, keys=[None])
    #%%
    return logit_pred, (model_logit, regressor)

def GBC(RV, df_norm, keys, kwrgs_GBR=None):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':1,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_bin = RV.RV_bin
    TrainIsTrue = df_norm['TrainIsTrue']
  
    X_train = X[TrainIsTrue]
    y_train = RV_bin[TrainIsTrue.values] 
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    regressor = GradientBoostingClassifier(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
        scoring   = 'brier_score_loss'
#        scoring   = 'neg_mean_squared_error'
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=5, refit=scoring, 
                  return_train_score=False)
        regressor.fit(X_train, y_train.values.ravel())
        results = regressor.cv_results_
        scores = results['mean_test_score'] / results['std_test_score']
        improv = int(100* (min(scores)-max(scores)) / max(scores))
        print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
              "best params {}".format(
                improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    

    prediction = pd.DataFrame(regressor.predict(X), index=X.index, columns=[0])

    #%%
    return prediction, regressor

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