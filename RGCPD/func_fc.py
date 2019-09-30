#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:54:45 2019

@author: semvijverberg
"""
import h5py
import pandas as pd
import numpy as np
import xarray as xr
import eofs
import stat_models
import validation as valid
import functions_pp
#from functions_pp import make_RVdatestr


def forecast_and_valid(RV, df_data, kwrgs_exp, stat_model=tuple, lags_i=list,
                       n_boot=0, verbosity=0):
    #%%
    # do forecasting accros lags
    splits  = df_data.index.levels[0]
    y_pred_all = []
    y_pred_c = []
    test_yrs = []
    c = 0

    if 'keys' in kwrgs_exp.keys():
        keys_d = kwrgs_exp['keys']


    if 'kwrgs_pp' in kwrgs_exp.keys():
        kwrgs_pp = kwrgs_exp['kwrgs_pp']
    else:
        kwrgs_pp = {}


    for lag in lags_i:

        y_pred_l = []

        for s in splits:
            c += 1
            progress = int(100 * (c) / (len(splits) * len(lags_i)))
            print(f"\rProgress {progress}%", end="")

            keys = keys_d[s].copy()
            model_name, kwrgs = stat_model

            df_split = df_data.loc[s]

            df_norm, keys = prepare_data(df_split, lag_i=int(lag),
                                                               keys=keys,
                                                               **kwrgs_pp)
            # data used to train and predict
            df_norm = df_norm[df_norm['fit_model_mask']]


            # forecasting models
            if model_name == 'logit':
                prediction, model = stat_models.logit(RV, df_norm, keys=keys)
            if model_name == 'GBR':
                kwrgs_GBR = kwrgs
                prediction, model = stat_models.GBR(RV, df_norm, keys,
                                                    kwrgs_GBR=kwrgs_GBR,
                                                    verbosity=verbosity)
            if model_name == 'logit-CV':
                kwrgs_logit = kwrgs
                prediction, model = stat_models.logit_skl(RV, df_norm, keys,
                                                          kwrgs_logit=kwrgs_logit)
            if model_name == 'GBR-logitCV':
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
            TrainRV = (df_norm['TrainIsTrue'])[df_norm['RV_mask']]
            TestRV  = (df_norm['TrainIsTrue']==False)[df_norm['RV_mask']]
            y_pred_l.append(prediction[TestRV.values])

            if lag == lags_i[0]:
                test_yr = np.unique(TestRV[TestRV].index.year)
                test_yrs.append(test_yr)
                # determining climatological prevailance in training data
                y_c_mask = np.logical_and(TrainRV, RV.RV_bin.squeeze().values==1)
                y_clim_val = RV.RV_bin[y_c_mask].size / RV.RV_bin[TrainRV].size
                # filling test years with clim of training data
                y_clim = RV.RV_bin[TestRV==True].copy()
                y_clim[:] = y_clim_val
                y_pred_c.append(y_clim)

        y_pred_l = pd.concat(y_pred_l)
        y_pred_l = y_pred_l.sort_index()

        if lag == lags_i[0]:
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

#    y_pred_all.iloc[1][y_pred_all.iloc[1]==1].size / RV.RV_bin[RV.RV_bin==1].size
    return df_valid, RV, y_pred_all

class RV_class:
    def __init__(self, RVfullts, RV_ts, kwrgs_events=None, only_RV_events=True,
                 fit_model_dates=None):
#        self.RV_ts = pd.DataFrame(df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] )
#        self.RVfullts = pd.DataFrame(df_data[df_data.columns[0]][0])
        self.RV_ts = RV_ts
        self.RVfullts = RVfullts
        self.dates_all = RVfullts.index
        self.dates_RV = RV_ts.index
        if fit_model_dates is None:
            bool_mask = [True if d in self.dates_RV else False for d in self.dates_all]
            self.fit_model_mask = pd.DataFrame(bool_mask, columns=['fit_model_mask'],
                                               index=self.dates_all)
            self.RV_ts_fit = self.RV_ts
            self.fit_dates = self.dates_RV
        else:
            startperiod, endperiod = fit_model_dates
            startyr = self.dates_all[0].year
            endyr   = self.dates_all[-1].year
            if self.dates_all.resolution == 'day':
                tfreq = (self.dates_all[1] - self.dates_all[0]).days
            ex = {'startperiod':startperiod, 'endperiod':endperiod,
                  'tfreq':tfreq}
            fit_dates = functions_pp.make_RVdatestr(self.dates_all,
                                                          ex, startyr, endyr)
            self.fit_dates = fit_dates
            bool_mask = [True if d in fit_dates else False for d in self.dates_all]
            self.fit_model_mask = pd.DataFrame(bool_mask, columns=['fit_model_mask'],
                                               index=self.dates_all)
            self.RV_ts_fit = self.RVfullts[self.fit_model_mask.values]
#        self.TrainIsTrue = df_data['TrainIsTrue']
#        self.RV_mask = df_data['RV_mask']
        self.n_oneRVyr = self.dates_RV[self.dates_RV.year == self.dates_RV.year[0]].size
        if kwrgs_events is not None:
            self.threshold = Ev_threshold(self.RV_ts,
                                              kwrgs_events['event_percentile'])
            self.threshold_ts_fit = Ev_threshold(self.RV_ts_fit,
                                              kwrgs_events['event_percentile'])
            if only_RV_events == True:

                self.RV_bin_fit = Ev_timeseries(self.RV_ts_fit,
                               threshold=self.threshold_ts_fit ,
                               min_dur=kwrgs_events['min_dur'],
                               max_break=kwrgs_events['max_break'],
                               grouped=kwrgs_events['grouped'])[0]
                self.RV_bin = self.RV_bin_fit.loc[self.dates_RV]
            elif only_RV_events == False:
                self.RV_b_full = Ev_timeseries(self.RVfullts,
                               threshold=self.threshold ,
                               min_dur=kwrgs_events['min_dur'],
                               max_break=kwrgs_events['max_break'],
                               grouped=kwrgs_events['grouped'])[0]
                self.RV_bin   = self.RV_b_full.loc[self.dates_RV]

            self.freq      = get_freq_years(self.RV_bin)



def load_hdf5(path_data):
    hdf = h5py.File(path_data,'r+')
    dict_of_dfs = {}
    for k in hdf.keys():
        dict_of_dfs[k] = pd.read_hdf(path_data, k)
    hdf.close()
    return dict_of_dfs

def prepare_data(df_split, lag_i=int, TrainIsTrue=None, RV_mask=None,
                 fit_model_dates=None, norm_datesRV=True, remove_RV=True, keys=None,
                 add_autocorr=True, EOF=False, expl_var=None):

    #%%
    '''
    TrainisTrue     : Specifies train and test dates.
    RV_mask         : Specifies what data will be predicted.
    fit_model_dates : It can be desirable to train on
                      more dates than what you want to predict.
    RV              : Response Variable (raw data).
    df_norm         : Normalized precursor data.
    remove_RV       : First column is the RV, and is removed.
    lag_i           : Data will be shifted with 'lag' periods, the index (dates).
                      will artificially be kept the same for each lag.

    '''

    dates_orig = df_split.index


    TrainIsTrue = df_split['TrainIsTrue']
    RV_mask = df_split['RV_mask']

    if fit_model_dates is None:
        fit_model_mask = RV_mask
    else:
        fit_model_mask = df_split['fit_model_mask']

    if remove_RV is True and add_autocorr==False:
        # completely remove RV timeseries
        RV_name = df_split.columns[0]
        df_prec = df_split.drop([RV_name], axis=1)
    else:
        # first column should contain RV timeseries
        RV_name = df_split.columns[0]
        df_RV = df_split[RV_name]
        df_prec = df_split.drop([RV_name], axis=1)
        keys = [k for k in keys if k != RV_name]

    # =============================================================================
    # Shifting data w.r.t. index dates
    # =============================================================================
#    df_prec.ix[:,0][:10]
    df_prec = df_prec.shift(periods=int(lag_i))
#    df_prec.ix[:,0][:10]

    if keys is None:
        x_keys = np.unique((df_prec.dtypes == np.float64).index)
    else:
        x_keys  = keys

    if add_autocorr:
        # ensure that autocorr never contains the RV timeseries at lag = 0
        df_prec.insert(0, 'RV_ac', df_RV.shift(periods=max(1,int(lag_i))))
        if 'RV_ac' not in keys:
            x_keys = np.insert(keys, 0, 'RV_ac')




    # drop nans
    mask_nonans = ~df_prec.iloc[:,0].isna().values
    # last dates are no longer present in shifted data. But we want restore
    # original dates. We are going to delete the first chronologicall dates
    # to make them equal length.
    dates_new  = dates_orig[mask_nonans]
    df_prec = pd.DataFrame(df_prec[mask_nonans].values, columns=df_prec.columns,
                           index=dates_new, dtype='float64')
    TrainIsTrue =   pd.Series(TrainIsTrue[mask_nonans], index=dates_new)
    RV_mask =       pd.Series(RV_mask[mask_nonans], index=dates_new)
    fit_model_mask =pd.Series(fit_model_mask[mask_nonans], index=dates_new)


    # =============================================================================
    # Select features / variables
    # =============================================================================
    df_prec = df_prec[x_keys]
    # =============================================================================
    # Normalize data using datesRV or all training data in dataframe
    # =============================================================================
    if norm_datesRV == False:
        # Normalize using all dates
        df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainIsTrue].mean(0)) \
                / df_prec[x_keys][TrainIsTrue].std(0)
    elif norm_datesRV == True:
        # Normalize only using the RV dates
        TrainRV = np.logical_and(TrainIsTrue,fit_model_mask).values
        df_prec[x_keys]  = (df_prec[x_keys] - df_prec[x_keys][TrainRV].mean(0)) \
                / df_prec[x_keys][TrainRV].std(0)


    if EOF:
        if expl_var is None:
            expl_var = 0.75
        else:
            expl_var = expl_var
        df_prec = transform_EOF(df_prec, TrainIsTrue, fit_model_mask, expl_var=0.8)
        df_prec.columns = df_prec.columns.astype(str)
        upd_keys = np.array(df_prec.columns.values.ravel(), dtype=str)
    else:
        upd_keys = keys

    # =============================================================================
    # Replace masks
    # =============================================================================
    df_prec['TrainIsTrue']  = TrainIsTrue
    df_prec['RV_mask']      = RV_mask
    df_prec['fit_model_mask']  = fit_model_mask
    df_prec.index = dates_new
    #%%
    return df_prec, upd_keys

def transform_EOF(df_prec, TrainIsTrue, RV_mask, expl_var=0.8):
    '''
    EOF is based upon all Training data.
    '''
    #%%
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

