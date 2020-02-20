#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:46:05 2020

@author: semvijverberg
"""

# import inspect, os
# curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory

import numpy as np
import pandas as pd
import xarray as xr

import functions_pp
import core_pp





class RV_class:
    def __init__(self, fullts, RV_ts, kwrgs_events=None, only_RV_events=True,
                 fit_model_dates=None):
        '''
        only_RV_events : bool. Decides whether to calculate the RV_bin on the
        whole fullts timeseries, or only on RV_ts
        '''
        #%%
#        self.RV_ts = pd.DataFrame(df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] )
#        self.fullts = pd.DataFrame(df_data[df_data.columns[0]][0])
        self.name = fullts.columns[0]
        self.RV_ts = RV_ts
        self.fullts = fullts
        self.dates_all = fullts.index
        self.dates_RV = RV_ts.index
        self.n_oneRVyr = self.dates_RV[self.dates_RV.year == self.dates_RV.year[0]].size
        self.tfreq = (self.dates_all[1] - self.dates_all[0]).days

        def handle_fit_model_dates(dates_RV, dates_all, RV_ts, fit_model_dates):
            if fit_model_dates is None:
                # RV_ts and RV_ts_fit are equal if fit_model_dates = None
                bool_mask = [True if d in dates_RV else False for d in dates_all]
                fit_model_mask = pd.DataFrame(bool_mask, columns=['fit_model_mask'],
                                                   index=dates_all)
                RV_ts_fit = RV_ts
                fit_dates = dates_RV
            else:
                startperiod, endperiod = fit_model_dates
                startyr = dates_all[0].year
                endyr   = dates_all[-1].year
#                if dates_all.resolution == 'day':
#                    tfreq = (dates_all[1] - dates_all[0]).days
                start_end_date = (startperiod, endperiod)
                start_end_year = (startyr, endyr)
                fit_dates = core_pp.get_subdates(dates_all,
                                                 start_end_date=start_end_date, 
                                                 start_end_year=start_end_year)
                bool_mask = [True if d in fit_dates else False for d in dates_all]
                fit_model_mask = pd.DataFrame(bool_mask, columns=['fit_model_mask'],
                                                   index=dates_all)

                RV_ts_fit = fullts[fit_model_mask.values]
                fit_dates = fit_dates
            return fit_model_mask, fit_dates, RV_ts_fit

        out = handle_fit_model_dates(self.dates_RV, self.dates_all, self.RV_ts, fit_model_dates)
        self.fit_model_mask, self.fit_dates, self.RV_ts_fit = out



        # make RV_bin for events based on aggregated daymeans
        if kwrgs_events is not None and (type(kwrgs_events) is not tuple or self.tfreq==1):

            if type(kwrgs_events) is tuple:
                kwrgs_events = kwrgs_events[1]
            # RV_ts and RV_ts_fit are equal if fit_model_dates = None
            self.threshold = Ev_threshold(self.RV_ts,
                                              kwrgs_events['event_percentile'])
            self.threshold_ts_fit = Ev_threshold(self.RV_ts_fit,
                                              kwrgs_events['event_percentile'])

            # unpack other optional arguments for defining event timeseries 
            kwrgs = {key:item for key, item in kwrgs_events.items() if key != 'event_percentile'}
            if only_RV_events == True:
                self.RV_bin_fit = Ev_timeseries(self.RV_ts_fit,
                               threshold=self.threshold_ts_fit ,
                               **kwrgs)[0]
                self.RV_bin = self.RV_bin_fit.loc[self.dates_RV]
            elif only_RV_events == False:
                self.RV_b_full = Ev_timeseries(self.fullts,
                               threshold=self.threshold ,
                               **kwrgs)[0]
                self.RV_bin   = self.RV_b_full.loc[self.dates_RV]

            self.freq_per_year      = RV_class.get_freq_years(self)


        # make RV_bin for extreme occurring in time window
        if kwrgs_events is not None and type(kwrgs_events) is tuple and self.tfreq !=1:



            filename_ts = kwrgs_events[0]
            kwrgs_events_daily = kwrgs_events[1]
            # unpack other optional arguments for defining event timeseries 
            kwrgs = {key:item for key, item in kwrgs_events_daily.items() if key != 'event_percentile'}
            # loading in daily timeseries
            fullts_xr = np.load(filename_ts, encoding='latin1',
                                     allow_pickle=True).item()['RVfullts95']


            dates_RVe = self.aggr_to_daily_dates(self.dates_RV)
            dates_alle  = self.aggr_to_daily_dates(self.dates_all)

            df_RV_ts_e = pd.DataFrame(fullts_xr.sel(time=dates_RVe).values,
                                      index=dates_RVe, columns=['RV_ts'])

            df_fullts_e = pd.DataFrame(fullts_xr.sel(time=dates_alle).values,
                                      index=dates_alle,
                                      columns=['fullts'])


            out = handle_fit_model_dates(dates_RVe, dates_alle, df_RV_ts_e, fit_model_dates)
            self.fit_model_mask, self.fit_dates, self.RV_ts_fit_e = out


            # RV_ts and RV_ts_fit are equal if fit_model_dates = None
            self.threshold = Ev_threshold(df_RV_ts_e,
                                              kwrgs_events_daily['event_percentile'])
            self.threshold_ts_fit = Ev_threshold(self.RV_ts_fit_e,
                                              kwrgs_events_daily['event_percentile'])

            if only_RV_events == True:
                # RV_bin_fit is defined such taht we can fit on RV_bin_fit
                # but validate on RV_bin
                self.RV_bin_fit = Ev_timeseries(df_RV_ts_e,
                               threshold=self.threshold_ts_fit, **kwrgs)[0]
                self.RV_bin = self.RV_bin_fit.loc[dates_RVe]
            elif only_RV_events == False:
                self.RV_b_full = Ev_timeseries(self.fullts,
                               threshold=self.threshold, **kwrgs)[0]
                self.RV_bin   = self.RV_b_full.loc[self.dates_RV]

            # convert daily binary to window probability binary
            if self.tfreq != 1:
                self.RV_bin, dates_gr = functions_pp.time_mean_bins(self.RV_bin.astype('float'),
                                                                self.tfreq,
                                                                None,
                                                                None)
                self.RV_bin_fit, dates_gr = functions_pp.time_mean_bins(self.RV_bin_fit.astype('float'),
                                                                        self.tfreq,
                                                                        None,
                                                                        None)

            # all bins, with mean > 0 contained an 'extreme' event
            self.RV_bin_fit[self.RV_bin_fit>0] = 1
            self.RV_bin[self.RV_bin>0] = 1
    #%%
    @staticmethod
    # Retrieve information on input timeseries
    def aggr_to_daily_dates(dates_precur_data):
        dates = functions_pp.get_oneyr(dates_precur_data)
        tfreq = (dates[1] - dates[0]).days
        start_date = dates[0] - pd.Timedelta(f'{int(tfreq/2)}d')
        end_date   = dates[-1] + pd.Timedelta(f'{int(-1+tfreq/2+0.5)}d')
        yr_daily  = pd.date_range(start=start_date, end=end_date,
                                        freq=pd.Timedelta('1d'))
        years = np.unique(dates_precur_data.year)
        ext_dates = functions_pp.make_dates(yr_daily, years)
    
        return ext_dates

    def get_freq_years(self, RV_bin=None):
        if RV_bin is None and hasattr(self, 'RV_bin'):
            RV_bin = self.RV_bin
        all_years = np.unique(RV_bin.index.year)
        binary = RV_bin.values
        freq = []
        for y in all_years:
            n_ev = int(binary[RV_bin.index.year==y].sum())
            freq.append(n_ev)
        return pd.Series(freq, index=all_years)

    def get_obs_clim(self):
        splits = self.TrainIsTrue.index.levels[0]
        RV_mask_s = self.RV_mask
        TrainIsTrue = self.TrainIsTrue
        y_prob_clim = self.RV_bin.copy()
        y_prob_clim = y_prob_clim.rename(columns={'RV_binary':'prob_clim'})
        for s in splits:
            RV_train_mask = TrainIsTrue[s][RV_mask_s[s]]
            y_b_train = self.RV_bin[RV_train_mask]
            y_b_test  = self.RV_bin[RV_train_mask==False]
    
            clim_prevail = y_b_train.sum() / y_b_train.size
            clim_arr = np.repeat(clim_prevail, y_b_test.size).values
            pdseries = pd.Series(clim_arr, index=y_b_test.index)
            y_prob_clim.loc[y_b_test.index, 'prob_clim'] = pdseries
        self.prob_clim = y_prob_clim
        return 


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


#    tfreq_RVts = pd.Timedelta((xarray.time[1]-xarray.time[0]).values)
    min_dur = min_dur ;
#    min_dur = pd.Timedelta(min_dur, 'd') / tfreq_RVts
#    max_break = pd.Timedelta(max_break, 'd') / tfreq_RVts

    if high_ano_events:
        Ev_ts = xarray.where( xarray.values > threshold)
    else:
        Ev_ts = xarray.where( xarray.values < threshold)

    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    events_idx = [list(xarray.time.values).index(E) for E in Ev_dates.values]
    n_timesteps = Ev_ts.size

    peak_o_thresh = Ev_binary(events_idx, n_timesteps, min_dur, max_break, grouped)
    event_binary_np  = np.array(peak_o_thresh != 0, dtype=int)

    # get duration of events
    if np.unique(peak_o_thresh).size == 2:
        dur = np.zeros( (peak_o_thresh.size) )
        for i in np.arange(1, max(peak_o_thresh)+1):
            size = peak_o_thresh[peak_o_thresh==i].size
            dur[peak_o_thresh==i] = size
    else:
        dur = 'dur_events_1'

    if np.sum(peak_o_thresh) < 1:
        Events = Ev_ts.where(peak_o_thresh > 0 ).dropna(how='all', dim='time').time
    else:
        peak_o_thresh[peak_o_thresh == 0] = np.nan
        Ev_labels = xr.DataArray(peak_o_thresh, coords=[Ev_ts.coords['time']])
        Ev_dates = Ev_labels.dropna(how='all', dim='time').time
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

    if min_dur != 1 or max_break > 1:
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
                # if i_steps >= max_break, same event
                peak_o_thresh[curr_ev] = ev_num
            elif abs(next_ev - curr_ev) > max_break:
                # elif i_steps > max_break, assign new event number
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
    else:
        peak_o_thresh[events_idx] = 1

    return peak_o_thresh


if __name__ == "__main__":
    pass