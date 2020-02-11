#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:46:05 2020

@author: semvijverberg
"""

import numpy as np
import pandas as pd
import func_fc
import functions_pp
import inspect, os
import core_pp
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
path_test = os.path.join(curr_dir, '..', 'data')



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
            self.threshold = func_fc.Ev_threshold(self.RV_ts,
                                              kwrgs_events['event_percentile'])
            self.threshold_ts_fit = func_fc.Ev_threshold(self.RV_ts_fit,
                                              kwrgs_events['event_percentile'])

            # unpack other optional arguments for defining event timeseries 
            kwrgs = {key:item for key, item in kwrgs_events.items() if key != 'event_percentile'}
            if only_RV_events == True:
                self.RV_bin_fit = func_fc.Ev_timeseries(self.RV_ts_fit,
                               threshold=self.threshold_ts_fit ,
                               **kwrgs)[0]
                self.RV_bin = self.RV_bin_fit.loc[self.dates_RV]
            elif only_RV_events == False:
                self.RV_b_full = func_fc.Ev_timeseries(self.fullts,
                               threshold=self.threshold ,
                               **kwrgs)[0]
                self.RV_bin   = self.RV_b_full.loc[self.dates_RV]

            self.freq      = func_fc.get_freq_years(self.RV_bin)


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
            self.threshold = func_fc.Ev_threshold(df_RV_ts_e,
                                              kwrgs_events_daily['event_percentile'])
            self.threshold_ts_fit = func_fc.Ev_threshold(self.RV_ts_fit_e,
                                              kwrgs_events_daily['event_percentile'])

            if only_RV_events == True:
                # RV_bin_fit is defined such taht we can fit on RV_bin_fit
                # but validate on RV_bin
                self.RV_bin_fit = func_fc.Ev_timeseries(df_RV_ts_e,
                               threshold=self.threshold_ts_fit, **kwrgs)[0]
                self.RV_bin = self.RV_bin_fit.loc[dates_RVe]
            elif only_RV_events == False:
                self.RV_b_full = func_fc.Ev_timeseries(self.fullts,
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

if __name__ == "__main__":
    pass