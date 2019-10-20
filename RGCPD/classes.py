#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:13:58 2019

@author: semvijverberg
"""

import numpy as np
import pandas as pd
import func_fc
import functions_pp


class RV_class:
    def __init__(self, RVfullts, RV_ts, kwrgs_events=None, only_RV_events=True,
                 fit_model_dates=None):
        '''
        only_RV_events : bool. Decides whether to calculate the RV_bin on the 
        whole RVfullts timeseries, or only on RV_ts
        '''
        #%%
#        self.RV_ts = pd.DataFrame(df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] )
#        self.RVfullts = pd.DataFrame(df_data[df_data.columns[0]][0])
        self.RV_ts = RV_ts
        self.RVfullts = RVfullts
        self.dates_all = RVfullts.index
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
                if dates_all.resolution == 'day':
                    tfreq = (dates_all[1] - dates_all[0]).days
                ex = {'startperiod':startperiod, 'endperiod':endperiod,
                      'tfreq':tfreq}
                fit_dates = functions_pp.make_RVdatestr(dates_all,
                                                              ex, startyr, endyr)
                bool_mask = [True if d in fit_dates else False for d in dates_all]
                fit_model_mask = pd.DataFrame(bool_mask, columns=['fit_model_mask'],
                                                   index=dates_all)
                
                RV_ts_fit = RVfullts[fit_model_mask.values]
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
            if only_RV_events == True:
                self.RV_bin_fit = func_fc.Ev_timeseries(self.RV_ts_fit,
                               threshold=self.threshold_ts_fit ,
                               min_dur=kwrgs_events['min_dur'],
                               max_break=kwrgs_events['max_break'],
                               grouped=kwrgs_events['grouped'])[0]
                self.RV_bin = self.RV_bin_fit.loc[self.dates_RV]
            elif only_RV_events == False:
                self.RV_b_full = func_fc.Ev_timeseries(self.RVfullts,
                               threshold=self.threshold ,
                               min_dur=kwrgs_events['min_dur'],
                               max_break=kwrgs_events['max_break'],
                               grouped=kwrgs_events['grouped'])[0]
                self.RV_bin   = self.RV_b_full.loc[self.dates_RV]

            self.freq      = func_fc.get_freq_years(self.RV_bin)
        
        
        # make RV_bin for extreme occurring in time window
        if kwrgs_events is not None and type(kwrgs_events) is tuple and self.tfreq !=1:
            
            
            
            filename_ts = kwrgs_events[0]
            kwrgs_events_daily = kwrgs_events[1]
            # loading in daily timeseries
            RVfullts_xr = np.load(filename_ts, encoding='latin1',
                                     allow_pickle=True).item()['RVfullts95']
        
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
        
        
            dates_RVe = aggr_to_daily_dates(self.dates_RV)
            dates_alle  = aggr_to_daily_dates(self.dates_all)
            
            df_RV_ts_e = pd.DataFrame(RVfullts_xr.sel(time=dates_RVe).values, 
                                      index=dates_RVe, columns=['RV_ts'])
            
            df_RVfullts_e = pd.DataFrame(RVfullts_xr.sel(time=dates_alle).values, 
                                      index=dates_alle, 
                                      columns=['RVfullts'])
            

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
                               threshold=self.threshold_ts_fit ,
                               min_dur=kwrgs_events_daily['min_dur'],
                               max_break=kwrgs_events_daily['max_break'],
                               grouped=kwrgs_events_daily['grouped'])[0]
                self.RV_bin = self.RV_bin_fit.loc[dates_RVe]
            elif only_RV_events == False:
                self.RV_b_full = func_fc.Ev_timeseries(self.RVfullts,
                               threshold=self.threshold ,
                               min_dur=kwrgs_events_daily['min_dur'],
                               max_break=kwrgs_events_daily['max_break'],
                               grouped=kwrgs_events_daily['grouped'])[0]
                self.RV_bin   = self.RV_b_full.loc[self.dates_RV]
            
            # convert daily binary to aggregated binary
            tfreq = (self.dates_all[1]  - self.dates_all[0]).days
            if tfreq != 1:
                self.RV_bin, dates_gr = functions_pp.time_mean_bins(self.RV_bin.astype('float'), 
                                                                tfreq,
                                                                None,
                                                                None)
                self.RV_bin_fit, dates_gr = functions_pp.time_mean_bins(self.RV_bin_fit.astype('float'), 
                                                                        tfreq,         
                                                                        None,
                                                                        None)
                                                                        
#                start_end_date = (ex['sstartdate'], ex['senddate'])
#                start_end_year = (ex['startyear'], ex['endyear'])
#                ds, dates = time_mean_bins(ds, tfreq,
#                                                start_end_date,
#                                                start_end_year,
#                                                seldays='part')
#            ex = dict(,
#                      startyear  = dates_RVe.year[0],
#                      endyear    = dates_RVe.year[-1])
            
            
            

            # all bins, with mean > 0 contained an 'extreme' event
            self.RV_bin_fit[self.RV_bin_fit>0] = 1
            self.RV_bin[self.RV_bin>0] = 1
    #%%

def Variable(self, ex):
    self.startyear = ex['startyear']
    self.endyear = ex['endyear']
    self.startmonth = 1
    self.endmonth = 12
    self.grid = ex['grid_res']
    self.dataset = ex['dataset']
    self.base_path = ex['base_path']
    self.path_raw = ex['path_raw']
    self.path_pp = ex['path_pp']
    return self


class Var_import_RV_netcdf:
    def __init__(self, ex):
        vclass = Variable(self, ex)

        vclass.name = ex['RVnc_name'][0]
        vclass.filename = ex['RVnc_name'][1]
        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name,
               vclass.startyear, vclass.endyear, vclass.grid)))

class Var_import_precursor_netcdf:
    def __init__(self, ex, idx):
        vclass = Variable(self, ex)

        vclass.name = ex['precursor_ncdf'][idx][0]
        vclass.filename = ex['precursor_ncdf'][idx][1]
        ex['vars'][0].append(vclass.name)      
            