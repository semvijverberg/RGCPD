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
import find_precursors
from pathlib import Path
import inspect, os
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
path_test = os.path.join(curr_dir, '..', 'data')

#list_of_name_path = [('t2m_eUS', os.path.join(path_test, 't2m_eUS.npy')),
#                     ('sst_test', os.path.join(path_test, 'data/sst_1979-2018_2.5deg_Pacific.nc'))]
#TV_period = ('06-15', '08-20')

class RGCPD:
    
    
    def __init__(self, list_of_name_path=None, start_end_TVdate=None, tfreq=10, 
                 start_end_date=None, start_end_year=None,
                 path_outmain=None, lags_i=np.array([1]),
                 kwrgs_pp=None, kwrgs_corr=None, kwrgs_cluster=None, verbosity=1):
        '''
        list_of_name_path : list of name, path tuples. 
        Convention: first entry should be (name, path) of target variable (TV).
        list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]
        
        TV period : tuple of start- and enddate in format ('mm-dd', 'mm-dd')
        
        '''
        if list_of_name_path is None:
            print('initializing with test data')
            list_of_name_path = [('t2m_eUS', 
                                  os.path.join(path_test, 't2m_eUS.npy')),
                                 ('sst_test', 
                                  os.path.join(path_test, 'sst_1979-2018_2.5deg_Pacific.nc'))]
        if start_end_TVdate is None:
            start_end_TVdate = ('06-15', '08-20')
        
        if kwrgs_pp is None:
            kwrgs_pp = dict(loadleap=False, seldates=None, selbox=None,
                            format_lon='east_west',
                            detrend=True, anomaly=True)

        if path_outmain is None:
            path_outmain = str(Path.home()) + '/Downloads/output_RGCPD'
            
        self.list_of_name_path = list_of_name_path
        self.start_end_TVdate  = start_end_TVdate
        self.start_end_date = start_end_date
        self.start_end_year = start_end_year
        self.verbosity  = verbosity
        self.tfreq      = tfreq
        self.lags_i     = lags_i
        self.lags       = np.array([l*self.tfreq for l in self.lags_i], dtype=int)
        self.kwrgs_pp   = kwrgs_pp
        self.path_outmain = path_outmain

        if kwrgs_corr is None:
                self.kwrgs_corr = dict(alpha=1E-2, # set significnace level for correlation maps
                                       lags=self.lags,
                                       FDR_control=True) # Accounting for false discovery rate
                
        # =============================================================================
        # settings precursor region selection
        # =============================================================================   
        # bigger distance_eps means more and smaller clusters
        # bigger min_area_in_degrees2 will interpet more small individual clusters as noise
        if kwrgs_cluster is None:
            self.kwrgs_cluster = dict(distance_eps=300,       # proportional to km apart from a core sample, standard = 400 km
                                 min_area_in_degrees2=2, # minimal size to become precursor region (core sample)
                                 group_split='together') # choose 'together' or 'seperate'
        else:
            self.kwrgs_cluster = kwrgs_cluster 
            
        return
    
    def pp_precursors(self):
        self.list_precur_pp = functions_pp.perform_post_processing(self.list_of_name_path, 
                                             kwrgs_pp=self.kwrgs_pp, 
                                             verbosity=self.verbosity)
    
    
    def pp_TV(self):
        self.fulltso = functions_pp.load_TV(self.list_of_name_path)
        self.fullts, self.TV_ts, inf = functions_pp.process_TV(self.fulltso, 
                                                              self.tfreq,
                                                              self.start_end_TVdate)
        self.input_freq = inf
        self.dates_or  = pd.to_datetime(self.fulltso.time.values)
        self.dates_all = pd.to_datetime(self.fullts.time.values)
        self.dates_TV = pd.to_datetime(self.TV_ts.time.values)    
        # Store added information in RV class to the exp dictionary
        if self.start_end_date is None:
            self.start_end_date = ('{}-{}'.format(self.dates_or.month[0],
                                                 self.dates_or[0].day),
                                '{}-{}'.format(self.dates_or.month[-1],
                                                 self.dates_or[-1].day))
        if self.start_end_year is None:
            self.start_end_year = (self.dates_or.year[0],
                                   self.dates_or.year[-1])
        months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                        7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
        RV_name_range = '{}{}-{}{}_'.format(self.dates_TV[0].day, 
                                         months[self.dates_TV.month[0]],
                                         self.dates_TV[-1].day, 
                                         months[self.dates_TV.month[-1]] )
        info_lags = 'lag{}-{}'.format(min(self.lags), max(self.lags))
        # Creating a folder for the specific spatial mask, RV period and traintest set
        self.path_outsub0 = os.path.join(self.path_outmain, RV_name_range + \
                                              info_lags ) 
    
        # =============================================================================
        # Test if you're not have a lag that will precede the start date of the year
        # =============================================================================
        # first date of year to be analyzed:
        if self.input_freq == 'daily':
            f = 'D'
        elif self.input_freq != 'monthly':
            f = 'M'
        firstdoy = self.dates_TV.min() - np.timedelta64(int(max(self.lags)), f)
        if firstdoy < self.dates_all[0] and (self.dates_all[0].month,self.dates_all[0].day) != (1,1):
            tdelta = self.dates_all.min() - self.dates_all.min()
            lag_max = int(tdelta / np.timedelta64(self.tfreq, 'D'))
            self.lags = self.lags[self.lags < lag_max]
            self.lags_i = self.lags_i[self.lags_i < lag_max]
            print(('Changing maximum lag to {}, so that you not skip part of the '
                  'year.'.format(max(self.lags)) ) )
    
    def traintest(self, kwrgs_TV=None):
        '''
        krwgs_TV has format:
            
        kwrgs_RV = dict(method=method,
                seed=seed,
                kwrgs_events=kwrgs_events,
                precursor_ts=precursor_ts)
            

        'method'        : str referring to method to split train test, see 
                          options for method below.        
        seed            : the seed to draw random samples for train test split
        kwrgs_events    : dict needed to create binary event timeseries, which 
                          is used to create stratified folds. 
                          See func_fc.Ev_timeseries? for more info.
        precursor_ts    : Load in precursor 1-d timeseries in format:
                          [(name1, path_to_h5_file1), [(name2, path_to_h5_file2)]]
                          precursor_ts should follow the RGCPD traintest format

        Options for method:
        (1) random{int}   :   with the int(ex['method'][6:8]) determining the amount of folds
        (2) ran_strat{int}:   random stratified folds, stratified based upon events, 
                              requires kwrgs_events.    
        (3) leave{int}    :   chronologically split train and test years.
        (4) split{int}    :   split dataset into single train and test set
        (5) no_train_test_split
        
        # Extra: RV events settings are needed to make balanced traintest splits
        '''
        if kwrgs_TV is None:
            kwrgs_TV = dict(method='no_train_test_split',
                    seed=1,
                    kwrgs_events=None,
                    precursor_ts=None)
            
        
        TV, df_splits = find_precursors.RV_and_traintest(self.fullts, 
                                                         self.TV_ts, 
                                                         verbosity=self.verbosity, 
                                                         **kwrgs_TV)
        self.TV = TV
        self.df_splits = df_splits
    
    def calc_corr_maps(self):
        keys = ['selbox', 'loadleap', 'seldates', 'format_lon']
        kwrgs_load = {k: self.kwrgs_pp[k] for k in keys}
        kwrgs_load['start_end_date']= self.start_end_date
        kwrgs_load['start_end_year']= self.start_end_year
        kwrgs_load['selbox']        = None
        kwrgs_load['loadleap']      = False
        kwrgs_load['format_lon']    = 'only_east'
        kwrgs_load['tfreq']         = self.tfreq
        self.kwrgs_load = kwrgs_load
        self.outdic_precur = find_precursors.calculate_corr_maps(self.TV, self.df_splits, 
                                            self.kwrgs_load, 
                                            self.list_precur_pp, 
                                            **self.kwrgs_corr)
                    
    def cluster_regions(self):
        for name, actor in self.outdic_precur.items():
            actor = find_precursors.cluster_DBSCAN_regions(actor, 
                                                           **self.kwrgs_cluster)
            self.outdic_precur[name] = actor
                   
#        kwrgs_load['tfreq'] = self.tfreq
#        kwrgs_load['start_end_date']
        

            


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

#def Variable(self, startyear, endyear, startmonth, endmonth, grid, dataset, 
#             path_outmain):
#    self.startyear = ex['startyear']
#    self.endyear = ex['endyear']
#    self.startmonth = 1
#    self.endmonth = 12
#    self.grid = ex['grid_res']
#    self.dataset = ex['dataset']
##    self.base_path = ex['base_path']
##    self.path_raw = ex['path_raw']
##    self.path_pp = ex['path_pp']
#    return self


class Var_import_RV_netcdf:
    def __init__(self, ex):
        vclass = Variable(self, ex)

        vclass.name = ex['RVnc_name'][0]
        vclass.filename = ex['RVnc_name'][1]
        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name,
               vclass.startyear, vclass.endyear, vclass.grid)))

class Var_import_precursor_netcdf:
    def __init__(self, tuple_name_path):
#        vclass = Variable(self, ex)

        vclass.name = tuple_name_path[0]
        vclass.filename = tuple_name_path[1] 
            