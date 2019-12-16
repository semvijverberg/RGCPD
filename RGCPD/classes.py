#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:13:58 2019

@author: semvijverberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import func_fc
import functions_pp
import plot_maps
import find_precursors
from pathlib import Path
import inspect, os
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
path_test = os.path.join(curr_dir, '..', 'data')


class RGCPD:

    def __init__(self, list_of_name_path=None, start_end_TVdate=None, tfreq=10, 
                 start_end_date=None, start_end_year=None,
                 path_outmain=None, lags_i=np.array([1]),
                 kwrgs_corr=None, verbosity=1):
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
        self.path_outmain = path_outmain

        if kwrgs_corr is None:
                self.kwrgs_corr = dict(alpha=1E-2, # set significnace level for correlation maps
                                       lags=self.lags,
                                       FDR_control=True) # Accounting for false discovery rate
                
            
        return
    
    def pp_precursors(self, kwrgs_pp=None):
        
        if kwrgs_pp is None:
            kwrgs_pp = dict(loadleap=False, seldates=None, selbox=None,
                            format_lon='east_west',
                            detrend=True, anomaly=True)
            
        self.kwrgs_pp = kwrgs_pp
        self.list_precur_pp = functions_pp.perform_post_processing(self.list_of_name_path, 
                                             kwrgs_pp=self.kwrgs_pp, 
                                             verbosity=self.verbosity)
    
    
    def pp_TV(self):
        self.fulltso = functions_pp.load_TV(self.list_of_name_path)
        self.fullts, self.TV_ts, inf = functions_pp.process_TV(self.fulltso, 
                                                              self.tfreq,
                                                              self.start_end_TVdate,
                                                              self.start_end_date,
                                                              self.start_end_year)
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
        ''' Splits the training and test dates, either via cross-validation or 
        via a simple single split.
 
        
        agrs:          
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
        
        Returns panda dataframe with traintest mask and Target variable mask 
        concomitant to each split. 
        '''
        if kwrgs_TV is None:
            self.kwrgs_TV = dict(method='no_train_test_split',
                    seed=1,
                    kwrgs_events=None,
                    precursor_ts=None)
        else:
            self.kwrgs_TV = kwrgs_TV
            
        
        TV, self.df_splits = find_precursors.RV_and_traintest(self.fullts, 
                                                         self.TV_ts, 
                                                         verbosity=self.verbosity, 
                                                         **self.kwrgs_TV)
        TV.name = self.list_of_name_path[0][0]
        self.TV = TV
        self.path_outsub1 = os.path.join(self.path_outsub0, 
                                      '_'.join([self.kwrgs_TV['method'], 
                                                's'+ str(self.kwrgs_TV['seed'])]))
        if os.path.isdir(self.path_outsub1) == False : os.makedirs(self.path_outsub1)
    
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
                    
    def cluster_regions(self, distance_eps=700, min_area_in_degrees2=2, 
                        group_split='together'):
        '''
        Settings precursor region selection.

        Bigger distance_eps means more and smaller clusters
        Bigger min_area_in_degrees2 will interpet more small individual clusters as noise
        '''
        self.kwrgs_cluster = dict(distance_eps=distance_eps,  # proportional to km apart from a core sample, standard = 400 km
                                 min_area_in_degrees2=min_area_in_degrees2, # minimal size to become precursor region (core sample)
                                 group_split=group_split) # choose 'together' or 'seperate
        
        for name, actor in self.outdic_precur.items():
            actor = find_precursors.cluster_DBSCAN_regions(actor, 
                                                           **self.kwrgs_cluster)
            self.outdic_precur[name] = actor
    
    def quick_view_labels(self, map_proj=None):
        for name, actor in self.outdic_precur.items():
            prec_labels = actor.prec_labels.copy()
            
            # colors of cmap are dived over min to max in n_steps. 
            # We need to make sure that the maximum value in all dimensions will be 
            # used for each plot (otherwise it assign inconsistent colors)
            max_N_regs = min(20, int(prec_labels.max() + 0.5))
            label_weak = np.nan_to_num(prec_labels.values) >=  max_N_regs
            contour_mask = None
            prec_labels.values[label_weak] = max_N_regs
            steps = max_N_regs+1
            cmap = plt.cm.tab20
            prec_labels.values = prec_labels.values-0.5
            clevels = np.linspace(0, max_N_regs,steps)
            
            if prec_labels.split.size == 1:
                cbar_vert = -0.1
            else:
                cbar_vert = -0.025
            
            if map_proj is None:            
                cen_lon = int(prec_labels.longitude.mean().values)
                map_proj = ccrs.LambertCylindrical(central_longitude=cen_lon)
            
            kwrgs_corr = {'row_dim':'split', 'col_dim':'lag', 'hspace':-0.35, 
                          'size':3, 'cbar_vert':cbar_vert, 'clevels':clevels,
                          'subtitles' : None, 'lat_labels':True, 
                          'cticks_center':True,
                          'cmap':cmap}        
            
            plot_maps.plot_corr_maps(prec_labels, 
                             contour_mask, 
                             map_proj, **kwrgs_corr)

    def get_ts_prec(self, import_prec_ts=None):
        
        
        self.outdic_precur = find_precursors.get_prec_ts(self.outdic_precur)
        self.df_data = find_precursors.df_data_prec_regs(self.TV, 
                                                         self.outdic_precur, 
                                                         self.df_splits)
        if import_prec_ts is not None:
            self.df_data_ext = find_precursors.import_precur_ts(import_prec_ts, 
                                                             self.df_splits, 
                                                             self.tfreq, 
                                                             self.start_end_date,
                                                             self.start_end_year)
            self.df_data = self.df_data.merge(self.df_data_ext, left_index=True, right_index=True)
        self.df_data = self.df_data.merge(self.df_splits, left_index=True, right_index=True)


    def PCMCI_df_data(self, path_txtoutput=None, tau_min=0, tau_max=1, 
                    pc_alpha=None, alpha_level=0.05, max_conds_dim=4,
                    max_combinations=1, max_conds_py=None, max_conds_px=None,
                    verbosity=4):

        import wrapper_PCMCI
        
        kwrgs_pcmci = dict(tau_min=tau_min,
                           tau_max=tau_max,
                           pc_alpha=pc_alpha,
                           alpha_level=alpha_level,
                           max_conds_dim=max_conds_dim,
                           max_combinations=max_combinations,
                           max_conds_py=max_conds_py,
                           max_conds_px=max_conds_px,
                           verbosity=4)
        

        
        
        if path_txtoutput is None:
            self.params_str = '{}_at{}_tau_{}-{}_conds_dim{}_combin{}'.format(pc_alpha, 
                          self.kwrgs_corr['alpha'], tau_min, tau_max, max_conds_dim, max_combinations)
            self.path_outsub2 = os.path.join(self.path_outsub1, self.params_str)
        else:
            self.path_outsub2 = path_txtoutput
        
        if os.path.isdir(self.path_outsub2) == False : os.makedirs(self.path_outsub2)
            
        self.pcmci_dict = wrapper_PCMCI.loop_train_test(self.df_data, self.path_outsub2, 
                                                          **kwrgs_pcmci)
        self.df_sum = wrapper_PCMCI.get_df_sum(self.pcmci_dict, kwrgs_pcmci['alpha_level'])
        print(self.df_sum)
        # get xarray dataset for each variable
        self.dict_ds = plot_maps.causal_reg_to_xarray(self.TV.name, self.df_sum, 
                                                      self.outdic_precur)
        
    def plot_maps_sum(self, map_proj=None, figpath=None, paramsstr=None):
        
        if map_proj is None:
            central_lon_plots = 200
            map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)
        
        if figpath is None:
            figpath = self.path_outsub1
        if paramsstr is None:
            paramsstr = self.params_str
        
        plot_maps.plot_labels_vars_splits(self.dict_ds, self.df_sum, map_proj, 
                                          figpath, paramsstr, self.TV.name)
        
        
        plot_maps.plot_corr_vars_splits(self.dict_ds, self.df_sum, map_proj, 
                                          figpath, paramsstr, self.TV.name)
        
        
        


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
                if dates_all.resolution == 'day':
                    tfreq = (dates_all[1] - dates_all[0]).days
                ex = {'startperiod':startperiod, 'endperiod':endperiod,
                      'tfreq':tfreq}
                fit_dates = functions_pp.make_RVdatestr(dates_all,
                                                              ex, startyr, endyr)
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
            if only_RV_events == True:
                self.RV_bin_fit = func_fc.Ev_timeseries(self.RV_ts_fit,
                               threshold=self.threshold_ts_fit ,
                               min_dur=kwrgs_events['min_dur'],
                               max_break=kwrgs_events['max_break'],
                               grouped=kwrgs_events['grouped'])[0]
                self.RV_bin = self.RV_bin_fit.loc[self.dates_RV]
            elif only_RV_events == False:
                self.RV_b_full = func_fc.Ev_timeseries(self.fullts,
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
            fullts_xr = np.load(filename_ts, encoding='latin1',
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
                               threshold=self.threshold_ts_fit ,
                               min_dur=kwrgs_events_daily['min_dur'],
                               max_break=kwrgs_events_daily['max_break'],
                               grouped=kwrgs_events_daily['grouped'])[0]
                self.RV_bin = self.RV_bin_fit.loc[dates_RVe]
            elif only_RV_events == False:
                self.RV_b_full = func_fc.Ev_timeseries(self.fullts,
                               threshold=self.threshold ,
                               min_dur=kwrgs_events_daily['min_dur'],
                               max_break=kwrgs_events_daily['max_break'],
                               grouped=kwrgs_events_daily['grouped'])[0]
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



    
#rg.get_ts_prec(import_prec_ts=[('CPPA', 
#                                '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5')])

#%%
#
#class Var_import_RV_netcdf:
#    def __init__(self, ex):
#        vclass = Variable(self, ex)
#
#        vclass.name = ex['RVnc_name'][0]
#        vclass.filename = ex['RVnc_name'][1]
#        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name,
#               vclass.startyear, vclass.endyear, vclass.grid)))
#
#class Var_import_precursor_netcdf:
#    def __init__(self, tuple_name_path):
##        vclass = Variable(self, ex)
#
#        vclass.name = tuple_name_path[0]
#        vclass.filename = tuple_name_path[1] 
            