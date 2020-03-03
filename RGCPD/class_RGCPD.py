#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:13:58 2019
@author: semvijverberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_pp
import plot_maps
import find_precursors
from class_RV import RV_class
from class_EOF import EOF
from class_BivariateMI import BivariateMI
from typing import List, Tuple, Union
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory

df_ana_func = os.path.join(curr_dir, '..', 'df_analysis/df_analysis/') # add df_ana path
sys.path.append(df_ana_func)
import df_ana
path_test = os.path.join(curr_dir, '..', 'data')


class RGCPD:

    def __init__(self, list_of_name_path=List[Tuple[str, str]], 
                 list_for_EOFS: List[Union[EOF]]=None, 
                 list_for_MI: List[Union[BivariateMI]]=None, 
                 import_prec_ts: List[Tuple[str, str]]=None, 
                 start_end_TVdate=None, 
                 tfreq: int=10, 
                 start_end_date: Tuple[str, str]=None, 
                 start_end_year: Tuple[int, int]=None, 
                 lags_i: np.ndarray=np.array([1]),
                 path_outmain: str=None, 
                 append_pathsub='',
                 verbosity: int=1):
        '''
        Class to study teleconnection of a Response Variable* of interest. 
        
        Methods to extract teleconnections/precursors:
            - correlation maps
            - EOF analysis
        
        Correlation maps
        One can calculate correlation maps for the 1-dimensional timeseries of
        interest (RV timeseries), cluster the correlation precursor regions 
        and extract their spatial mean timeseries. 
        
        
        *Sometimes Response Variable is also called Target Variable.

        Parameters
        ----------
        list_of_name_path : list, optional
            list of (name, path) tuples. If None, test data is loaded.
            Convention: first entry should be (name, path) of target variable (TV).
        e.g. list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]
        list_for_EOFS : list, optional
            list of EOF classes, see docs EOF?
        import_prec_ts : list, optional
            Load in precursor 1-d timeseries in format:
                          [(name1, path_to_h5_file1), [(name2, path_to_h5_file2)]]
                          precursor_ts should follow the RGCPD traintest format
        start_end_TVdate : tuple, optional
            tuple of start- and enddate for target variable in 
            format ('mm-dd', 'mm-dd').
        tfreq : int, optional
            The default is 10.
        start_end_date : tuple, optional
            tuple of start- and enddate for data to load in 
            format ('mm-dd', 'mm-dd'). default is ('01-01' - '12-31')
        start_end_year : tuple, optional
            default is to load all years
        lags_i : nparray, optional
            The default is np.array([1]).            
        path_outmain : str, optional
            Root folder for output. Default is your 
            '/users/{username}/Download'' path
        append_pathsub: str, optional
            The first subfolder will be created below path_outmain, to store
            output data & figures. The append_pathsub1 argument allows you to 
            manually add some hash or string refering to some experiment.
        verbosity : int, optional
            Regulate the amount of feedback given by the code.
            The default is 1.

        Returns
        -------
        initialization of the RGCPD class

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
            user_download_path = get_download_path()
            path_outmain = user_download_path + '/output_RGCPD/'
        if os.path.isdir(path_outmain) != True : os.makedirs(path_outmain)

        self.list_of_name_path = list_of_name_path
        self.list_for_EOFS = list_for_EOFS
        self.list_for_MI = list_for_MI
        self.import_prec_ts = import_prec_ts

        self.start_end_TVdate   = start_end_TVdate
        self.start_end_date     = start_end_date
        self.start_end_year     = start_end_year
        self.verbosity          = verbosity
        self.tfreq              = tfreq
        self.lags_i             = lags_i
        self.lags               = np.array([l*self.tfreq for l in self.lags_i], dtype=int)
        self.path_outmain       = path_outmain
        self.append_pathsub    = append_pathsub
        self.figext             = '.pdf'
        self.orig_stdout        = sys.stdout
        return

    def pp_precursors(self, loadleap=False, seldates=None, selbox=None,
                            format_lon='only_east',
                            detrend=True, anomaly=True):
        '''
        in format 'only_east':
        selbox assumes [lowest_east_lon, highest_east_lon, south_lat, north_lat]
        '''
        loadleap = loadleap
        seldates = seldates
        selbox = selbox
        format_lon = format_lon
        detrend = detrend
        anomaly = anomaly

        self.kwrgs_pp = dict(loadleap=loadleap, seldates=seldates, selbox=selbox,
                            format_lon=format_lon, detrend=detrend, anomaly=anomaly)

        self.list_precur_pp = functions_pp.perform_post_processing(self.list_of_name_path,
                                             kwrgs_pp=self.kwrgs_pp,
                                             verbosity=self.verbosity)

    def get_clust(self, name_ds='ts'):
        f = functions_pp
        self.df_clust, self.ds = f.nc_xr_ts_to_df(self.list_of_name_path[0][1],
                                                  name_ds=name_ds)

    def apply_df_ana_plot(self, df=None, name_ds='ts', func=None, kwrgs_func={}):
        if df is None:
            self.get_clust(name_ds=name_ds)
            df = self.df_clust
        if func is None:
            func = df_ana.plot_ac ; kwrgs_func = {'AUC_cutoff':(14,30),'s':60}
        return df_ana.loop_df(df, function=func, sharex=False, 
                             colwrap=2, hspace=.5, kwrgs=kwrgs_func)
        

    def plot_df_clust(self):
        self.get_clust()
        plot_maps.plot_labels(self.ds['xrclustered'])
        
    def pp_TV(self, name_ds='ts', loadleap=False):
        self.name_TVds = name_ds
        f = functions_pp
        self.fulltso, self.hash = f.load_TV(self.list_of_name_path,
                                            loadleap=loadleap,
                                            name_ds=self.name_TVds)
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
        self.path_outsub0 = os.path.join(self.path_outmain, self.fulltso.name \
                                         +'_' +self.hash +'_'+RV_name_range \
                                         + info_lags + self.append_pathsub)
                                         

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


    def traintest(self, method='no_train_test_split', seed=1,
                  kwrgs_events=None):
        ''' Splits the training and test dates, either via cross-validation or
        via a simple single split.
        agrs:
        'method'        : str referring to method to split train test, see
                          options for method below.
        seed            : the seed to draw random samples for train test split
        kwrgs_events    : dict needed to create binary event timeseries, which
                          is used to create stratified folds.
                          See func_fc.Ev_timeseries? for more info.

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


        self.kwrgs_TV = dict(method=method,
                    seed=seed,
                    kwrgs_events=kwrgs_events,
                    precursor_ts=self.import_prec_ts)

        TV, self.df_splits = RV_and_traintest(self.fullts,
                                              self.TV_ts,
                                              verbosity=self.verbosity,
                                              **self.kwrgs_TV)
        self.TV = TV
        self.path_outsub1 = self.path_outsub0 + '_'.join(['', self.TV.method \
                            + 's'+ str(self.TV.seed)])
        if os.path.isdir(self.path_outsub1) == False : os.makedirs(self.path_outsub1)



        
    def calc_corr_maps(self, list_for_MI: List[Union[BivariateMI]]=None):
        keys = ['selbox', 'loadleap', 'seldates', 'format_lon']
        kwrgs_load = {k: self.kwrgs_pp[k] for k in keys}
        kwrgs_load['start_end_date']= self.start_end_date
        kwrgs_load['start_end_year']= self.start_end_year
        kwrgs_load['tfreq']         = self.tfreq
        self.kwrgs_load = kwrgs_load
        if list_for_MI is None:
            list_for_MI = self.list_for_MI
        
        # self.list_for_MI = []
        for precur in list_for_MI:
            precur.filepath = [l for l in self.list_precur_pp if l[0]==precur.name][0][1]
            precur.lags = self.lags
            find_precursors.calculate_region_maps(precur, 
                                                  self.TV, 
                                                  self.df_splits,
                                                  self.kwrgs_load)
                                                  
            # self.list_for_MI.append(precur)
    
    def cluster_list_MI(self):
        for precur in self.list_for_MI:                
            precur = find_precursors.cluster_DBSCAN_regions(precur)
                                                          

    def quick_view_labels(self, map_proj=None):
        for precur in self.list_for_MI:  
            prec_labels = precur.prec_labels.copy()
            if all(np.isnan(prec_labels.values.flatten()))==False:
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

                kwrgs = {'row_dim':'split', 'col_dim':'lag', 'hspace':-0.35,
                              'size':3, 'cbar_vert':cbar_vert, 'clevels':clevels,
                              'subtitles' : None, 'lat_labels':True,
                              'cticks_center':True,
                              'cmap':cmap}

                plot_maps.plot_corr_maps(prec_labels,
                                 contour_mask,
                                 map_proj, **kwrgs)
            else:
                print(f'no {precur.name} regions that pass distance_eps and min_area_in_degrees2 citeria')

    def get_EOFs(self):
        for i, e_class in enumerate(self.list_for_EOFS):
            print(f'Retrieving {e_class.neofs} EOF(s) for {e_class.name}')
            filepath = [l for l in self.list_precur_pp if l[0]==e_class.name][0][1]
            e_class.get_pattern(filepath=filepath, df_splits=self.df_splits)
            
    def get_ts_prec(self, precur_aggr=None):
        if precur_aggr is None:
            self.precur_aggr = self.tfreq
        else:
            self.precur_aggr = precur_aggr
        
        if precur_aggr is not None:
            # retrieving timeseries at different aggregation, TV and df_splits
            # need to redefined on new tfreq using the same arguments
            print(f'redefine target variable on {self.precur_aggr} day means')
            self.fulltso, self.TV_tso, inf = functions_pp.process_TV(self.fulltso,
                                                              self.precur_aggr,
                                                              self.start_end_TVdate,
                                                              self.start_end_date,
                                                              self.start_end_year)
            TV, df_splits = RV_and_traintest(self.fulltso, 
                                             self.TV_tso, **self.kwrgs_TV)
        else:
            # use original TV timeseries
            TV = self.TV ; df_splits = self.df_splits
        
        if hasattr(self, 'list_for_MI'):
            print('\nGetting MI timeseries')
            for i, precur in enumerate(self.list_for_MI):
                precur.get_prec_ts(precur_aggr=self.precur_aggr,
                                   kwrgs_load=self.kwrgs_load)
            self.df_data = find_precursors.df_data_prec_regs(self.list_for_MI, 
                                                             TV, 
                                                             df_splits)
                                                
        # Append (or only load in) external timeseries
        if self.import_prec_ts is not None:
            print('\nGetting external timeseries')
            self.df_data_ext = find_precursors.import_precur_ts(self.import_prec_ts,
                                                             df_splits,
                                                             self.precur_aggr,
                                                             self.start_end_date,
                                                             self.start_end_year)
            if hasattr(self, 'df_data'):
                self.df_data = self.df_data.merge(self.df_data_ext, left_index=True, right_index=True)
            else:
                self.df_data = self.df_data_ext.copy()
        
        # Append (or only load) EOF timeseries
        if hasattr(self, 'list_for_EOFS'):
            print('\nGetting EOF timeseries')
            for i, e_class in enumerate(self.list_for_EOFS):
                e_class.get_ts(tfreq_ts=self.precur_aggr, df_splits=df_splits)
                keys = np.array(e_class.df.dtypes.index[e_class.df.dtypes != bool], dtype='object')
                if hasattr(self, 'df_data'):
                    self.df_data = self.df_data.merge(e_class.df[keys], 
                                                      left_index=True, 
                                                      right_index=True)
                else:
                    self.df_data = e_class.df[keys]
            
        # Append Traintest and RV_mask as last columns
        self.df_data = self.df_data.merge(df_splits, left_index=True, right_index=True)


    def PCMCI_init(self):
        import wrapper_PCMCI
        self.pcmci_dict = wrapper_PCMCI.init_pcmci(self.df_data)

    def PCMCI_df_data(self, path_txtoutput=None, tau_min=0, tau_max=1,
                    pc_alpha=None, alpha_level=0.05, max_conds_dim=None,
                    max_combinations=2, max_conds_py=None, max_conds_px=None,
                    verbosity=4):
        import wrapper_PCMCI
        

        self.kwrgs_pcmci = dict(tau_min=tau_min,
                           tau_max=tau_max,
                           pc_alpha=pc_alpha,
                           alpha_level=alpha_level,
                           max_conds_dim=max_conds_dim,
                           max_combinations=max_combinations,
                           max_conds_py=max_conds_py,
                           max_conds_px=max_conds_px,
                           verbosity=4)




        if path_txtoutput is None:
            self.params_str = '{}_at{}_tau_{}-{}_conds_dim{}_combin{}_dt{}'.format(
                          pc_alpha, self.kwrgs_pcmci['alpha_level'], tau_min, tau_max, 
                          max_conds_dim, max_combinations, self.precur_aggr)
            self.path_outsub2 = os.path.join(self.path_outsub1, self.params_str)
        else:
            self.path_outsub2 = path_txtoutput

        if os.path.isdir(self.path_outsub2) == False : os.makedirs(self.path_outsub2)
        
        if hasattr(self, 'pcmci_dict')==False:
            self.pcmci_dict = wrapper_PCMCI.init_pcmci(self.df_data)
        
        out = wrapper_PCMCI.loop_train_test(self.pcmci_dict, self.path_outsub2,
                                                          **self.kwrgs_pcmci)
        self.pcmci_results_dict = out
    
    def PCMCI_get_links(self, alpha_level: float=None):
        import wrapper_PCMCI
        
        if hasattr(self, 'pcmci_results_dict')==False:
            print('first perform PCMCI_df_data to get pcmci_results_dict')
        
        if alpha_level is None:
            alpha_level = self.kwrgs_TV['alpha_level']
        self.parents_dict = wrapper_PCMCI.get_links_pcmci(self.pcmci_dict, 
                                                          self.pcmci_results_dict,
                                                          alpha_level)
        self.df_sum, self.mapping_links = wrapper_PCMCI.get_df_sum(self.parents_dict)

        # get xarray dataset for each variable
        self.dict_ds = plot_maps.causal_reg_to_xarray(self.TV.name, self.df_sum,
                                                      self.list_for_MI)

    def store_df_PCMCI(self):
        import wrapper_PCMCI
        if self.tfreq != self.precur_aggr:
            path = self.path_outsub2 + f'_dtd{self.precur_aggr}'
        else:
            path = self.path_outsub2
        wrapper_PCMCI.store_ts(self.df_data, self.df_sum, self.dict_ds,
                               path+'.h5')
        self.df_data_filename = path+'.h5'
                               
    def store_df(self):
        if len(self.list_for_MI) != 0:
            varstr = '_' + '_'.join([p.name for p in self.list_for_MI])
        else:
            varstr = ''
        if hasattr(self, 'df_data_ext'):
            varstr = '_'.join([n[0] for n in self.import_prec_ts]) + varstr
        filename = os.path.join(self.path_outsub1, f'df_data_{varstr}_'
                                f'dt{self.precur_aggr}_{self.hash}.h5')
        functions_pp.store_hdf_df({'df_data':self.df_data}, filename)
        print('Data stored in \n{}'.format(filename))
        self.df_data_filename = filename
        
    def plot_maps_corr(self, precursors=None, mask_xr=None, map_proj=None,
                       row_dim='split', col_dim='lag', clim='relaxed', 
                       hspace=-0.6, wspace=.02, size=2.5, cbar_vert=-0.01, units='units',
                       cmap=None, clevels=None, cticks_center=None, drawbox=None,
                       title=None, subtitles=None, zoomregion=None, lat_labels=True,
                       aspect=None, save=False):

        
        if precursors is None:
            precursors = [p.name for p in self.list_for_MI]
        for precur_name in precursors:
            pclass = [p for p in self.list_for_MI if p.name == precur_name][0]
            plot_maps.plot_corr_maps(pclass.corr_xr,
                                     mask_xr=pclass.corr_xr['mask'], map_proj=map_proj,
                                   row_dim=row_dim, col_dim=col_dim, clim=clim, 
                                   hspace=hspace, wspace=wspace, size=size, cbar_vert=cbar_vert, 
                                   units=units, cmap=cmap, clevels=clevels, 
                                   cticks_center=cticks_center, drawbox=drawbox,
                                   title=None, subtitles=subtitles, 
                                   zoomregion=zoomregion, 
                                   lat_labels=lat_labels, aspect=aspect)
            if save is True:
                f_name = 'corr_map_{}'.format(precur_name)
                                                 
                fig_path = os.path.join(self.path_outsub1, f_name)+self.figext
                plt.savefig(fig_path, bbox_inches='tight')

    def plot_maps_sum(self, var='all', map_proj=None, figpath=None, 
                      paramsstr=None, kwrgs_plot={}):

#         if map_proj is None:
#             central_lon_plots = 200
#             map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)

        if figpath is None:
            figpath = self.path_outsub1
        if paramsstr is None:
            paramsstr = self.params_str
        if var == 'all':
            dict_ds = self.dict_ds
        else:
            dict_ds = {f'{var}':self.dict_ds[var]} # plot single var            
        plot_maps.plot_labels_vars_splits(dict_ds, self.df_sum, map_proj,
                                          figpath, paramsstr, self.TV.name,
                                           kwrgs_plot=kwrgs_plot)


        plot_maps.plot_corr_vars_splits(dict_ds, self.df_sum, map_proj,
                                          figpath, paramsstr, self.TV.name, 
                                          kwrgs_plot=kwrgs_plot)

    def _get_testyrs(self, df_splits):
    #%%
        if df_splits is None:
            df_splits = self.df_splits
        traintest_yrs = []
        splits = df_splits.index.levels[0]
        for s in splits:
            df_split = df_splits.loc[s]
            test_yrs = np.unique(df_split[df_split['TrainIsTrue']==False].index.year)
            traintest_yrs.append(test_yrs)
        return traintest_yrs

def RV_and_traintest(fullts, TV_ts, method=str, kwrgs_events=None, precursor_ts=None, 
                     seed=int, verbosity=1): #, method=str, kwrgs_events=None, precursor_ts=None, seed=int, verbosity=1

    # Define traintest:
    df_fullts = pd.DataFrame(fullts.values, 
                            index=pd.to_datetime(fullts.time.values),
                            columns=[fullts.name])
    df_RV_ts = pd.DataFrame(TV_ts.values,
                            index=pd.to_datetime(TV_ts.time.values),
                            columns=['RV'+fullts.name])

    if method[:9] == 'ran_strat' and kwrgs_events is None and precursor_ts is not None:
            # events need to be defined to enable stratified traintest.
            kwrgs_events = {'event_percentile': 66,
                            'min_dur' : 1,
                            'max_break' : 0,
                            'grouped' : False}
            if verbosity == 1:
                print("kwrgs_events not given, creating stratified traintest split "
                    "based on events defined as exceeding the {}th percentile".format(
                        kwrgs_events['event_percentile']))

    TV = RV_class(df_fullts, df_RV_ts, kwrgs_events)
    
    
    if precursor_ts is not None:
        print('Retrieve same train test split as imported ts')
        method = 'from_import' ; seed = ''
        path_data = ''.join(precursor_ts[0][1])
        df_splits = functions_pp.load_hdf5(path_data)['df_data'].loc[:,['TrainIsTrue', 'RV_mask']]
        test_yrs_imp  = functions_pp.get_testyrs(df_splits)
        df_splits = functions_pp.rand_traintest_years(TV, test_yrs=test_yrs_imp,
                                                        method=method,
                                                        seed=seed, 
                                                        kwrgs_events=kwrgs_events, 
                                                        verb=verbosity)

        test_yrs_set  = functions_pp.get_testyrs(df_splits)
        assert (np.equal(test_yrs_imp, test_yrs_set)).all(), "Train test split not equal"
    else:
        df_splits = functions_pp.rand_traintest_years(TV, method=method,
                                                        seed=seed, 
                                                        kwrgs_events=kwrgs_events, 
                                                        verb=verbosity)
    TV.method = method
    TV.seed   = seed
    return TV, df_splits


def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'Downloads')
