#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:13:58 2019
@author: semvijverberg
"""
import inspect, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_pp
import plot_maps
import find_precursors
import core_pp
from class_RV import RV_class
from class_EOF import EOF
from class_BivariateMI import BivariateMI
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
fc_dir = os.path.join(main_dir, 'forecasting/')

if fc_dir not in sys.path:
    sys.path.append(fc_dir)
import func_models as fc_utils

from typing import List, Tuple, Union

def get_timestr(formatstr='%Y-%m-%d_%Hhr_%Mmin'):
    import datetime
    return datetime.datetime.today().strftime(formatstr)


try:
    import wrapper_PCMCI as wPCMCI
except:
    # raise(ModuleNotFoundError)
    print('Not able to load in Tigramite modules, to enable causal inference '
          'features, install Tigramite from '
          'https://github.com/jakobrunge/tigramite/')

try:
    from tigramite import plotting as tp
except:
    # raise(ModuleNotFoundError)
    print('Not able to load in plotting modules, check installment of networkx')

df_ana_dir = os.path.join(curr_dir, '..', 'df_analysis/df_analysis/') # add df_ana path
fc_dir       = os.path.join(curr_dir, '..', 'forecasting/') # add df_ana path
sys.path.append(df_ana_dir) ; sys.path.append(fc_dir)
import df_ana
import stat_models_cont as sm
path_test = os.path.join(curr_dir, '..', 'data')


class RGCPD:

    def __init__(self, list_of_name_path: List[Tuple[str, str]]=None,
                 list_for_EOFS: List[Union[EOF]]=None,
                 list_for_MI: List[Union[BivariateMI]]=None,
                 list_import_ts: List[Tuple[str, str]]=None,
                 start_end_TVdate=None,
                 tfreq: int=10,
                 start_end_date: Tuple[str, str]=None,
                 start_end_year: Tuple[int, int]=None,
                 path_outmain: str=None,
                 append_pathsub='', save: bool=True,
                 verbosity: int=1):
        '''
        Class to study teleconnections of a Response Variable* of interest.

        Methods to extract teleconnections/precursors:
            - BivariateMI (supporting (partial) correlation maps)
            - EOF analysis

        BivariateMI (MI = Mutual Information) is class which allows for a
        statistical test in the form:
        MI(lon,lat) = for gc in map: func(x(t), y(t)),
        where map is a (time,lon,lat) map and gc stands for each gridcell/coordinate
        in that map. The y timeseries is always the same 1-dimensional timeseries of
        interest (i.e. the Response Variable). At this point, only supports the
        correlation analysis. Once the significance is attributed, it is stored
        in the MI map. Precursor regions are found by clustering the
        significantly (correlating) gridcells (+ and - regions are separated)
        and extract their spatial mean (or spatial covariance) timeseries.

        *Sometimes Response Variable is also called Target Variable.

        Parameters
        ----------
        list_of_name_path : list, optional
            list of (name, path) tuples defining the input data.

            Convention: first entry should be (name, path) of target variable (TV).
            e.g. list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]

            TVpath input data supports .nc/.h5 or .csv file format.
            if using output of the clustering:
                'TVname' should refer the name
                you have given the timeseries on the dimesion 'cluster', i.e.
                xrTV.sel(cluster=TVname)
            elif using .h5 the index should contain a datetime axis.
            elif using .csv the first columns should be [year, month, day, value]


            prec_path input data supports only .nc
            'prec_name' is a string/key that can be chosen freely, does not have
            to refer to the variable in the .nc file.
            Each prec_path .nc file should contain only a single variable
            of format (time, lat, lon).
        list_for_EOFS : list, optional
            list of EOF classes, see docs EOF?
        list_import_ts : list, optional
            Load in precursor 1-d timeseries in format:
            [(name1, path_to_h5_file1), [(name2, path_to_h5_file2)]]
            precursor_ts can handle the RGCPD cross-validation format.
        start_end_TVdate : tuple, optional
            tuple of start- and enddate for target variable in
            format ('mm-dd', 'mm-dd').
        tfreq : int, optional
            The default is 10, if using time_mean_periods, tfreq should be None.
        start_end_date : tuple, optional
            tuple of start- and enddate for data to load in
            format ('mm-dd', 'mm-dd'). default is ('01-01' - '12-31')
        start_end_year : tuple, optional
            default is to load all years
        path_outmain : [str, bool], optional
            Root folder for output. If None, default is your
            '/users/{username}/Download' path.
        append_pathsub: str, optional
            The first subfolder will be created below path_outmain, to store
            output data & figures. The append_pathsub1 argument allows you to
            manually add some hash or string refering to some experiment.
        save : bool, optional
            If you want to save figures, data and text output automatically.
        verbosity : int, optional
            Regulate the amount of feedback given by the code.
            The default is 1.

        Returns
        -------
        initialization of the RGCPD class

        '''
        if list_of_name_path is None:
            print('initializing with test data')
            list_of_name_path = [(3,
                                  os.path.join(path_test, 'tf5_nc5_dendo_80d77.nc')),
                                 ('sst_test',
                                  os.path.join(path_test, 'sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc'))]

        if start_end_TVdate is None:
            start_end_TVdate = ('06-01', '08-31')

        if path_outmain is None:
            user_download_path = get_download_path()
            path_outmain = user_download_path + '/output_RGCPD/'



        self.list_of_name_path = list_of_name_path
        self.list_for_EOFS = list_for_EOFS
        self.list_for_MI = list_for_MI
        self.list_import_ts = list_import_ts

        self.start_end_TVdate   = start_end_TVdate
        self.start_end_date     = start_end_date
        self.start_end_year     = start_end_year


        self.verbosity          = verbosity
        self.tfreq              = tfreq
        self.path_outmain       = path_outmain
        self.append_pathsub     = append_pathsub
        self.figext             = '.pdf'
        self.save               = save
        self.orig_stdout        = sys.stdout
        if self.save == True and os.path.isdir(path_outmain) != True:
            os.makedirs(path_outmain)
        return

    def pp_precursors(self, loadleap=False, seldates=None,
                      selbox=None, format_lon='only_east',
                      auto_detect_mask=False,
                      detrend=True, anomaly=True):
        '''
        in format 'only_east':
        selbox assumes [lowest_east_lon, highest_east_lon, south_lat, north_lat]
        '''
        # loadleap=False;seldates=None;selbox=None;format_lon='only_east',
        # detrend=True; anomaly=True; auto_detect_mask=False
        self.kwrgs_load = dict(loadleap=loadleap, seldates=seldates,
                               selbox=selbox, format_lon=format_lon)
        self.kwrgs_pp = self.kwrgs_load.copy()
        self.kwrgs_pp.update(dict(detrend=detrend, anomaly=anomaly,
                                  auto_detect_mask=auto_detect_mask))

        self.kwrgs_load.update(dict(start_end_date=self.start_end_date,
                                    start_end_year=self.start_end_year,
                                    start_end_TVdate=self.start_end_TVdate,
                                    tfreq=self.tfreq))

        self.list_precur_pp = functions_pp.perform_post_processing(self.list_of_name_path,
                                             kwrgs_pp=self.kwrgs_pp,
                                             verbosity=self.verbosity)
        if self.list_for_MI is not None:
            for precur in self.list_for_MI:
                precur.filepath = [l for l in self.list_precur_pp if l[0]==precur.name][0][1]

    def pp_TV(self, name_ds='ts', detrend=False, anomaly=False,
              kwrgs_core_pp_time: dict={}, ext_annual_to_mon: bool=True,
              TVdates_aggr: bool=False):
        '''
        Load and pre-process target variable/response variable.

        Parameters
        ----------
        name_ds : str, optional
            name of 1-d timeseries in .nc file. The default is 'ts'.
        detrend : bool, optional
            linear detrend. The default is False.
        anomaly : bool, optional
            calculate anomaly verus climatology. The default is False.
        kwrgs_core_pp_time: dict, {}
            see xr_core_pp_time? for optional arguments
            - start_end_year selection is done prior to detrend & anomoly.
        ext_annual_to_mon : bool, optional
            if tfreq is None and target variable contain one-value-per-year,
            the target is extended to match the percursor time-axis.
            If precursors are monthly means, then the the target is also extended
            to monthly values, else daily values. Both are then aggregated to
            {tfreq} day/monthly means.  The default is True.
        TVdates_aggr : bool, optional
            If True, set rg.tfreq to None. Target Variable will be aggregated
            to a single-value-per-year "period mean". start_end_TVdate defines
            the period to aggregate over.

        Returns
        -------
        fulltso: the original 1-d timeseries
        fullts: the pre-processed 1-d timeseries
        TV_ts: the pre-processed 1-d timeseries within target period
        dates_all: all dates of fullts
        dates_TV: all dates of TV_ts


        '''
        self.name_TVds = name_ds
        self.RV_anomaly = anomaly
        self.RV_detrend = detrend
        f = functions_pp
        self.fulltso, self.hash = f.load_TV(self.list_of_name_path,
                                            name_ds=self.name_TVds)
        from core_pp import xr_core_pp_time
        self.fulltso = xr_core_pp_time(self.fulltso, **kwrgs_core_pp_time)
        out = f.process_TV(self.fulltso,
                            tfreq=self.tfreq,
                            start_end_TVdate=self.start_end_TVdate,
                            start_end_date=self.start_end_date,
                            start_end_year=self.start_end_year,
                            RV_detrend=self.RV_detrend,
                            RV_anomaly=self.RV_anomaly,
                            ext_annual_to_mon=ext_annual_to_mon,
                            TVdates_aggr=TVdates_aggr)
        self.fullts, self.TV_ts, inf, self.traintestgroups = out

        self.input_freq = inf
        self.dates_or  = pd.to_datetime(self.fulltso.time.values)
        self.dates_all = pd.to_datetime(self.fullts.time.values)
        self.dates_TV = pd.to_datetime(self.TV_ts.time.values)
        # if self.start_end_date is None and self.input_freq != 'annual':
        #     self.start_end_date = ('{}-{}'.format(self.dates_or.month[0],
        #                                          self.dates_or[0].day),
        #                            '{}-{}'.format(self.dates_or.month[-1],
        #                                          self.dates_or[-1].day))
        if self.start_end_year is None:
            self.start_end_year = (self.dates_or.year[0],
                                   self.dates_or.year[-1])


    def traintest(self, method: Union[str, bool]=None, seed=1,
                  kwrgs_events=None, subfoldername=None):
        ''' Splits the training and test dates, either via cross-validation or
        via a simple single split.

        method : str or bool, optional
            Referring to method to split train test, see options for method below.
            default is False.
        seed : int, optional
            The seed to draw random samples for train test split, default is 1.
        kwrgs_events : dict, optional
            Kwrgs needed to create binary event timeseries, which was used to
            create stratified folds. See func_fc.Ev_timeseries? for more info.

        Options for method:
        (1) random_{int}   :   with the int(ex['method'][6:8]) determining the amount of folds
        (2) ranstrat_{int}:   random stratified folds, stratified based upon events,
                              requires kwrgs_events.
        (3) leave_{int}    :   chronologically split train and test years.
        (4) split_{int}    :   (should be updated) split dataset into single train and test set
        (5) no_train_test_split or False
        '''

        if method is None or method is False:
            method = 'no_train_test_split'
        self.kwrgs_TV = dict(method=method,
                    seed=seed,
                    kwrgs_events=kwrgs_events,
                    precursor_ts=self.list_import_ts)

        self.TV, self.df_splits = RV_and_traintest(self.fullts,
                                                   self.TV_ts,
                                                   self.traintestgroups,
                                                   verbosity=self.verbosity,
                                                   **self.kwrgs_TV)
        self.n_spl = self.df_splits.index.levels[0].size
        if subfoldername is None:
            RV_name_range = '{}-{}_'.format(*list(self.start_end_TVdate))
            var = '_'.join([np[0] for np in self.list_of_name_path[1:]])
            # Creating a folder for the specific target, RV period and traintest set
            part1 = os.path.join(self.fulltso.name \
                                 +'_' +self.hash +'_'+RV_name_range \
                                 +var)
            subfoldername = part1 + '_'.join(['', self.TV.method \
                                  + 's'+ str(self.TV.seed)])
            if self.append_pathsub is not None:
                subfoldername += '_' + self.append_pathsub
        self.path_outsub1 = os.path.join(self.path_outmain, subfoldername)
        if self.save and os.path.isdir(self.path_outsub1)==False:
            os.makedirs(self.path_outsub1)

    def calc_corr_maps(self, var: Union[str, list]=None):

        if var is None:
            if type(var) is str:
                var = [var]
            var = [MI.name for MI in self.list_for_MI]
        kwrgs_load = self.kwrgs_load
        for precur in self.list_for_MI:
            if precur.name in var:
                find_precursors.calculate_region_maps(precur,
                                                      self.TV,
                                                      self.df_splits,
                                                      kwrgs_load)

    def cluster_list_MI(self, var: Union[str, list]=None):
        if var is None:
            if type(var) is str:
                var = [var]
            var = [MI.name for MI in self.list_for_MI]
        for precur in self.list_for_MI:
            if precur.name in var:
                if hasattr(precur, 'corr_xr'):
                    precur = find_precursors.cluster_DBSCAN_regions(precur)
                else:
                    print(f'No MI map available for {precur.name}')

    def get_EOFs(self):
        for i, e_class in enumerate(self.list_for_EOFS):
            print(f'Retrieving {e_class.neofs} EOF(s) for {e_class.name}')
            filepath = [l for l in self.list_precur_pp if l[0]==e_class.name][0][1]
            e_class.get_pattern(filepath=filepath, df_splits=self.df_splits)

    def plot_EOFs(self, mean=True, kwrgs: dict=None):
        for i, e_class in enumerate(self.list_for_EOFS):
            print(f'Retrieving {e_class.neofs} EOF(s) for {e_class.name}')
            e_class.plot_eofs(mean=mean, kwrgs=kwrgs)

    def get_ts_prec(self, precur_aggr: int=None, keys_ext: list=None,
                    start_end_TVdate: tuple=None):
        '''
        Aggregate target and precursors to binned means.

        Parameters
        ----------
        precur_aggr : int, optional
            bin window size to calculate time mean bins. If None, self.tfreq
            value is choosen.
        keys_ext : list, optional
            list with column names to load and aggregate a subset of the
            .h5 pandas dataframe. The default is None.
        start_end_TVdate : tuple, optional
            Allows to change the target start end period. Using format
            format ('mm-dd', 'mm-dd'). The default is None.


        Returns
        -------
        None.

        '''
        if precur_aggr is None:
            self.precur_aggr = self.tfreq
        else:
            self.precur_aggr = precur_aggr

        if precur_aggr is not None or start_end_TVdate is not None:
            if start_end_TVdate is None:
                start_end_TVdate = self.start_end_TVdate
                # for loading aggregated ncdf is aligned
            else:
                self.kwrgs_load['start_end_TVdate'] = start_end_TVdate
            # retrieving timeseries at different aggregation, TV and df_splits
            # need to redefined on new tfreq using the same arguments
            print(f'redefine target variable on {self.precur_aggr} day means')
            f = functions_pp
            self.fullts, self.TV_ts = f.process_TV(self.fulltso,
                                                tfreq=self.precur_aggr,
                                                start_end_TVdate=start_end_TVdate,
                                                start_end_date=self.start_end_date,
                                                start_end_year=self.start_end_year,
                                                RV_detrend=self.RV_detrend,
                                                RV_anomaly=self.RV_anomaly)[:2]
            TV, df_splits = RV_and_traintest(self.fullts,
                                             self.TV_ts,
                                             self.traintestgroups, **self.kwrgs_TV)
        else:
            # use original TV timeseries
            start_end_TVdate = self.start_end_TVdate
            TV = self.TV ; df_splits = self.df_splits
        self.df_data = pd.DataFrame(TV.fullts.values, columns=[TV.name],
                                    index=TV.dates_all)
        self.df_data = pd.concat([self.df_data]*self.df_splits.index.levels[0].size,
                                 keys=self.df_splits.index.levels[0])
        if self.list_for_MI is not None:
            print('\nGetting MI timeseries') ; c = 0
            for i, precur in enumerate(self.list_for_MI):
                if hasattr(precur, 'prec_labels'):
                    precur.get_prec_ts(precur_aggr=self.precur_aggr,
                                       kwrgs_load=self.kwrgs_load)
                else:
                    print(f'{precur.name} not clustered yet')
                    c += i
            if c == len(self.list_for_MI):
                print('No precursors clustered')
            else:
                MI_ts_corr = [MI for MI in self.list_for_MI if hasattr(MI, 'ts_corr')]
                check_ts = np.unique([MI.ts_corr.size for MI in MI_ts_corr])
                any_MI_ts = (~np.equal(check_ts, 0)).any() # ts_corr.size != 0?
                if any_MI_ts:
                    df_data_MI = find_precursors.df_data_prec_regs(self.list_for_MI,
                                                                   TV,
                                                                   df_splits)
                    self.df_data = self.df_data.merge(df_data_MI, left_index=True,
                                                      right_index=True)
                else:
                    print('No precursor regions significant')

        # Append (or only load in) external timeseries
        if self.list_import_ts is not None:
            print('\nGetting external timeseries')
            f = find_precursors
            self.df_data_ext = f.import_precur_ts(self.list_import_ts,
                                                  df_splits.copy(),
                                                  self.start_end_date,
                                                  self.start_end_year,
                                                  cols=keys_ext,
                                                  precur_aggr=self.precur_aggr,
                                                  closed_on_date=start_end_TVdate[-1])
            self.df_data = self.df_data.merge(self.df_data_ext,
                                              left_index=True, right_index=True)


        # Append (or only load) EOF timeseries
        if self.list_for_EOFS is not None:
            print('\nGetting EOF timeseries')
            for i, e_class in enumerate(self.list_for_EOFS):
                e_class.get_ts(tfreq_ts=self.precur_aggr, df_splits=df_splits)
                keys = np.array(e_class.df.dtypes.index[e_class.df.dtypes != bool], dtype='object')
                self.df_data = self.df_data.merge(e_class.df[keys],
                                                      left_index=True,
                                                      right_index=True)

        # Append Traintest and RV_mask as last columns
        self.df_data = self.df_data.merge(df_splits, left_index=True, right_index=True)
        allkeys = [list(self.df_data.loc[s].dropna(axis=1).columns[1:-2]) for s in range(self.n_spl)]
        allkeys = functions_pp.flatten(allkeys)
        {k:allkeys.count(k) for k in allkeys}
        self._df_count = pd.Series({k:allkeys.count(k) for k in allkeys})

    def get_subdates_df(self, df_data: pd.DataFrame=None,
                        start_end_date: tuple=None,
                        years: Union[list, tuple]=None):
        if df_data is None:
            df_data = self.df_data.copy()
        dates = df_data.loc[0].index
        if type(years) is tuple or start_end_date is None:
            seldates = functions_pp.core_pp.get_subdates(dates, start_end_date,
                                            years)
        elif type(years) in [np.ndarray, list]:
            seldates = functions_pp.get_oneyr(dates, years)
        return df_data.loc[pd.MultiIndex.from_product([range(self.n_spl), seldates])]

    def merge_df_on_df_data(self, df, columns: list=None):
        '''
        Merges self.df_data with given df[columns]. Ensures that first column
        remains target var and last (two) column(s) are TrainIsTrue, (RV_mask).
        self.df_data and df must be on same time axis.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list, optional

        Returns
        -------
        df_data_merged.

        '''

        if columns is None:
            columns = list(df.columns)
        if hasattr(df.index, 'levels') == False:
            print('No traintest split in df, copying to traintest splits')
            splits = self.df_data.index.levels[0]
            df = pd.concat([df]*splits.size, keys=splits)
        df_mrg = pd.merge(df, self.df_data, left_index=True, right_index=True)
        order = list(self.df_data.columns) ; order[1:1] = columns
        return df_mrg[order]


    def PCMCI_init(self, df_data: pd.DataFrame=None, keys: list=None, verbosity=4):
        if df_data is None:
            df_data = self.df_data.copy()
        if keys is None:
            keys = df_data.columns
        elif 'TrainIsTrue' not in keys and 'RV_mask' not in keys:
            keys = keys.copy()
            keys.append('TrainIsTrue') ; keys.append('RV_mask')

        self.pcmci_dict = wPCMCI.init_pcmci(self.df_data[keys],
                                            verbosity=verbosity)

    def PCMCI_df_data(self, df_data: pd.DataFrame=None, keys: list=None,
                      path_txtoutput=None, tigr_function_call='run_pcmci',
                      kwrgs_tigr: dict=None,
                      replace_RV_mask: np.ndarray=None,
                      verbosity=4):

        if df_data is None:
            df_data = self.df_data.copy()
        if keys is None:
            keys = df_data.columns

        if kwrgs_tigr is None: # Some reasonable defaults
            self.kwrgs_tigr = dict(tau_min=0,
                                   tau_max=1,
                                   pc_alpha=[.01,.05,.1,.2],
                                   max_conds_dim=len(keys)-4,
                                   max_combinations=2,
                                   max_conds_py=2,
                                   max_conds_px=2)
        else:
            self.kwrgs_tigr = kwrgs_tigr
        if tigr_function_call == 'run_pcmciplus' and kwrgs_tigr is None:
            self.kwrgs_tigr = self.kwrgs_tigr.pop('max_combinations')

        if path_txtoutput is None:
            kwrgs_tigr = self.kwrgs_tigr.copy() ;
            if 'selected_links' in kwrgs_tigr.keys():
                kwrgs_tigr['selected_links'] = True
            kd = sorted(kwrgs_tigr.items()) ;
            d = [list(d) for d in kd if type(d[1]) not in [np.ndarray,list]]
            dl = [[d[0],str(d[1]).replace(' ','')] for d in kd if type(d[1]) in [np.ndarray,list]]
            p =''.join(''.join(np.array(d+dl,str).flatten()))
            self.params_str = '{}_{}_dt{}'.format(tigr_function_call.split('_')[1],
                                               p, self.precur_aggr)

            self.path_outsub2 = os.path.join(self.path_outsub1, self.params_str)
        else:
            self.path_outsub2 = path_txtoutput
        if self.save:
            if os.path.isdir(self.path_outsub2)==False:
                os.makedirs(self.path_outsub2)
            path_outsub2 = self.path_outsub2
        else:
            path_outsub2 = False # not textfile written



        if type(replace_RV_mask) is np.ndarray:
            self._replace_RV_mask(df_data=df_data,
                                  replace_RV_mask=replace_RV_mask,
                                  plot=True)

        self.PCMCI_init(df_data, keys, verbosity=verbosity)

        out = wPCMCI.loop_train_test(self.pcmci_dict, path_outsub2,
                                     tigr_function_call=tigr_function_call,
                                     kwrgs_tigr=self.kwrgs_tigr)
        self.pcmci_results_dict = out

    def PCMCI_get_links(self, var: str=None, alpha_level: float=.05):
        '''


        Parameters
        ----------
        var : str, optional
            Specify variable you want to retrieve links for. If None, returns
            links toward target variable
        alpha_level : float, optional
            significance threshold. The default is .05.

        Returns
        -------
        DataFrame of MCI coefficients and alpha values _toward_ var.

        '''

        if hasattr(self, 'pcmci_results_dict')==False:
            print('first perform PCMCI_df_data to get pcmci_results_dict')
        if var is None:
            var = self.TV.name

        self.parents_dict = wPCMCI.get_links_pcmci(self.pcmci_dict,
                                                   self.pcmci_results_dict,
                                                   alpha_level)
        self.df_links = wPCMCI.get_df_links(self.parents_dict, variable=var)
        lags = np.arange(0, self.kwrgs_tigr['tau_max']+1)
        self.df_MCIc, self.df_MCIa = wPCMCI.get_df_MCI(self.pcmci_dict,
                                                 self.pcmci_results_dict,
                                                 lags, variable=var)
        # # get xarray dataset for each variable
        self.dict_ds = plot_maps.causal_reg_to_xarray(self.df_links,
                                                      self.list_for_MI)

    def PCMCI_plot_graph(self, variable: str=None, s: int=None, kwrgs: dict=None,
                         figshape: tuple=(10,10), min_link_robustness: int=1,
                         append_figpath: str=None):

        out = wPCMCI.get_traintest_links(self.pcmci_dict,
                                         self.parents_dict,
                                         self.pcmci_results_dict,
                                         variable=variable,
                                         s=s,
                                         min_link_robustness=min_link_robustness)
        links_plot, val_plot, weights, var_names = out

        if kwrgs is None:
            kwrgs = {'link_colorbar_label':'cross-MCI',
                     'node_colorbar_label':'auto-MCI',
                     'curved_radius':.4,
                     'arrowhead_size':4000,
                     'arrow_linewidth':50,
                     'label_fontsize':14}
        if 'link_width' in kwrgs.keys():
            link_width = kwrgs.pop('link_width')
        else:
            link_width = 1
        if 'weights_squared' in kwrgs.keys():
            link_width = link_width+weights**kwrgs.pop('weights_squared')
        fig = plt.figure(figsize=figshape, facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        fig, ax = tp.plot_graph(val_matrix=val_plot,
                      var_names=var_names,
                      link_width=link_width,
                      link_matrix=links_plot,
                      fig_ax=(fig, ax),
                      **kwrgs)
        f_name = f'CEN_{variable}_s{s}'
        if append_figpath is not None:
            fig_path = os.path.join(self.path_outsub1, f_name+append_figpath)
        else:
            fig_path = os.path.join(self.path_outsub1, f_name)
        if self.save:
            fig.savefig(fig_path+self.figext, bbox_inches='tight')
        plt.show()

    def PCMCI_get_ParCorr_from_txt(self, variable=None, pc_alpha='auto'):

        if variable is None:
            variable = self.TV.name

        # lags = range(0, self..kwrgs_tigr['tau_max']+1)
        splits = self.df_splits.index.levels[0]
        df_ParCorr_s = np.zeros( (splits.size) , dtype=object)
        for s in splits:
            filepath_txt = os.path.join(self.path_outsub2,
                                        f'split_{s}_PCMCI_out.txt')

            df = wPCMCI.extract_ParCorr_info_from_text(filepath_txt,
                                                       variable=variable)
            df_ParCorr_s[s] = df
        df_ParCorr = pd.concat(list(df_ParCorr_s), keys= range(splits.size))
        df_ParCorr_sum = pd.concat([df_ParCorr['coeff'].mean(level=1),
                                    df_ParCorr['coeff'].min(level=1),
                                    df_ParCorr['coeff'].max(level=1),
                                    df_ParCorr['pval'].mean(level=1),
                                    df_ParCorr['pval'].max(level=1)],
                                   keys = ['coeff mean', 'coeff min', 'coeff max',
                                           'pval mean', 'pval max'], axis=1)
        all_options = np.unique(df_ParCorr['ParCorr'])[::-1]
        list_of_series = []
        for op in all_options:
            newseries = (df_ParCorr['ParCorr'] == op).sum(level=1).astype(int)
            newseries.name = f'ParCorr {op}'
            list_of_series.append(newseries)

        self.df_ParCorr_sum = pd.merge(df_ParCorr_sum,
                                  pd.concat(list_of_series, axis=1),
                                  left_index=True, right_index=True)
        return self.df_ParCorr_sum



    def store_df_PCMCI(self):
        import wrapper_PCMCI
        if self.tfreq != self.precur_aggr:
            path = self.path_outsub2 + f'_dtd{self.precur_aggr}'
        else:
            path = self.path_outsub2
        wrapper_PCMCI.store_ts(self.df_data, self.df_links, self.dict_ds,
                               path+'.h5')
        self.path_df_data = path+'.h5'

    def store_df(self, filename: str=None, append_str: str=None):
        if self.list_for_MI is not None:
            varstr = '_'.join([p.name for p in self.list_for_MI])
        else:
            varstr = ''
        if hasattr(self, 'df_data_ext'):
            varstr = '_'.join([n[0] for n in self.list_import_ts]) + varstr
        if filename is None:
            filename = os.path.join(self.path_outsub1,
                                f'{get_timestr()}_df_data_{varstr}_'
                                f'dt{self.precur_aggr}_tf{self.tfreq}_{self.hash}')
            if append_str is not None:
                filename += '_'+append_str
        functions_pp.store_hdf_df({'df_data':self.df_data}, filename+'.h5')
        print('Data stored in \n{}'.format(filename+'.h5'))
        self.path_df_data = filename

    def get_clust(self, name_ds='ts'):
        f = functions_pp
        self.df_clust, self.ds = f.nc_xr_ts_to_df(self.list_of_name_path[0][1],
                                                  name_ds=name_ds)

    def apply_df_ana_plot(self, df=None, name_ds='ts', func=None, kwrgs_func={},
                          colwrap=2):
        if df is None:
            self.get_clust(name_ds=name_ds)
            df = self.df_clust
        if func is None:
            func = df_ana.plot_ac ; kwrgs_func = {'AUC_cutoff':(14,30),'s':60}
        return df_ana.loop_df(df, function=func, sharex=False,
                             colwrap=colwrap, hspace=.5, kwrgs=kwrgs_func)

    def plot_df_clust(self, save=False):
        self.get_clust()
        plot_maps.plot_labels(self.ds['xrclustered'])
        if save and hasattr(self, 'path_sub1'):
            fig_path = os.path.join(self.path_outsub1, 'RV_clusters')
            plt.savefig(fig_path+self.figext, bbox_inches='tight')

    def quick_view_labels(self, var=None, mean=True, save=False,
                          kwrgs_plot: dict={}, min_detect_gc: float=.5,
                          append_str: str=None, region_labels=None,
                          replacement_labels=None):
        '''
        Parameters
        ----------
        var : TYPE, optional
            DESCRIPTION. The default is None.
        map_proj : TYPE, optional
            DESCRIPTION. The default is None.
        median : TYPE, optional
            DESCRIPTION. The default is True.
        save : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        if type(var) is str:
            var = [var]
        if var is None:
            var = [p.name for p in self.list_for_MI]
        for precur_name in var:
            try:
                pclass = [p for p in self.list_for_MI if p.name == precur_name][0]
            except IndexError as e:
                print(e)
                print('var not in list_for_MI')
            if hasattr(pclass, 'prec_labels')==False:
                continue
            if region_labels is not None:
                f = find_precursors.view_or_replace_labels
                prec_labels = f(pclass.prec_labels.copy(),
                   region_labels,
                   replacement_labels)
            else:
                prec_labels = pclass.prec_labels.copy()

            if mean:
                n_splits = self.df_splits.index.levels[0].size
                min_d = round(n_splits * (1- min_detect_gc),0)
                # 1 == non-significant, 0 == significant
                mask = (~np.isnan(pclass.prec_labels)).sum(dim='split') > min_d
                prec_labels = prec_labels.mean(dim='split')
                if min_detect_gc<.1 or min_detect_gc>1.:
                    raise ValueError( 'give value between .1 en 1.0')
                prec_labels = prec_labels.where(mask)
                cbar_vert = -0.1
            else:
                prec_labels = prec_labels
                if prec_labels.split.size == 1:
                    cbar_vert = -0.1
                else:
                    cbar_vert = -0.025
            if all(np.isnan(prec_labels.values.flatten()))==False:
                # colors of cmap are dived over min to max in n_steps.
                # We need to make sure that the maximum value in all dimensions will be
                # used for each plot (otherwise it assign inconsistent colors)
                max_N_regs = min(20, int(prec_labels.max() + 0.5))
                label_weak = np.nan_to_num(prec_labels.values) >=  max_N_regs
                xrmask = None
                prec_labels.values[label_weak] = max_N_regs
                steps = max_N_regs+1
                cmap = plt.cm.tab20
                prec_labels.values = prec_labels.values-0.5
                clevels = np.linspace(0, max_N_regs,steps)

                kwrgs = {'row_dim':'split', 'col_dim':'lag', 'hspace':-0.35,
                              'size':3, 'cbar_vert':cbar_vert, 'clevels':clevels,
                              'subtitles' : None,
                              'cticks_center':True,
                              'cmap':cmap}
                kwrgs.update(kwrgs_plot)

                plot_maps.plot_corr_maps(prec_labels,
                                 xrmask,
                                 **kwrgs)
                if save == True:
                    if replacement_labels is not None:
                        r = ''.join(np.array(replacement_labels, dtype=str))
                    else:
                        r = ''
                    f_name = 'clusterlabels{}_{}_eps{}_mingc{}'.format(r,
                                                        pclass._name,
                                                        pclass.distance_eps,
                                                        pclass.min_area_in_degrees2)
                    if append_str is not None:
                        f_name += f'_{append_str}'
                    fig_path = os.path.join(self.path_outsub1, f_name)+self.figext
                    plt.savefig(fig_path, bbox_inches='tight')
            else:
                print(f'no {pclass.name} regions that pass distance_eps and min_area_in_degrees2 citeria')


    def plot_maps_corr(self, var=None, plotlags: list=None, kwrgs_plot: dict={},
                       mean: bool=True, min_detect_gc: float=.5,
                       mask_xr=None, region_labels: Union[int,list]=None,
                       save: bool=False, append_str: str=None):

        if type(var) is str:
            var = [var]
        if var is None:
            var = [p.name for p in self.list_for_MI]
        for precur_name in var:
            try:
                pclass = [p for p in self.list_for_MI if p.name == precur_name][0]
            except IndexError as e:
                print(e, '\nvar not in list_for_MI')
            if plotlags is None:
                plotlags = pclass.corr_xr.lag.values
            if region_labels is not None and mask_xr is None:
                f = find_precursors.view_or_replace_labels
                mask_xr = np.isnan(f(pclass.prec_labels.copy(), region_labels))
            if mask_xr is not None:
                xrmask = (pclass.corr_xr['mask'] + mask_xr).astype(bool)
            else:
                xrmask = pclass.corr_xr['mask']
            if mean==False:
                xrvals = pclass.corr_xr.sel(lag=plotlags)
                xrmask = xrmask.sel(lag=plotlags)
            else:
                xrvals = pclass.corr_xr.mean(dim='split').sel(lag=plotlags)
                if min_detect_gc<.1 or min_detect_gc>1.:
                    raise ValueError( 'give value between .1 en 1.0')
                n_splits = self.df_splits.index.levels[0].size
                min_d = round(n_splits * (1- min_detect_gc),0)
                # 1 == non-significant, 0 == significant
                xrmask = xrmask.sel(lag=plotlags).sum(dim='split') > min_d


            plot_maps.plot_corr_maps(xrvals,
                                     mask_xr=xrmask, **kwrgs_plot)
            if save == True:
                if append_str is not None:
                    f_name = '{}_a{}'.format(pclass._name,
                                              pclass.alpha)+'_'+append_str
                else:
                    f_name = '{}_a{}'.format(precur_name,
                                             pclass.alpha)
                if mean:
                    f_name += f'_md{min_detect_gc}'

                fig_path = os.path.join(self.path_outsub1, f_name)+self.figext
                plt.savefig(fig_path, bbox_inches='tight')

    def plot_maps_sum(self, var='all', figpath=None, paramsstr=None,
                      cols: List=['corr', 'C.D.'], save: bool=False,
                      kwrgs_plot={}):

        if figpath is None:
            figpath = self.path_outsub1
        if paramsstr is None:
            paramsstr = self.params_str
        if cols != ['corr', 'C.D.']:
            paramsstr = cols[0] +'_'+paramsstr
        if var == 'all':
            dict_ds = self.dict_ds
        else:
            dict_ds = {f'{var}':self.dict_ds[var]} # plot single var
        plot_maps.plot_labels_vars_splits(dict_ds, self.df_links,
                                          figpath, paramsstr, self.TV.name,
                                          save=self.save, cols=cols,
                                          kwrgs_plot=kwrgs_plot)


        plot_maps.plot_corr_vars_splits(dict_ds, self.df_links,
                                          figpath, paramsstr, self.TV.name,
                                          save=self.save, cols=cols,
                                          kwrgs_plot=kwrgs_plot)

    def _replace_RV_mask(self, df_data=None, replace_RV_mask=False, plot=False):
        if df_data is None:
            df_data = self.df_data
        if type(replace_RV_mask) is tuple:
            print(f'replacing RV_mask for dates {replace_RV_mask}')
            orig = pd.to_datetime(self.fullts.time.values)
            newdates = functions_pp.core_pp.get_subdates(orig,
                                                         replace_RV_mask)
            replace_RV_mask = [True if d in newdates else False for d in orig]

        n_splits = df_data.index.levels[0].size
        new = pd.DataFrame(data=(np.array([replace_RV_mask]*n_splits)).flatten(),
                           index=df_data.index, columns=['RV_mask'])
        df_data['RV_mask'] = new
        if plot:
            df_data['RV_mask'].loc[0].astype(int).plot()
        return df_data

    def _get_testyrs(self, df_splits=None):
    #%%
        if df_splits is None:
            df_splits = self.df_splits
        return functions_pp.get_testyrs(df_splits)

    def transform_df_data(self, df_data: pd.DataFrame=None, transformer=None,
                          keys : list=None):
        if df_data is None:
            df_data = self.df_data.copy()
        if transformer is None:
            transformer = fc_utils.standardize_on_train_and_RV
        if keys is None:
            keys = [k for k in df_data.columns if k not in ['RV_mask', 'TrainIsTrue']]
        df_trans = np.zeros( self.n_spl, dtype=object)
        for s in range(self.n_spl):
            df_s = df_data.loc[s]
            if transformer.__name__ == 'standardize_on_train_and_RV':
                df_trans[s] = df_s[keys].apply(fc_utils.standardize_on_train_and_RV,
                              args=[df_s[['RV_mask', 'TrainIsTrue']], 0])

            else:
                df_trans[s] = df_s[keys].apply(transformer,
                                          args=[df_s['TrainIsTrue']])
        return pd.concat(df_trans, keys=range(self.n_spl))

    def fit_df_data_ridge(self,
                          df_data: pd.DataFrame=None,
                          keys: Union[list, np.ndarray]=None,
                          target: Union[str,pd.DataFrame]=None,
                          tau_min: int=1,
                          tau_max: int=1,
                          match_lag_region_to_lag_fc=False,
                          transformer=None,
                          fcmodel=None,
                          kwrgs_model: dict={'scoring':'neg_mean_squared_error'}):
        '''
        Perform cross-validated Ridge regression to predict the target.

        Parameters
        ----------
        keys : Union[list, np.ndarray]
            list of nparray of predictors that you want to merge into one.
        target : str, optional
            target timeseries to predict. The default target is the RV.
        tau_max : int, optional
            prediction is repeated at lags 0 to tau_max. It might be that the
            signal stronger at a certain lag. Relationship is established for
            all lags in range(tau_min, tau_max+1) and the strongest is
            relationship is kept.
        fcmodel : function of stat_models, None
            Give function that satisfies stat_models format of I/O, default
            Ridge regression
        kwrgs_model : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        predict (DataFrame), weights (DataFrame), models_lags (dict).

        '''
        # df_data=None;keys=None;target=None;tau_min=1;tau_max=1;transformer=None
        # kwrgs_model={'scoring':'neg_mean_squared_error'}
        if df_data is None:
            df_data = self.df_data.copy()
        lags = range(tau_min, tau_max+1)
        if keys is None:
            keys = df_data.columns[df_data.dtypes != bool]
        splits = df_data.index.levels[0]
        # data_new_s   = np.zeros( (splits.size) , dtype=object)

        RV_mask = df_data.loc[0]['RV_mask'] # not changing
        if target is None: # not changing
            target_ts = df_data.loc[0].iloc[:,[0]][RV_mask]


        models_lags = dict()

        for il, lag in enumerate(lags):
            preds = np.zeros( (splits.size), dtype=object)
            wghts = np.zeros( (splits.size) , dtype=object)
            models_splits_lags = dict()
            for isp, s in enumerate(splits):
                fit_masks = df_data.loc[s][['RV_mask', 'TrainIsTrue']]
                TrainIsTrue = df_data.loc[s]['TrainIsTrue']

                df_s = df_data.loc[s]
                if type(keys) is dict:
                    _ks = keys[s]
                    _ks = [k for k in _ks if k in df_s.columns] # keys split
                else:
                    _ks = [k for k in keys if k in df_s.columns] # keys split

                if match_lag_region_to_lag_fc:
                    ks = [k for k in _ks if k.split('..')[0] == str(lag)]
                    l = lag ; valid = len(ks) !=0 and ~df_s[ks].isna().values.all()
                    while valid == False:
                        ks = [k for k in _ks if k.split('..')[0] == str(l)]
                        if len(ks) !=0 and ~df_s[ks].isna().values.all():
                            valid = True
                        else:
                            l -= 1
                        print(f"\rNot found lag {lag}, using lag {l}", end="")
                        assert l > 0, 'ts @ lag not found or nans'
                else:
                    ks = _ks

                if transformer is not None and transformer != False:
                    df_trans = df_s[ks].apply(transformer,
                                            args=[TrainIsTrue])
                elif transformer == False:
                    df_trans = df_s[ks] # no transformation
                else: # transform to standard normal
                    df_trans = df_s[ks].apply(fc_utils.standardize_on_train_and_RV,
                                              args=[fit_masks, lag])
                                            # result_type='broadcast')

                if type(target) is str:
                    target_ts = df_data.loc[s][[target]][RV_mask]
                elif type(target) is pd.DataFrame:
                    target_ts = target


                df_norm = df_trans.merge(fc_utils.apply_shift_lag(fit_masks, lag),
                                          left_index=True,
                                          right_index=True)


                if fcmodel is None:
                    fcmodel = sm.ScikitModel()
                pred, model = fcmodel.fit_wrapper({'ts':target_ts},
                                                  df_norm, ks, kwrgs_model)

                # if len(lags) > 1:
                models_splits_lags[f'split_{s}'] = model


                if il == 0:#  and isp == 0:
                # add truth
                    prediction = target_ts.copy()
                    prediction = prediction.merge(pred.rename(columns={0:lag}),
                                                  left_index=True,
                                                  right_index=True)
                else:
                    prediction = pred.rename(columns={0:lag})

                coeff = fc_utils.SciKitModel_coeff(model, lag)

                preds[isp] = prediction
                wghts[isp] = coeff
            if il == 0:
                predict = pd.concat(list(preds), keys=splits)
                weights = pd.concat(list(wghts), keys=splits)
            else:
                predict = predict.merge(pd.concat(list(preds), keys=splits),
                              left_index=True, right_index=True)
                weights = weights.merge(pd.concat(list(wghts), keys=splits),
                              left_index=True, right_index=True)
            models_lags[f'lag_{lag}'] = models_splits_lags

        # weights_norm = weights.mean(axis=0, level=1)
        # weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')
        return predict, weights, models_lags



def RV_and_traintest(fullts, TV_ts, traintestgroups, method=str, kwrgs_events=None,
                     precursor_ts=None, seed: int=1, verbosity=1):
    # fullts = rg.fullts ; TV_ts = rg.TV_ts
    # method='random_10'; kwrgs_events=None; precursor_ts=None; seed=1; verbosity=1

    # Define traintest:
    df_fullts = pd.DataFrame(fullts.values,
                            index=pd.to_datetime(fullts.time.values),
                            columns=[fullts.name])
    df_RV_ts = pd.DataFrame(TV_ts.values,
                            index=pd.to_datetime(TV_ts.time.values),
                            columns=['RV'+fullts.name])

    if precursor_ts is not None:
        path_data = ''.join(precursor_ts[0][1])
        df_ext = functions_pp.load_hdf5(path_data)['df_data'].loc[:,:]
        if 'TrainIsTrue' in df_ext.columns:
            print('Retrieve same train test split as imported ts')
            method = 'from_import' ; seed = ''

    if method[:9] == 'ran_strat' and kwrgs_events is None and method != 'from_import':
            # events need to be defined to enable stratified traintest.
            kwrgs_events = {'event_percentile': 66,
                            'min_dur' : 1,
                            'max_break' : 0,
                            'grouped' : False,
                            'window':'mean'}
            if verbosity == 1:
                print("kwrgs_events not given, creating stratified traintest split "
                    "based on events defined as exceeding the {}th percentile".format(
                        kwrgs_events['event_percentile']))

    TV = RV_class(df_fullts, df_RV_ts, kwrgs_events)
    # TV.traintestgroups = traintestgroups


    if method == 'from_import':
        df_splits = functions_pp.load_hdf5(path_data)['df_data'].loc[:,['TrainIsTrue', 'RV_mask']]
        test_yrs_imp  = functions_pp.get_testyrs(df_splits)
        df_splits = functions_pp.rand_traintest_years(TV,
                                                      test_yrs=test_yrs_imp,
                                                      method=method,
                                                      seed=seed,
                                                      kwrgs_events=kwrgs_events,
                                                      verb=verbosity)
        test_yrs_set  = functions_pp.get_testyrs(df_splits)
        assert (np.equal(test_yrs_imp, test_yrs_set)).all(), "Train test split not equal"

    if method != 'from_import':
        df_splits = functions_pp.cross_validation(df_RV_ts,
                                                  traintestgroups=traintestgroups,
                                                  method=method,
                                                  seed=seed)
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
