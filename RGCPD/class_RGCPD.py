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
from class_RV import RV_class
from class_EOF import EOF
from class_BivariateMI import BivariateMI
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
fc_dir = os.path.join(main_dir, 'forecasting/')

if fc_dir not in sys.path:
    sys.path.append(fc_dir)
from class_fc import apply_shift_lag

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
import func_models
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
                 lags_i: np.ndarray=np.array([0]),
                 path_outmain: str=None,
                 append_pathsub='',
                 verbosity: int=1):
        '''
        Class to study teleconnections of a Response Variable* of interest.

        Methods to extract teleconnections/precursors:
            - BivariateMI (now only supporting correlation maps)
            - EOF analysis

        BivariateMI (MI = Mutual Information) is class which allows for a
        statistical test in the form:
        MI(lon,lat) = for gc in map: func(x(t), y(t)),
        where map is a (time,lon,lat) map and gc stands for each gridcell/coordinate
        in that map. The y is always the same 1-dimensional timeseries of interest
        (i.e. the Response Variable). At this point, only supports the
        correlation analysis. Once the significance is attributed, it is stored
        in the MI map. Precursor regions are found by clustering the
        significantly (correlating) gridcells (+ and - regions are separated)
        and extract their spatial mean timeseries.

        *Sometimes Response Variable is also called Target Variable.

        Parameters
        ----------
        list_of_name_path : list, optional
            list of (name, path) tuples defining the input data (.nc).

            Convention: first entry should be (name, path) of target variable (TV).
            e.g. list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]

            'prec_name' is a string/key that can be chosen freely, does not have to refer
            to the variable in the .nc file.
            Each prec_path .nc file should contain only a single variable
            of format (time, lat, lon).
            'TVname' should refer the name you have given the timeseries on
            the dimesion 'cluster', i.e. xrTV.sel(cluster=TVname)
            The TVpath Netcdf file assumes that it contains a xr.DataArray
            called xrclustered containing a spatial map with clustered regions.
            It must contain an xr.DataArray which contains timeseries for each
            cluster. See RGCPD.pp_TV().

        list_for_EOFS : list, optional
            list of EOF classes, see docs EOF?
        list_import_ts : list, optional
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
            The default is np.array([0]).
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
            list_of_name_path = [(3,
                                  os.path.join(path_test, 'tf5_nc5_dendo_80d77.nc')),
                                 ('sst_test',
                                  os.path.join(path_test, 'sst_1979-2018_2.5deg_Pacific.nc'))]

        if start_end_TVdate is None:
            start_end_TVdate = ('06-01', '08-31')



        if path_outmain is None:
            user_download_path = get_download_path()
            path_outmain = user_download_path + '/output_RGCPD/'
        if os.path.isdir(path_outmain) != True : os.makedirs(path_outmain)

        self.list_of_name_path = list_of_name_path
        self.list_for_EOFS = list_for_EOFS
        self.list_for_MI = list_for_MI
        self.list_import_ts = list_import_ts

        self.start_end_TVdate   = start_end_TVdate
        self.start_end_date     = start_end_date
        self.start_end_year     = start_end_year


        self.verbosity          = verbosity
        self.tfreq              = tfreq
        self.lags_i             = lags_i
        self.lags               = np.array([l*self.tfreq for l in self.lags_i], dtype=int)
        self.path_outmain       = path_outmain
        self.append_pathsub     = append_pathsub
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


        self.kwrgs_load = dict(loadleap=loadleap, seldates=seldates,
                               selbox=selbox, format_lon=format_lon)
        self.kwrgs_pp = self.kwrgs_load.copy()
        self.kwrgs_pp.update(dict(detrend=detrend, anomaly=anomaly))

        self.kwrgs_load.update(dict(start_end_date=self.start_end_date,
                                    start_end_year=self.start_end_year,
                                    closed_on_date=self.start_end_TVdate[-1],
                                    tfreq=self.tfreq))

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

    def plot_df_clust(self, save=False):
        self.get_clust()
        plot_maps.plot_labels(self.ds['xrclustered'])
        if save and hasattr(self, 'path_sub1'):
            fig_path = os.path.join(self.path_outsub1, 'RV_clusters')
            plt.savefig(fig_path+self.figext, bbox_inches='tight')

    def pp_TV(self, name_ds='ts', loadleap=False, detrend=False, anomaly=False):
        self.name_TVds = name_ds
        self.RV_anomaly = anomaly
        self.RV_detrend = detrend
        f = functions_pp
        self.fulltso, self.hash = f.load_TV(self.list_of_name_path,
                                            loadleap=loadleap,
                                            name_ds=self.name_TVds)
        out = f.process_TV(self.fulltso,
                            tfreq=self.tfreq,
                            start_end_TVdate=self.start_end_TVdate,
                            start_end_date=self.start_end_date,
                            start_end_year=self.start_end_year,
                            RV_detrend=self.RV_detrend,
                            RV_anomaly=self.RV_anomaly)
        self.fullts, self.TV_ts, inf, start_end_TVdate = out

        self.input_freq = inf
        self.dates_or  = pd.to_datetime(self.fulltso.time.values)
        self.dates_all = pd.to_datetime(self.fullts.time.values)
        self.dates_TV = pd.to_datetime(self.TV_ts.time.values)
        # Store added information in RV class to the exp dictionary
        # if self.start_end_date is None and self.input_freq == 'annual':
        #     self.start_end_date = self.start_end_TVdate
        if self.start_end_date is None and self.input_freq != 'annual':
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
        if self.input_freq == 'daily' or self.input_freq == 'annual':
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


    def traintest(self, method: str=None, seed=1,
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

        if method is None:
            method = 'no_train_test_split'
        self.kwrgs_TV = dict(method=method,
                    seed=seed,
                    kwrgs_events=kwrgs_events,
                    precursor_ts=self.list_import_ts)

        TV, self.df_splits = RV_and_traintest(self.fullts,
                                              self.TV_ts,
                                              verbosity=self.verbosity,
                                              **self.kwrgs_TV)
        self.TV = TV
        self.path_outsub1 = self.path_outsub0 + '_'.join(['', self.TV.method \
                            + 's'+ str(self.TV.seed)])
        if os.path.isdir(self.path_outsub1) == False : os.makedirs(self.path_outsub1)




    def calc_corr_maps(self, var: Union[str, list]=None):

        if var is None:
            if type(var) is str:
                var = [var]
            var = [MI.name for MI in self.list_for_MI]
        kwrgs_load = self.kwrgs_load
        # self.list_for_MI = []
        for precur in self.list_for_MI:
            if precur.name in var:
                precur.filepath = [l for l in self.list_precur_pp if l[0]==precur.name][0][1]
                precur.lags = self.lags
                if hasattr(precur, 'selbox'):
                    kwrgs_load['selbox'] = precur.selbox
                find_precursors.calculate_region_maps(precur,
                                                      self.TV,
                                                      self.df_splits,
                                                      kwrgs_load)

            # self.list_for_MI.append(precur)

    def cluster_list_MI(self, var: Union[str, list]=None):
        if var is None:
            if type(var) is str:
                var = [var]
            var = [MI.name for MI in self.list_for_MI]
        for precur in self.list_for_MI:
            if precur.name in var:
                precur = find_precursors.cluster_DBSCAN_regions(precur)


    def get_EOFs(self):
        for i, e_class in enumerate(self.list_for_EOFS):
            print(f'Retrieving {e_class.neofs} EOF(s) for {e_class.name}')
            filepath = [l for l in self.list_precur_pp if l[0]==e_class.name][0][1]
            e_class.get_pattern(filepath=filepath, df_splits=self.df_splits)

    def plot_EOFs(self, mean=True, kwrgs: dict=None):
        for i, e_class in enumerate(self.list_for_EOFS):
            print(f'Retrieving {e_class.neofs} EOF(s) for {e_class.name}')
            e_class.plot_eofs(mean=mean, kwrgs=kwrgs)

    def get_ts_prec(self, precur_aggr=None, keys_ext=None):
        if precur_aggr is None:
            self.precur_aggr = self.tfreq
        else:
            self.precur_aggr = precur_aggr

        if precur_aggr is not None:
            # retrieving timeseries at different aggregation, TV and df_splits
            # need to redefined on new tfreq using the same arguments
            print(f'redefine target variable on {self.precur_aggr} day means')
            f = functions_pp
            self.fullts, self.TV_ts = f.process_TV(self.fulltso,
                                                tfreq=self.precur_aggr,
                                                start_end_TVdate=self.start_end_TVdate,
                                                start_end_date=self.start_end_date,
                                                start_end_year=self.start_end_year,
                                                RV_detrend=self.RV_detrend,
                                                RV_anomaly=self.RV_anomaly)[:2]
            TV, df_splits = RV_and_traintest(self.fullts,
                                             self.TV_ts, **self.kwrgs_TV)
        else:
            # use original TV timeseries
            TV = self.TV ; df_splits = self.df_splits
        self.df_data = pd.DataFrame(TV.fullts.values, columns=[TV.name],
                                    index=TV.dates_all)
        self.df_data = pd.concat([self.df_data]*self.df_splits.index.levels[0].size,
                                 keys=self.df_splits.index.levels[0])
        if self.list_for_MI is not None:
            print('\nGetting MI timeseries')
            for i, precur in enumerate(self.list_for_MI):
                if hasattr(precur, 'prec_labels'):
                    precur.get_prec_ts(precur_aggr=precur_aggr,
                                   kwrgs_load=self.kwrgs_load)
                else:
                    print(f'{precur.name} not clustered yet')
            df_data_MI = find_precursors.df_data_prec_regs(self.list_for_MI,
                                                             TV,
                                                             df_splits)
            self.df_data = self.df_data.merge(df_data_MI, left_index=True, right_index=True)

        # Append (or only load in) external timeseries
        if self.list_import_ts is not None:
            print('\nGetting external timeseries')
            f = find_precursors
            self.df_data_ext = f.import_precur_ts(self.list_import_ts,
                                                  df_splits,
                                                  self.start_end_date,
                                                  self.start_end_year,
                                                  cols=keys_ext,
                                                  precur_aggr=self.precur_aggr,
                                                  start_end_TVdate=self.start_end_TVdate)
            self.df_data = self.df_data.merge(self.df_data_ext, left_index=True, right_index=True)


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


    def PCMCI_init(self, keys: list=None):
        if keys is None:
            keys = self.df_data.columns
        else:
            keys.append('TrainIsTrue') ; keys.append('RV_mask')
        self.pcmci_dict = wPCMCI.init_pcmci(self.df_data[keys])

    def PCMCI_df_data(self, keys: list=None, path_txtoutput=None,
                      tau_min=0, tau_max=1, pc_alpha=None,
                      max_conds_dim=None, max_combinations=2,
                      max_conds_py=None, max_conds_px=None,
                      replace_RV_mask: np.ndarray=None,
                      verbosity=4):

        if max_conds_dim is None:
            max_conds_dim = self.df_data.columns.size - 2 # -2 for bool masks

        self.kwrgs_pcmci = dict(tau_min=tau_min,
                           tau_max=tau_max,
                           pc_alpha=pc_alpha,
                           max_conds_dim=max_conds_dim,
                           max_combinations=max_combinations,
                           max_conds_py=max_conds_py,
                           max_conds_px=max_conds_px,
                           verbosity=4)

        if path_txtoutput is None:
            self.params_str = '{}_tau_{}-{}_conds_dim{}_combin{}_dt{}'.format(
                          pc_alpha, tau_min, tau_max,
                          max_conds_dim, max_combinations, self.precur_aggr)
            self.path_outsub2 = os.path.join(self.path_outsub1, self.params_str)
        else:
            self.path_outsub2 = path_txtoutput

        if os.path.isdir(self.path_outsub2) == False : os.makedirs(self.path_outsub2)

        if keys is None:
            keys = self.df_data.columns

        df_data = self.df_data.copy()
        if type(replace_RV_mask) is np.ndarray:
            print('replacing RV_mask')
            new = pd.DataFrame(data=(np.array([replace_RV_mask]*10)).flatten(),
                               index=df_data.index, columns=['RV_mask'])
            df_data['RV_mask'] = new
            df_data['RV_mask'].loc[0].astype(int).plot()

        self.pcmci_dict = wPCMCI.init_pcmci(df_data[keys])

        out = wPCMCI.loop_train_test(self.pcmci_dict, self.path_outsub2,
                                                          **self.kwrgs_pcmci)
        self.pcmci_results_dict = out

    def PCMCI_get_links(self, var: str=None, alpha_level: float=.05):


        if hasattr(self, 'pcmci_results_dict')==False:
            print('first perform PCMCI_df_data to get pcmci_results_dict')
        if var is None:
            var = self.TV.name

        self.parents_dict = wPCMCI.get_links_pcmci(self.pcmci_dict,
                                                   self.pcmci_results_dict,
                                                   alpha_level)
        self.df_links = wPCMCI.get_df_links(self.parents_dict, variable=var)
        lags = np.arange(self.kwrgs_pcmci['tau_min'], self.kwrgs_pcmci['tau_max']+1)
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
        fig = plt.figure(figsize=figshape, facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        tp.plot_graph(val_matrix=val_plot,
                      var_names=var_names,
                      link_width=weights,
                      link_matrix=links_plot,
                      fig_ax=(fig, ax),
                      **kwrgs)
        f_name = f'CEN_{variable}_s{s}'
        if append_figpath is not None:
            fig_path = os.path.join(self.path_outsub1, f_name+append_figpath)
        else:
            fig_path = os.path.join(self.path_outsub1, f_name)
        fig.savefig(fig_path+self.figext, bbox_inches='tight')
        plt.show()

    def PCMCI_get_ParCorr_from_txt(self, variable=None, pc_alpha='auto'):

        if variable is None:
            variable = self.TV.name

        # lags = range(0, self..kwrgs_pcmci['tau_max']+1)
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

    def store_df(self, append_str: str=None):
        if self.list_for_MI is not None:
            varstr = '_'.join([p.name for p in self.list_for_MI])
        else:
            varstr = ''
        if hasattr(self, 'df_data_ext'):
            varstr = '_'.join([n[0] for n in self.list_import_ts]) + varstr
        filename = os.path.join(self.path_outsub1,
                                f'{get_timestr()}_df_data_{varstr}_'
                                f'dt{self.precur_aggr}_tf{self.tfreq}_{self.hash}')
        if append_str is not None:
            filename += '_'+append_str
        functions_pp.store_hdf_df({'df_data':self.df_data}, filename+'.h5')
        print('Data stored in \n{}'.format(filename+'.h5'))
        self.path_df_data = filename

    def quick_view_labels(self, var=None, map_proj=None, median=True,
                          save=False):
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
                precur = [p for p in self.list_for_MI if p.name == precur_name][0]
            except IndexError as e:
                print(e)
                print('var not in list_for_MI')
            prec_labels = precur.prec_labels.copy()
            if median:
                prec_labels = prec_labels.median(dim='split')
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

                if median==False:
                    if prec_labels.split.size == 1:
                        cbar_vert = -0.1
                    else:
                        cbar_vert = -0.025
                else:
                    cbar_vert = -0.1

                kwrgs = {'row_dim':'split', 'col_dim':'lag', 'hspace':-0.35,
                              'size':3, 'cbar_vert':cbar_vert, 'clevels':clevels,
                              'subtitles' : None, 'lat_labels':True,
                              'cticks_center':True,
                              'cmap':cmap}

                plot_maps.plot_corr_maps(prec_labels,
                                 contour_mask,
                                 map_proj, **kwrgs)
                if save == True:
                    f_name = 'cluster_labels_{}'.format(precur_name)
                    fig_path = os.path.join(self.path_outsub1, f_name)+self.figext
                    plt.savefig(fig_path, bbox_inches='tight')
            else:
                print(f'no {precur.name} regions that pass distance_eps and min_area_in_degrees2 citeria')


    def plot_maps_corr(self, var=None, mean=True, mask_xr=None, map_proj=None,
                       row_dim='split', col_dim='lag', clim='relaxed',
                       hspace=-0.6, wspace=.02, size=2.5, cbar_vert=-0.01, units='units',
                       cmap=None, clevels=None, cticks_center=None, drawbox=None,
                       title=None, subtitles=None, zoomregion=None, lat_labels=True,
                       aspect=None, n_xticks=5, n_yticks=3,
                       x_ticks: np.ndarray=None, y_ticks: np.ndarray=None,
                       save=False,
                       append_str: str=None):

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
            if mean:
                xrvals = pclass.corr_xr.mean(dim='split')
                xrmask = pclass.corr_xr['mask'].mean(dim='split')
            else:
                xrvals = pclass.corr_xr
                xrmask = pclass.corr_xr['mask']
            plot_maps.plot_corr_maps(xrvals,
                                     mask_xr=xrmask, map_proj=map_proj,
                                    row_dim=row_dim, col_dim=col_dim, clim=clim,
                                    hspace=hspace, wspace=wspace, size=size, cbar_vert=cbar_vert,
                                    units=units, cmap=cmap, clevels=clevels,
                                    cticks_center=cticks_center, drawbox=drawbox,
                                    title=None, subtitles=subtitles,
                                    zoomregion=zoomregion,
                                    lat_labels=lat_labels, aspect=aspect,
                                    n_xticks=n_xticks, n_yticks=n_yticks,
                                    x_ticks=x_ticks, y_ticks=y_ticks)
            if save == True:
                if append_str is not None:
                    f_name = 'corr_map_{}'.format(precur_name)+'_'+append_str
                else:
                    f_name = 'corr_map_{}'.format(precur_name)

                fig_path = os.path.join(self.path_outsub1, f_name)+self.figext
                plt.savefig(fig_path, bbox_inches='tight')

    def plot_maps_sum(self, var='all', map_proj=None, figpath=None,
                      paramsstr=None, cols: List=['corr', 'C.D.'], kwrgs_plot={}):

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
        plot_maps.plot_labels_vars_splits(dict_ds, self.df_links, map_proj,
                                          figpath, paramsstr, self.TV.name,
                                          cols=cols, kwrgs_plot=kwrgs_plot)


        plot_maps.plot_corr_vars_splits(dict_ds, self.df_links, map_proj,
                                          figpath, paramsstr, self.TV.name,
                                          cols=cols, kwrgs_plot=kwrgs_plot)


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

    def fit_df_data_ridge(self, keys: Union[list, np.ndarray],
                             target: Union[str,pd.DataFrame]=None,
                             tau_min: int=1,
                             tau_max: int=3,
                             newname:str = None, transformer=None,
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
        newname : str, optional
            new column name of the predicted timeseries. The default is None.
        kwrgs_model : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        # self.df_data_all = self.df_data.copy()
        lags = range(tau_min, tau_max+1)
        if keys is None:
            keys = self.df_data.columns[self.df_data.dtypes != bool]
        splits = self.df_data.index.levels[0]
        # data_new_s   = np.zeros( (splits.size) , dtype=object)

        RV_mask = self.df_data.loc[0]['RV_mask'] # not changing
        if target is None: # not changing
            target_ts = self.df_data.loc[0].iloc[:,[0]][RV_mask]

        preds = np.zeros( (splits.size), dtype=object)
        wghts = np.zeros( (splits.size) , dtype=object)
        for isp, s in enumerate(splits):
            fit_masks = self.df_data.loc[s][['RV_mask', 'TrainIsTrue']]
            TrainIsTrue = self.df_data.loc[s]['TrainIsTrue']

            df_s = self.df_data.loc[s]
            ks = [k for k in keys if k in df_s.columns] # keys split

            if transformer is not None:
                df_trans = df_s[ks].apply(transformer,
                                        args=[TrainIsTrue],
                                        result_type='broadcast')
            else:
                df_trans = df_s[ks] # no transformation

            if type(target) is str:
                target_ts = self.df_data.loc[s][target][RV_mask]
            elif type(target) is pd.DataFrame:
                target_ts = target

            # make prediction for each lag
            for il, lag in enumerate(lags):
                df_train = df_trans.merge(apply_shift_lag(fit_masks, lag),
                                          left_index=True,
                                          right_index=True)



                pred, model = sm.ridgeCV({'ts':target_ts},
                                               df_train, ks, kwrgs_model)
                if il == 0:
                    # add truth
                    prediction = target_ts.copy()
                    prediction = prediction.merge(pred.rename(columns={0:lag}),
                                                  left_index=True,
                                                  right_index=True)
                    coeff = pd.DataFrame(model.coef_, index=model.X_pred.columns,
                                         columns=[lag])
                else:
                    prediction = prediction.merge(pred.rename(columns={0:lag}),
                                     left_index=True,
                                     right_index=True)
                    coeff = coeff.merge(pd.DataFrame(model.coef_,
                                                     index=model.X_pred.columns,
                                                     columns=[lag]),
                                         left_index=True,
                                         right_index=True)

            preds[isp] = prediction
            wghts[isp] = coeff

        predict = pd.concat(list(preds), keys=splits)
        weights = pd.concat(list(wghts), keys=splits)
        weights_norm = weights.mean(axis=0, level=1)
        weights_norm.div(weights_norm.max(axis=0)).T.plot()
        return predict, weights, model



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
                            'grouped' : False,
                            'window':'mean'}
            if verbosity == 1:
                print("kwrgs_events not given, creating stratified traintest split "
                    "based on events defined as exceeding the {}th percentile".format(
                        kwrgs_events['event_percentile']))

    TV = RV_class(df_fullts, df_RV_ts, kwrgs_events)


    if precursor_ts is not None:
        path_data = ''.join(precursor_ts[0][1])
        df_ext = functions_pp.load_hdf5(path_data)['df_data'].loc[:,:]
        if 'TrainIsTrue' in df_ext.columns:
            print('Retrieve same train test split as imported ts')
            method = 'from_import' ; seed = ''

            df_splits = functions_pp.load_hdf5(path_data)['df_data'].loc[:,['TrainIsTrue', 'RV_mask']]
            test_yrs_imp  = functions_pp.get_testyrs(df_splits)
            df_splits = functions_pp.rand_traintest_years(TV, test_yrs=test_yrs_imp,
                                                            method=method,
                                                            seed=seed,
                                                            kwrgs_events=kwrgs_events,
                                                            verb=verbosity)

            test_yrs_set  = functions_pp.get_testyrs(df_splits)
            assert (np.equal(test_yrs_imp, test_yrs_set)).all(), "Train test split not equal"
    if method != 'from_import':
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
