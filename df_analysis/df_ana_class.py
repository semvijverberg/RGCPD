# -*- coding: utf-8 -*-
import sys, os, inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
subdates_dir = os.path.join(main_dir, 'RGCPD/')
fc_dir = os.path.join(main_dir, 'forecasting/')

if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(subdates_dir)
    sys.path.append(fc_dir)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.plotting.matplotlib.register_converters = True # apply custom format as well to matplotlib
plt.style.use('seaborn')
import scipy as sp
from scipy import stats
import seaborn as sns
import matplotlib as mpl
from matplotlib import cycler
import xarray
try:
    import mtspec
except:
    print('could not import mtspec')
from typing import List, Tuple, Union
import warnings
import tables
import h5py
# from forecasting import *
from statsmodels.tsa import stattools 
from core_pp import get_subdates


class DataFrameAnalysis:
    def __init__(self):
        self.keys = None
        self.time_steps = None 
        self.methods = dict() 
        self.period = None
        self.window_size = 0 
        self.threshold_bin = 0
        self.target_variable = None
        self.file_path = ""
        self.selection_year = 0
        self.alpha = 0.01
        self.max_lag = None
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    def __str__(self):
        return f'{self.__class__.__name__}{self.__dict__}'

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__dict__!r}'

    @staticmethod
    def autocorrelation_stats_meth(time_serie, max_lag=None, alpha=0.01):
        " Autocorrelation for 1D-arrays"
        if max_lag == None :
            max_lag = time_serie.size
        return stattools.acf(time_serie.values, nlags=max_lag - 1, unbiased=False, alpha=alpha, fft=True)
    
    def __get_keys(self, data_frame, keys):
        if keys == None:
            # retrieve only float series
            keys = self.keys
            type_check = np.logical_or(data_frame.dtypes == 'float',data_frame.dtypes == 'float32')
            keys = type_check[type_check].index
        return keys

    
    def loop_df_ana(self, df, function, keys=None, to_np=False, kwargs=None):
        # Should be analysis from any function which return non-tuple like results
        keys = self.__get_keys(df, keys)
        if to_np:
            output = df.apply(function, raw=True, **kwargs)
        output = df.apply(function, **kwargs)
        return pd.DataFrame(output, columns=keys)
    
    def loop_df(self, df, functions, args=None, kwargs=None, keys=None):
        # TODO Test this functionality
        # Should be analysis from any function with methods from functions that might return tuples
        # method should be dict with labels as function name and values as function calls
        # keys = self.__get_keys(df, keys)
        # df = df.loc[:,keys]
        # print(type(functions), functions)
        # # if not isinstance(functions, list) or not isinstance(functions, dict) or isinstance(functions, property):
        # return self.apply_concat_series(df, functions, arg=args)
        raise NotImplementedError
    
    def subset_pdseries(self, df_serie, time_steps=None, 
                   select_years: Union[int,list]=None):
        # if isinstance(df_serie.index, pd.core.indexes.datetimes.DatetimeIndex):
        #     date_time = df_serie.index 
        if time_steps == None:
            _, conf_intval = self.apply_concat_series(df_serie, self.autocorrelation_stats_meth)
            conf_low = [np.where(conf_intval[i][:, 0] < 0)[0] for i in range(len(conf_intval))]
            numb_times = [[] for _ in range(len(conf_low))]
            for i in range(len(conf_low)):
                for idx in conf_low[i]:
                    numb_times[i].append(idx + 1 - conf_low[i][0])
            cut_off = [conf_low[i][np.where(np.array(numb_times[i]) == 1)][0] for i in range(len(conf_low))]
            time_steps = [20 * i for i in cut_off]
            list_pdseries = [df_serie.iloc[:s,i] for i,s in enumerate(time_steps)]
        else:
            time_steps = [time_steps] * df_serie.columns.size
            list_pdseries = [df_serie.iloc[:s,i] for i,s in enumerate(time_steps)]

        if select_years != None:
            if not isinstance(select_years, list):
                select_years = [select_years]
            date_time = self.get_one_year(df_serie.index, *select_years)
            list_pdseries = [df_serie.loc[date_time,c] for c in df_serie.columns]
        return list_pdseries

    def spectrum(self, y, methods):
        # TODO Fix the try except logic here, freq_dframe gets changed twice.
        try:
            freq_dframe = None
            one_year_size = None 
            if methods == None:
                methods = {'periodogram': self.periodogram, 'mtspec':self.multi_tape_spectr}
            
            assert isinstance(methods, dict), "Methods needs to be dict datatype"
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                freq_dframe = (y.index[1] - y.index[0]).days

                one_year_size = self.get_one_year(y.index).size
                xlim = (freq_dframe, one_year_size * freq_dframe)
            else:
                freq_dframe = 1
                raise ValueError('Data not Dataframe object or pandas Series')
        except AssertionError as as_err:
            print(as_err)
        except ValueError as v_err:
            print(v_err)
        finally:
            xlim = (freq_dframe, 365)
            results = {}
            for label, func in methods.items():
                freq, spec =  self.apply_concat_series(y, func=func, arg=1)
                period = [1 * freq_dframe / freq[i][1:] for i in range(len(freq))]
                spec = [ spec[i][1:] for i in range(len(spec))]
                results[label] = [period, spec]          
        return [results, xlim]

    def accuracy(self, y, sample='auto', auc_cutoff=None):
        acc, conf_intval = self.autocorrelation_stats_meth(y)
        cut_off  = 0
        if sample =='auto':
            val_below_zero = np.where(conf_intval[:, 0] < 0)[0]
            numb_of_occurence  = np.array([idx + 1 - val_below_zero[0] for idx in val_below_zero])
            cut_off = int(val_below_zero[np.where(numb_of_occurence == 1)[0][0]])
            sample = 2 * cut_off
        else:
            sample = 5
        if auc_cutoff == None or isinstance(auc_cutoff, int):
            auc_cutoff = cut_off
            auc = np.trapz(acc[:auc_cutoff], x=range(auc_cutoff))
            text = f'AUC {auc} range lag {auc_cutoff}'
        else:
            auc = np.trapz(acc[auc_cutoff[0]:auc_cutoff[1]], x=range(auc_cutoff[0], auc_cutoff[1]))
            text = f'AUC {auc} range lag {auc_cutoff[0]}-{auc_cutoff[1]}'
        return [auc, acc,  auc_cutoff, sample, conf_intval, text,((y.index[1] - y.index[0]).days, 'time [days]') if isinstance(y.index , pd.core.indexes.datetimes.DatetimeIndex) else (1,'timesteps')]

    def resample(self,df, window_size=20, lag=0, columns=list):

        splits = df.index.levels[0]
        df_train_is_true = df['TrainIsTrue']
        test_list = [ df[columns].loc[split][df_train_is_true[split] == False] for split in range(splits.size) ]

        df_test = pd.concat(test_list).sort_index()
        for pre_cursor in df_test.columns[1:]:
            df_test[pre_cursor] = df_test[pre_cursor].shift(periods=- lag)
        
        return df_test.resample(f'{window_size}D').mean()

    def select_period(self, df, targ_var_mask, start_date, end_date, start_end_year, leap_year, rename=False):
        
        dates_full_origin = df.loc[0].index 
        dates_target_var_origin = df.loc[0].index[df.loc[0]['RV_mask'] == True ]
        df_resample  = self.resample(df=df)
        df_period  = get_subdates(dates_target_var_origin, start_date, end_date, start_end_year, leap_year)

        if rename:
             df_period = df_period.rename(rename, axis= 1)
             return df_period
        
        return df_period

    def apply_concat_dFrame(self, df, field, func, col_name):
        return pd.concat((df, df[field].apply(lambda cell : pd.Series(func(cell), index=col_name))), axis=1)

    def apply_concat_series(self, series, func, arg=None):
        return zip(*series.apply(func, args=(arg,)))

    def multi_tape_spectr(self, time_serie, sampling_period=1, band_width=4, numb_tapers=4):
       spectrum, frequence, _, _, _ =  mtspec.mtspec(data=time_serie, delta=sampling_period, 
                                    time_bandwidth=band_width, number_of_tapers=numb_tapers, statistics=True)
       return [frequence, spectrum]

    def periodogram(self, time_serie, sampling_period=1):
        frequence = np.fft.rfftfreq(len(time_serie))
        spectrum = 2 * np.abs(np.fft.rfft(time_serie))**2 / len(time_serie)
        return [frequence, spectrum]

    def fft_np(self, data, sampling_period=1.0):
        yfft = sp.fftpack.fft(data)
        ypsd = np.abs(yfft)**2
        ypsd = 2.0 / len(data) * ypsd
        fft_frequence = sp.fftpack.fftfreq(len(ypsd), sampling_period)
        return [fft_frequence, ypsd]
        
    def cross_corr_p_val(self, data, alpha=0.05):
        def pearson_pval(x, y):
            return stats.pearsonr(x, y)[1]

        if isinstance(data, pd.DataFrame):
            pval_matrix = data.corr(method=pearson_pval).to_numpy()
            sig_mask = pval_matrix < alpha
            cross_cor = data.corr(method='pearson')
            return [cross_cor, sig_mask, pval_matrix]
        else:
            raise ValueError('Please provide correct datatype', sys.exc_info())

    def get_one_year(self, pd_date_time, *args):
        pd_date_time = pd.to_datetime(pd_date_time)
        first_year =  pd_date_time.year[0]
        if len(args) != 0:
            dates = [pd_date_time.where(pd_date_time.year == arg).dropna() for arg in args]
            return  pd.to_datetime(self.flatten(dates))
        else:
            return pd_date_time.where(pd_date_time.year == first_year).dropna() 

    def remove_leap_period(self, date_time_or_xr):
        no_leap_month = None
        mask = None
        try:
            if pd.to_datetime(date_time_or_xr, format='%Y-%b-%d', errors='coerce'):
                date_time = pd.to_datetime(date_time_or_xr.time.values)
                mask = np.logical_and((date_time.month == 2), (date_time.day == 29))
                no_leap_month = date_time[mask==False]
            else: 
                raise ValueError('Not dataframe datatype')
            
        except ValueError as v_err_1:
            try:
                if isinstance(date_time_or_xr, xarray):
                    mask = np.logical_and((date_time_or_xr.month == 2), (date_time_or_xr.day == 29))
                    no_leap_month = date_time_or_xr[mask==False]
                    no_leap_month = date_time_or_xr.sel(time=no_leap_month)
                else:
                    raise ValueError('Not xarray datatype')
            except ValueError as v_err_2:
                print("Twice ValueError generated ", sys.exc_info(), v_err_1, v_err_2)
                sys.exit(1)
        return no_leap_month


    def load_hdf(self, file_path):
        hdf = h5py.File(file_path, 'r+')
        dict_of_dfs = {k:pd.read_hdf(file_path, k) for k in hdf.keys()}
        hdf.close()
        return dict_of_dfs

    def save_hdf(self, dict_of_dfs, file_path):
        with pd.HDFStore(file_path, 'w') as hdf :
            for k, items in dict_of_dfs.items():
                hdf.put(k, items,  format='table', data_columns=True)
        return 

class VisualizeAnalysis:
    # TODO untangle the inheritance and make two stand alone classes to avoid inheritance cluster fuck.
    def __init__(self, col_wrap=3, sharex='col', sharey='row'):
        self.nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
        self.colors_nice = cycler('color',
                        self.nice_colors)
        self.colors_datasets = sns.color_palette('deep')

        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
            axisbelow=True, grid=True, prop_cycle=self.colors_nice)
        plt.rc('grid', color='w', linestyle='solid')
        plt.rc('xtick', direction='out', color='black')
        plt.rc('ytick', direction='out', color='black')
        plt.rc('patch', edgecolor='#E6E6E6')
        plt.rc('lines', linewidth=2)

        mpl.rcParams['figure.figsize'] = [7.0, 5.0]
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 600

        mpl.rcParams['font.size'] = 13
        mpl.rcParams['legend.fontsize'] = 'medium'
        mpl.rcParams['figure.titlesize'] = 'medium'
        self.col_wrap = col_wrap
        self.gridspec_kw = {'wspace':0.5, 'hspace':0.4}
        self.sharex = sharex
        self.sharey = sharey

    def __str__(self):
        return f'{self.__class__.__name__}{self.__dict__}'

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__dict__!r}'

    def _column_wrap(self, df):
        if df.columns.size % self.col_wrap == 0:
            return int(df.columns.size / self.col_wrap)
        return int(df.columns.size / self.col_wrap) + 1

    def _subplots_func_adjustment(self, col=None):
        if col == None:
            return plt.subplots(constrained_layout=True)
        else:
            if col % self.col_wrap == 0:
                return plt.subplots(int(col/self.col_wrap),self.col_wrap, 
                                    constrained_layout=True)
            else:
                return plt.subplots(int(col/self.col_wrap) + 1, self.col_wrap, 
                                    constrained_layout=True)

    def subplots_fig_settings(self, df):
        row = self._column_wrap(df)
        return plt.subplots(row, self.col_wrap, sharex=self.sharex, 
        sharey=self.sharey, figsize= (3* self.col_wrap, 2.5* row), gridspec_kw=self.gridspec_kw, constrained_layout=True)

    # def plot(self, df):
    #     titles = list(df.columns)
    #     # fig.suptitle(str(function))
    #     df.plot(subplots=True, title=titles)
    #     plt.show()

    def accuracy(self, values, title):
        auc, aut_corr, auc_cutoffs, sample_cutoff, conf_intval, text, time_freq = values
        fig, ax = self._subplots_func_adjustment()
        x_labels = [ i * time_freq[0] for i in range(sample_cutoff)]

        # High confindence interval
        high_conf = [min(1, h) for h in conf_intval[:, 1][:sample_cutoff] ]
        # Low confidennce interval
        low_conf = [min(1, l) for l in conf_intval[:, 0][:sample_cutoff]]
        numb_labels = max(1, int(sample_cutoff / 5))

        ax.plot(x_labels, high_conf, color='orange')
        ax.plot(x_labels, low_conf, color='orange')
        ax.plot(x_labels, aut_corr[:sample_cutoff])
        ax.scatter(x_labels, aut_corr[:sample_cutoff])

        ax.hlines(y= 0, xmin=min(x_labels), xmax=max(x_labels))
        ax.text(0.99, 0.99, text, transform=ax.transAxes, horizontalalignment='right', fontdict={'fontsize':8})
        ax.set_xticks(x_labels[::numb_labels])
        ax.set_xticklabels(x_labels[::numb_labels], fontsize=10)
        ax.set_xlabel(time_freq[1], fontsize=10)

        if title:
            ax.set_title(title, fontsize=10)
        plt.show()

    def vis_timeseries(self, list_pdseries, s=0):
        fig, axes = self._subplots_func_adjustment(len(list_pdseries))
        for i, pdseries in enumerate(list_pdseries):
            ax = axes.flatten()[i]
            ax.plot(pdseries.index, pdseries)
            every_nth = round(len( ax.xaxis.get_ticklabels() )/ 3)
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0 :
                    label.set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_title(pdseries.name, fontsize=10)
        plt.show()
    
    def scatter(self, df, target_var, aggr, title):
        fig, ax = self._subplots_func_adjustment()

        if aggr == 'annual':
            df_gr = df.groupby(df.index.year).mean()
            target_var_gr = target_var.grouby(target_var.index.year).mean()
            ax.scatter(df_gr, target_var_gr)
        else:
            ax.scatter(df, target_var)
        if title:
            ax.set_title(title, fontsize=10)
        plt.show()
    
    def time_serie_matrix(self, df_period, cross_corr, sig_mask, pval):
        fig, ax = self._subplots_func_adjustment()
        plt.figure(figsize=(10, 10))

        # Generate mask for upper triangle matrix 
        mask_triangle = np.zeros_like(cross_corr, dtype=bool)
        mask_triangle[np.triu_indices_from(mask_triangle)] = True
        mask_signal = mask_triangle.copy()
        mask_signal[sig_mask == False] = True

        # Removing meaningless row and column 
        cross_corr = cross_corr.columns.drop(cross_corr.columns[0], axis=0).drop(cross_corr.columns[-1], axis=1)
        mask_signal = mask_signal[1: , :-1]
        mask_triangle = mask_triangle[1: , :-1]

        # Custom cmap for corr matrix plot
        cust_cmap = sns.diverging_palette(220, 10, n=9, l=30, as_cmap=True)

        signf_labels = self.significance_annotation(cross_corr, mask_signal)

        ax = sns.heatmap(cross_corr, ax=ax, mask=mask_triangle, cmap=cust_cmap, vmax=1, center=0,
        square=True, linewidths=.5, cbar_kws={'shrink': .8}, annot=signf_labels, annot_kws={'size':30}, cbar=False, fmt='s')

        ax.tick_params(axis='both', labelsize=15, bottom=True, top=False, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.set_xticklabels(cross_corr.columns, fontdict={'fontweight': 'bold',
                                                        'fontsize': 25})
        ax.set_yticklabels(cross_corr.index, fontdict={'fontweight':'bold',
                                                    'fontsize':25}, rotation=0)

        plt.show()

    def spectrum(self, title,subtitle, results, xlim, ylim=(1e-4,1e3)):
        fig, ax = self._subplots_func_adjustment(col=len(subtitle))
        
        # TODO Better way to plot all results tuples  instead of double for-loops
        counter = len(results)
        label = list(results.keys())
        for _,values in results.items():
            for idx in range(len(subtitle)):
                ax[idx].plot(values[0][idx], values[1][idx], ls='-', c=self.nice_colors[counter], label=label)
                ax[idx].set_title(subtitle[idx])
                ax[idx].loglog()
                ax[idx].set_ylim(ylim)
                loc_maj = mpl.ticker.LogLocator(base=10.0, numticks=int(-1E-99 + xlim[-1]/100) + 1)
                loc_min = mpl.ticker.LogLocator(base=10.0,subs=tuple(np.arange(0,1,0.1)[1:]),numticks=int(-1E-99+xlim[-1]/100) + 1)
                ax[idx].xaxis.set_major_locator(loc_maj)
                ax[idx].xaxis.set_minor_locator(loc_min)
                ax[idx].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
                ax[idx].set_xlim(xlim)
                ax[idx].legend(loc=0, fontsize='small')
                counter -=  1
        if title is not None:
            fig.suptitle(title, fontsize=10)
        return fig, ax
           
    def significance_annotation(self, corr, pvals):
        corr_str = np.zeros_like( corr, dtype=str ).tolist()
        for i1, r in enumerate(corr.values):
            for i2, c in enumerate(r):
                if pvals[i1, i2] <= 0.05 and pvals[i1, i2] > 0.01:
                    corr_str[i1][i2] = '{:.2f}*'.format(c)
                if pvals[i1, i2] <= 0.01:
                    corr_str[i1][i2]= '{:.2f}**'.format(c)
                elif pvals[i1, i2] > 0.05:
                    corr_str[i1][i2]= '{:.2f}'.format(c)
        return np.array(corr_str)


class DFA(DataFrameAnalysis, VisualizeAnalysis):
    
    def __init__(self, df=pd.DataFrame):
        DataFrameAnalysis.__init__(self)
        VisualizeAnalysis.__init__(self)
        self.df = df
        self.index = self.df.index
        self.multiindex = hasattr(self.df.index, 'levels')
        
    def plot_timeseries(self, s=0, cols: list=None):
        '''
        Moet ook nog plot timeseries different traintest
        moet alle argumenten definieren

        Parameters
        ----------
        s : TYPE, optional
            DESCRIPTION. The default is 0.
        cols : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if self.multiindex:
            df = self.df.loc[s]
        if cols is None:
            cols = df.columns[df.dtypes != bool]
        list_pdseries = self.subset_pdseries(df[cols])
        self.vis_timeseries(list_pdseries)


if __name__ == "__main__":
        
    # df = DFA(df=rg.df_data)
    # df.dataframe(df.df)
    # df_ana = DataFrameAnalysis()
    # df_vis = VisualizeAnalysis()
    # print(repr(df_ana))
    # print(repr(df_vis))
    pass