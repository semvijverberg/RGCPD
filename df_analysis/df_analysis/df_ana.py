#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import matplotlib.dates as mdates
# import mtspec
flatten = lambda l: [item for sublist in l for item in sublist]
from typing import List, Tuple, Union

from functions_pp import time_mean_bins, get_oneyr



import matplotlib as mpl
from matplotlib import cycler
nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
colors_nice = cycler('color',
                nice_colors)
colors_datasets = sns.color_palette('deep')

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors_nice)
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


def loop_df_ana(df, function, keys=None, to_np=False, kwrgs=None):
    if keys is None:
        # retrieve only float series
        type_check = np.logical_or(df.dtypes == 'float',df.dtypes == 'float32')
        keys = type_check[type_check].index
    df = df.loc[:,keys]
    out_list = []
    for header in df:
        if to_np == False:
            y = df[header]
        elif to_np == True:
            y = df[header].values
        out_i = function(y, **kwrgs)
        out_list.append(out_i)
    return pd.DataFrame(np.array(out_list).T, index=df.index, columns=keys)


def loop_df(df: pd.DataFrame(), function=None, keys=None, colwrap=3, sharex='col',
            sharey='row', hspace=.4, figsize: tuple=None, kwrgs=None):
    #%%
    assert type(df) == type(pd.DataFrame()), ('df should be DataFrame, '
                'not pd.Series or ndarray')

    if function is None:
        function = plot_timeseries

    if keys is None:
        # retrieve only float series
        type_check = np.logical_or(df.dtypes == 'float',
                                   df.dtypes == 'float32')

        keys = type_check[type_check].index

    df = df.loc[:,keys]

    if (df.columns.size) % colwrap == 0:
        rows = int(df.columns.size / colwrap)
    elif (df.columns.size) % colwrap != 0:
        rows = int(df.columns.size / colwrap) + 1

    if figsize is None:
        figsize = (3*colwrap,rows*2.5)
    gridspec_kw = {'hspace':hspace}
    fig, ax = plt.subplots(rows, colwrap, sharex=sharex, sharey=sharey,
                           figsize = figsize,
                           gridspec_kw=gridspec_kw)

    for i, ax in enumerate(fig.axes):
        if i >= df.columns.size:
            ax.axis('off')
        else:
            header = df.columns[i]

            y = df[header]
            if kwrgs is None:
                kwrgs = {'title': header,
                         'ax'   : ax}
            else:
                kwrgs['title'] = header
                kwrgs['ax'] = ax

            function(y, **kwrgs)
    return
    #%%

def plot_df(df: pd.DataFrame(), function=None, keys=None, title: str=None,
            figsize: tuple=None, kwrgs=None):

    assert type(df) == type(pd.DataFrame()), ('df should be DataFrame, '
                'not pd.Series or ndarray')

    if function is None:
        function = plot_timeseries

    if keys is None:
        # retrieve only float series
        type_check = np.logical_or(df.dtypes == 'float',
                                   df.dtypes == 'float32')

        keys = type_check[type_check].index

    df = df.loc[:,keys]

    # if (df.columns.size) % colwrap == 0:
    #     rows = int(df.columns.size / colwrap)
    # elif (df.columns.size) % colwrap != 0:
    #     rows = int(df.columns.size / colwrap) + 1

    if figsize is None:
        figsize = (10, 5)
    fig, ax = plt.subplots(1, 1, figsize = figsize)

    for i, key in enumerate(keys):

        y = df[key]
        if kwrgs is None:
            if title is None:
                title = key
            kwrgs = {'title': title,
                     'ax'   : ax}
        else:
            kwrgs['title'] = key
            kwrgs['ax'] = ax

        function(y, **kwrgs)
    return fig, ax

def autocorr_sm(ts, max_lag=None, alpha=0.01):
    from statsmodels.tsa import stattools
    if max_lag == None:
        max_lag = ts.size
    ac, con_int = stattools.acf(ts.values, nlags=max_lag-1,
                                unbiased=False, alpha=0.01,
                                 fft=True)
    return ac, con_int

def plot_ac(y=pd.Series, s='auto', title=None, AUC_cutoff=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if hasattr(y.index,'levels'):
        y = y.loc[0]
    ac, con_int = autocorr_sm(y)

    time = y.index
    if type(time) is pd.core.indexes.datetimes.DatetimeIndex:
        tfreq = (time[1] - time[0]).days
        xaxlabel = 'time [days]'
    else:
        xaxlabel = 'timesteps'
        tfreq = 1

    # auto xlabels
    if s=='auto':
        where = np.where(con_int[:,0] < 0 )[0]
        # has to be below 0 for n times (not necessarily consecutive):
        n = 1
        n_of_times = np.array([idx+1 - where[0] for idx in where])
        try:
            cutoff = int(where[np.where(n_of_times == n)[0][0] ])
        except:
            cutoff = tfreq * 20

        s = 2*cutoff
    else:
        cutoff = int(s/2)
        s = s
    if AUC_cutoff != False:
        if AUC_cutoff is None:
            AUC_cutoff = cutoff
        if type(AUC_cutoff) is int:
            AUC = np.trapz(ac[:AUC_cutoff], x=range(AUC_cutoff))
            text = 'AUC {:.2f} up to lag {}'.format(AUC, AUC_cutoff)
        elif type(AUC_cutoff) is tuple:
            AUC = np.trapz(ac[AUC_cutoff[0]:AUC_cutoff[1]],
                           x=range(AUC_cutoff[0], AUC_cutoff[1]))
            text = 'AUC {:.2f} range lag {}-{}'.format(AUC, AUC_cutoff[0],
                                                       AUC_cutoff[1])
        ax.text(0.99, 0.90,
        text,
        transform=ax.transAxes, horizontalalignment='right',
        fontdict={'fontsize':8})

    xlabels = [x * tfreq for x in range(s)]
    # con high
    high = [ min(1,h) for h in con_int[:,1][:s]]
    ax.plot(xlabels, high, color='orange')
    # con low
    ax.plot(xlabels, con_int[:,0][:s], color='orange')
    # ac values
    ax.plot(xlabels,ac[:s])
    ax.scatter(xlabels,ac[:s])
    ax.hlines(y=0, xmin=min(xlabels), xmax=max(xlabels))


    xlabels = [x * tfreq for x in range(s)]
    n_labels = max(1, int(s / 5))
    ax.set_xticks(xlabels[::n_labels])
    ax.set_xticklabels(xlabels[::n_labels])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel(xaxlabel, fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=10)
    return ax

def plot_timeseries(y, timesteps: list=None,
                    selyears: Union[list, int]=None, title=None,
                    legend: bool=True, nth_xyear: int=10, ax=None):
    # ax=None
    #%%


    if hasattr(y.index,'levels'):
        y_ac = y.loc[0]
    else:
        y_ac = y

    if type(y_ac.index) == pd.core.indexes.datetimes.DatetimeIndex:
        datetimes = y_ac.index

    if timesteps is None and selyears is None:
        ac, con_int = autocorr_sm(y_ac)
        where = np.where(con_int[:,0] < 0 )[0]
        # has to be below 0 for n times (not necessarily consecutive):
        n = 1
        n_of_times = np.array([idx+1 - where[0] for idx in where])
        if n_of_times.size != 0:
            cutoff = where[np.where(n_of_times == n)[0][0] ]
        else:
            cutoff = 100

        timesteps = min(y_ac.index.size, 10*cutoff)
        datetimes = y_ac.iloc[:timesteps].index

    if selyears is not None and timesteps is None:
        if type(selyears) is int:
            selyears = [selyears]
        datetimes = get_oneyr(y.index, *selyears)

    if timesteps is not None and selyears is None:
        datetimes = datetimes[:timesteps]

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if hasattr(y.index,'levels'):
        for fold in y.index.levels[0]:
            if legend:
                label = f'f {fold+1}' ; color = None ; alpha=.5
            else:
                label = None ; color = 'red' ; alpha=.1
            ax.plot(datetimes, y.loc[fold, datetimes], alpha=alpha,
                    label=label, color=color)
        if legend:
            ax.legend(prop={'size':6})
    else:
        ax.plot(datetimes, y.loc[datetimes])

    if nth_xyear is None:
        nth_xtick = round(len(ax.xaxis.get_ticklabels())/5)
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % nth_xtick != 0:
                label.set_visible(False)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1)) # set tick every year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # format %Y
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % nth_xyear != 0:
                label.set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=8)
    if title is not None:
        ax.set_title(title, fontsize=10)
    #%%
    return ax

def plot_scatter(y, tv=pd.Series, aggr=None, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if aggr == 'annual':
        y_gr = y.groupby(y.index.year).mean()
        tv_gr  = tv.groupby(y.index.year).mean()
    else:
        y_gr = y
        tv_gr = tv
    ax.scatter(y_gr, tv_gr)
    if title is not None:
        ax.set_title(title, fontsize=10)
    return ax

def mtspectrum(ts, d=1., tb=4, nt=4):
    import mtspec
    """ multi-taper spectrum

    input:
    ts .. time series
    d  .. sampling period
    tb .. time bounds (bandwidth)
    nt .. number of tapers
    """
    spec, freq, jackknife, _, _ = mtspec.mtspec(
                data=ts, delta=d, time_bandwidth=tb,
                number_of_tapers=nt, statistics=True)
    return freq, spec

def periodogram(ts, d=1.):
    """ naive absolute squared Fourier components

    input:
    ts .. time series
    d  .. sampling period
    """
    freq = np.fft.rfftfreq(len(ts))
    Pxx = 2*np.abs(np.fft.rfft(ts))**2/len(ts)
    return freq, Pxx

def fft_np(y, sampling_period=1.):
    yfft = sp.fftpack.fft(y)
    ypsd = np.abs(yfft)**2
    ypsd = 2.0/len(y) * ypsd
    fftfreq = sp.fftpack.fftfreq(len(ypsd), sampling_period)
    return fftfreq, ypsd

def plot_spectrum(y, methods: List[tuple]=[('periodogram', periodogram)],
                  vlines=None, y_lim=None,
                  year_max=.5, title=None, ax=None):


    # ax=None ;
    if hasattr(y.index,'levels'):
        y = y.loc[0]

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if methods is None:
        methods = [('periodogram', periodogram)]


    try:
        freq_df = (y.index[1] - y.index[0]).days
        if freq_df in [28, 29, 30, 31]:
            freq_df = 'month'
        elif type(freq_df) == int:
            freq_df = int(365 / freq_df)

    except:
        freq_df = 1

    def freq_to_period(xfreq, freq_df):
        if freq_df == 'month':
            periods = 1/(xfreq * 12)
        elif type(freq_df) is int:
            periods = 1/(xfreq*freq_df)
        return np.round(periods, 3)

    def period_to_fred(periods, freq_df):
        if freq_df == 'month':
            freq = 1 / (periods * 12)
        else:
            freq = 1 / (periods * freq_df)
        return np.round(freq, 1)


    for i, method in enumerate(methods):
        label, func_ = method
        freq, spec = func_(y)
        _periods = freq_to_period(freq[1:], freq_df)
        idx = int(np.argwhere(_periods-year_max ==min(abs(_periods - year_max)))[0])
        periods = _periods[:idx+1]
        ax.plot(periods, spec[1:idx+2], ls='-', c=nice_colors[i], label=label)
        ax.set_xscale('log')
        ax.set_xticks(periods[np.logical_or(periods%2 == 0, periods==1)])
        ax.set_xticklabels(np.array(periods[np.logical_or(periods%2 == 0, periods==1)], dtype=int))
        ax.set_xlim((periods[0], periods[-1]))
        ax.set_xlabel('Periods [years]', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

        ax2 = ax.twiny()
        ax2.plot(periods, spec[1:idx+2], ls='-', c=nice_colors[i], label=label)
        ax2.set_xscale('log')
        ax2.set_xticks(periods[:][np.logical_or(periods%2 == 0, periods==1)])
        ax2.set_xticklabels(np.round(freq[1:idx+2][np.logical_or(periods%2 == 0, periods==1)], 3))
        ax2.set_xlim((periods[0], periods[-1]))

        ax2.tick_params(axis='both', labelsize=8)
        ax.set_xlabel('Periods [years]', fontsize=8)
        if freq_df == 'month':
            ax2.set_xlabel('Frequency [1/months]', fontsize=8)
        else:
            ax2.set_xlabel(f'Frequency [1/ {freq_df} days]', fontsize=6)
        # ax.set_ylabel('Power Spectrum Density', fontsize=8)

    ax.legend(fontsize='xx-small')
    if title is not None:
        ax.set_title(title, fontsize=9)
    return ax

def resample(df, to_freq='M'):
    return df.resample(to_freq).mean()

def corr_matrix_pval(df, alpha=0.05):
    from scipy import stats
    if type(df) == type(pd.DataFrame()):
        cross_corr = np.zeros( (df.columns.size, df.columns.size) )
        pval_matrix = np.zeros_like(cross_corr)
        for i1, col1 in enumerate(df.columns):
            for i2, col2 in enumerate(df.columns):
                pval = stats.pearsonr(df[col1].values, df[col2].values)
                pval_matrix[i1, i2] = pval[-1]
                cross_corr[i1, i2]  = pval[0]
        # recreate pandas cross corr
        cross_corr = pd.DataFrame(data=cross_corr, columns=df.columns,
                                  index=df.columns)
    sig_mask = pval_matrix < alpha
    return cross_corr, sig_mask, pval_matrix

def plot_ts_matric(df_init, win: int=None, lag=0, columns: list=None, rename: dict=None,
                   period='fullyear', plot_sign_stars=True, fontsizescaler=0):
    #%%
    '''
    period = ['fullyear', 'summer60days', 'pre60days']
    '''
    if columns is None:
        columns = list(df_init.columns[(df_init.dtypes != bool).values])


    df_cols = df_init[columns]


    if hasattr(df_init.index, 'levels'):
        splits = df_init.index.levels[0]
        print('extracting RV dates from test set')
        dates_RV_orig   = df_init.loc[0].index[df_init.loc[0]['RV_mask']==True]
        TrainIsTrue = df_init['TrainIsTrue']
        dates_full_orig = df_init.loc[0].index
        list_test = []
        for s in range(splits.size):
            TestIsTrue = TrainIsTrue[s]==False
            list_test.append(df_cols.loc[s][TestIsTrue])
        df_test = pd.concat(list_test).sort_index()
    else:
        df_test = df_init
        dates_full_orig = df_init.index

    if lag != 0:
        # shift precursor vs. tmax
        for c in df_test.columns[1:]:
            df_test[c] = df_test[c].shift(periods=-lag)

    # bin means
    if win is not None:
        oneyr = get_oneyr(df_test.index)
        start_end_date = (f'{oneyr[0].month:02d}-{oneyr[0].day:02d}',
                          f'{oneyr[-1].month:02d}-{oneyr[-1].day:02d}')
        df_test = time_mean_bins(df_test, win, start_end_date=start_end_date)[0]


    if period=='fullyear':
        dates_sel = dates_full_orig.strftime('%Y-%m-%d')
    if 'RV_mask' in df_init.columns:
        if period == 'RV_mask':
            dates_sel = dates_RV_orig.strftime('%Y-%m-%d')
        elif period == 'RM_mask_lag60':
            dates_sel = (dates_RV_orig - pd.Timedelta(60, unit='d')).strftime('%Y-%m-%d')

    # after resampling, not all dates are in their:
    dates_sel =  pd.to_datetime([d for d in dates_sel if d in df_test.index] )
    df_period = df_test.loc[dates_sel, :].dropna()

    if rename is not None:
        df_period = df_period.rename(rename, axis=1)

    corr, sig_mask, pvals = corr_matrix_pval(df_period, alpha=0.01)

    # Generate a mask for the upper triangle
    mask_tri = np.zeros_like(corr, dtype=np.bool)
    mask_tri[np.triu_indices_from(mask_tri)] = True
    mask_sig = mask_tri.copy()
    mask_sig[sig_mask==False] = True

    # removing meaningless row and column
    cols = corr.columns
    corr = corr.drop(cols[0], axis=0).drop(cols[-1], axis=1)
    mask_sig = mask_sig[1:, :-1]
    mask_tri = mask_tri[1:, :-1]
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, n=9, l=30, as_cmap=True)

    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1E99, center=0,
                square=True, linewidths=.5,
                 annot=False, annot_kws={'size':30+fontsizescaler}, cbar=False)

    if plot_sign_stars:
        sig_bold_labels = sig_bold_annot(corr, mask_sig)
    else:
        sig_bold_labels = corr.round(2).astype(str).values
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                 annot=sig_bold_labels, annot_kws={'size':30+fontsizescaler}, cbar=False, fmt='s')

    ax.tick_params(axis='both', labelsize=15+fontsizescaler,
                   bottom=True, top=False, left=True, right=False,
                   labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    ax.set_xticklabels(corr.columns, fontdict={'fontweight':'bold',
                                               'fontsize':20+fontsizescaler})
    ax.set_yticklabels(corr.index, fontdict={'fontweight':'bold',
                                               'fontsize':20+fontsizescaler}, rotation=0)
    #%%
    return

def sig_bold_annot(corr, pvals):
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


def get_oneyr(pddatetime, *args):
    if hasattr(pddatetime,'levels'):
        pddatetime = pddatetime.levels[1]
    dates = []
    pddatetime = pd.to_datetime(pddatetime)
    year = pddatetime.year[0]

    for arg in args:
        year = arg
        dates.append(pddatetime.where(pddatetime.year==year).dropna())
    dates = pd.to_datetime(flatten(dates))
    if len(dates) == 0:
        dates = pddatetime.where(pddatetime.year==year).dropna()
    return dates

def remove_leapdays(datetime_or_xr):
   if type(datetime_or_xr) != type(pd.to_datetime(['2000-01-01'])):
       datetime = pd.to_datetime(datetime_or_xr.time.values)
   else:
       datetime = datetime_or_xr
   mask_lpyrfeb = np.logical_and((datetime.month == 2), (datetime.day == 29))
   noleap = datetime[mask_lpyrfeb==False]
   if type(datetime_or_xr) != type(pd.to_datetime(['2000-01-01'])):
       noleap = datetime_or_xr.sel(time=noleap)

   return noleap

def store_hdf_df(dict_of_dfs, file_path):
    import warnings
    import tables

    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    with pd.HDFStore(file_path, 'w') as hdf:
        for key, item in  dict_of_dfs.items():
            hdf.put(key, item, format='table', data_columns=True)
        hdf.close()
    return

def load_hdf5(path_data):
    import h5py, time
    attempt = 'Fail'
    c = 0
    while attempt =='Fail':
        c += 1
        try:
            hdf = h5py.File(path_data,'r+')
            dict_of_dfs = {}
            for k in hdf.keys():
                dict_of_dfs[k] = pd.read_hdf(path_data, k)
            hdf.close()
            attempt = 1
        except:
            time.sleep(1)
        assert c!= 5, print('loading in hdf5 failed')

    return dict_of_dfs






pi = np.pi


def square_function(N, square_width):
    """Generate a square signal.

    Args:
        N (int): Total number of points in the signal.
        square_width (int): Number of "high" points.

    Returns (ndarray):
        A square signal which looks like this:

              |____________________
              |<-- square_width -->
              |                    ______________
              |
              |^                   ^            ^
        index |0             square_width      N-1

        In other words, the output has [0:N]=1 and [N:]=0.
    """
    signal = np.zeros(N)
    signal[0:square_width] = 1
    return signal


def check_num_coefficients_ok(N, num_coefficients):
    """Make sure we're not trying to add more coefficients than we have."""
    limit = None
    if N % 2 == 0 and num_coefficients > N // 2:
        limit = N/2
    elif N % 2 == 1 and num_coefficients > (N - 1)/2:
        limit = (N - 1)/2
    if limit is not None:
        raise ValueError(
            "num_coefficients is {} but should not be larger than {}".format(
                num_coefficients, limit))


def reconstruct_fft(signal, coefficients:list=None,
                    list_of_harm: list=[1, 1/2, 1/3], square_width: int=None):
    """Test partial (i.e. filtered) Fourier reconstruction of a square signal.

    Args:
        N (int): Number of time (and frequency) points. We support both even
            and odd N.
        square_width (int): Number of "high" points in the time domain signal.
            This number must be less than or equal to N.
        num_coefficients (int): Number of frequencies, in addition to the dc
            term, to use in Fourier reconstruction. This is the number of
            positive frequencies _and_ the number of negative frequencies.
            Therefore, if N is odd, this number cannot be larger than
            (N - 1)/2, and if N is even this number cannot be larger than
            N/2.
        list_of_harm (int):
    """
    if square_width is not None:
        N = 2*square_width
        if square_width > N:
            raise ValueError("square_width cannot be larger than N")
        check_num_coefficients_ok(N, len(coefficients))

        signal = square_function(N, square_width)
    else:
        N = signal.shape[0]



    time_axis = np.linspace(0, N-1, N)
    ft = np.fft.fft(signal, axis=0)

    freqs = np.fft.fftfreq(signal.size)
    periods = np.zeros_like(freqs)
    periods[1:] = 1/(freqs[1:]*365)

    def get_harmonics(periods, list_of_harm=[1, 1/2, 1/3, 1/4, 1/5, 1/6]):
        harmonics = []
        for h in list_of_harm:
            harmonics.append(np.argmin((abs(periods - h))))
        harmonics = np.array(harmonics) - 1 # subtract 1 because loop below is adding 1
        return harmonics

    if coefficients is None:
        if list_of_harm is [1, 1/2, 1/3]:
            print('using default first 3 annual harmonics, expecting cycles of 365 days')
        coefficients = get_harmonics(periods, list_of_harm=list_of_harm)
    elif coefficients is not None:
        coefficients = coefficients


        # # determine coefficients based up first 3 annual harmonics
        # two_yearly_harmonics = np.logical_or(periods==1,
        #                                        periods==.5)
        # third_harmonic = np.argmin((abs(periods - 1/3)))
        # three_harm_year = np.concatenate([np.argwhere(two_yearly_harmonics).squeeze(),
        #                                   [third_harmonic]])
        # coefficients = three_harm_year - 1


    reconstructed_signal = np.zeros(N, dtype='complex128')
    reconstructed_signal += ft[0] * np.ones(N, dtype='complex128')
    # Adding the dc term explicitly makes the looping easier in the next step.

    for k in coefficients:
        k += 1  # Bump by one since we already took care of the dc term.
        if k == N-k:
            reconstructed_signal += ft[k] * np.exp(
                1.0j*2 * np.pi * (k) * time_axis / N)
        # This catches the case where N is even and ensures we don't double-
        # count the frequency k=N/2.

        else:
            reconstructed_signal += ft[k] * np.exp(
                1.0j*2 * np.pi * (k) * time_axis / N)
            reconstructed_signal += ft[N-k] * np.exp(
                1.0j*2 * np.pi * (N-k) * time_axis / N)
        # In this case we're just adding a frequency component and it's
        # "partner" at minus the frequency

    reconstructed_signal = reconstructed_signal / N
    # Normalize by the number of points in the signal. numpy's discete Fourier
    # transform convention puts the (1/N) normalization factor in the inverse
    # transform, so we have to do it here.

    plt.plot(time_axis[:365*3], signal[:365*3],
             'b-.',
             label='original first 3 years')
    plt.plot(time_axis[:365*3], reconstructed_signal.real[:365*3],
             'r-', linewidth=3,
             label='reconstructed first 3 years')
    # The imaginary part is zero anyway. We take the real part to
    # avoid matplotlib warnings.

    plt.grid()
    plt.legend(loc='upper right')
    return reconstructed_signal.real



