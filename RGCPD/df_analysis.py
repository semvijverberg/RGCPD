#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

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


def loop_df(df, function, keys=None, colwrap=3, sharex='col', kwrgs=None):
    #%%
    if keys is None:
        # retrieve only float series
        type_check = np.logical_or(df.dtypes == 'float',df.dtypes == 'float32')
        keys = type_check[type_check].index

    df = df.loc[:,keys]
    
    if (df.columns.size) % colwrap == 0:
        rows = int(df.columns.size / colwrap)
    elif (df.columns.size) % colwrap != 0:
        rows = int(df.columns.size / colwrap) + 1
        
    gridspec_kw = {'hspace':0.5}
    fig, ax = plt.subplots(rows, colwrap, sharex=sharex, sharey='row',
                           figsize = (3*colwrap,rows*2.5), gridspec_kw=gridspec_kw)

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
    return fig
    #%%

def autocorr_sm(ts, max_lag=None, alpha=0.01):
    import statsmodels as sm
    if max_lag == None:
        max_lag = ts.size
    ac, con_int = sm.tsa.stattools.acf(ts.values, nlags=max_lag-1,
                                unbiased=True, alpha=0.01,
                                 fft=True)
    return ac, con_int

def plot_ac(y=pd.Series, s='auto', title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    ac, con_int = autocorr_sm(y)

    time = y.index
    tfreq = (time[1] - time[0]).days

    # auto xlabels
    if s=='auto':
        where = np.where(con_int[:,0] < 0 )[0]
        # has to be below 0 for n times (not necessarily consecutive):
        n = 1
        n_of_times = np.array([idx+1 - where[0] for idx in where])
        cutoff = where[np.where(n_of_times == n)[0][0] ]
        s = 2*cutoff
    else:
        s = 5

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
    ax.set_xticklabels(xlabels[::n_labels], fontsize=10)
    ax.set_xlabel('time [days]', fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=10)
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


def periodogram(ts, d=1.):
    """ naive absolute squared Fourier components 
    
    input:
    ts .. time series
    d  .. sampling period
    """
    freq = np.fft.rfftfreq(len(ts))
    Pxx = 2*np.abs(np.fft.rfft(ts))**2/len(ts)
    return freq, Pxx

def fft_np(y, freq):
    yfft = sp.fftpack.fft(y)
    ypsd = np.abs(yfft)**2
    fftfreq = sp.fftpack.fftfreq(len(ypsd), freq)
    ypsd = 2.0/len(y) * ypsd
    return fftfreq, yfft, ypsd

def fft_powerspectrum(df, freq):
    
    df_fft = df[:].copy()
    df_psd = df[:].copy()
    
    list_freq = []
    for reg in df.columns:
        fftfreq, yfft, ypsd = fft_np(np.array(df[reg]), freq)
#        plt.plot(yfft)
        
        df_fft[reg] = yfft[:]
        df_psd[reg] = ypsd[:]
        df_fft.index = fftfreq[:]
        df_psd.index = fftfreq[:]
        i = fftfreq > 0
        idx = np.argmax(ypsd[i])
        text = '{} fft {:.1f}, T = {:.0f}'.format(
                reg,
                fftfreq[idx], 
                1 / (fftfreq[i][idx] * freq))
        list_freq.append(text)
    df_psd.columns = list_freq
    
    return fftfreq, df_fft, df_psd

def plot_powerspectrum(df, freq, title=None, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    
    fftfreq, df_fft, df_psd = fft_powerspectrum(df, freq)

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

def build_ts_matric(df_init, win=20, lag=0, columns=list, rename=dict, period='fullyear'):
    #%%
    '''
    period = ['fullyear', 'summer60days', 'pre60days']
    '''
    splits = df_init.index.levels[0]
    dates_full_orig = df_init.loc[0].index
    dates_RV_orig   = df_init.loc[0].index[df_init.loc[0]['RV_mask']==True]
    if columns is None:
        columns = df_init.columns

    df_cols = df_init[columns]
    TrainIsTrue = df_init['TrainIsTrue']

    list_test = []
    for s in range(splits.size):
        TestIsTrue = TrainIsTrue[s]==False
        list_test.append(df_cols.loc[s][TestIsTrue])

    df_test = pd.concat(list_test).sort_index()
    # shift precursor vs. tmax
    for c in df_test.columns[1:]:
        df_test[c] = df_test[c].shift(periods=-lag)

    # bin means
    df_test = df_test.resample(f'{win}D').mean()

    if period=='fullyear':
        dates_sel = dates_full_orig.strftime('%Y-%m-%d')
    elif period == 'summer60days':
        dates_sel = dates_RV_orig.strftime('%Y-%m-%d')
    elif period == 'pre60days':
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
                 annot=False, annot_kws={'size':30}, cbar=False)


    sig_bold_labels = sig_bold_annot(corr, mask_sig)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                 annot=sig_bold_labels, annot_kws={'size':30}, cbar=False, fmt='s')

    ax.tick_params(axis='both', labelsize=15,
                   bottom=True, top=False, left=True, right=False,
                   labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    ax.set_xticklabels(corr.columns, fontdict={'fontweight':'bold',
                                               'fontsize':25})
    ax.set_yticklabels(corr.index, fontdict={'fontweight':'bold',
                                               'fontsize':25}, rotation=0)
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
