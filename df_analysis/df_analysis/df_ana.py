#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
# import mtspec
flatten = lambda l: [item for sublist in l for item in sublist]
from typing import List, Tuple, Union


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
    
    out_list = []
    for header in df:
        if to_np == False:
            y = df[header]
        elif to_np == True:
            y = df[header].values
        out_i = function(y, **kwrgs) 
        out_list.append(out_i)
    return pd.DataFrame(np.array(out_list).T, index=df.index, columns=keys)
    

def loop_df(df: pd.DataFrame(), function, keys=None, colwrap=3, sharex='col', 
            sharey='row', hspace=.4, kwrgs=None):
    #%%
    assert type(df) == type(pd.DataFrame()), ('df should be DataFrame, '
                'not pd.Series or ndarray')
    if keys is None:
        # retrieve only float series
        type_check = np.logical_or(df.dtypes == 'float',df.dtypes == 'float32')
        keys = type_check[type_check].index

    df = df.loc[:,keys]
    
    if (df.columns.size) % colwrap == 0:
        rows = int(df.columns.size / colwrap)
    elif (df.columns.size) % colwrap != 0:
        rows = int(df.columns.size / colwrap) + 1
        
    gridspec_kw = {'hspace':hspace}
    fig, ax = plt.subplots(rows, colwrap, sharex=sharex, sharey=sharey,
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
    from statsmodels.tsa import stattools
    if max_lag == None:
        max_lag = ts.size
    ac, con_int = stattools.acf(ts.values, nlags=max_lag-1,
                                unbiased=False, alpha=0.01,
                                 fft=True)
    return ac, con_int

def plot_ac(y=pd.Series, s='auto', title=None, AUC_cutoff=None, ax=None):
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
        cutoff = int(where[np.where(n_of_times == n)[0][0] ])
        
        s = 2*cutoff
    else:
        cutoff = int(s/2)
        s = s
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
    
    ax.text(0.99, 0.90, 
            text, 
            transform=ax.transAxes, horizontalalignment='right',
            fontdict={'fontsize':8})
    xlabels = [x * tfreq for x in range(s)]
    n_labels = max(1, int(s / 5))
    ax.set_xticks(xlabels[::n_labels])
    ax.set_xticklabels(xlabels[::n_labels], fontsize=10)
    ax.set_xlabel(xaxlabel, fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=10)
    return ax

def plot_timeseries(y, timesteps=None, selyears: Union[list, int]=None, title=None, ax=None):
    # ax=None
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if type(y.index) == pd.core.indexes.datetimes.DatetimeIndex:
        datetimes = y.index
        
    if timesteps is None and selyears is None:
        if hasattr(y.index,'levels'):
            y_ac = y.loc[0]
        else:
            y_ac = y
        ac, con_int = autocorr_sm(y_ac)
        where = np.where(con_int[:,0] < 0 )[0]
        # has to be below 0 for n times (not necessarily consecutive):
        n = 1
        n_of_times = np.array([idx+1 - where[0] for idx in where])
        cutoff = where[np.where(n_of_times == n)[0][0] ]
        timesteps = 20*cutoff
        datetimes = y_ac.iloc[:timesteps].index
    
    if selyears is not None and timesteps is None:
        if type(selyears) is not list:
            selyears = [selyears]
        datetimes = get_oneyr(y.index, *selyears)
        
    
    if hasattr(y.index,'levels'):
        for fold in y.index.levels[0]:
            ax.plot(datetimes, y.loc[fold, datetimes], alpha=.5,
                    label=f'f {fold}')
        ax.legend(prop={'size':6})
    else:
        ax.plot(datetimes, y.loc[datetimes])
    
    every_nth = round(len(ax.xaxis.get_ticklabels())/3)
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=8)
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

def plot_spectrum(y, methods=None, vlines=None, y_lim=(1e-4,1e3), 
                  x_lim=None, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        
    if methods is None:
        methods = [('periodogram', periodogram),
                   ('MT-spec', mtspectrum)]

    try:
        freq_df = (y.index[1] - y.index[0]).days
    except:
        freq_df = 1
        
        
    if x_lim is None:
        try:
            oneyrsize = get_oneyr(y.index).size
            x_lim = (freq_df, oneyrsize*freq_df)
        except:
            x_lim = (1,365)
        
    
    for i, method in enumerate(methods):
        label, func_ = method
        freq, spec = func_(y)
        periods = 1*freq_df/(freq[1:])
        ax.plot(periods, spec[1:], ls='-', c=nice_colors[i], label=label)  
        ax.loglog()
        ax.set_ylim(y_lim)   
        locmaj = mpl.ticker.LogLocator(base=10,numticks=int(-1E-99+x_lim[-1]/100) + 1) 
        ax.xaxis.set_major_locator(locmaj)
        locmin = mpl.ticker.LogLocator(base=10.0,subs=tuple(np.arange(0,1,0.1)[1:]),numticks=int(-1E-99+x_lim[-1]/100) + 1)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(x_lim)
    ax.legend(fontsize='small')
    if title is not None:
        ax.set_title(title, fontsize=10)
    return ax
        


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
    import h5py
    hdf = h5py.File(path_data,'r+')
    dict_of_dfs = {}
    for k in hdf.keys():
        dict_of_dfs[k] = pd.read_hdf(path_data, k)
    hdf.close()
    return dict_of_dfs

def df_figures(df_data, keys, analysis, line_dim='model', group_line_by=None,
                  met='default', wspace=0.08, col_wrap=None, threshold_bin=None):
    #%%
    '''
    3 dims to plot: [metrics, experiments, stat_models]
    2 can be assigned to row or col, the third will be lines in the same axes.
    '''

    dims = ['exper', 'models', 'met']
    col_dim = [s for s in dims if s not in [line_dim, 'met']][0]
    if met == 'default':
        met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Precision', 'Rel. Curve', 'ts']



    if line_dim == 'model':
        lines = models
        cols  = expers
    elif line_dim == 'exper':
        lines = expers
        cols  = models
    assert line_dim in ['model', 'exper'], ('illegal key for line_dim, '
                           'choose \'exper\' or \'model\'')

    if len(cols) == 1 and group_line_by is not None:
        group_s = len(group_line_by)
        cols = group_line_by
        lines_grouped = []
        for i in range(0,len(lines),group_s):
            lines_grouped.append(lines[i:i+group_s])



    grid_data = np.zeros( (2, len(met)), dtype=str)
    grid_data = np.stack( [np.repeat(met, len(cols)),
                           np.repeat(cols, len(met))])


    df = pd.DataFrame(grid_data.T, columns=['met', col_dim])
    if len(cols) != 1 or col_wrap is None:
        g = sns.FacetGrid(df, row='met', col=col_dim, height=3, aspect=1.4,
                      sharex=False,  sharey=False)
        # Only if 1 column is requisted, col_wrap is allowed
    if len(cols) == 1 and col_wrap is not None:
#        cols = met
        g = sns.FacetGrid(df, col='met', height=3, aspect=1.4,
                      sharex=False,  sharey=False, col_wrap=col_wrap)




    for col, c_label in enumerate(cols):

        if col_wrap is None:
            g.axes[0,col].set_title(c_label)
        if len(models) == 1 and group_line_by is not None:
            lines = lines_grouped[col]


        for row, metric in enumerate(met):

            if col_wrap is None:
                ax = g.axes[row,col]
            else:
                ax = g.axes[row]


            for l, line in enumerate(lines):

                if line_dim == 'model':
                    model = line
                    exper = c_label
                    color = nice_colors[l]

                elif line_dim == 'exper':
                    model = c_label
                    exper = line
                    if len(models) == 1 and group_line_by is not None:
                        exper = line
                        model = models[0]
                    color = colors_datasets[l]

#                if col_wrap is not None:
#                    metric = c_label # metrics on rows
                    # exper is normally column, now we only have 1 expers
#                    exper = expers[0]



                df_valid, RV, y_pred_all = dict_experiments[exper][model]
                tfreq = (y_pred_all.iloc[1].name - y_pred_all.iloc[0].name).days
                lags_i     = list(dict_experiments[exper][model][2].columns.astype(int))
                lags_tf = [l*tfreq for l in lags_i]
                if tfreq != 1:
                    # the last day of the time mean bin is tfreq/2 later then the centerered day
                    lags_tf = [l_tf- int(tfreq/2) if l_tf!=0 else 0 for l_tf in lags_tf]



                if metric in ['AUC-ROC', 'AUC-PR', 'BSS', 'Precision', 'Accuracy']:
                    df_metric = df_valid.loc[metric]
                    if metric in ['AUC-PR', 'Precision', 'Accuracy']:
                        clim = RV.RV_bin.values[RV.RV_bin==1].size / RV.RV_bin.size
                        if metric == 'Accuracy':
                            import validation as valid
                            # threshold upper 3/4 of above clim
                            threshold = int(100 * (1 - 0.75*clim))
                            df_ran = valid.get_metrics_confusion_matrix(RV, y_pred_all.loc[:,:0],
                                                    thr=[threshold], n_shuffle=400)
                            clim = df_ran[threshold/100]['fc shuf'].loc[:,'Accuracy'].mean()

                    else:
                        clim = None
                    plot_score_lags(df_metric, metric, color, lags_tf,
                                    linestyle=line_styles[l], clim=clim,
                                    cv_lines=False, col=col, 
                                    threshold_bin=threshold_bin, ax=ax)
                                    
                if metric == 'Rel. Curve':
                    if l == 0:
                        ax, n_bins = rel_curve_base(RV, lags_tf, col=col, ax=ax)
                    print(l,line)

                    rel_curve(RV, y_pred_all, color, lags_i, n_bins,
                              linestyle=line_styles[l], mean_lags=True,
                              ax=ax)

                if metric == 'ts':
                    if l == 0:
                        ax, dates_ts = plot_events(RV, color=nice_colors[-1], n_yrs=6,
                                         col=col, ax=ax)
                    plot_ts(RV, y_pred_all, dates_ts, color, line_styles[l], lag_i=1, ax=ax)

                # legend conditions
                same_models = np.logical_and(row==0, col==0)
                grouped_lines = np.logical_and(row==0, group_line_by is not None)
                if same_models or grouped_lines:
#                    legend.append(patches.Rectangle((0,0),0.5,0.5,facecolor=color))

                    ax.legend(ax.lines, lines,
                          loc = 'lower left', fancybox=True,
                          handletextpad = 0.2, markerscale=0.1,
                          borderaxespad = 0.1,
                          handlelength=2.5, handleheight=1, prop={'size': 12})

    #%%
    g.fig.subplots_adjust(wspace=wspace)

    return g.fig