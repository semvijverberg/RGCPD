#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import pandas as pd
import xarray as xr

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import func_models as fc_utils
import functions_pp; import df_ana

# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']


target = 'easterntemp'
if target == 'easterntemp':
    TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/east_forecast/')

path_data = os.path.join(main_dir, 'publications/paper2/data/')
cluster_label = 2
name_ds='ts'



start_end_date = ('1-1', '10-31')
tfreq = 15
precur_aggr = tfreq
alpha_corr = .05
experiment = 'fixed_corr'
# experiment = 'adapt_corr'
method     = 'leave_2'
n_boot = 2000
#%% run RGPD
start_end_TVdate = ('06-01', '08-31')
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.parcorr_map_time,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={'precursor':True},
                            distance_eps=800, min_area_in_degrees2=10,
                            calc_ts='region mean', selbox=(130,260,-10,60),
                            lags=np.array([0]))]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=None,
           tfreq=tfreq,
           path_outmain=path_out_main,
           append_pathsub='_' + experiment)

#%%
if experiment == 'fixed_corr':
    rg.pp_TV(name_ds=name_ds, detrend=False)
    rg.traintest(method=method)
    rg.calc_corr_maps()
    rg.cluster_list_MI()
    rg.quick_view_labels(save=True, append_str=experiment)

# rg.get_ts_prec()
#%%
months = {'May'         : ('05-01', '05-30'),
          'June'        : ('06-01', '06-30'),
          'July'        : ('07-01', '07-30'),
          'August'      : ('08-01', '08-30'),
          'september'   : ('09-01', '09-30')}

months = {'May-June'    : ('05-01', '06-30'),
          'June-July'   : ('06-01', '07-30'),
           'July-Aug'    : ('07-01', '08-31'),
           'Aug-Sept'    : ('08-01', '09-30'),
           'Sept-Okt'    : ('09-01', '10-31')}

monthkeys= list(months.keys()) ; oneyrsize = 0


if precur_aggr == 15:
    blocksize=2
    lag = 2
elif precur_aggr==60:
    blocksize=1
    lag = 1

list_test = []
list_test_b = []
dm = {} # dictionairy months
for month, start_end_TVdate in months.items():

    if experiment == 'fixed_corr':
        # overwrite RV_mask
        rg.get_ts_prec(precur_aggr=precur_aggr,
                       start_end_TVdate=start_end_TVdate)
    elif experiment == 'adapt_corr':
        rg.start_end_TVdate = start_end_TVdate
        rg.pp_TV(name_ds=name_ds, detrend=False)
        rg.pp_precursors()
        rg.traintest(method=method)
        rg.calc_corr_maps()
        rg.cluster_list_MI()
        rg.quick_view_labels(save=True, append_str=experiment)
        rg.get_ts_prec(precur_aggr=precur_aggr)

        # plotting corr_map
        SST_green_bb = (140,235,20,59)#(170,255,11,60)
        title = r'$parcorr(SST_t, mx2t_t\ |\ SST_{t-1},mx2t_{t-1})$'
        subtitles = np.array([['']]) #, f'lag 2 (15 day lead)']] )
        kwrgs_plot = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
                      'wspace':-.15, 'size':3, 'cbar_vert':-.08,
                      'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
                      'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
                      'n_yticks':6, 'x_ticks':np.arange(130, 280, 25),
                      'subtitles':subtitles, 'title':title,
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
        rg.plot_maps_corr(var='sst', save=True,
                          kwrgs_plot=kwrgs_plot,
                          min_detect_gc=1.0,
                          append_str=experiment+'_'+month)
        precur = rg.list_for_MI[0]
        dm[month] = rg



    if monthkeys.index(month) >= 1:
        nextyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
        if nextyr.size != oneyrsize:
            raise ValueError

    oneyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
    oneyrsize = oneyr.size


    kwrgs_model = {'scoring':'neg_mean_squared_error',
                   'alphas':np.logspace(.1, 2, num=25),
                   'normalize':False}

    keys = [k for k in rg.df_data.columns[:-2] if k != rg.TV.name]
    target_ts = rg.df_data.iloc[:,[0]].loc[0][rg.df_data.iloc[:,-1].loc[0]]
    target_ts = (target_ts - target_ts.mean()) / target_ts.std()

    out = rg.fit_df_data_ridge(target=target_ts,
                               keys=keys,
                               tau_min=lag, tau_max=lag,
                               kwrgs_model=kwrgs_model,
                               transformer=fc_utils.standardize_on_train)

    predict, weights, models_lags = out
    prediction = predict.rename({predict.columns[0]:'temp',lag:'Prediction'},
                                axis=1)
    score_func_list = [metrics.mean_squared_error, fc_utils.corrcoef]
    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                             rg.df_data.iloc[:,-2:],
                                                             score_func_list,
                                                             n_boot = n_boot,
                                                             blocksize=blocksize,
                                                             rng_seed=1)
    print(df_test_m)
    df_boot['mean_squared_error'] = 1-df_boot['mean_squared_error']
    list_test_b.append(df_boot)
    list_test.append(df_test_m)
    m = models_lags[f'lag_{lag}']['split_0']
    print(m.alpha_)
    idx_alpha = np.argwhere(kwrgs_model['alphas']==m.alpha_)[0][0]
    if idx_alpha in [0,24]:
        print(f'\nadapt alphas, idx is {idx_alpha}\n')


    # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
    #                       function=df_ana.plot_timeseries,
    #                       kwrgs={'timesteps':rg.fullts.size,
    #                                   'nth_xyear':5})

df_test = functions_pp.get_df_test(prediction.merge(rg.df_data.iloc[:,-2:],
                                                        left_index=True,
                                                        right_index=True)).iloc[:,:2]

corrvals = [test.values[0,1] for test in list_test]
MSE_SS_vals = [1-test.values[0,0] for test in list_test]

df_scores = pd.DataFrame({'RMSE-SS':MSE_SS_vals, 'Corr. Coef.':corrvals},
                         index=monthkeys)
df_test_b = pd.concat(list_test_b, keys = monthkeys,axis=1)

yerr = [] ; alpha = .10
for i in range(df_test_b.columns.size):
    Eh = 1 - alpha/2 ; El = alpha/2
    tup = [df_test_b.iloc[:,i].quantile(El), df_test_b.iloc[:,i].quantile(Eh)]
    mean = df_scores.values.flatten()[i]
    tup = abs(mean-tup)
    yerr.append(tup)
ax = df_scores.plot.bar(rot=0, yerr=np.array(yerr).reshape(2,2,5),
                        capsize=8, error_kw=dict(capthick=1))
ax.set_ylabel('Skill Score', fontsize=16)
# ax.set_xlabel('Months', fontsize=16)
ax.set_title(f'Seasonal dependence of {precur_aggr}-day mean temperature predictions',
             fontsize=16)
ax.tick_params(axis='both', labelsize=13)
ax.legend(fontsize=16, frameon=True, facecolor='grey',
              framealpha=.5)
ax.set_ylim(-0.5, 1)
plt.savefig(os.path.join(rg.path_outsub1,
             f'skill_score_vs_months_{precur_aggr}tf_lag{lag}_nb{n_boot}_blsz{blocksize}_{alpha_corr}.pdf'))
#%%
if experiment == 'adapt_corr':
    import plot_maps;

    corr = rg.list_for_MI[0].corr_xr.mean(dim='split').drop('time')
    list_xr = [corr.expand_dims('months', axis=0) for i in range(len(monthkeys))]
    corr = xr.concat(list_xr, dim = 'months')
    corr['months'] = ('months', monthkeys)

    np_data = np.zeros_like(corr.values)
    np_mask = np.zeros_like(corr.values)
    for i, f in enumerate(monthkeys):
        rg = dm[f]
        vals = rg.list_for_MI[0].corr_xr.mean(dim='split').values
        np_data[i] = vals
        mask = rg.list_for_MI[0].corr_xr.mask.mean(dim='split')
        np_mask[i] = mask

    corr.values = np_data
    mask = (('months', 'lag', 'latitude', 'longitude'), np_mask )
    corr.coords['mask'] = mask

    kwrgs_plot = {'aspect':2, 'hspace':.3,
                  'wspace':-.3, 'size':3, 'cbar_vert':0,
                  'units':'Corr. Coeff. [-]',
                  'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=0),
                  'y_ticks':np.arange(-90,91,60),
                  'title':title,
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

    if precur.lag_as_gap:
        corr = corr.rename({'lag':'gap'}) ; dim = 'gap'

    fig = plot_maps.plot_corr_maps(corr, mask_xr=corr.mask, col_dim='months',
                                   row_dim=corr.dims[1],
                                   **kwrgs_plot)

    f_name = 'corr_map_{}_a{}'.format(precur.name,
                                      precur.alpha) + '_' + \
                                      f'{experiment}_gap{precur.lag_as_gap}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext

    plt.savefig(fig_path, bbox_inches='tight')


#%%

# df = df_test_b.stack().reset_index(level=1)
# dfx = df.groupby(['level_1'])
# axes = dfx.boxplot()
# axes[0].set_ylim(-0.5, 1)
#%%
# import seaborn as sns
# df_ap = pd.concat(list_test_b, axis=0, ignore_index=True)
# df_ap['months'] = np.repeat(monthkeys, list_test_b[0].index.size)
# # df_ap.boxplot(by='months')
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['mean_squared_error'])
# ax.set_ylim(-0.5, 1)
# plt.figure()
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['corrcoef'])
# ax.set_ylim(-0.5, 1)

# #%%
# columns_my_order = monthkeys
# fig, ax = plt.subplots()
# for position, column in enumerate(columns_my_order):
#     ax.boxplot(df_test_b.loc[column], positions=[position,position+.25])

# ax.set_xticks(range(position+1))
# ax.set_xticklabels(columns_my_order)
# ax.set_xlim(xmin=-0.5)
# plt.show()


#%%

