#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
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
from RGCPD import EOF
import plot_maps

TVpathRW = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-15_ts_random10s1/2020-07-14_15hr_10min_df_data_v200_z500_dt1_0ff31_z500_140-300-20-73.h5'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/east_forecast/')
name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 60
#%%
list_of_name_path = [(name_or_cluster_label, TVpathRW),
                      ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                       ('OLR', os.path.join(path_raw, 'OLRtrop_1979-2018_1_12_daily_2.5deg.nc'))]



list_for_MI   = [BivariateMI(name='v200', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=600, min_area_in_degrees2=1,
                              calc_ts='pattern cov', selbox=(0,360,-10,90),
                              use_sign_pattern=True),
                   BivariateMI(name='z500', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.05, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(0,360,-10,90),
                                use_sign_pattern=True),
                   BivariateMI(name='OLR', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.05, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov'),
                   BivariateMI(name='sst', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.05, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(120,260,-10,90))]


rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0,1]),
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors()

rg.traintest('random10')
#%%

import cartopy.crs as ccrs ; import matplotlib.pyplot as plt

rg.calc_corr_maps()

v200_green_bb = (170,359,23,73)
units = 'Corr. Coeff. [-]'
subtitles = np.array([[f'lag {l}: v-wind-200 hPa vs eastern U.S. RW'] for l in rg.lags_i])
rg.plot_maps_corr(var='v200', row_dim='lag', col_dim='split',
                  aspect=2, size=5, hspace=-0.58, cbar_vert=.18, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  drawbox=[(0,0), v200_green_bb],
                  clim=(-.6,.6))

# z500_green_bb = (140,260,20,73) #: Pacific box
# z500_green_bb = (140,300,20,73) #: RW box
z500_green_bb = (140,300,20,65) #: testing forecasting skill
subtitles = np.array([[f'lag {l}: z-500 hPa vs eastern U.S. RW'] for l in rg.lags_i])
rg.plot_maps_corr(var='z500', row_dim='lag', col_dim='split',
                  aspect=2, size=5, hspace=-0.63, cbar_vert=.2, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  drawbox=[(0,0), z500_green_bb],
                  clim=(-.6,.6),
                  append_str=''.join(map(str, z500_green_bb)))

SST_green_bb = (140,235,20,59)#(170,255,11,60)
subtitles = np.array([[f'lag {l}: SST vs eastern U.S. RW' for l in rg.lags_i]])
rg.plot_maps_corr(var='sst', row_dim='split', col_dim='lag',
                  aspect=2, hspace=-.47, wspace=-.18, size=3, cbar_vert=-.08, save=True,
                  subtitles=subtitles, units=units, zoomregion=(130,260,-10,60),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  x_ticks=np.arange(130, 280, 25),
                  clim=(-.6,.6))

#%% Only SST

list_of_name_path = [(name_or_cluster_label, TVpathRW),
                       ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]



list_for_MI   = [BivariateMI(name='sst', func=BivariateMI.corr_map,
                            kwrgs_func={'alpha':.05, 'FDR_control':True},
                            distance_eps=1000, min_area_in_degrees2=1,
                            calc_ts='pattern cov', selbox=(120,260,-10,90))]


rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0,1]),
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors()

rg.traintest('random10')

rg.calc_corr_maps()

rg.cluster_list_MI()

rg.get_ts_prec()

rename = {'z5000..0..z500_sp': 'Rossby wave (z500)',
          '0..0..v200_sp':'Rossby wave (v300) lag 0',
          '60..0..v200_sp': 'Rossby wave (v300) lag 1',
          '0..0..z500_sp': 'Rossby wave (z500) lag 0',
          '60..0..z500_sp':'Rossby wave (z500) lag 1',
          '0..0..sst_sp': 'SST lag 0',
          f'{rg.tfreq}..0..sst_sp': 'SST lag 1',
          f'{rg.tfreq}..1..sst': 'SST r1 lag 1',
          f'{rg.tfreq}..2..sst': 'SST r2 lag 1',
          '0..1..sst': 'SST r1 lag 0',
          '0..2..sst': 'SST r2 lag 0'}
rg.df_data = rg.df_data.rename(rename, axis=1)

#%% Ridge
import df_ana; import sklearn, functions_pp, func_models
keys = ['SST r1 lag 1', 'SST r2 lag 1'] # lag 1
keys = ['SST lag 1']
kwrgs_model = {'scoring':'neg_mean_squared_error',
               'alphas':np.logspace(-3, 1, num=10)}
s = 0
target_ts = rg.df_data.loc[s].iloc[:,[0]][rg.df_data.loc[s]['RV_mask']].copy()
target_mean = target_ts.mean().squeeze()
target_std = target_ts.std().squeeze()
# standardize :
target_ts = (target_ts - target_mean) / target_std
predict, coef, model = rg.fit_df_data_ridge(keys=keys, target=target_ts, tau_min =1,
                                            tau_max=1,
                                            transformer=func_models.standardize_on_train,
                                            kwrgs_model=kwrgs_model)
prediction = predict.rename({1:'Prediction', 'RVz5000..0..z500_sp':'Rossby wave (z500)'}, axis=1)

# AR1
AR1, c, m = rg.fit_df_data_ridge(keys=['Rossby wave (z500)'], target=target_ts,
                                 tau_min =1,
                                 tau_max=1,
                                 transformer=func_models.standardize_on_train,
                                 kwrgs_model=kwrgs_model)
AR1 = AR1.rename({1:'AR1 fit'}, axis=1)

#%%
import matplotlib.dates as mdates

df_splits = rg.df_data[['TrainIsTrue', 'RV_mask']]
df_AR1test = functions_pp.get_df_test(AR1.merge(df_splits,
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:2]
df_test = functions_pp.get_df_test(prediction.merge(df_splits,
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:2]

# Plot
fig, ax = plt.subplots(1, 1, figsize = (15,5))
ax.plot(df_test[['Prediction']], label='SST pattern lag 1',
        color='red',#ax.lines[0].get_color(),
        linewidth=1)

y = prediction['Prediction']
for fold in y.index.levels[0]:
    label = None ; color = 'red' ;
    ax.plot(y.loc[fold].index, y.loc[fold], alpha=.1,
            label=label, color=color)


ax.xaxis.set_major_locator(mdates.YearLocator(5, month=6, day=3))   # every year)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim(pd.to_datetime('1979-01-01'), xmax=pd.to_datetime('2020-12-31'))
ax.tick_params(axis='both', labelsize=14)
fig.autofmt_xdate()

ax.scatter(df_test[['Prediction']].index,
           df_test[['Prediction']].values, label=None,
           color=ax.lines[0].get_color(),
           s=15)
ax.plot(df_test[['Rossby wave (z500)']], label='Truth', color='black',
        linewidth=1)

ax.plot(df_AR1test[['AR1 fit']], label='AR1', color='grey',
        linewidth=1, linestyle='--')
ax.set_title('Out of sample (10-fold) 60-day aggr. RW prediction from lagged SST pattern',
             fontsize=16)
ax.hlines(y=0,xmin=pd.to_datetime('1979-06-03'),
          xmax=pd.to_datetime('2018-08-02'), color='black')


ax.legend()
ax.set_ylabel('Standardized E-U.S. RW timeseries', fontsize=16)
ax.set_ylim(-3,3)


MSE_func = sklearn.metrics.mean_squared_error
fullts = rg.df_data.loc[0].iloc[:,0]
Persistence = fullts.shift(1)[rg.df_data.loc[0]['RV_mask']]
Persistence = (Persistence - target_mean) / target_std
MSE_model = MSE_func(df_test.iloc[:,0],df_test.iloc[:,1], squared=False)
MSE_pers  = MSE_func(df_test.iloc[:,0],Persistence, squared=False)
MSE_AR1  = MSE_func(df_test.iloc[:,0],df_AR1test.iloc[:,1], squared=False)
Corr_pers = np.corrcoef(Persistence.values.squeeze(), df_test.iloc[:,0].values)
Corr_AR1 = np.corrcoef(df_test.iloc[:,0].values, df_AR1test.iloc[:,1].values)

text1 =  'Corr. coeff. model          : {:.2f}\n'.format(df_test.corr().iloc[0,1])
text1 +=  'Corr. coeff. persistence : {:.2f}\n'.format(Corr_pers[0][1])
text1 +=  'Corr. coeff. AR1             : {:.2f}'.format(Corr_AR1[0][1])
text2 = r'RMSE model          : {:.2f} $\sigma$'.format(MSE_model) + '\n'
text2 += r'RMSE persistence : {:.2f} $\sigma$'.format(MSE_pers) + '\n'
text2 += r'RMSE AR1            : {:.2f} $\sigma$'.format(MSE_AR1)

ax.text(.038, .05, text1,
        transform=ax.transAxes, horizontalalignment='left',
        fontdict={'fontsize':14},
        bbox = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor='black'))
ax.text(.395, .05, text2,
        transform=ax.transAxes, horizontalalignment='left',
        fontdict={'fontsize':14},
        bbox = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor='black'))

figname = rg.path_outsub1 +f'/forecast_lagged_SST_tf{rg.precur_aggr}.pdf'
plt.savefig(figname, bbox_inches='tight')


#%% correlate with PDO part 1
import climate_indices
rg.tfreq = 1
rg.pp_TV(name_ds=name_ds, detrend=False)
rg.traintest('random10')
df_PDO, PDO_patterns = climate_indices.PDO(rg.list_precur_pp[2][1], rg.df_splits)
#%% correlate with PDO part 2
rg.cluster_list_MI('sst')
rg.get_ts_prec()
df = rg.df_data.mean(axis=0, level=1)[['0..0..sst_sp']]
df = df.rename({'0..0..sst_sp':'N-Pacifc SST (60d)'},axis=1)

df_PDO_corr = df.merge(df_PDO.mean(axis=0, level=1),left_index=True, right_index=True)
df_PDO_corr.corr()

#%% Store daily data
rg.get_ts_prec(precur_aggr=1)
rg.df_data = rg.df_data.rename(rename, axis=1)
rg.store_df()
#%% interannual variability events?
import class_RV
RV_ts = rg.fulltso.sel(time=rg.TV.aggr_to_daily_dates(rg.dates_TV))
threshold = class_RV.Ev_threshold(RV_ts, event_percentile=85)
RV_bin, np_dur = class_RV.Ev_timeseries(RV_ts, threshold=threshold, grouped=True)
plt.hist(np_dur[np_dur!=0])

#%%


freqs = [1, 5, 15, 30, 60]
for f in freqs:
    rg.get_ts_prec(precur_aggr=f)
    rg.df_data = rg.df_data.rename({'0..0..z500_sp':'Rossby wave (z500)',
                               '0..0..sst_sp':'Pacific SST',
                               '15..0..sst_sp':'Pacific SST (lag 15)',
                               '0..0..v200_sp':'Rossby wave (v200)'}, axis=1)

    keys = [['Rossby wave (z500)', 'Pacific SST'], ['Rossby wave (v200)', 'Pacific SST']]
    k = keys[0]
    name_k = ''.join(k[:2]).replace(' ','')
    k.append('TrainIsTrue') ; k.append('RV_mask')

    rg.PCMCI_df_data(keys=k,
                     pc_alpha=None,
                     tau_max=5,
                     max_conds_dim=10,
                     max_combinations=10)
    rg.PCMCI_get_links(var=k[0], alpha_level=.01)

    rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(3,2),
                        kwrgs={'vmax_nodes':1.0,
                               'vmax_edges':.6,
                               'vmin_edges':-.6,
                               'node_ticks':.3,
                               'edge_ticks':.3,
                               'curved_radius':.5,
                               'arrowhead_size':1000,
                               'label_fontsize':10,
                               'link_label_fontsize':12,
                               'node_label_size':16},
                        append_figpath=f'_tf{rg.precur_aggr}_{name_k}')

    rg.PCMCI_get_links(var=k[1], alpha_level=.01)
    rg.df_links.astype(int).sum(0, level=1)
    MCI_ALL = rg.df_MCIc.mean(0, level=1)

#%% Conditional probability summer RW

rg.cluster_list_MI()

rg.get_ts_prec()
rename = {'z5000..0..z500_sp': 'Rossby wave (z500)',
          '0..0..v200_sp':'Rossby wave (v300) lag 0',
          '60..0..v200_sp': 'Rossby wave (v300) lag 1',
          '0..0..z500_sp': 'Rossby wave (z500) lag 0',
          '60..0..z500_sp':'Rossby wave (z500) lag 1',
          '0..0..sst_sp': 'SST lag 0',
          '60..0..sst_sp': 'SST lag 1',
          '60..1..sst': 'SST r1 lag 1',
          '60..2..sst': 'SST r2 lag 1',
          '0..1..sst': 'SST r1 lag 0',
          '0..2..sst': 'SST r2 lag 0'}
rg.df_data = rg.df_data.rename(rename, axis=1)
#%% (Conditional) Probability Density Function

import func_models
import functions_pp

k = list(rename.values())
# s = 9
# df_std = func_models.standardize_on_train(rg.df_data[k], np.logical_and(df_test['RV_mask']))
df_test = functions_pp.get_df_test(rg.df_data)

shift = 1
mask_standardize = df_test['RV_mask']
df = func_models.standardize_on_train(df_test[k], mask_standardize)
SST_lag_summer = df['SST lag 60'].shift(shift)[df_test['RV_mask']]
RV_and_SST_mask = SST_lag_summer > np.percentile(SST_lag_summer, 85)
fig = df[df_test['RV_mask']][RV_and_SST_mask][k].hist(sharex=True)
fig[0,0].set_xlim(-3,3)

mask_summer = (df['SST lag 1'].shift(shift) > np.percentile(df['SST lag 1'], 85))[df_test['RV_mask']]
#%% Test MJO signal in OLR
tfreq = 60
list_of_name_path = [(name_or_cluster_label, TVpathRW),
                       ('OLR', os.path.join(path_raw, 'OLRtrop_1979-2018_1_12_daily_2.5deg.nc'))]



list_for_MI   = [BivariateMI(name='OLR', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.2, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(150,360,-30,30))]

list_for_EOFS = [EOF(name='OLR', neofs=2, selbox=(150,360,-30,30),
                     n_cpu=1, start_end_date=start_end_TVdate)]

rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            list_for_EOFS=list_for_EOFS,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0,1]),
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors(anomaly=False, detrend=False)

rg.traintest('random10')


import cartopy.crs as ccrs ; import matplotlib.pyplot as plt

rg.calc_corr_maps()

rg.get_EOFs()

rg.cluster_list_MI()

rg.get_ts_prec()


df_test = functions_pp.get_df_test(rg.df_data)
corr = df_test.corr()