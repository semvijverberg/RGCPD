#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:33:52 2020

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs

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
import class_BivariateMI
import functions_pp, core_pp


TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/1ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-06-11_15hr_16min_df_data_z500_dt1_0ff31.h5'
TVpathRV= '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/west/1ts_0ff31_10jun-24aug_lag0-15_ts_random10s1/2020-07-14_15hr_08min_df_data_v200_z500_dt1_0ff31_z500_145-325-20-62.h5'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')
name_or_cluster_label = 'z500'
name_ds='0..0..z500_sp'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 15


#%%

list_of_name_path = [(name_or_cluster_label, TVpathRV),
                     ('N-Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=(-180,360,-10,90),
                                use_sign_pattern=True, lags=np.array([0])),
                 BivariateMI(name='N-Pac. SST', func=class_BivariateMI.parcorr_map_time,
                              alpha=.05, FDR_control=True,
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=(130,260,-10,90),
                              lags=np.array([0]))]

list_import_ts = None #[('versusmx2t', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/west/1ts_0ff31_10jun-24aug_lag0-15_ts_random10s1/2020-07-09_09hr_48min_df_data_v200_z500_sst_dt1_0ff31.h5')]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=list_import_ts,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq,
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)




# selbox = [None, {'NorthPac':(115, 250, 0, 70),
#                  'NorthAtl':(360-83, 6, 0, 70),
#                  'v200':[130,350,10,90]}]
rg.pp_TV(name_ds=name_ds)

rg.pp_precursors(selbox=None, anomaly=True)



rg.traintest(method='random10')

rg.calc_corr_maps()
#%%
SST_WestUS_bb = (140,235,20,59)
subtitles = np.array([['SST vs western RW']])
units = 'Corr. Coeff. [-]'
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag',
              'aspect':2, 'hspace':-.57, 'wspace':-.22, 'size':2, 'cbar_vert':-.02,
              'subtitles':subtitles, 'units':units, 'zoomregion':(130,260,-10,60),
              'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6,
              'x_ticks':np.array([]), 'y_ticks':np.array([]),
              'drawbox':[(0,0), SST_WestUS_bb],
              'clim':(-.6,.6)}
rg.plot_maps_corr(var='N-Pac. SST', save=True, kwrgs_plot=kwrgs_plot)

z500_boxPac = (140,260,20,62)
subtitles = np.array([['z500 vs Rossby wave (z500)']])
rg.plot_maps_corr(var='z500',
                  aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', z500_boxPac], clim=(-.6,.6),
                  append_str=''.join(map(str, z500_boxPac)))

rg.tfreq = 60 ; rg.lags_i = np.array([0, 1]) ; rg.lags = np.array([0, 60])
rg.calc_corr_maps('N-Pac. SST')
# RW when tfreq = 60
SST_box = (140,235,20,59)
subtitles = np.array([[f'lag {l}: SST vs western U.S. RW' for l in rg.lags]])
rg.plot_maps_corr(var='N-Pac. SST', row_dim='split', col_dim='lag',
                  aspect=2, hspace=-.47, wspace=-.18, size=3, cbar_vert=-.08, save=True,
                  subtitles=subtitles, units=units, zoomregion=(130,260,-10,60),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  x_ticks=np.arange(130, 280, 25),
                  clim=(-.6,.6),
                  append_str='60_daymean')
 #%%
# rg.list_for_MI[0].selbox = greenrectangle_WestUS_bb
rg.list_for_MI = [BivariateMI(name='N-Pac. SST', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=SST_WestUS_bb)]

rg.calc_corr_maps('N-Pac. SST')
rg.cluster_list_MI('N-Pac. SST')
rg.quick_view_labels(median=True)
rg.get_ts_prec(precur_aggr=1)
rg.store_df(append_str='RW_and_SST_feedback')
#%%
# rg.cluster_list_MI()
# rg.list_for_MI[0].calc_ts = 'pattern cov'
freqs = [1, 5, 10, 15, 30, 60]
for f in freqs:
    rg.get_ts_prec(precur_aggr=f)
    rg.df_data = rg.df_data.rename({'z5000..0..z500_sp':'Rossby wave (z500)',
                                    '0..0..N-Pac. SST_sp':'N-Pacific SST'}, axis=1)


    keys = [['Rossby wave (z500)','N-Pacific SST']]
    k = keys[0]
    k.append('TrainIsTrue') ; k.append('RV_mask')
    name_k = ''.join(k[:2]).replace(' ','')

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
    rg.df_links.mean(0, level=1)
    MCI_ALL = rg.df_MCIc.mean(0, level=1)


#%%
import func_models
k = ['Rossby Wave (z500)', '0..0..z500_sp', 'SSTvsZ500']
shift = 2
mask_standardize = np.logical_and(rg.df_data.loc[0]['TrainIsTrue'], rg.df_data.loc[0]['RV_mask'])
df = func_models.standardize_on_train(rg.df_data[k].loc[0], mask_standardize)
RV_and_SST_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], df['SSTvsZ500'].shift(-shift) > .5)
fig = df[RV_and_SST_mask][k[:]].hist(sharex=True)
fig[0,0].set_xlim(-3,3)
# df_ParCorr_sum = rg.PCMCI_get_ParCorr_from_txt()

#%% Adapt RV_mask
import matplotlib.pyplot as plt

# new_mask = None
keys = ['z5000..0..z500_sp',
       '0..0..NorthPacAtl_sp', 'TrainIsTrue',
       'RV_mask']

freqs = [1, 15, 30, 60]
for f in freqs:
    rg.get_ts_prec(precur_aggr=f)

    # when both SST and RW above threshold
    RW_ts = rg.df_data.loc[0].iloc[:,0]
    RW_mask = RW_ts > float(rg.TV.RV_ts.quantile(q=.75))
    new_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], RW_mask)
    sst = functions_pp.get_df_test(rg.df_data, cols=['0..0..NorthPacAtl_sp'])
    sst_mask = (sst > sst.quantile(q=.75).values).squeeze()
    new_mask = np.logical_and(sst_mask, new_mask)
    sumyears = new_mask.groupby(new_mask.index.year).sum()
    sumyears = list(sumyears.index[sumyears > 25])
    RV_mask = rg.df_data.loc[0]['RV_mask']
    m = np.array([True if y in sumyears else False for y in RV_mask.index.year])
    new_mask = np.logical_and(m, RV_mask)
    new_mask.astype(int).plot()
    plt.savefig(os.path.join(rg.path_outsub1, 'subset_dates.pdf'))
    print(f'{new_mask[new_mask].size} datapoints')

    keys = ['z5000..0..z500_sp',
           '0..0..NorthPacAtl_sp', 'TrainIsTrue',
           'RV_mask']

    rg.PCMCI_df_data(keys=keys,
                     replace_RV_mask=new_mask.values,
                     pc_alpha=None,
                     tau_max=5,
                     max_conds_dim=10,
                     max_combinations=10)

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
                        append_figpath=f'_subset_dates_tf{rg.precur_aggr}')

    rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
    rg.df_links.mean(0, level=1)
    MCI_subset = rg.df_MCIc.mean(0, level=1)


