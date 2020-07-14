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
import functions_pp


TVpathPac = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-07_18hr_48min_df_data_z500_v200_dt1_0ff31.h5'
# TVpathRW = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-01_17hr_04min_df_data_z500_dt1_0ff31.h5'
TVpathall = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-07-07_18hr_48min_df_data_z500_v200_dt1_0ff31.h5'
TVpathRWvsRW = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_0ff31_10jun-24aug_lag0-15_0..0..z500_sp_random10s1/2020-07-09_11hr_42min_df_data_N-Pac. SST_Trop. Pac. SST_z500_dt1_0ff31.h5'
TVpathRW = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-15_ts_random10s1/2020-07-14_15hr_10min_df_data_v200_z500_dt1_0ff31_z500_140-300-20-73.h5'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/east/')
name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 60

#%%

list_of_name_path = [(name_or_cluster_label, TVpathRW),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('N-Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]
                     # ('Trop. Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

selbox = (-180,360,-10,90)
list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.05, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=selbox,
                                use_sign_pattern=True),
                 BivariateMI(name='N-Pac. SST', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=selbox)]
                 # BivariateMI(name='Trop. Pac. SST', func=BivariateMI.corr_map,
                 #              kwrgs_func={'alpha':.01, 'FDR_control':True},
                 #              distance_eps=500, min_area_in_degrees2=5,
                 #              calc_ts='pattern cov', selbox=selbox)]



# list_import_ts = [('versusmx2t', TVpathall)]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds)

rg.pp_precursors(anomaly=True)


rg.traintest(method='random10')

rg.calc_corr_maps()



# subtitles = np.array([['SST vs spat. covariance Rossby wave (z500)']])

# rg.plot_maps_corr(var='Pacific SST',
#                   aspect=2, size=5, cbar_vert=.19, save=True,
#                   subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
#                   map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
#                   drawbox=['all', sst_green_bb],
#                   clim=(-.6,.6))
if tfreq > 15: sst_green_bb = (140,240,-9,59) # (180, 240, 30, 60): original warm-code focus
if tfreq <= 15: sst_green_bb = (140,235,20,59) # same as for West
save = True
units = 'Corr. Coeff. [-]'
subtitles = np.array([[f'lag {l}: SST vs Rossby wave ({name_or_cluster_label})' for l in rg.lags]])
rg.plot_maps_corr(var='N-Pac. SST', row_dim='split', col_dim='lag',
                  aspect=2, hspace=-.57, wspace=-.22, size=3.5, cbar_vert=-.08, save=True,
                  subtitles=subtitles, units=units, zoomregion=(130,260,-10,60),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  n_xticks=6,
                  drawbox=[(0,0), sst_green_bb],
                  clim=(-.6,.6))


# sst_tropbox = (140, 250, 0, 30)
# subtitles = np.array([[f'lag {l}: SST vs Rossby wave ({name_or_cluster_label})'] for l in rg.lags])
# units = 'Corr. Coeff. [-]'
# rg.plot_maps_corr(var='Trop. Pac. SST', row_dim='lag', col_dim='split',
#                   aspect=2, hspace=-.57, size=5, cbar_vert=.175, save=save,
#                   subtitles=subtitles, units=units, #zoomregion=(-180,360,-10,70),
#                   map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
#                   drawbox=['all', sst_tropbox], clim=(-.6,.6),
#                   append_str=''.join(map(str, sst_tropbox)))

z500_green_bb = (155, 310, 10, 80)
subtitles = np.array([[f'lag {l}: z 500hpa vs Rossby wave ({name_or_cluster_label})'] for l in rg.lags])
rg.plot_maps_corr(var='z500', row_dim='lag', col_dim='split',
                  aspect=2, hspace=-.57, size=5, cbar_vert=.175, save=save,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  drawbox=['all', z500_green_bb], clim=(-.6,.6),
                  append_str=''.join(map(str, z500_green_bb)))



 #%%
rg.list_for_MI = [BivariateMI(name='N-Pac. SST', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=sst_green_bb)]

rg.calc_corr_maps(var='N-Pac. SST')
rg.cluster_list_MI(var='N-Pac. SST')
rg.quick_view_labels(median=True)
rg.get_ts_prec(precur_aggr=1)
rg.store_df(append_str=f'RW_and_SST_fb_tf{rg.tfreq}')

#%%
# rg.cluster_list_MI()
# rg.list_for_MI[0].calc_ts = 'pattern cov'
freqs = [1, 5, 10, 15, 30, 60]
for f in freqs[:]:
    rg.get_ts_prec(precur_aggr=f)
    rg.df_data = rg.df_data.rename({'z5000..0..z500_sp':'Rossby wave (z500)', '0..0..N-Pac. SST_sp':'N-Pacific SST',
                                    '15..0..Trop. Pac. SST_sp':'Trop. Pac. SST',
                                    '15..2..Trop. Pac. SST':'Nina'}, axis=1)

    keys = [['Rossby wave (z500)','N-Pacific SST']]
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

#%%
import func_models

shift = 2
mask_standardize = np.logical_and(rg.df_data.loc[0]['TrainIsTrue'], rg.df_data.loc[0]['RV_mask'])
df = func_models.standardize_on_train(rg.df_data[k].loc[0], mask_standardize)
RV_and_SST_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], df['N-Pacific SST'].shift(-shift) > .5)
fig = df[RV_and_SST_mask][k[:-2]].hist(sharex=True)
fig[0,0].set_xlim(-3,3)

#%% Adapt RV_mask
import matplotlib.pyplot as plt


freqs = [1, 15, 30, 60]
for f in freqs:
    rg.get_ts_prec(precur_aggr=f)

    keys = ['z5000..0..z500_sp',
           '0..0..NorthPacAtl_sp', 'TrainIsTrue',
           'RV_mask']

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

    rg.PCMCI_df_data(keys=keys,
                     replace_RV_mask=new_mask.values,
                     pc_alpha=None,
                     tau_max=5,
                     max_conds_dim=10,
                     max_combinations=10)

    rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(8,2),
                        kwrgs={'vmax_nodes':1.0,
                               'vmax_edges':.6,
                               'vmin_edges':-.6,
                               'node_ticks':.2,
                               'edge_ticks':.1},
                        append_figpath=f'_subset_dates_tf{rg.precur_aggr}')

    rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
    rg.df_links.mean(0, level=1)
    MCI_subset = rg.df_MCIc.mean(0, level=1)




