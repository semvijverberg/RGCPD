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


TVpath = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-06-15_12hr_28min_df_data_z500_dt1_0ff31.h5'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/east/')
name_or_cluster_label = 'z500'
name_ds='0..0..z500_sp'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 15

#%%

list_of_name_path = [(name_or_cluster_label, TVpath),
                     ('NorthPacAtl', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='NorthPacAtl', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.001, 'FDR_control':True},
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov')]



rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)


selbox = (130,350,10,90)
anomaly = True

rg.pp_precursors(selbox=selbox, anomaly=anomaly)

rg.pp_TV(name_ds=name_ds)

rg.traintest(method='random10')

rg.calc_corr_maps()

sst_green_bb = (180, 240, 25, 60)
rg.plot_maps_corr(var='NorthPacAtl', drawbox=['all', sst_green_bb],
                  cbar_vert=.02, save=True)

 #%%
selbox = sst_green_bb
anomaly = True

rg.pp_precursors(selbox=selbox, anomaly=anomaly)

rg.pp_TV(name_ds=name_ds)

rg.traintest(method='random10')

rg.calc_corr_maps()

sst_green_bb = (180, 240, 25, 60)
rg.plot_maps_corr(var='NorthPacAtl', drawbox=['all', sst_green_bb],
                  cbar_vert=.02)

rg.cluster_list_MI()
rg.quick_view_labels(median=True)

rg.get_ts_prec(precur_aggr=1)

keys = ['z5000..0..z500_sp',
       '0..0..NorthPacAtl_sp', 'TrainIsTrue',
       'RV_mask']

rg.PCMCI_df_data(keys=keys,
                 pc_alpha=None,
                 tau_max=1,
                 max_conds_dim=10,
                 max_combinations=10)
rg.PCMCI_get_links(var=keys[0], alpha_level=.01)
rg.df_links.mean(0, level=1)
print(rg.df_MCIc.mean(0, level=1))

rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(5,2),
                    kwrgs={'vmax_nodes':.5,
                           'vmax_edges':.5,
                           'vmin_edges':-.5})

#%% Adapt RV_mask

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
# new_mask = None
keys = ['z5000..0..z500_sp',
       '0..0..NorthPacAtl_sp', 'TrainIsTrue',
       'RV_mask']

rg.PCMCI_df_data(keys=keys,
                 replace_RV_mask=new_mask.values,
                 pc_alpha=None,
                 tau_max=1,
                 max_conds_dim=10,
                 max_combinations=10)

rg.PCMCI_get_links(var=keys[0], alpha_level=.01)
rg.df_links.mean(0, level=1)
print(rg.df_MCIc.mean(0, level=1))

rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(5,2),
                    kwrgs={'vmax_nodes':.5,
                           'vmax_edges':.5,
                           'vmin_edges':-.5})



rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
rg.df_links.mean(0, level=1)
rg.df_MCIc.mean(0, level=1)



# =============================================================================
#%% Hovmoller diagram
# =============================================================================
tfreq = 15
TVpathHM = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-06-15_12hr_28min_df_data_z500_dt1_0ff31_RW_for_HM.h5'
list_of_name_path = [(name_or_cluster_label, TVpathHM),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov')]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)

rg.pp_TV(name_ds=name_ds)

wide_WestUS_bb = (0,360,0,73)
rg.pp_precursors(selbox=wide_WestUS_bb, anomaly=True)


kwrgs_events = {'event_percentile':66, 'window':'mean'}#,
                # 'min_dur':7,'max_break':3, 'grouped':True,'reference_group':'center'}
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

rg.calc_corr_maps()

greenrectangle_EastUS_bb = (170,300,15,73)
subtitles = np.array([['Eastern U.S. one-point correlation map Z 500hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='z500', aspect=2, size=5, cbar_vert=.19, save=False,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', greenrectangle_EastUS_bb],
                  clim=(-.6,.6))

rg.cluster_list_MI()

rg.get_ts_prec()


event_dates = rg.TV.RV_bin[rg.TV.RV_bin.astype(bool).values].index
one_ev_peryear = functions_pp.remove_duplicates_list(list(event_dates.year))[1]
event_dates = event_dates[one_ev_peryear]

#%%
from class_hovmoller import Hovmoller
kwrgs_load = rg.kwrgs_load.copy()
kwrgs_load['tfreq'] = 1
HM = Hovmoller(kwrgs_load=kwrgs_load, event_dates=event_dates,
               seldates=rg.TV.aggr_to_daily_dates(rg.dates_TV), standardize=True, lags_prior=40,
               lags_posterior=20, rollingmeanwindow=7)
self = HM
HM.get_HM_data(rg.list_precur_pp[0][1])


fname1 = 'HM_'+'_'.join(['{}_{}'.format(*ki) for ki in kwrgs_events.items()])
fname2 = '_'.join(np.array(HM.kwrgs_load['selbox']).astype(str)) + \
                    f'_w{self.rollingmeanwindow}_std{self.standardize}.pdf'
fig_path = os.path.join(rg.path_outsub1, '_'.join([fname1, fname2]))
HM.plot_HM(drawbox=greenrectangle_EastUS_bb, clevels=np.arange(-.5, .51, .1),
           fig_path=fig_path)