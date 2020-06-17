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
from RGCPD import EOF
import functions_pp


TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/1ts_0ff31_10jun-24aug_lag0-0_ts_no_train_test_splits1/2020-06-11_15hr_16min_df_data_z500_dt1_0ff31.h5'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')
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
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=path_out_main,
           append_pathsub='_' + name_ds)


selbox = [None, {'NorthPacAtl':[130,350,10,90]}]

# selbox = [None, {'NorthPac':(115, 250, 0, 70),
#                  'NorthAtl':(360-83, 6, 0, 70),
#                  'v200':[130,350,10,90]}]
rg.pp_TV(name_ds=name_ds)

rg.pp_precursors(selbox=selbox, anomaly=True)



rg.traintest(method='random10')

rg.calc_corr_maps()


subtitles = np.array([['Correlation map SST vs Western U.S. z500 Rossby wave']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='NorthPacAtl', aspect=2, size=5, cbar_vert=.0, save=True,
                  subtitles=subtitles, units=units,
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  clim=(-.6,.6))

 #%%
rg.cluster_list_MI()
rg.quick_view_labels(median=True)

rg.get_ts_prec(precur_aggr=1)
#%%
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
rg.df_MCIc.mean(0, level=1)

rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(5,2),
                    kwrgs={'vmax_nodes':.5,
                           'vmax_edges':.5,
                           'vmin_edges':-.5})

rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
rg.df_links.mean(0, level=1)
rg.df_MCIc.mean(0, level=1)

# df_ParCorr_sum = rg.PCMCI_get_ParCorr_from_txt()

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
rg.df_MCIc.mean(0, level=1)

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

list_of_name_path = [(name_or_cluster_label, TVpath),
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
greenrectangle_WestUS_bb = (140,325,24,62)
wide_WestUS_bb = (0,360,0,62)
rg.pp_precursors(selbox=wide_WestUS_bb, anomaly=True)


kwrgs_events = {'event_percentile':85, 'window':'mean'}#,
                # 'min_dur':7,'max_break':3, 'grouped':True,'reference_group':'center'}
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

rg.calc_corr_maps()


subtitles = np.array([['Western U.S. one-point correlation map Z 500hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='z500', aspect=2, size=5, cbar_vert=.19, save=False,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', greenrectangle_WestUS_bb],
                  clim=(-.6,.6))



event_dates = rg.TV.RV_bin[rg.TV.RV_bin.astype(bool).values].index
one_ev_peryear = functions_pp.remove_duplicates_list(list(event_dates.year))[1]
event_dates = event_dates[one_ev_peryear]

#%%
from class_hovmoller import Hovmoller
kwrgs_load = rg.kwrgs_load.copy()
kwrgs_load['tfreq'] = 1
HM = Hovmoller(kwrgs_load=kwrgs_load, event_dates=event_dates,
               seldates=rg.TV.aggr_to_daily_dates(rg.dates_TV), standardize=True, lags_prior=40,
               lags_posterior=20, rollingmeanwindow=5)
self = HM
HM.get_HM_data(rg.list_precur_pp[0][1])


fname1 = 'HM_'+'_'.join(['{}_{}'.format(*ki) for ki in kwrgs_events.items()])
fname2 = '_'.join(np.array(HM.kwrgs_load['selbox']).astype(str)) + \
                    f'_w{self.rollingmeanwindow}_std{self.standardize}.pdf'
fig_path = os.path.join(rg.path_outsub1, '_'.join([fname1, fname2]))
HM.plot_HM(drawbox=greenrectangle_WestUS_bb, clevels=np.arange(-.5, .51, .1),
           fig_path=fig_path)

# HM.quick_HM_plot()
