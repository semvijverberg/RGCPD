#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:33:52 2020

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


TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
cluster_label = 1
name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 15

#%%

list_of_name_path = [(cluster_label, TVpath),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('NorthPacAtl', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=700, min_area_in_degrees2=7,
                             calc_ts='pattern cov'),
                   # BivariateMI(name='sm12', func=BivariateMI.corr_map,
                   #               kwrgs_func={'alpha':.01, 'FDR_control':True},
                   #               distance_eps=900, min_area_in_degrees2=5),
                 # BivariateMI(name='snow', func=BivariateMI.corr_map,
                 #               kwrgs_func={'alpha':.01, 'FDR_control':True},
                 #               distance_eps=700, min_area_in_degrees2=7),
                 # BivariateMI(name='NorthPac', func=BivariateMI.corr_map,
                 #              kwrgs_func={'alpha':1E-4, 'FDR_control':True},
                 #              distance_eps=700, min_area_in_degrees2=5,
                 #              calc_ts='pattern cov'),
                  BivariateMI(name='NorthPacAtl', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':1E-4, 'FDR_control':True},
                              distance_eps=700, min_area_in_degrees2=5,
                              calc_ts='pattern cov')]



list_import_ts = None #[('OMI', '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/OMI.h5')]



rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=list_import_ts,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
           append_pathsub='_' + name_ds)


selbox = [None, {'NorthPacAtl':(115, 359, 0, 70),
                 'z500':[130,350,10,90]}]

# selbox = [None, {'NorthPac':(115, 250, 0, 70),
#                  'NorthAtl':(360-83, 6, 0, 70),
#                  'v200':[130,350,10,90]}]

anomaly = [True, {'sm12':False, 'OLRtrop':False}]
rg.pp_precursors(selbox=selbox, anomaly=anomaly)

rg.pp_TV(name_ds=name_ds)

rg.traintest(method='random10')

rg.calc_corr_maps()

 #%%
rg.cluster_list_MI()
rg.quick_view_labels(median=True)

rg.get_ts_prec(precur_aggr=1)

keys = ['0..0..z500_sp',
       '0..0..NorthPac_sp', 'TrainIsTrue',
       'RV_mask']

rg.PCMCI_df_data(keys=keys,
                 pc_alpha=None,
                 tau_max=1,
                 max_conds_dim=10,
                 max_combinations=10)
rg.PCMCI_get_links(var=keys[0], alpha_level=.01)
rg.df_links.mean(0, level=1)
rg.df_MCIc.mean(0, level=1)

rg.PCMCI_plot_graph(min_link_robustness=5)

df_ParCorr_sum = rg.PCMCI_get_ParCorr_from_txt()


# rg.quick_view_labels(median=True)

# rg.plot_maps_corr(var=['z500'], save=False)

# rg.plot_maps_sum(var='sm12',
#                  kwrgs_plot={'aspect': 2, 'wspace': -0.02})
# rg.plot_maps_sum(var='st2',
#                  kwrgs_plot={'aspect': 2, 'wspace': -0.02})
# rg.plot_maps_sum(var='snow',
#                  kwrgs_plot={'aspect': 2, 'wspace': -0.02})
# rg.plot_maps_sum(var='sst',
#                  kwrgs_plot={'cbar_vert':.02})
# rg.plot_maps_sum(var='z500',
#                  kwrgs_plot={'cbar_vert':.02})

# rg.get_ts_prec(precur_aggr=1)
# rg.store_df_PCMCI()