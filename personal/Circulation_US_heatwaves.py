#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

@author: semvijverberg
"""


#%%
import os, inspect, sys
import numpy as np

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
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


TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf5_nc5_dendo_80d77.nc'
cluster_label = 3
name_ds='q75tail'
start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')
#%%
list_of_name_path = [(cluster_label, TVpath), 
                     ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]




list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map, 
                              kwrgs_func={'alpha':.01, 'FDR_control':True}, 
                              distance_eps=600, min_area_in_degrees2=7),
                 BivariateMI(name='v200', func=BivariateMI.corr_map, 
                               kwrgs_func={'alpha':.01, 'FDR_control':True}, 
                               distance_eps=600, min_area_in_degrees2=5)]





rg = RGCPD(list_of_name_path=list_of_name_path, 
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=1,
           tfreq=14, lags_i=np.array([0,1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
           append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds)

rg.pp_precursors(selbox=(130,350,10,90))

rg.traintest('no_train_test_split')


rg.calc_corr_maps()
rg.plot_maps_corr(aspect=4.5, cbar_vert=-.1)


#%%
from RGCPD import RGCPD

list_of_name_path = [(cluster_label, TVpath), 
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm2', os.path.join(path_raw, 'sm2_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm3', os.path.join(path_raw, 'sm3_1979-2018_1_12_daily_1.0deg.nc')),
                     ('snow',os.path.join(path_raw, 'snow_1979-2018_1_12_daily_1.0deg.nc')),
                     ('st2',  os.path.join(path_raw, 'lsm_st2_1979-2018_1_12_daily_1.0deg.nc')),
                     ('OLRtrop',  os.path.join(path_raw, 'OLRtrop_1979-2018_1_12_daily_2.5deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map, 
                             kwrgs_func={'alpha':.01, 'FDR_control':True}, 
                             distance_eps=600, min_area_in_degrees2=7, 
                             calc_ts='pattern cov'),
                 BivariateMI(name='sst', func=BivariateMI.corr_map, 
                             kwrgs_func={'alpha':.0001, 'FDR_control':True}, 
                             distance_eps=600, min_area_in_degrees2=5),
                 BivariateMI(name='sm2', func=BivariateMI.corr_map, 
                              kwrgs_func={'alpha':.05, 'FDR_control':True}, 
                              distance_eps=600, min_area_in_degrees2=5),
                 BivariateMI(name='sm3', func=BivariateMI.corr_map, 
                              kwrgs_func={'alpha':.05, 'FDR_control':True}, 
                              distance_eps=700, min_area_in_degrees2=7),
                 BivariateMI(name='snow', func=BivariateMI.corr_map, 
                              kwrgs_func={'alpha':.05, 'FDR_control':True}, 
                              distance_eps=700, min_area_in_degrees2=7),
                 BivariateMI(name='st2', func=BivariateMI.corr_map, 
                               kwrgs_func={'alpha':.05, 'FDR_control':True}, 
                               distance_eps=700, min_area_in_degrees2=5)]

list_for_EOFS = [EOF(name='OLR', neofs=1, selbox=[-180, 360, -15, 30])]



rg = RGCPD(list_of_name_path=list_of_name_path, 
           list_for_MI=list_for_MI,
           list_for_EOFS=list_for_EOFS,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=14, lags_i=np.array([1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
           append_pathsub='_' + name_ds)

rg.pp_TV(name_ds=name_ds)
selbox = [None, {'sst':[-180,360,-10,90], 'z500':[130,350,10,90], 'v200':[130,350,10,90]}]
rg.pp_precursors(selbox=selbox)

rg.traintest(method='random10')

rg.calc_corr_maps()

 #%%
rg.cluster_list_MI()
rg.quick_view_labels() 
rg.plot_maps_corr(precursors=['st2'], save=True)

rg.get_EOFs()


rg.get_ts_prec(precur_aggr=None)
rg.PCMCI_df_data(pc_alpha=None, 
                 tau_max=2,
                 alpha_level=.05, 
                 max_dims=6,
                 max_combinations=1)
rg.PCMCI_get_links(alpha_level=.05)
rg.df_sum.loc[0]



rg.plot_maps_sum(var='sm2', 
                 kwrgs_plot={'aspect': 2, 'wspace': -0.02})
rg.plot_maps_sum(var='sm3', 
                 kwrgs_plot={'aspect': 2, 'wspace': -0.02})
rg.plot_maps_sum(var='st2', 
                 kwrgs_plot={'aspect': 2, 'wspace': -0.02})
rg.plot_maps_sum(var='snow', 
                 kwrgs_plot={'aspect': 2, 'wspace': -0.02})
rg.plot_maps_sum(var='sst', 
                 kwrgs_plot={'cbar_vert':.02})

rg.get_ts_prec(precur_aggr=1)
rg.store_df_PCMCI()
#%%
from class_fc import fcev
import os
logitCV = ('logitCV',
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs',
           'max_iter':125,
           'refit':False})

path_data = rg.df_data_filename
name = rg.TV.name
# path_data = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/3_80d77_26jun-21aug_lag14-14_q75tail_random10s1/None_at0.05_tau_0-1_conds_dimNone_combin2_dt14_dtd1.h5'
# name = '3'
kwrgs_events = {'event_percentile': 66}


lags_i = np.array([0, 10, 14, 21, 28, 35])
precur_aggr = 16
use_fold = None


list_of_fc = [fcev(path_data=path_data, precur_aggr=precur_aggr, 
                    use_fold=None, start_end_TVdate=None,
                    stat_model=logitCV, 
                    kwrgs_pp={}, 
                    dataset=f'{precur_aggr} day means exper 1',
                    keys_d='persistence'),
              fcev(path_data=path_data, precur_aggr=precur_aggr, 
                   use_fold=None, start_end_TVdate=None,
                   stat_model=logitCV, 
                   kwrgs_pp={}, 
                   dataset=f'{precur_aggr} day means exper 2',
                   keys_d='all')]
           

                  
for i, fc in enumerate(list_of_fc):


    fc.get_TV(kwrgs_events=kwrgs_events)
    
    fc.fit_models(lead_max=lags_i, verbosity=1)

for i, fc in enumerate(list_of_fc):
    fc.perform_validation(n_boot=500, blocksize='auto', alpha=0.05,
                          threshold_pred=(1.5, 'times_clim'))
    


df_valid, RV, y_pred_all = fc.dict_sum



import valid_plots as dfplots
kwrgs = {'wspace':0.25, 'col_wrap':None, 'threshold_bin':fc.threshold_pred}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision']
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
line_dim = 'dataset'

fig = dfplots.valid_figures(list_of_fc, 
                          line_dim=line_dim,
                          group_line_by=None,
                          met=met, **kwrgs)


working_folder, filename = fc._print_sett(list_of_fc=list_of_fc)

f_format = '.pdf'
pathfig_valid = os.path.join(filename + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600