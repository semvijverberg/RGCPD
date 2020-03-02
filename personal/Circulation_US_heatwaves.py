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
#%%

TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf5_nc5_dendo_80d77.nc'
cluster_label = 3
list_of_name_path = [(cluster_label, TVpath), 
                     ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]




list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map, 
                              kwrgs_func={'alpha':.01, 'FDR_control':True}, 
                              distance_eps=600, min_area_in_degrees2=7),
                 BivariateMI(name='v200', func=BivariateMI.corr_map, 
                               kwrgs_func={'alpha':.01, 'FDR_control':True}, 
                               distance_eps=600, min_area_in_degrees2=5)]

start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')
name_ds='q75tail'


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
                     ('snow',os.path.join(path_raw, 'snow_1979-2018_1_12_daily_1.0deg.nc'))]

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
                              distance_eps=700, min_area_in_degrees2=7)]

list_for_EOFS = [EOF(name='OLR', neofs=1, selbox=[-180, 360, -15, 30])]

start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')


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
rg.plot_maps_corr(save=True)


rg.get_ts_prec(precur_aggr=None)
rg.PCMCI_df_data(pc_alpha=None, 
                 tau_max=2,
                 alpha_level=.05, 
                 max_combinations=3)
rg.df_sum.loc[0]



rg.plot_maps_sum()

rg.get_ts_prec(precur_aggr=1)
rg.store_df_PCMCI()
#%%
from class_fc import fcev
logitCV = ('logitCV',
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs',
           'max_iter':100,
           'refit':False})

GBC = ('GBC',
      {'max_depth':1,
       'learning_rate':.05,
       'n_estimators' : 500,
       'min_samples_split':.25,
       'max_features':.4,
       'subsample' : .6,
       'random_state':60,
       'n_iter_no_change':20,
       'tol':1E-4,
       'validation_fraction':.3,
       'scoringCV':'brier_score_loss'
       } )

path_data = rg.df_data_filename
name = rg.TV.name
datasets_path = {f'cluster {name}':(path_data, ['persistence', None])}
kwrgs_events = {'event_percentile': 66}
stat_model_l = [logitCV, GBC]
kwrgs_pp     = {'add_autocorr' : True, 'normalize':False}

lags_i = np.array([0, 10, 15, 21])
precur_aggr = 15
use_fold = None



dict_experiments = {} ; list_of_fc = []
for dataset, tuple_sett in datasets_path.items():
    path_data = tuple_sett[0]
    keys_d_list = tuple_sett[1]
    for keys_d in keys_d_list:

        fc = fcev(path_data=path_data, precur_aggr=precur_aggr, use_fold=use_fold)
        fc.get_TV(kwrgs_events=kwrgs_events)
        fc.fit_models(stat_model_l=stat_model_l, lead_max=lags_i,
                           keys_d=keys_d, kwrgs_pp=kwrgs_pp, verbosity=1)

        fc.perform_validation(n_boot=500, blocksize='auto', alpha=0.05,
                              threshold_pred=(1.5, 'times_clim'))
        dict_experiments[dataset+'_'+str(keys_d)] = fc.dict_sum
        list_of_fc.append(fc)

y_pred_all, y_pred_c = fc.dict_preds[fc.stat_model_l[0][0]]




import valid_plots as dfplots
kwrgs = {'wspace':0.25, 'col_wrap':None, 'threshold_bin':fc.threshold_pred}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision']
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
expers = list(dict_experiments.keys())
models   = list(dict_experiments[expers[0]].keys())
line_dim = 'exper'

fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
                          line_dim=line_dim,
                          group_line_by=None,
                          met=met, **kwrgs)


working_folder, filename = fc._print_sett(list_of_fc=list_of_fc)

f_format = '.pdf'
pathfig_valid = os.path.join(filename + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600