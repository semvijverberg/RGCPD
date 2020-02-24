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
#%%
from RGCPD import RGCPD

TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW_dendo_f30ff.nc'
cluster_label = 4
list_of_name_path = [(cluster_label, TVpath), 
                     ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]


start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')
kwrgs_corr = {'alpha':1E-2}

rg = RGCPD(list_of_name_path=list_of_name_path, 
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([0,1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW')

rg.pp_TV()
rg.pp_precursors(selbox=(-180, 360, -10, 90))

rg.traintest('no_train_test_split')

rg.calc_corr_maps(**kwrgs_corr) 

rg.plot_maps_corr(save=True)


#%%
from RGCPD import RGCPD

list_of_name_path = [(cluster_label, TVpath), 
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm2', os.path.join(path_raw, 'sm2_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm3', os.path.join(path_raw, 'sm3_1979-2018_1_12_daily_1.0deg.nc'))]


start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')
kwrgs_corr = {'alpha':1E-3}

rg = RGCPD(list_of_name_path=list_of_name_path, 
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW')

rg.pp_TV()
selbox = [None, {'sst':[-180,360,-10,90]}]
rg.pp_precursors(selbox=selbox)

rg.traintest(method='random10')

rg.calc_corr_maps(alpha=1E-3)
 #%%
rg.cluster_regions(distance_eps=700, min_area_in_degrees2=5)
rg.quick_view_labels() 
rg.get_ts_prec(precur_aggr=1)
rg.plot_maps_corr(save=True)
rg.df_data

rg.store_df()

#%%
from class_fc import fcev
logitCV = ('logitCV',
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs',
           'max_iter':100,
           'refit':False})

path_data = rg.df_data_filename
name = rg.TV.name
datasets_path = {f'cluster {name}':(path_data, [None])}
kwrgs_events = {'event_percentile': 66}
stat_model_l = [logitCV]
kwrgs_pp     = {'add_autocorr' : True, 'normalize':'datesRV'}

lags_i = np.array([0, 10, 15, 21])
precur_aggr = 16
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

# In[8]:


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