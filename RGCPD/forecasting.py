#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:58:26 2019

@author: semvijverberg
"""
#%%
import os, datetime
import pandas as pd
import numpy as np
import func_fc
import matplotlib.pyplot as plt
import validation as valid
import exp_fc

# =============================================================================
# load data 
# =============================================================================

path_data =  '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_m01-08_dt14/9jun-18aug_t2mmax_E-US_lag0-0/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'
rand_10d_sm = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt10/21jun-20aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'
rand_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-25.h5'
path_data_3d_sp = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-30.h5'
strat_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-09-03.h5'
strat_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt10/21jun-20aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-09-03.h5'

n_boot = 500










#%%

def forecast_wrapper(datasets=dict, kwrgs_exp=dict, kwrgs_events=dict, stat_model_l=list, lags=list, n_boot=0):
    '''
    dict should have splits (as keys) and concomitant list of keys of that particular split 
    '''
    #%%
    
    
    df_data = func_fc.load_hdf5(path_data)['df_data']
    
    
    RV = func_fc.RV_class(df_data, kwrgs_events)
    
    dict_sum = {}
    for stat_model in stat_model_l:
        name = stat_model[0]
        df_valid, RV, y_pred_all = func_fc.forecast_and_valid(RV, df_data, kwrgs_exp, 
                                                              stat_model=stat_model, 
                                                              lags=lags, n_boot=n_boot)
        dict_sum[name] = (df_valid, RV, y_pred_all)
    #%%    
    return dict_sum  
#%%
logit = ('logit', None)

GBR_model = ('GBR', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 1250,
               'max_features':'sqrt',
               'subsample' : 0.6} )
    
logitCV = ('logit-CV', { 'class_weight':{ 0:1, 1:1},
                'scoring':'brier_score_loss',
                'penalty':'l2',
                'solver':'lbfgs'})
    
GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 750,
               'max_features':'sqrt',
               'subsample' : 0.6} )  
    
stat_model_l = [logit, GBR_logitCV]


#datasets_path = {'ERA-5 30d strat':path_data_strat, 'ERA-5 30d sp':path_data_3d_sp}
datasets_path = {'ERA-5 10d strat':strat_10d, 'ERA-5 30d strat':strat_30d}



causal = False
keys_d_sets = {} ; experiments = {}
for dataset, path_data in datasets_path.items():
#    keys_d = exp_fc.compare_use_spatcov(path_data, causal=causal)
    keys_d = exp_fc.normal_precursor_regions(path_data, causal=causal)
    keys_d_sets[dataset] = keys_d
    for master_key, feature_keys in keys_d.items():
        kwrgs_pp = {'EOF':False, 'expl_var':0.5}
        experiments[dataset] = (path_data, {'keys':feature_keys,
                                           'kwrgs_pp':kwrgs_pp
                                           })
    
kwrgs_events = {'event_percentile': 80,
                'min_dur' : 1,
                'max_break' : 0,
                'grouped' : False}



dict_datasets = {}
for dataset, tuple_sett in experiments.items():
    path_data = tuple_sett[0]
    kwrgs_exp = tuple_sett[1]
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    tfreq = (df_data.loc[0].index[1] - df_data.loc[0].index[0]).days
    lags = np.arange(0, 90+1E-9, tfreq)/tfreq 
    
    df_sum  = dict_of_dfs['df_sum']
    
    if 'keys' not in kwrgs_exp:
        # if keys not defined, getting causal keys
        kwrgs_exp['keys'] = exp_fc.normal_precursor_regions(path_data, causal=True)['normal_precursor_regions']

        
    dict_sum = forecast_wrapper(path_data, kwrgs_exp=kwrgs_exp, kwrgs_events=kwrgs_events, 
                            stat_model_l=stat_model_l, 
                            lags=lags, n_boot=n_boot)

    dict_datasets[dataset] = dict_sum
    
df_valid, RV, y_pred = dict_sum[stat_model_l[-1][0]]



def print_sett(experiments, stat_model_l, filename):
    f= open(filename+".txt","w+")
    lines = []
    lines.append(f'Models used:\n')
    for m in stat_model_l:
        lines.append(m)
    e = 1
    for k, item in experiments.items():
        
        lines.append(f'\n\n***Experiment {e}***\n\n')
        lines.append(f'Title \t : {k}')
        lines.append(f'file \t : {item[0]}')
        for key, it in item[1].items():
            lines.append(f'{key} : {it}')
        e+=1
    
    [print(n, file=f) for n in lines]

    f.close()

        
    

working_folder = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/forecasting'
fig = valid.valid_figures(dict_datasets, met='default')
today = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
f_name = f'{RV.RV_ts.name}_{tfreq}d_{today}'
f_format = '.png' 
filename = os.path.join(working_folder, f_name)
fig.savefig(os.path.join(filename + f_format), bbox_inches='tight') 
print_sett(experiments, stat_model_l, filename)

#for stat_model in stat_model_l:
#    name = stat_model[0]
#    
#    df_valid, RV, y_pred_all = dict_sum[name]
#    print(df_valid)


#plt.figure()
#plt.plot_date(y_pred_all.index, y_pred_all[1], 'b-')
#plt.plot_date(y_pred_all.index, RV.RV_bin, 'b-', alpha=0.5)