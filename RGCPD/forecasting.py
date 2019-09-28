#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:58:26 2019

@author: semvijverberg
"""
#%%
import time
start_time = time.time()
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
script_dir = "/Users/semvijverberg/surfdrive/Scripts/RGCPD/RGCPD" # script directory
# To link modules in RGCPD folder to this script
os.chdir(script_dir)
sys.path.append(script_dir)
from importlib import reload as rel
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

#path_data =  '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_m01-08_dt14/9jun-18aug_t2mmax_E-US_lag0-0/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'
#rand_10d_sm = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt10/21jun-20aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'
#rand_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-25.h5'
#path_data_3d_sp = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-30.h5'
#strat_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-09-03.h5'
#strat_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt10/21jun-20aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-09-03.h5'
strat_1d_CPPA_era5 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
strat_1d_CPPA_EC   = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/ran_strat10_s30/data/EC_16-09-19_19hr_lag_0.h5'
CPPA_v_sm_20d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_v200hpa_sm123_m01-09_dt20/13jun-12aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-18.h5'
CPPA_v_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_v200hpa_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-20.h5'
CPPA_sm_10d   = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
n_boot = 2000












def forecast_wrapper(datasets=dict, kwrgs_exp=dict, kwrgs_events=dict, stat_model_l=list, lags_i=list, n_boot=0):
    '''
    dict should have splits (as keys) and concomitant list of keys of that particular split 
    '''

    
    
    df_data = func_fc.load_hdf5(path_data)['df_data']
    splits  = df_data.index.levels[0]
    RVfullts = pd.DataFrame(df_data[df_data.columns[0]][0])
    RV_ts    = pd.DataFrame(df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] )
    fit_model_dates = kwrgs_exp['kwrgs_pp']['fit_model_dates']
    RV = func_fc.RV_class(RVfullts, RV_ts, kwrgs_events, 
                          fit_model_dates=fit_model_dates)
    
    RV.TrainIsTrue = df_data['TrainIsTrue']
    RV.RV_mask = df_data['RV_mask']
    fit_model_mask = pd.concat([RV.fit_model_mask] * 10, keys=splits)
    df_data = df_data.merge(fit_model_mask, left_index=True, right_index=True)
    RV.prob_clim = func_fc.get_obs_clim(RV)
    
    dict_sum = {}
    for stat_model in stat_model_l:
        name = stat_model[0]
        df_valid, RV, y_pred_all = func_fc.forecast_and_valid(RV, df_data, kwrgs_exp, 
                                                              stat_model=stat_model, 
                                                              lags_i=lags_i, n_boot=n_boot)
        dict_sum[name] = (df_valid, RV, y_pred_all)
   
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

GBR_logitCV_tuned = ('GBR-logitCV', 
          {'max_depth':[3,5,7],
           'learning_rate':1E-3,
           'n_estimators' : 750,
           'max_features':'sqrt',
           'subsample' : 0.6} ) 
    

# format {'dataset' : (path_data, list(keys_options) ) }

ERA_and_EC  = {'ERA-5':(strat_1d_CPPA_era5, ['PEP', 'CPPA']),
                 'EC-earth 2.3':(strat_1d_CPPA_EC, ['PEP', 'CPPA'])}
stat_model_l = [GBR_logitCV]

#
#ERA         = {'ERA-5:':(CPPA_sm_10d, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}
#ERA_Bram         = {'ERA-5:':(CPPA_sm_10d, ['sst(CPPA)+sm'])}
#stat_model_l = [logit, GBR_logitCV]
#
#ERA_sp      = {'ERA-5:':(CPPA_sm_10d, ['CPPAregs+sm', 'CPPApattern+sm', 'sst(CPPA)+sm'])}
#stat_model_l = [logit, GBR_logitCV]

datasets_path = ERA_and_EC

causal = False
experiments = {} #; keys_d_sets = {}
for dataset, path_key in datasets_path.items():
#    keys_d = exp_fc.compare_use_spatcov(path_data, causal=causal)
    
    path_data = path_key[0]
    keys_options = path_key[1]
    
    keys_d = exp_fc.CPPA_precursor_regions(path_data, 
                                           keys_options=keys_options)
    
#    keys_d = exp_fc.normal_precursor_regions(path_data, 
#                                             keys_options=keys_options,
#                                             causal=causal)
    
    

    for master_key, feature_keys in keys_d.items():
        kwrgs_pp = {'EOF':False, 
                    'expl_var':0.5,
                    'fit_model_dates' : None}
        experiments[dataset+' '+master_key] = (path_data, {'keys':feature_keys,
                                           'kwrgs_pp':kwrgs_pp
                                           })

    
kwrgs_events = {'event_percentile': 'std',
                'min_dur' : 1,
                'max_break' : 0,
                'grouped' : False}



dict_experiments = {}
for dataset, tuple_sett in experiments.items():
    path_data = tuple_sett[0]
    kwrgs_exp = tuple_sett[1]
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    tfreq = (df_data.loc[0].index[1] - df_data.loc[0].index[0]).days
    if tfreq == 1:
        lags_i = np.arange(0, 70+1E-9, max(10,tfreq), dtype=int)
    else:
        lags_i = np.array(np.arange(0, 70+1E-9, max(10,tfreq))/max(10,tfreq), dtype=int)

#    lags_i = np.array([0], dtype=int)
    
    if 'keys' not in kwrgs_exp:
        # if keys not defined, getting causal keys
        kwrgs_exp['keys'] = exp_fc.normal_precursor_regions(path_data, causal=True)['normal_precursor_regions']

    print(kwrgs_events)
    dict_sum = forecast_wrapper(path_data, kwrgs_exp=kwrgs_exp, kwrgs_events=kwrgs_events, 
                            stat_model_l=stat_model_l, 
                            lags_i=lags_i, n_boot=n_boot)

    dict_experiments[dataset] = dict_sum
    
df_valid, RV, y_pred = dict_sum[stat_model_l[-1][0]]



def print_sett(experiments, stat_model_l, filename):
    f= open(filename+".txt","w+")
    lines = []
    
    lines.append("\nEvent settings:")
    lines.append(kwrgs_events)
    
    lines.append(f'\nModels used:')  
    for m in stat_model_l:
        lines.append(m)
        
    lines.append(f'\nnboot: {n_boot}')
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
    [print(n) for n in lines]

        
  
RV_name = 't2mmax'
working_folder = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/forecasting'
pdfs_folder = os.path.join(working_folder,'pdfs')
if os.path.isdir(pdfs_folder) != True : os.makedirs(pdfs_folder)
today = datetime.datetime.today().strftime('%Hhr_%Mmin_%d-%m-%Y')
f_name = f'{RV_name}_{tfreq}d_{today}'


#%%

#rename_ERA =    {'ERA-5: sst(PEP)+sm':'PEP+sm', 
#             'ERA-5: sst(PDO,ENSO)+sm':'PDO+ENSO+sm', 
#             'ERA-5: sst(CPPA)+sm':'CPPA+sm'}
#
#for old, new in rename_ERA.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

#rename_EC = {'ERA-5 PEP':'PEP', 
#             'ERA-5 CPPA':'CPPA', 
#             'EC-earth 2.3 PEP':'PEP ', 
#             'EC-earth 2.3 CPPA':'CPPA '}
#
#for old, new in rename_EC.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

#rename_CPPA_comp =    {'ERA-5: CPPAregs+sm' : 'precursor regions + sm', 
#                       'ERA-5: CPPApattern+sm': 'precursor pattern + sm', 
#                       'ERA-5: sst(CPPA)+sm' : 'CPPA (all) + sm'}
#
#for old, new in rename_CPPA_comp.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

f_format = '.png' 
filename = os.path.join(working_folder, f_name)




group_line_by = None
#group_line_by = ['ERA-5', 'EC']
kwrgs = {'wspace':0.08}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
expers = list(dict_experiments.keys())
models   = list(dict_experiments[expers[0]].keys())

fig = valid.valid_figures(dict_experiments, expers=expers, models=models,
                          line_dim='exper', 
                          group_line_by=group_line_by,  
                          met=met, **kwrgs)
if f_format == '.png':
    fig.savefig(os.path.join(filename + f_format), 
                bbox_inches='tight') # dpi auto 600
elif f_format == '.pdf':
    fig.savefig(os.path.join(pdfs_folder,f_name+ f_format), 
                bbox_inches='tight')
    
print_sett(experiments, stat_model_l, filename)


np.save(filename + '.npy', dict_experiments)
#%%
# =============================================================================
# Cross-correlation matrix
# =============================================================================
#f_format = '.pdf' 
#
#path_data = strat_1d_CPPA_era5
#win = 1
#
#period = ['fullyear', 'summer60days', 'pre60days'][1]
#df_data = func_fc.load_hdf5(path_data)['df_data']
##df_data['0_104_PDO'] = df_data['0_104_PDO'] * -1
#f_name = f'Cross_corr_strat_1d_CPPA_era5_win{win}_{period}'
#columns = ['t2mmax', '0_100_CPPAspatcov', '0_101_PEPspatcov', '0_901_PDO', '0_900_ENSO34']
#rename = {'t2mmax':'T95', 
#          '0_100_CPPAspatcov':'CPPA', 
#          '0_101_PEPspatcov':'PEP',
#          '0_901_PDO' : 'PDO',
#          '0_900_ENSO34': 'ENSO'}
#valid.build_ts_matric(df_data, win=win, lag=0, columns=columns, rename=rename, period=period)
#if f_format == '.png':
#    plt.savefig(os.path.join(working_folder, f_name + f_format), 
#                bbox_inches='tight') # dpi auto 600
#elif f_format == '.pdf':
#    plt.savefig(os.path.join(pdfs_folder,f_name+ f_format), 
#                bbox_inches='tight')

# =============================================================================
#%% Translation to extremes
# =============================================================================

# import original timeseries:
path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
RVts_filename = 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4.npy'
RVfullts = np.load(os.path.join(path_ts, RVts_filename),
                        encoding='latin1', allow_pickle=True).item()['RVfullts95']

import functions_pp
dates = functions_pp.get_oneyr(y_pred.index)
#dates = y_pred.index
tfreq = (dates[1] - dates[0]).days
fc_values = np.repeat(y_pred[2].values, tfreq)
start_date = dates[0] - pd.Timedelta(f'{tfreq/2}d')
end_date   = dates[-1] + pd.Timedelta(f'{-1+tfreq/2}d')
daily  = pd.DatetimeIndex(start=start_date, end=end_date,
                                freq=pd.Timedelta('1d'))
ext_dates = functions_pp.make_dates(y_pred.index, daily, 2018)
df_fc_ext = pd.DataFrame(fc_values, index=ext_dates, columns=['fc'])
RVts_daily = RVfullts.sel(time=ext_dates)
threshold = func_fc.Ev_threshold(RVts_daily, 90)
RV_bin    = func_fc.Ev_timeseries(RVts_daily,
                               threshold=threshold ,
                               min_dur=4,
                               max_break=1,
                               grouped=True)[0]     

df_fc_ext['RV'] = RV_bin                                         
fraction_of_positives,mean_predicted_values = calibration_curve(df_fc_ext['RV'], df_fc_ext['fc'], n_bins=5); 
clim_prob = RV_bin[RV_bin==1].size/RV_bin.size
plt.hlines(1, 0, 1)
plt.scatter(mean_predicted_values, fraction_of_positives/clim_prob) 
