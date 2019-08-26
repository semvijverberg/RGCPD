#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:58:26 2019

@author: semvijverberg
"""
#%%

import pandas as pd
import numpy as np
import func_fc
import matplotlib.pyplot as plt
import validation as valid

# =============================================================================
# load data 
# =============================================================================

path_data =  '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_m01-08_dt14/9jun-18aug_t2mmax_E-US_lag0-0/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'
path_data_sm = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt10/21jun-20aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'
path_data_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt30/11jun-10aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-25.h5'


lags = [0,1,2]

n_boot = 500



    
stat_model = ('GBR', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 500,
               'max_features':'sqrt',
               'subsample' : 0.6} )
  

#stat_model = ('GBR_classes', 
#              {'max_depth':5,
#               'learning_rate':1E-3,
#               'n_estimators' : 500,
#               'max_features':'sqrt',
#               'subsample' : 0.6} )     

kwrgs_events = {'event_percentile': 'std',
                'min_dur' : 1,
                'max_break' : 0,
                'grouped' : False}




#%%

def forecast_wrapper(datasets=dict, keys_d=dict, kwrgs_events=dict, stat_model_l=list, lags=list, n_boot=0):
    '''
    dict should have splits (as keys) and concomitant list of keys of that particular split 
    '''
    #%%
    
    
    df_data = func_fc.load_hdf5(path_data)['df_data']
    
    
    # create info on class
    class RV_class:
        def __init__(self, df_data, kwrgs_events=None):
            self.RV_ts = df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] 
            self.RVfullts = df_data[df_data.columns[0]][0]
            if kwrgs_events != None:
                self.threshold = func_fc.Ev_threshold(self.RV_ts, 
                                                  kwrgs_events['event_percentile'])
            self.RV_b_full = func_fc.Ev_timeseries(df_data[df_data.columns[0]][0], 
                                   threshold=self.threshold , 
                                   min_dur=kwrgs_events['min_dur'],
                                   max_break=kwrgs_events['max_break'], 
                                   grouped=kwrgs_events['grouped'])[0]
            self.RV_bin   = self.RV_b_full[df_data['RV_mask'][0]] 
            self.dates_all = self.RV_b_full.index
            self.dates_RV = self.RV_bin.index
            self.TrainIsTrue = df_data['TrainIsTrue']
            self.RV_mask = df_data['RV_mask']
            self.prob_clim = func_fc.get_obs_clim(self)
            
    RV = RV_class(df_data, kwrgs_events)
    dict_sum = {}
    for stat_model in stat_model_l:
        name = stat_model[0]
        df_valid, RV, y_pred_all = func_fc.forecast_and_valid(RV, df_data, keys_d, stat_model=stat_model, lags=lags, n_boot=n_boot)
        dict_sum[name] = (df_valid, RV, y_pred_all)
    #%%    
    return dict_sum  
#%%
logit_model = ('logit', None)

GBR_model = ('GBR', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 500,
               'max_features':'sqrt',
               'subsample' : 0.6} )
    
logitCV = ('logit-CV', { 'class_weight':{ 0:1, 1:1},
                'scoring':'brier_score_loss',
                'penalty':'l2',
                'solver':'lbfgs'})
    
GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 500,
               'max_features':'sqrt',
               'subsample' : 0.6} )  
    
stat_model_l = [logitCV, GBR_logitCV]


datasets = {'ERA-510d':path_data_sm, 'ERA-5 30d':path_data_30d}
dict_datasets = {}
for dataset, path_data in datasets.items():
    
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    
    df_sum  = dict_of_dfs['df_sum']
    # determine keys per dataset
    keys_d = {}
    for s in splits:
        keys_ = df_sum[df_sum['causal']].loc[s].index
        keys_ = df_sum.loc[s].index.delete(0)
        keys_d[s] = np.array((keys_))
        
    dict_sum = forecast_wrapper(path_data, keys_d=keys_d, kwrgs_events=kwrgs_events, 
                            stat_model_l=stat_model_l, 
                            lags=lags, n_boot=n_boot)

    dict_datasets[dataset] = dict_sum

valid.valid_figures(dict_datasets, met='default')
#for stat_model in stat_model_l:
#    name = stat_model[0]
#    
#    df_valid, RV, y_pred_all = dict_sum[name]
#    print(df_valid)


#plt.figure()
#plt.plot_date(y_pred_all.index, y_pred_all[1], 'b-')
#plt.plot_date(y_pred_all.index, RV.RV_bin, 'b-', alpha=0.5)