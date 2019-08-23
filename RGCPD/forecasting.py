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
path_data = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_u500hpa_sm3_m01-08_dt10/21jun-20aug_lag0-0_random10_s30/pcA_none_ac0.01_at0.05_subinfo/fulldata_pcA_none_ac0.01_at0.05_2019-08-22.h5'

dict_of_dfs = func_fc.load_hdf5(path_data)
df_data = dict_of_dfs['df_data']
splits  = df_data.index.levels[0]

df_sum  = dict_of_dfs['df_sum']


kwrgs_events = {'event_percentile': 66,
                'min_dur' : 1,
                'max_break' : 0,
                'grouped' : False}

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
        
RV = RV_class(df_data, kwrgs_events)


#RV_ts.mean()




#%%
lags = [1,2]

stat_model = 'GBC'

kwrgs_GBR = {'max_depth':3,
             'learning_rate':1E-3,
             'n_estimators' : 1250,
             'max_features':'sqrt',
             'subsample' : 0.5}

#kwrgs_GBR = {'max_depth':[1,2,3,4,5,6],
#             'learning_rate':1E-3,
#             'n_estimators' : 1250,
#             'max_features':['sqrt'],
#             'subsample' : [0.5, 0.8]}
n_boot = 500

y_pred_all = []
y_pred_c = []
c = 0
for lag in lags:
    
    y_pred_l = []

    for s in splits:
        c += 1
        progress = 100 * (c-1) / (len(splits) * len(lags))
        print(f"\rProgress {progress}%", end="")
        
        df_norm, RV_mask, TrainIsTrue = func_fc.prepare_data(df_data.loc[s])
        
        
        keys = df_sum[df_sum['causal']].loc[s].index
        
        
#        dates_all = pd.to_datetime(RV.index.values)

        dates_l  = func_fc.func_dates_min_lag(RV.dates_RV, lag, indays=False)[1]
        df_norm = df_norm.loc[dates_l]
#        if s == 0:
#            print(df_norm.iloc[0][0])
#            print(df_norm[df_norm['RV_mask']].iloc[0][0])
        if stat_model == 'logit':
            prediction, model = func_fc.logit_model(RV.RV_bin, df_norm, keys)
        if stat_model == 'GBR':
            prediction, model = func_fc.GBR(RV, df_norm, keys, kwrgs_GBR=kwrgs_GBR)
        if stat_model == 'GBC':
            prediction, model = func_fc.GBC(RV, df_norm, keys, kwrgs_GBR=kwrgs_GBR)
            
        prediction = pd.DataFrame(prediction.values, index=RV.dates_RV,
                                  columns=[lag])
        y_pred_l.append(prediction[(df_norm['TrainIsTrue']==False).values])  
        
        if lag == lags[0]:
            # determining climatological prevailance in training data
            y_c_mask = np.logical_and(df_norm['TrainIsTrue'].values, RV.RV_bin.squeeze().values==1)
            y_clim_val = RV.RV_bin[y_c_mask].size / RV.RV_bin[df_norm['TrainIsTrue'].values].size
            y_clim = RV.RV_bin[df_norm['TrainIsTrue'].values==False].copy()
            y_clim[:] = y_clim_val
            y_pred_c.append(y_clim)
        
    y_pred_l = pd.concat(y_pred_l) 
    y_pred_l = y_pred_l.sort_index()
    
    if lag == lags[0]:
        y_pred_c = pd.concat(y_pred_c) 
        y_pred_c = y_pred_c.sort_index()


    y_pred_all.append(y_pred_l)
y_pred_all = pd.concat(y_pred_all, axis=1) 
print("\n")

#%%

blocksize = valid.get_bstrap_size(RV.RVfullts)
out = valid.get_metrics_sklearn(RV, y_pred_all, y_pred_c, n_boot=n_boot,
                                blocksize=blocksize)
df_auc, df_KSS, df_brier, metrics_dict = out

print(df_brier)




