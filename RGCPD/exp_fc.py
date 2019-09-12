#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:04:46 2019

@author: semvijverberg
"""
import func_fc
import numpy as np
from  more_itertools import unique_everseen


#experiments = { 'ERA-5 30d Only_all_spatcov':(path_data_3d_sp,
#                            {'keys': keys_d['Only_all_spatcov'] } ),
#                'ERA-5 30d Regions_all_spatcov':(path_data_3d_sp,
#                            {'keys': keys_d['Regions_all_spatcov'] } ),
#                'ERA-5 30d only_sp_caus_no_all_sp':(path_data_3d_sp,
#                            {'keys': keys_d['only_sp_caus_no_all_sp'] } ),
#                'ERA-5 30d Regions_sp_caus_no_all_sp':(path_data_3d_sp,
#                            {'keys': keys_d['Regions_sp_caus_no_all_sp'] } )}

def compare_use_spatcov(path_data, causal=True):
    #%%
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    df_sum  = dict_of_dfs['df_sum']
    
    keys_d = {}
    
    keys_d_ = {}
    for s in splits:
        if causal == True:
            # causal
            all_keys = df_sum[df_sum['causal']].loc[s].index
        elif causal == False:
            # correlated
            all_keys = df_sum.loc[s].index.delete(0)
            
        # Regions + all_spatcov(_caus)
        keys_ = [k for k in all_keys if ('spatcov_caus' in k) and k[:3] == 'all']
        keys_d_[s] = np.array((keys_))
        
    keys_d['Only_all_spatcov'] = keys_d_
    
    
    keys_d_ = {}
    for s in splits:
        if causal == True:
            # causal
            all_keys = df_sum[df_sum['causal']].loc[s].index
        elif causal == False:
            # correlated
            all_keys = df_sum.loc[s].index.delete(0)
            
        # Regions + all_spatcov(_caus)
        keys_ = [k for k in all_keys if ('spatcov_caus' not in k) or k[:3] == 'all']
        keys_d_[s] = np.array((keys_))
        
    keys_d['Regions_all_spatcov'] = keys_d_
    
    keys_d_ = {}
    for s in splits:
        if causal == True:
            # causal
            all_keys = df_sum[df_sum['causal']].loc[s].index
        elif causal == False:
            # correlated
            all_keys = df_sum.loc[s].index.delete(0)

        # only spatcov(_caus) (no all_spatcov)
        keys_ = [k for k in all_keys if ('spatcov' in k) and ('all' not in k)]

        keys_d_[s] = np.array((keys_))
    keys_d['only_sp_caus_no_all_sp'] = keys_d_
    

    keys_d_ = {}
    for s in splits:
        if causal == True:
            # causal
            all_keys = df_sum[df_sum['causal']].loc[s].index
        elif causal == False:
            # correlated
            all_keys = df_sum.loc[s].index.delete(0)

        # only spatcov(_caus) (no all_spatcov)
        keys_ = [k for k in all_keys if ('all' not in k)]

        keys_d_[s] = np.array((keys_))
    keys_d['Regions_sp_caus_no_all_sp'] = keys_d_
    
    
    #%%
    return keys_d

def normal_precursor_regions(path_data, causal=True):
    #%%
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    df_sum  = dict_of_dfs['df_sum']
    
    keys_d = {}
    keys_d_ = {}
    for s in splits:
        if causal == True:
            # causal
            all_keys = df_sum[df_sum['causal']].loc[s].index
        elif causal == False:
            # correlated
            all_keys = df_sum.loc[s].index.delete(0)
            
        # Regions + all_spatcov(_caus)
        keys_ = [k for k in all_keys if ('spatcov' not in k)]
        keys_d_[s] = np.array(list(unique_everseen(keys_)))
        
    keys_d['normal_precursor_regions'] = keys_d_
        
    #%%
    return keys_d

def CPPA_precursor_regions(path_data, option='all'):
    #%%
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    
    keys_d = {}
    keys_d_ = {}
    for s in splits:
        if option == 'all':
            all_keys = df_data.loc[s].columns
#            keys_ = all_keys
            keys_ = [k for k in all_keys if k[0] == '0' ] # remove later
        elif option == 'only_ts':
            all_keys = df_data.loc[s].columns
            keys_ = [k for k in all_keys if ('spatcov' not in k)]
        keys_d_[s] = np.array(list(unique_everseen(keys_)))
        
    keys_d['CPPA_precursor_regions'] = keys_d_
        
    #%%
    return keys_d