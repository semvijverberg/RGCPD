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

def normal_precursor_regions(path_data, keys_options=['all'], causal=True):
    #%%
    '''
    keys_options=['all', 'only_db_regs', 'sp_and_regs', 'sst+sm+RWT',
                  'sst(CPPA)+sm', 'sst(PEP)+sm', 'sst(PDO,ENSO)+sm']
    '''
    
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    df_sum  = dict_of_dfs['df_sum']
    
    skip = ['all_spatcov', '0_2_sm123', '0_101_PEPspatcov', 'sm123_spatcov']
    
    
    
    keys_d = {}
    for option in keys_options:
        keys_d_ = {}
        for s in splits:
            
            if causal == True:
                # causal
                keys_ = df_sum[df_sum['causal']].loc[s].index
                
            elif causal == False:
                # correlated
                all_keys = df_sum.loc[s].index.delete(0)
                # remove spatcov_causals
                all_keys = [k for k in all_keys if k[-4:] != 'caus']
                
                
            if option == 'all':
                keys_ = [k for k in all_keys if k not in skip]
            elif option == 'only_db_regs':                
                # Regions + all_spatcov(_caus)
                keys_ = [k for k in all_keys if ('spatcov' not in k)]
                keys_ = [k for k in keys_ if k not in skip]
            elif option == 'sp_and_regs': 
                keys_ = [k for k in all_keys if k not in skip]
            elif option == 'sst+sm+RWT': 
                keys_ = [k for k in all_keys if k[-7:] != 'v200hpa']
                keys_ = [k for k in keys_ if k not in skip]
            elif option == 'sst(CPPA)+sm': 
                keys_ = [k for k in all_keys if 'v200hpa' not in k]
                keys_ = [k for k in keys_ if k not in skip]
            elif option == 'sst(PEP)+sm': 
                keys_ = [k for k in all_keys if 'sm' in k or 'PEP' in k]
                keys_ = [k for k in keys_ if k != 'sm123_spatcov']
#            elif option == 'sst(PDO,ENSO)+sm':
#                keys_ = [k for k in all_keys if 'sm' in k or 'PEP' in k]
#                keys_ = [k for k in keys_ if k != 'sm123_spatcov']
                
            keys_d_[s] = np.array(list(unique_everseen(keys_)))
            
        keys_d[option] = keys_d_
        
    #%%
    return keys_d


def CPPA_precursor_regions(path_data, keys_options=['CPPA']):
    #%%
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    skip = ['TrainIsTrue', 'RV_mask']
    keys_d = {}
    
    for option in keys_options:
        keys_d_ = {}
        for s in splits:
            if option == 'CPPA':
                not_robust = ['0_101_PEPspatcov', 'PDO', 'ENSO_34']
                all_keys = df_data.loc[s].columns[1:]
                all_keys = [k for k in all_keys if k not in skip]
                all_keys = [k for k in all_keys if k not in not_robust]
                keys_ = all_keys
                
            elif option == 'robust':
                not_robust = ['0_101_PEPspatcov', 'PDO', 'ENSO_34']
                all_keys = df_data.loc[s].columns[1:]
                all_keys = [k for k in all_keys if k not in skip]
                all_keys = [k for k in all_keys if k not in not_robust]
                
                robust = ['0_100_CPPAspatcov', '2', '7', '9' ]
                sst_regs = [k for k in all_keys if len(k.split('_')) == 3]
                other    = [k for k in all_keys if len(k.split('_')) != 3]
                keys_ = [k for k in sst_regs if k.split('_')[1] in robust ] 
                [keys_.append(k) for k in other]
                
            elif option == 'PEP':
                all_keys = df_data.loc[s].columns[1:]
                all_keys = [k for k in all_keys if k not in skip]
                keys_ = [k for k in all_keys if k.split('_')[-1] == 'PEPspatcov']
        
            keys_d_[s] = np.array(list(unique_everseen(keys_)))        
        keys_d[option] = keys_d_
        
    #%%
    return keys_d