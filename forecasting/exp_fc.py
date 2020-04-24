#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:04:46 2019

@author: semvijverberg
"""
import functions_pp
import numpy as np
import itertools 
flatten = lambda l: list(itertools.chain.from_iterable(l))


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
    dict_of_dfs = functions_pp.load_hdf5(path_data)
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

def normal_precursor_regions(path_data, keys_options=['all'], causal=False):
    #%%
    '''
    keys_options=['all', 'only_db_regs', 'sp_and_regs', 'sst+sm+RWT',
                  'CPPA+sm', 'sst(PEP)+sm', 'sst(PDO,ENSO)+sm',
                  'CPPA', 'PEP', 'sst combined', 'sst combined + sm', 
                  'sst(CPPA) expert knowledge', 'sst(CPPA Pattern)'
                  'CPPA Pattern', 'PDO+ENSO', 'persistence', 'CPPA+PEP+sm']
                  
    '''
    
    
    dict_of_dfs = functions_pp.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    try:
        df_sum  = dict_of_dfs['df_sum']
    except:
        pass

#    skip = ['all_spatcov', '0_2_sm123', '0_101_PEPspatcov', 'sm123_spatcov']
    skip = ['all_spatcov']
    

    keys_d = {}
    for option in keys_options:
        keys_d_ = {}
        for s in splits:

            if causal == True or 'causal' in option:
                # causal
                all_keys = df_sum[df_sum['causal']].loc[s].index

            elif causal == False and 'causal' not in option:
                # correlated
                df_s = df_data.loc[s]
                all_keys = df_s.columns.delete(0)
                # extract only float columns
                mask_f = np.logical_or(df_s.dtypes == 'float64', df_s.dtypes == 'float32')
                all_keys = all_keys[mask_f[1:].values]
                # remove spatcov_causals
                all_keys = [k for k in all_keys if k[-4:] != 'caus']
                

            if option == 'all':
                # extract only float columns
                keys_ = [k for k in all_keys if k not in skip]
                
            elif 'only_db_regs' in option:
                # Regions + all_spatcov(_caus)
                keys_ = [k for k in all_keys if ('spatcov' not in k)]
                keys_ = [k for k in keys_ if k not in skip]
            elif option == 'sp_and_regs':
                keys_ = [k for k in all_keys if k not in skip]
            elif option == 'CPPA':
                skip_ex = ['0..103..PEPsv',  'sm123_spatcov', 'all_spatcov']
                keys_ = [k for k in all_keys if 'v200hpa' not in k]
                keys_ = [k for k in keys_ if 'sm' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]# or 'PDO' not in k]
                keys_ = [k for k in keys_ if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'PEPsv' not in k]
                keys_ = [k for k in keys_ if 'OLR' not in k] 
                keys_ = [k for k in keys_ if k not in skip_ex]
            elif option == 'sst combined':
                keys_ = [k for k in all_keys if 'sm' not in k]
            elif option == 'sst combined+sm':
                keys_ = all_keys
            elif option == 'sst(CPPA Pattern)' or option == 'CPPA Pattern':
                keys_ = [k for k in all_keys if 'CPPAsv' in k]
            elif option == 'sst+sm+z500':
                keys_ = []
                keys_.append([k for k in all_keys if '..sst' in k])
                keys_.append([k for k in all_keys if '..sm' in k])
                keys_.append([k for k in all_keys if '..z500' in k])
                keys_ = flatten(keys_)

            elif option == 'CPPA+sm':
                keys_ = [k for k in all_keys if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]
                keys_ = [k for k in keys_ if 'PEP' not in k]  
                keys_ = [k for k in keys_ if 'OLR' not in k]  
                keys_ = [k for k in keys_ if k not in skip]
            elif option == 'CPPA+PEP+sm':
                keys_ = [k for k in all_keys if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]
            elif option == 'CPPA+sm+OLR':
                keys_ = [k for k in all_keys if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]
                keys_ = [k for k in keys_ if 'PEP' not in k]  
                keys_ = [k for k in keys_ if k not in skip]
            elif option == 'CPPAregs+sm':
                keys_ = [k for k in all_keys if 'v200hpa' not in k]
                keys_ = [k for k in keys_ if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]
                keys_ = [k for k in keys_ if 'PEP' not in k]  
                keys_ = [k for k in keys_ if ('CPPAsv' not in k)]
            elif option == 'CPPApattern+sm':
                skip_ex = ['0..100..ENSO34','0..101..PDO']
                keys_ = [k for k in all_keys if 'v200hpa' not in k]
                keys_ = [k for k in keys_ if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]
                keys_ = [k for k in keys_ if 'PEP' not in k]  
                keys_ = [k for k in keys_ if 'OLR' not in k] 
                keys_ = [k for k in keys_ if ('spatcov' in k or 'sm' in k)]
            elif option == 'sm':
                keys_ = [k for k in all_keys if 'sm' in k]
                keys_ = [k for k in keys_ if 'spatcov' not in k]
            elif option == 'sst(PEP)+sm':
                keys_ = [k for k in all_keys if 'sm' in k or 'PEP' in k]
                keys_ = [k for k in keys_ if k != 'sm123_spatcov']
            elif option == 'PEP':
                keys_ = [k for k in all_keys if 'PEP' in k]
            elif option == 'sst(PDO,ENSO)+sm':
                keys_ = [k for k in all_keys if 'sm' in k or 'PDO' in k or 'ENSO' in k]
                keys_ = [k for k in keys_ if 'spatcov' not in k]
            elif option == 'PDO+ENSO':
                keys_ = [k for k in all_keys if 'PDO' in k or 'ENSO' in k]
                keys_ = [k for k in keys_ if 'spatcov' not in k]
            elif option == 'sst(CPPA) expert knowledge':
                keys_ = [k for k in all_keys if 'sm' not in k]
                keys_ = [k for k in keys_ if 'PDO' not in k]
                keys_ = [k for k in keys_ if 'ENSO' not in k]
                keys_ = [k for k in keys_ if 'PEP' not in k]                
                expert = ['CPPAsv', '..9..sst', '..2..sst', '..6..sst', '..1..sst', '..7..sst']
                keys_ = [k for k in keys_ for e in expert if e in k]
            if option == 'persistence':
                keys_ = []

            
            keys_d_[s] = np.unique(keys_)

        keys_d[option] = keys_d_

    #%%
    return keys_d


def CPPA_precursor_regions(path_data, keys_options=['CPPA']):
    #%%
    dict_of_dfs = functions_pp.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    skip = ['TrainIsTrue', 'RV_mask']
    keys_d = {}

    for option in keys_options:
        keys_d_ = {}
        for s in splits:

            if option == 'robust':
                not_robust = ['0_101_PEPspatcov', 'PDO', 'ENSO_34',
                              'ENSO_34', 'PDO']
                all_keys = df_data.loc[s].columns[1:]
                all_keys = [k for k in all_keys if k not in skip]
                all_keys = [k for k in all_keys if k not in not_robust]

                robust = ['0_100_CPPAspatcov', '2', '7', '9' ]
                sst_regs = [k for k in all_keys if len(k.split('_')) == 3]
                other    = [k for k in all_keys if len(k.split('_')) != 3]
                keys_ = [k for k in sst_regs if k.split('_')[1] in robust ]
                [keys_.append(k) for k in other]

            elif option == 'CPPA':
                not_robust = ['0_101_PEPspatcov', '0_104_PDO', '0_103_ENSO34',
                              'ENSO_34', 'PDO', '0_900_ENSO34', '0_901_PDO']
                all_keys = df_data.loc[s].columns[1:]
                all_keys = [k for k in all_keys if k not in skip]
                all_keys = [k for k in all_keys if k not in not_robust]
                keys_ = all_keys

            elif option == 'PEP':
                all_keys = df_data.loc[s].columns[1:]
                all_keys = [k for k in all_keys if k not in skip]
                keys_ = [k for k in all_keys if k.split('_')[-1] == 'PEPspatcov']

            keys_d_[s] = np.unique(keys_)
        keys_d[option] = keys_d_

    #%%
    return keys_d
