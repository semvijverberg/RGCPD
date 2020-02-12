#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:20:31 2019

@author: semvijverberg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:58:26 2019

@author: semvijverberg
"""

import time
start_time = time.time()
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
python_dir = os.path.join(main_dir, 'RGCPD')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(python_dir)
    sys.path.append(df_ana_dir)
import os, datetime
import numpy as np
import func_fc
import valid_plots as dfplots


# =============================================================================
# load data 
# =============================================================================

era5_1d_CPPA = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_0_Xzkup1.h5'
era5_1d_CPPA_l10 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_10_Xzkup1.h5'
strat_1d_CPPA_era5 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
strat_1d_CPPA_era5_l10 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_10.h5'
#strat_1d_CPPA_EC   = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/ran_strat10_s30/data/EC_16-09-19_19hr_lag_0.h5'
#CPPA_v_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_v200hpa_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-20.h5'
#CPPA_sm_10d   = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
#RGCPD_sst_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.01_subinfo/fulldata_pcA_none_ac0.01_at0.01_2019-10-04.h5'
verbosity = 1



#%%
logit = ('logit', None)

logitCV = ('logitCV', 
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs'})

GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 750,
               'max_features':'sqrt',
               'subsample' : 0.6} ) 

# format {'dataset' : (path_data, list(keys_options) ) }


ERA_daily = {'ERA-5':(era5_1d_CPPA_l10, ['sst(CPPA)'])}

   

datasets_path = ERA_daily

#causal = False
#experiments = {} #; keys_d_sets = {}
#for dataset, path_key in datasets_path.items():
##    keys_d = exp_fc.compare_use_spatcov(path_data, causal=causal)
#    
#    path_data = path_key[0]
#    keys_options = path_key[1]
#    
#    keys_d = exp_fc.CPPA_precursor_regions(path_data, 
#                                           keys_options=keys_options)
#    
##    keys_d = exp_fc.normal_precursor_regions(path_data, 
##                                             keys_options=keys_options,
##                                             causal=causal)
#
#    for master_key, feature_keys in keys_d.items():
#        kwrgs_pp = {'EOF':False, 
#                    'expl_var':0.5,
#                    'fit_model_dates' : None}
#        experiments[dataset+' '+master_key] = (path_data, {'keys':feature_keys,
#                                           'kwrgs_pp':kwrgs_pp
#                                           })

#%%
## import original Response Variable timeseries:
#path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
#RVts_filename = 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19.npy'
#filename_ts = os.path.join(path_ts, RVts_filename)
#kwrgs_events_daily =    (filename_ts, 
#                         {  'event_percentile': 90,
#                        'min_dur' : 1,
#                        'max_break' : 0,
#                        'grouped' : False   }
#                         )
#
#kwrgs_events = kwrgs_events_daily
#    
#kwrgs_events = {'event_percentile': 66,
#                'min_dur' : 1,
#                'max_break' : 0,
#                'grouped' : False}


n_boot = 500
dict_experiments = {}

# LAG_DAY = 25
# def get_freqs_same_lag(LAG_DAY):
#     d_t_l = {}
#     o_freq = np.arange(1,200)
#     for f in o_freq:
#         if f == 1:
#             s = LAG_DAY/f
#         else:
#             s = (0.5 + LAG_DAY/f)
#             sm1 = (0.5 + (LAG_DAY-1)/f)
#             sp1 = (0.5 + (LAG_DAY+1)/f)
#         if s == int(s) or sm1 == int(s) or sp1 == int(s):
#             d_t_l[f] = int(s)
#         if f > 1.5*LAG_DAY:
#             break
#     return d_t_l

# LAG_DAY = 10
# def get_val_close_lag(LAG_DAY, tfreqs):
#     d_t_l = {}
#     for tfreq in tfreqs:
#         lags_t = [int((l-1) * tfreq + tfreq/2) for l in [0,1,2,3,4]]
#         diff = abs(np.array(lags_t)-LAG_DAY)
#         index = int(np.argwhere(diff == min(diff))[0])
#         d_t_l[tfreq] = index
#         print(tfreq, index, lags_t[int(index)])
#     return d_t_l
#np.array([l * tfreq for l in [0,1,2,3,4]])       
# dictionairy _ temporal frequency _ lag
LAG_DAY = 10
frequencies = np.arange(10, 14, 2)
percentiles = [50, 66, 80]
percentiles = [50,55,60,66,70,75,80,84.2]
frequencies = np.arange(4, 34, 2)

#d_t_l = {f:1 for f in range(15,27)}
#d_t_l = {f:1 for f in range(27,35)}
#d_t_l = {f:2 for f in range(16,21)}
#d_t_l = {f:1 for f in [18,20,25,30]}
# frequencies = list(d_t_l.keys())



#d_t_l = {10:1, 18:1}
# print(d_t_l)
#frequencies = list(d_t_l.keys())
#percentiles = [50,66]

kwrgs_pp={'add_autocorr':False}
stat_model_l = [logitCV]
folds = -9
seed=30

list_of_fc = [] ; 
dict_experiments = {}

#for i, fold in enumerate(folds):

dict_perc = {}
for perc in percentiles:
    kwrgs_events = {'event_percentile': perc}
    dict_freqs = {}
    for freq in frequencies:
        for dataset, tuple_sett in datasets_path.items():
            lags_i = np.array([LAG_DAY])
            path_data = tuple_sett[0]
            keys_d_list = tuple_sett[1]
            for keys_d in keys_d_list:
                fc = func_fc.fcev(path_data=path_data, precur_aggr=freq, use_fold=folds)

                print(f'{fc.fold} {fc.test_years[0]} {perc}')
                fc.get_TV(kwrgs_events=kwrgs_events)
                
                fc.fit_models(stat_model_l=stat_model_l, lead_max=lags_i, 
                               keys_d=keys_d, causal=False, kwrgs_pp=kwrgs_pp)
             
                fc.perform_validation(n_boot=n_boot, blocksize='auto', 
                                              threshold_pred='upper_clim')
                dict_freqs[freq] = fc.dict_sum
#                if i==0:
                # lags_t.append(fc.lags_t[0])
                list_of_fc.append(fc)
    dict_perc[perc] = dict_freqs



#%%
#df_valid, RV, y_pred = fc.dict_sum[stat_model_l[-1][0]]



def print_sett(list_of_fc, filename):
    f= open(filename+".txt","w+")
    file= open(filename+".txt","w+")
    lines = []
    
    lines.append("\nEvent settings:")        
    
    e = 1
    for i, fc_i in enumerate(list_of_fc):
        
        lines.append(f'\n\n***Experiment {e}***\n\n')
        lines.append(f'Title \t : {fc_i.name}')
        lines.append(f'file \t : {fc_i.path_data}')
        lines.append(f'kwrgs_events \t : {fc_i.kwrgs_events}')
        lines.append(f'kwrgs_pp \t : {fc_i.kwrgs_pp}')
        lines.append(f'Title \t : {fc_i.name}')
        lines.append(f'file \t : {fc_i.path_data}')
        lines.append(f'kwrgs_events \t : {fc_i.kwrgs_events}')
        lines.append(f'kwrgs_pp \t : {fc_i.kwrgs_pp}')
        lines.append(f'keys_d: \n{fc_i.keys_d}')
        lines.append(f'nboot: {fc_i.n_boot}')
        lines.append(f'stat_model_l: {fc_i.stat_model_l}')
        lines.append(f'fold: {fc_i.fold}')
        lines.append(f'keys_used: \n{fc_i._get_precursor_used()}')
        
        e+=1
    
    [print(n, file=f) for n in lines]
    f.close()
    [print(n, file=file) for n in lines]
    file.close()
    [print(n) for n in lines]

RV_name = 't2mmax'
working_folder = f'/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/{fc.hash}_forecast_expers'
if os.path.isdir(working_folder) != True : os.makedirs(working_folder)
today = datetime.datetime.today().strftime('%Hhr_%Mmin_%d-%m-%Y')
if type(kwrgs_events) is tuple:
    percentile = kwrgs_events[1]['event_percentile']
else:
    percentile = kwrgs_events['event_percentile']
folds_used = [f.fold for f in list_of_fc]
if np.unique(folds_used).size == 1:
    folds_used = str(folds_used[0])
else:
    folds_used =  str(folds_used).replace(' ','')
if np.unique(lags_t).size == 1:
    lagstr = f'lag{int(np.mean(lags_t))}'
else:
    lagstr = f'lag_r{int(lags_t[0])}-{int(lags_t[-1])}'
if np.unique(folds).size == 1:
    foldstr = f'fold{int(np.mean(folds))}'
else:
    foldstr = f'fold_r{int(folds[0])}-{int(folds[-1])}'
f_name = f'{RV_name}_{percentile}p_{foldstr}_{lagstr}_{today}'
filename = os.path.join(working_folder, f_name)

print_sett(list_of_fc, filename)




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
import valid_plots as dfplots
f_format = '.pdf'

filename = os.path.join(working_folder, f_name)


metric = 'BSS'
if type(kwrgs_events) is tuple:
    x_label = 'Temporal window [days]'
else:
    x_label = 'Temporal Aggregation [days]'
x_label2 = 'Lag in days'

path_data, dict_of_dfs = dfplots.get_score_matrix(d_expers=dict_perc, 
                                                  model=stat_model_l[0][0], 
                                                  metric=metric, lags_t=LAG_DAY)
fig = dfplots.plot_score_matrix(path_data, col=0, 
                                x_label=x_label, x_label2=x_label2, ax=None)
                      
    

fig.savefig(os.path.join(filename + f_format), 
            bbox_inches='tight') # dpi auto 600

    



    
    
#np.save(filename + '.npy', dict_experiments)
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
#dfplots.build_ts_matric(df_data, win=win, lag=0, columns=columns, rename=rename, period=period)
#if f_format == '.png':
#    plt.savefig(os.path.join(working_folder, f_name + f_format), 
#                bbox_inches='tight') # dpi auto 600
#elif f_format == '.pdf':
#    plt.savefig(os.path.join(pdfs_folder,f_name+ f_format), 
#                bbox_inches='tight')


#for freq in frequencies:
#    
#    import valid_plots as dfplots
#    kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
#    met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision', 'Accuracy']
#    expers = list(dict_experiments.keys())
#    models   = list(dict_experiments[expers[0]].keys())
#    line_dim = 'model'
#    
#    
#    fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
#                              line_dim=line_dim, 
#                              group_line_by=None,  
#                              met=met, **kwrgs)


#for freq in frequencies:
#    for dataset, tuple_sett in experiments.items():
#        '''
#        Format output is 
#        dict(
#                exper_name = dict( statmodel=tuple(df_valid, RV, y_pred) ) 
#            )
#        '''
#        path_data = tuple_sett[0]
#        kwrgs_exp = tuple_sett[1]
#        dict_of_dfs = func_fc.load_hdf5(path_data)
#        df_data = dict_of_dfs['df_data']
#        splits  = df_data.index.levels[0]
#    
#        
#        if 'keys' not in kwrgs_exp:
#            # if keys not defined, getting causal keys
#            kwrgs_exp['keys'] = exp_fc.normal_precursor_regions(path_data, causal=True)['normal_precursor_regions']
#    
#        print(kwrgs_events)
#        
#        if type(kwrgs_events) is tuple:
#            kwrgs_events_ = kwrgs_events[1]
#        else:
#            kwrgs_events_ = kwrgs_events
#            
#        df_data  = func_fc.load_hdf5(path_data)['df_data']
#        df_data_train = df_data.loc[fold][df_data.loc[fold]['TrainIsTrue'].values]
#        df_data_train, dates = functions_pp.time_mean_bins(df_data_train, 
#                                                           to_freq=freq, 
#                                                           start_end_date=None, 
#                                                           start_end_year=None, 
#                                                           verbosity=0)
#        
#        
#        # insert fake train test split to make RV
#        df_data_train = pd.concat([df_data_train], axis=0, keys=[0]) 
#        RV = func_fc.df_data_to_RV(df_data_train, kwrgs_events=kwrgs_events)
#        df_data_train = df_data_train.loc[0][df_data_train.loc[0]['TrainIsTrue'].values]
#        df_data_train = df_data_train.drop(['TrainIsTrue', 'RV_mask'], axis=1)
#        # create CV inside training set
#        df_splits = functions_pp.rand_traintest_years(RV, method=method,
#                                                      seed=seed, 
#                                                      kwrgs_events=kwrgs_events_, 
#                                                      verb=0)
#        # add Train test info
#        splits = df_splits.index.levels[0]
#        df_data_s   = np.zeros( (splits.size) , dtype=object)
#        for s in splits:
#            df_data_s[s] = pd.merge(df_data_train, df_splits.loc[s], left_index=True, right_index=True)
#            
#        df_data  = pd.concat(list(df_data_s), keys= range(splits.size))
#
#    
#        tfreq = (df_data.loc[0].index[1] - df_data.loc[0].index[0]).days
#
#        
#        lags_i = np.array([d_t_l[freq]])
#            
#        print(f'tfreq: {tfreq}, lag: {lags_i[0]}')
#        if tfreq == 1: 
#            lags_t.append(lags_i[0] * tfreq)
#        else:
#            lags_t.append((lags_i[0]-1) * tfreq + tfreq/2)
#
#        
#        fc.fit_models(stat_model_l=stat_model_l, lead_max=45, 
#                   keys_d=None, kwrgs_pp={})
#        
##        dict_sum = func_fc.forecast_wrapper(df_data, kwrgs_exp=kwrgs_exp, kwrgs_events=kwrgs_events, 
##                                stat_model_l=stat_model_l, 
##                                lags_i=lags_i, n_boot=n_boot)
#     
#        fc.perform_validation(n_boot=100, blocksize='auto', 
#                                      threshold_pred='upper_clim')
#
#        dict_experiments[freq] = fc.dict_sum