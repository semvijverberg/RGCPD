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
import inspect, os, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
    
import time
start_time = time.time()
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
python_dir = os.path.join(main_dir, 'RGCPD')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(python_dir)
    sys.path.append(df_ana_dir)
user_dir = os.path.expanduser('~')


    
import numpy as np
import func_fc
import valid_plots as dfplots


# =============================================================================
# load data 
# =============================================================================

era5_1d_CPPA = user_dir + '/surfdrive/output_RGCPD/easternUS/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_0_Xzkup1.h5'
era5_1d_CPPA_l10 = user_dir + '/surfdrive/output_RGCPD/easternUS/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_10_Xzkup1.h5'
CPPA_s5_l10 = user_dir + '/surfdrive/output_RGCPD/easternUS/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s5/data/ERA5_15-02-20_15hr_lag_10_Xzkup1.h5'
CPPA_s5_l10_sm_OLR = user_dir + '/surfdrive/output_RGCPD/easternUS/Xzkup1_20jun-19aug_lag10-10/random10_s1/df_data_sst_CPPAs5_sm2_sm3_OLR_dt1_Xzkup1.h5'
# strat_1d_CPPA_era5 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
# strat_1d_CPPA_era5_l10 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_10.h5'
#strat_1d_CPPA_EC   = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/ran_strat10_s30/data/EC_16-09-19_19hr_lag_0.h5'
#CPPA_v_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_v200hpa_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-20.h5'
#CPPA_sm_10d   = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
#RGCPD_sst_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.01_subinfo/fulldata_pcA_none_ac0.01_at0.01_2019-10-04.h5'
verbosity = 1




logit = ('logit', None)

logitCV = ('logitCV', 
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs'})


# format {'dataset' : (path_data, list(keys_options) ) }
ERA_daily = {'ERA-5':(CPPA_s5_l10_sm_OLR, ['sst(CPPA)+sm+OLR'])}


datasets_path = ERA_daily


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
LAG_DAY = 21
frequencies = np.arange(5, 6, 2)
percentiles = [50]
percentiles = [50,55,60,66,70,75,80,84.2]
frequencies = np.arange(4, 34, 2)



kwrgs_pp={'add_autocorr':False}
stat_model_l = [logitCV]
folds = -9
seed=30

list_of_fc = [] ; 

dict_experiments = {}
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
    dict_experiments[perc] = dict_freqs



#%%

subfoldername='forecast_optimal_freq'
f_name = '{}_freqs{}-{}_perc{}-{}'.format(fc.hash, frequencies[0], frequencies[-1], 
                                          percentiles[0], percentiles[-1])
working_folder, filename = fc._print_sett(list_of_fc=list_of_fc, 
                                          subfoldername=subfoldername, f_name=f_name)



f_format = '.pdf'

metric = 'BSS'
if type(kwrgs_events) is tuple:
    x_label = 'Temporal window [days]'
else:
    x_label = 'Temporal Aggregation [days]'
x_label2 = 'Lag in days'

path_data, dict_of_dfs = dfplots.get_score_matrix(d_expers=dict_experiments, 
                                                  model=stat_model_l[0][0], 
                                                  metric=metric, lags_t=LAG_DAY)
fig = dfplots.plot_score_matrix(path_data, col=0, 
                                x_label=x_label, x_label2=x_label2, ax=None)
                      
    

fig.savefig(os.path.join(filename + f_format), 
            bbox_inches='tight') # dpi auto 600

    
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
