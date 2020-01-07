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
import numpy as np
#import func_fc

# =============================================================================
# load data 
# =============================================================================

strat_1d_CPPA_era5 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
strat_1d_CPPA_EC   = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/ran_strat10_s30/data/EC_16-09-19_19hr_lag_0.h5'
CPPA_v_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_v200hpa_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-20.h5'
CPPA_sm_10d   = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
CPPA_sm_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt30/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.01_subinfo/fulldata_pcA_none_ac0.01_at0.01_2019-10-18.h5'
RGCPD_sst_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.01_subinfo/fulldata_pcA_none_ac0.01_at0.01_2019-10-04.h5'
RGCPD_sst_sm_z500_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/list_tf_8_11_19/t2mmax_E-US_sst_sm123_z500hpa_m01-08_dt10/18jun-27aug_lag10-10_random10_s30/pcA_none_ac0.002_at0.05_subinfo/fulldata_pcA_none_ac0.002_at0.05_2019-11-17.h5'
RGCPD_sst_sm_z500_20d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/list_tf_8_11_19/t2mmax_E-US_sst_sm123_z500hpa_m01-08_dt20/14may-22aug_lag20-20_random10_s30/pcA_none_ac0.002_at0.05_subinfo/fulldata_pcA_none_ac0.002_at0.05_2019-11-19.h5'
RGCPD_sst_sm_z500_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/list_tf_8_11_19/t2mmax_E-US_sst_sm123_z500hpa_m01-08_dt30/19may-17aug_lag30-30_random10_s30/pcA_none_ac0.002_at0.05_subinfo/fulldata_pcA_none_ac0.002_at0.05_2019-11-17.h5'





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
                'solver':'lbfgs',
                'max_iter':150}) #100 is default
    
GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 750,
               'max_features':'sqrt',
               'subsample' : 0.6,
               'random_state':60} )  

GBR_logitCV_tuned = ('GBR-logitCV', 
          {'max_depth':[3,5,7],
           'learning_rate':1E-3,
           'n_estimators' : 750,
           'max_features':'sqrt',
           'subsample' : 0.6} ) 
    

# format {'dataset' : (path_data, list(keys_options) ) }

ERA_and_EC_daily  = {'ERA-5':(strat_1d_CPPA_era5, ['PEP', 'CPPA']),
                 'EC-earth 2.3':(strat_1d_CPPA_EC, ['PEP', 'CPPA'])} # random_state 60



   
#
#ERA5         = {'ERA-5:':(CPPA_sm_10d, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}
#ERA_Bram         = {'ERA-5:':(CPPA_sm_10d, ['all'])}
#stat_model_l = [GBR_logitCV, logit]

#ERA5_sm         = {'ERA-5:':(CPPA_sm_10d, ['sst(CPPA)', 'sm', 'sst(CPPA)+sm'])}
#stat_model_l = [GBR_logitCV]

#CPPA_sm_30d
#ERA5_sm_30d         = {'ERA-5:':(CPPA_sm_30d, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}
#ERA5_sm_30d         = {'ERA-5:':(CPPA_sm_30d, ['sst(CPPA)+sm'])}
#stat_model_l = [logit, logitCV]

ERA_Bram         = {'ERA-5:':(CPPA_sm_10d, ['sst(CPPA)+sm'])}


#RGCPD       = {'RGCPD:' : (RGCPD_sst_sm_z500_10d, ['only_db_regs'])}
#stat_model_l = [logitCV, GBR_logitCV]


#RGCPD_20    = {'RGCPD:' : (RGCPD_sst_sm_z500_20d, ['only_db_regs'])}

#RGCPD_30       = {'RGCPD:' : (RGCPD_sst_sm_z500_30d, ['only_db_regs', 'causal only_db_regs'])}
#RGCPD_30       = {'RGCPD:' : (RGCPD_sst_sm_z500_30d, ['only_db_regs'])}
#stat_model_l = [logit, logitCV]
#
#ERA_sp      = {'ERA-5:':(CPPA_sm_10d, ['CPPAregs+sm', 'CPPApattern+sm', 'sst(CPPA)+sm'])}
#stat_model_l = [GBR_logitCV]

datasets_path = ERA_Bram

causal = False
stat_model_l = [GBR_logitCV]

#%%
# import original Response Variable timeseries:
path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
RVts_filename = 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19.npy'
filename_ts = os.path.join(path_ts, RVts_filename)
kwrgs_events_daily =    (filename_ts, 
                         {  'event_percentile': 80,
                        'min_dur' : 1,
                        'max_break' : 0,
                        'grouped' : False   }
                         )

kwrgs_events = kwrgs_events_daily
    
#kwrgs_events = {'event_percentile': 66,
#                'min_dur' : 1,
#                'max_break' : 0,
#                'grouped' : False}

kwrgs_pp = {'EOF':False, 
            'expl_var':0.5,
            'add_autocorr':True,
            'normalize':'datesRV'}



#%%
n_boot = 100
verbosity = 0
lead_max = 75 # np.array([0,1])
from func_fc import fcev
#stat_model_l = [logit]
dict_experiments = {} ; list_fc = []
for dataset, path_key in datasets_path.items():

    
    path_data = path_key[0]
    keys_options = path_key[1]
    for i, keys_d in enumerate(keys_options):
        name = dataset+' '+keys_d
        
        fc = fcev(path_data, name=name)
        
        fc.get_TV(kwrgs_events=kwrgs_events)
        
        fc.fit_models(stat_model_l=stat_model_l, lead_max=lead_max, 
                   keys_d=keys_d, causal=False, kwrgs_pp=kwrgs_pp)
        
        fc.perform_validation(n_boot=n_boot, blocksize='auto', 
                              threshold_pred=(2, 'times_clim'))
        
        list_fc.append(fc)

        dict_sum = fc.dict_sum
        dict_experiments[name] = dict_sum
    
df_valid, RV, y_pred_all = dict_sum[stat_model_l[0][0]]
#%%

def print_sett(list_fc, stat_model_l, filename=None):
    if filename is not None:
        file= open(filename+".txt","w+")
    lines = []
    
    lines.append("\nEvent settings:")
    lines.append(kwrgs_events)
    
    lines.append(f'\nModels used:')  
    for m in stat_model_l:
        lines.append(m)
        
    lines.append(f'\nnboot: {n_boot}')
    e = 1
    for i, fc_i in enumerate(list_fc):
        
        lines.append(f'\n\n***Experiment {e}***\n\n')
        lines.append(f'Title \t : {fc_i.name}')
        lines.append(f'file \t : {fc_i.path_data}')
        lines.append(f'kwrgs_events \t : {fc_i.kwrgs_events}')
        lines.append(f'kwrgs_pp \t : {fc_i.kwrgs_pp}')
        lines.append(f'keys_d: \n{fc.keys_d}')
#        lines.append(f'kwrgs_pp \t : {f.kwrgs_pp}')
        
        e+=1
    if filename is not None:
        [print(n, file=file) for n in lines]
        file.close()
    [print(n) for n in lines]



        
  
RV_name = 't2mmax'
working_folder = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/forecasting'
pdfs_folder = os.path.join(working_folder,'pdfs')
if os.path.isdir(pdfs_folder) != True : os.makedirs(pdfs_folder)
today = datetime.datetime.today().strftime('%Hhr_%Mmin_%d-%m-%Y')
f_name = f'{RV_name}_{fcev.tfreq}d_{today}'
filename = os.path.join(working_folder, f_name)
print_sett(list_fc, stat_model_l, filename)

#%%
import valid_plots as dfplots
#rename_ERA =    {'ERA-5: sst(PEP)+sm':'PEP+sm', 
#             'ERA-5: sst(PDO,ENSO)+sm':'PDO+ENSO+sm', 
#             'ERA-5: sst(CPPA)+sm':'CPPA+sm'}
#
#for old, new in rename_ERA.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

#rename_EC_vs_CPPA = {'ERA-5 PEP':'PEP', 
#             'ERA-5 CPPA':'CPPA', 
#             'EC-earth 2.3 PEP':'PEP ', 
#             'EC-earth 2.3 CPPA':'CPPA '}
#
#for old, new in rename_EC_vs_CPPA.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

#rename_CPPA_comp =    {'ERA-5: CPPAregs+sm' : 'precursor regions + sm', 
#                       'ERA-5: CPPApattern+sm': 'precursor pattern + sm', 
#                       'ERA-5: sst(CPPA)+sm' : 'CPPA (all) + sm'}
#
#for old, new in rename_CPPA_comp.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

#rename_sm_impact =    {'ERA-5: sst(CPPA)' : 'sst(CPPA)', 
#                       'ERA-5: sm': 'sm', 
#                       'ERA-5: sst(CPPA)+sm' : 'CPPA (all) + sm'}
#
#for old, new in rename_sm_impact.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

#rename_RGCPD =    {'RGCPD: only_db_regs' : 'RGCPD: correlated regions', 
#                       'RGCPD: causal only_db_regs': 'RGCPD: causal regions'}
#for old, new in rename_RGCPD.items():
#    if new not in dict_experiments.keys():
#        try:
#            dict_experiments[new] = dict_experiments.pop(old)
#        except:
#            pass
        
f_formats = ['.pdf']
#f_format = '.png' 
#f_format = None
for f_format in f_formats:
    filename = os.path.join(working_folder, f_name)
    
    group_line_by = None
#    group_line_by = ['ERA-5', 'EC-Earth']
    col_wrap = None
    wspace = 0.05
    kwrgs = {'wspace':wspace, 'col_wrap':col_wrap}
    met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
    kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
    met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision', 'Accuracy']
    expers = list(dict_experiments.keys())
    models   = list(dict_experiments[expers[0]].keys())
    line_dim = 'model'

    
    fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
                              line_dim=line_dim, 
                              group_line_by=group_line_by,  
                              met=met, **kwrgs)

    if f_format == '.png':
        fig.savefig(os.path.join(filename + f_format), 
                    bbox_inches='tight') # dpi auto 600
    elif f_format == '.pdf':
        fig.savefig(os.path.join(pdfs_folder,f_name+ f_format), 
                    bbox_inches='tight')
    
print_sett(list_fc, stat_model_l, filename)


np.save(filename + '.npy', dict_experiments)
#%%
#fcev.plot_scatter()
#%%

#valid.loop_df(fcev.df_data.loc[0], valid.plot_ac, sharex='none')

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

# =============================================================================
#%% Translation to extremes
# =============================================================================
