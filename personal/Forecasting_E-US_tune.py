#!/usr/bin/env python
# coding: utf-8

# # Forecasting
# Below done with test data, same format as df_data

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import os, inspect, sys
import numpy as np
import pandas as pd
import datetime
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
python_dir = os.path.join(main_dir, 'RGCPD')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(python_dir)
    sys.path.append(df_ana_dir)

user_dir = os.path.expanduser('~')
# In[2]:


from func_fc import fcev


# In[3]:
old_CPPA = user_dir + '/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
old = user_dir + '/Downloads/output_RGCPD/20jun-19aug_lag10-10/ran_strat10_s1/None_at0.001_tau_0-1_conds_dim4_combin1.h5'
era5_10d_CPPA_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_18jun-17aug_lag10-10/ran_strat10_s1/df_data_sst_CPPA_sm123_Xzkup1.h5'
era5_1d_CPPA_lag0 =  user_dir + '/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_0_Xzkup1.h5'
era5_1d_CPPA_l10 = user_dir + '/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_10_Xzkup1.h5'
era5_16d_CPPA_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_19jun-22aug_lag16-16/ran_strat10_s1/df_data_sst_CPPA_sm123_dt16_Xzkup1.h5'
era5_16d_RGCPD_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_19jun-22aug_lag16-16/ran_strat10_s1/df_data__sm123_sst_dt16_Xzkup1.h5'
era5_12d_RGCPD_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_24may-28aug_lag12-12/ran_strat20_s1/df_data__sm123_sst_dt12_Xzkup1.h5'
era5_10d_RGCPD_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_10jun-29aug_lag20-20/random10_s1/df_data__sm1_sm2_sm3_OLR_sst_dt10_Xzkup1.h5'
era5_10d_RGCPD_sm_uv = user_dir + '/Downloads/output_RGCPD/Xzkup1_10jun-29aug_lag20-20/random10_s1/df_data__sm123_u500_v200_sst_dt10_Xzkup1.h5'
# In[4]:


#ERA_and_EC_daily  = {'ERA-5':(strat_1d_CPPA_era5, ['PEP', 'CPPA']),
#                 'EC-earth 2.3':(strat_1d_CPPA_EC, ['PEP', 'CPPA'])}
ERA_10d = {'ERA-5':(era5_10d_CPPA_sm, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}
ERA_10d_sm = {'ERA-5':(era5_10d_CPPA_sm, ['sst(CPPA)+sm', None])}
ERA_1d_CPPA = {'ERA-5':(era5_1d_CPPA_lag0, ['sst(PDO,ENSO)', 'all'])}
ERA_10d_RGCPD = {'ERA-5':(era5_10d_RGCPD_sm, ['all'])}
ERA_10d_RGCPD_all = {'ERA-5':(era5_10d_RGCPD_sm_uv, ['all'])}
ERA_16d_RGCPD = {'ERA-5':(era5_16d_RGCPD_sm, [None, 'sst(CPPA)'])}
ERA_12d_RGCPD = {'ERA-5':(era5_12d_RGCPD_sm, ['sst(CPPA)+sm', 'sst(CPPA)'])}
ERA_vs_PEP = {'ERA-5':(era5_1d_CPPA_lag0, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}

datasets_path  = ERA_10d_sm


# Define statmodel:
logit = ('logit', None)

logitCV = ('logitCV', 
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs'})
    
logitCVfs = ('logitCV', 
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs',
           'feat_sel':{'model':None}})
    
GBC_tfs = ('GBC', 
          {'max_depth':[1, 2, 3, 4],
           'learning_rate':[1E-2, 5E-3, 1E-3, 5E-4],
           'n_estimators' : [200, 300, 400, 500, 600, 700, 800, 1000],
           'min_samples_split':[.15, .25],
           'max_features':[.2,'sqrt', .5],
           'subsample' : [.3, .4, .5, 0.6],
           'random_state':60,
           'scoringCV':'brier_score_loss',
           'feat_sel':{'model':None} } )
        
GBC_t = ('GBC', 
{'max_depth':[1, 2, 3, 4],
           'learning_rate':[1E-2, 5E-3, 1E-3, 5E-4],
           'n_estimators' : [200, 300, 400, 500, 600, 700, 800, 1000],
           'min_samples_split':[.15, .25],
           'max_features':[.2,'sqrt', .5],
           'subsample' : [.3, .4, .5, 0.6],
           'random_state':60,
           'scoringCV':'brier_score_loss' } )
    
GBC = ('GBC', 
      {'max_depth':1,
       'learning_rate':.01,
       'n_estimators' : [200, 400],
       'min_samples_split':.1,
       'max_features':'sqrt',
       'subsample' : .5,
       'random_state':60,
       'scoringCV':'brier_score_loss',
       'feat_sel':{'model':None} } )
    
# In[6]:
path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
RVts_filename = '/Users/semvijverberg/surfdrive/MckinRepl/RVts/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'
filename_ts = os.path.join(path_ts, RVts_filename)
kwrgs_events_daily =    (filename_ts, 
                         {'event_percentile': 90})
    
kwrgs_events = {'event_percentile': 66}

kwrgs_events = kwrgs_events

#stat_model_l = [logitCVfs, logitCV, GBC_tfs, GBC_t, GBC]
stat_model_l = [logit, logit]
kwrgs_pp     = {'add_autocorr' : True}
lags_i = np.array([0])
tfreq = None
use_fold = -9



dict_experiments = {} ; list_of_fc = []  
for dataset, tuple_sett in datasets_path.items():
    path_data = tuple_sett[0]
    keys_d_list = tuple_sett[1] 
    for keys_d in keys_d_list:
        
        fc = fcev(path_data=path_data, daily_to_aggr=tfreq, use_fold=use_fold)
        fc.get_TV(kwrgs_events=kwrgs_events)
        fc.fit_models(stat_model_l=stat_model_l, lead_max=lags_i, 
                           keys_d=keys_d, kwrgs_pp=kwrgs_pp, verbosity=1)

        fc.perform_validation(n_boot=500, blocksize='auto', alpha=0.05,
                              threshold_pred=(1.5, 'times_clim'))
        dict_experiments[dataset+'_'+str(keys_d)] = fc.dict_sum
        list_of_fc.append(fc)

# In[7]:

#
#dict_experiments = {}       
#fc.perform_validation(n_boot=100, blocksize='auto', 
#                              threshold_pred=(1.5, 'times_clim'))
#dict_experiments['test'] = fc.dict_sum
y_pred_all, y_pred_c = fc.dict_preds[fc.stat_model_l[0][0]]

# In[8]:


import valid_plots as dfplots
kwrgs = {'wspace':0.25, 'col_wrap':None, 'threshold_bin':fc.threshold_pred}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision']
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
expers = list(dict_experiments.keys())
models   = list(dict_experiments[expers[0]].keys())
line_dim = 'model'


fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
                          line_dim=line_dim, 
                          group_line_by=None,  
                          met=met, **kwrgs)



def print_sett(list_of_fc, filename):
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
        lines.append(f'alpha \t : {fc_i.alpha}')
        lines.append(f'nboot: {fc_i.n_boot}')
        lines.append(f'stat_models:')
        lines.append('\n'.join(str(m) for m in fc_i.stat_model_l))
        lines.append(f'fold: {fc_i.fold}')        
        lines.append(f'keys_d: \n{fc_i.keys_d}')
        lines.append(f'keys_used: \n{fc_i._get_precursor_used()}')
        
        e+=1
    
    [print(n, file=file) for n in lines]
    file.close()
    [print(n) for n in lines[:-2]]

RV_name = 't2mmax'
subfolder = 'forecasts'
working_folder = '/'.join(fc.path_data.split('/')[:-1]) 
working_folder = os.path.join(working_folder, subfolder)
if os.path.isdir(working_folder) != True : os.makedirs(working_folder)
today = datetime.datetime.today().strftime('%Hhr_%Mmin_%d-%m-%Y')
if type(kwrgs_events) is tuple:
    percentile = kwrgs_events[1]['event_percentile']
else:
    percentile = kwrgs_events['event_percentile']
folds_used = [f.fold for f in list_of_fc]
f_name = f'{RV_name}_{tfreq}d_{percentile}p_fold{folds_used}_{today}'
filename = os.path.join(working_folder, f_name)

print_sett(list_of_fc, filename)

f_format = '.pdf'
pathfig_valid = os.path.join(filename + f_format)
fig.savefig(f_format, 
            bbox_inches='tight') # dpi auto 600




## In[9]:
## =============================================================================
## Feature Importance Analysis
## =============================================================================
#
##keys = list(rename_labels.values())
#keys = None
#fc.plot_GBR_feature_importances(lag=None, keys=keys)
#
#
## In[10]:
#
#
#import stat_models
##keys = tuple(rename_labels.values())
#keys = None
#GBR_models_split_lags = fc.dict_models['GBR-logitCV']
#stat_models.plot_oneway_partial_dependence(GBR_models_split_lags,
#                                          keys=keys,
#                                          lags=[0,1,2])
#
#
## In[18]:
#
#
#import stat_models
#GBR_models_split_lags = fc.dict_models['GBR-logitCV']
##keys = tuple(rename_labels.values())
##plot_pairs = [(keys[2], keys[1])]
#df_all = stat_models.plot_twoway_partial_dependence(GBR_models_split_lags, lag_i=2, keys=keys,
#                                   plot_pairs=None, min_corrcoeff=0.1)
#
#
## In[12]:
#
#
#from IPython.display import Image
#Image(filename=os.path.join(main_dir, "docs/images/pcA_none_ac0.002_at0.05_t2mmax_E-US_vs_sst_labels_mean.png"),
#      width=1000, height=200)
#
#
## In[13]:
#
#
## Soil Moisture labels
#Image(filename=os.path.join(main_dir, "docs/images/pcA_none_ac0.002_at0.05_t2mmax_E-US_vs_sm123_labels_mean.png"),
#      width=1000, height=400)
#
#
## In[14]:
#
#
#Image(filename=os.path.join(main_dir, "docs/images/pcA_none_ac0.002_at0.05_t2mmax_E-US_vs_z500hpa_labels_mean.png"),
#      width=1000, height=200)

#%%


import valid_plots as dfplots
model = 'logitCV'
model = None
fig = dfplots.visual_analysis(fc, lag=2, model=model)
if model is None:
    model = fc.stat_model_l[0][0]
f_name = f'{RV_name}_{tfreq}d_{percentile}p_fold{folds_used}_{today}_va_{model}'
f_format = '.pdf'
pathfig_vis = os.path.join(working_folder, f_name) + f_format

fig.savefig(pathfig_vis, bbox_inches='tight') # dpi auto 600
            
try:
    f_name = f'{RV_name}_{tfreq}d_{percentile}p_fold{folds_used}_{today}_deviance'
    
    fig = dfplots.plot_deviance(fc, lag=None, model=model)
    f_format = '.pdf'
    path_fig_GBC = os.path.join(working_folder, f_name) + f_format
    fig.savefig(path_fig_GBC, 
            bbox_inches='tight') # dpi auto 600
except:
    pass

