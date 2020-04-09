#!/usr/bin/env python
# coding: utf-8

# # Forecasting
# Below done with test data, same format as df_data

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import os, inspect, sys
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_dir = os.path.join(main_dir, 'RGCPD')
fc_dir = os.path.join(main_dir, 'forecasting')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if fc_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_dir)
    sys.path.append(df_ana_dir)
    sys.path.append(fc_dir)

user_dir = os.path.expanduser('~')
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
    n_cpu = 16
else:
    n_cpu = None

from class_fc import fcev

# Define statmodel:
logit = ('logit', None)

logitCV = ('logitCV',
          {'Cs':10, #np.logspace(-4,1,10)
          'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs',
           'max_iter':100,
           'kfold':5,
           'seed':2})



# In[6]:
EC_data  = user_dir + '/surfdrive/output_RGCPD/easternUS_EC/EC_tas_tos_Northern/958dd_ran_strat10_s30/data/EC_21-03-20_16hr_lag_0_958dd.h5'
ERA_data = user_dir + '/surfdrive/output_RGCPD/easternUS/ERA5_mx2t_sst_Northern/ff393_ran_strat10_s30/data/ERA5_21-03-20_12hr_lag_0_ff393.h5'

# path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
# RVts_filename = '/Users/semvijverberg/surfdrive/MckinRepl/RVts/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'
# filename_ts = os.path.join(path_ts, RVts_filename)
# kwrgs_events_daily =    (filename_ts,
#                          {'event_percentile': 90})

kwrgs_events = {'event_percentile': 50}

kwrgs_events = kwrgs_events
precur_aggr = 15
add_autocorr = False
use_fold = None
n_boot = 1000
lags_i = np.array([0, 10, 15, 20, 25, 30])
start_end_TVdate = None # ('7-04', '8-22')


list_of_fc = [fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model=logitCV, 
                    kwrgs_pp={'add_autocorr':add_autocorr, 'normalize':'datesRV'}, 
                    dataset=f'CPPA vs PEP',
                    keys_d='PEP',
                    n_cpu=n_cpu),              
                fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
                      use_fold=use_fold, start_end_TVdate=None,
                      stat_model=logitCV, 
                      kwrgs_pp={'add_autocorr':add_autocorr, 'normalize':'datesRV'}, 
                      dataset=f'CPPA vs PEP',
                      keys_d='CPPA',
                      n_cpu=n_cpu),   
               fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
                     use_fold=use_fold, start_end_TVdate=None,
                     stat_model=logitCV, 
                     kwrgs_pp={'add_autocorr':add_autocorr, 'normalize':'datesRV'}, 
                     dataset=f'CPPA vs PDO+ENSO',
                     keys_d='PDO+ENSO',
                     n_cpu=n_cpu),              
               fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
                     use_fold=use_fold, start_end_TVdate=None,
                     stat_model=logitCV, 
                     kwrgs_pp={'add_autocorr':add_autocorr, 'normalize':'datesRV'}, 
                     dataset=f'CPPA vs PDO+ENSO',
                     keys_d='CPPA',
                    n_cpu=n_cpu)]

# list_of_fc = [fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
#                     use_fold=use_fold, start_end_TVdate=None,
#                     stat_model=logitCV, 
#                     kwrgs_pp={'add_autocorr':add_autocorr, 'normalize':'datesRV'}, 
#                     dataset=f'CPPA vs PDO+ENSO',
#                     keys_d='PDO+ENSO'),
#               fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
#                     use_fold=use_fold, start_end_TVdate=None,
#                     stat_model=logitCV, 
#                     kwrgs_pp={'add_autocorr':add_autocorr, 'normalize':'datesRV'}, 
#                     dataset=f'CPPA vs PDO+ENSO',
#                     keys_d='persistence')]

              

              
fc = list_of_fc[0]
#%%
for i, fc in enumerate(list_of_fc):
    
    fc.get_TV(kwrgs_events=kwrgs_events)
    
    fc.fit_models(lead_max=lags_i, verbosity=1)

    fc.perform_validation(n_boot=n_boot, blocksize='auto', alpha=0.05,
                          threshold_pred=(1.5, 'times_clim'))
    

# In[8]:
working_folder, filename = fc._print_sett(list_of_fc=list_of_fc)

store = False
if __name__ == "__main__":
    filename = fc.filename 
    store = True

import valid_plots as dfplots
import functions_pp


dict_all = dfplots.merge_valid_info(list_of_fc, store=store)
if store:
    dict_merge_all = functions_pp.load_hdf5(filename+'.h5')


kwrgs = {'wspace':0.15, 'col_wrap':None, 'skip_redundant_title':True, 
         'lags_relcurve':[10,20]}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
line_dim = None
group_line_by = 'dataset'
# line_dim = 'exper' ; group_line_by = None

fig = dfplots.valid_figures(dict_merge_all, 
                          line_dim=line_dim,
                          group_line_by=group_line_by,
                          met=met, **kwrgs)


f_format = '.pdf'
pathfig_valid = os.path.join(filename + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600



#%%

im = 0
il = 1
ifc = 0
f_format = '.pdf'
if os.path.isdir(fc.filename) == False : os.makedirs(fc.filename)
import valid_plots as dfplots
if __name__ == "__main__":
    for ifc, fc in enumerate(list_of_fc):
        for im, m in enumerate([n[0] for n in fc.stat_model_l]):
            for il, l in enumerate(fc.lags_i):
                fc = list_of_fc[ifc]
                m = [n[0] for n in fc.stat_model_l][im]
                l = fc.lags_i[il]
                # visual analysis
                f_name = os.path.join(filename, f'ifc{ifc}_va_l{l}_{m}')
                fig = dfplots.visual_analysis(fc, lag=l, model=m)
                fig.savefig(os.path.join(working_folder, f_name) + f_format, 
                            bbox_inches='tight') # dpi auto 600
                # plot deviance
                if m[:3] == 'GBC':
                    fig = dfplots.plot_deviance(fc, lag=l, model=m)
                    f_name = os.path.join(filename, f'ifc{ifc}_deviance_l{l}')
                    

                    fig.savefig(os.path.join(working_folder, f_name) + f_format,
                                bbox_inches='tight') # dpi auto 600
                    
                    fig = fc.plot_oneway_partial_dependence()
                    f_name = os.path.join(filename, f'ifc{ifc}_partial_depen_l{l}')
                    fig.savefig(os.path.join(working_folder, f_name) + f_format,
                                bbox_inches='tight') # dpi auto 600
                    
                if m[:7] == 'logitCV':
                    fig = fc.plot_logit_regularization(lag_i=l)
                    f_name = os.path.join(filename, f'ifc{ifc}_logitregul_l{l}')
                    fig.savefig(os.path.join(working_folder, f_name) + f_format,
                            bbox_inches='tight') # dpi auto 600
                
            df_importance, fig = fc.plot_feature_importances()
            f_name = os.path.join(filename, f'ifc{ifc}_feat_l{l}_{m}')
            fig.savefig(os.path.join(working_folder, f_name) + f_format, 
                        bbox_inches='tight') # dpi auto 600



#%%
            
# import df_ana ; import matplotlib.pyplot as plt
# flatten = lambda l: [item for sublist in l for item in sublist]
# ERA_data = user_dir + '/surfdrive/output_RGCPD/easternUS/ERA5_mx2t_sst_Northern/ff393_ran_strat10_s30/data/ERA5_21-03-20_12hr_lag_0_ff393.h5'
# working_folder = user_dir + '/surfdrive/output_RGCPD/easternUS/ERA5_mx2t_sst_Northern/ff393_ran_strat10_s30/'


# fc = fcev(path_data=ERA_data, precur_aggr=1, 
#                     use_fold=None, start_end_TVdate=None,
#                     stat_model=logitCV, 
#                     kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
#                     dataset=f'CPPA vs PEP',
#                     keys_d=None,
#                     n_cpu=n_cpu)

# columns = ['mx2t', '0..CPPAsv', '0..PEPsv', 'PDO', 'ENSO34']

# rename = {'mx2t':'T90m', '0..CPPAsv':'CPPAsp', '0..PEPsv':'PEP'}
# df_corr = fc.df_data.loc[:,['mx2t', 'ENSO34','PDO', '0..CPPAsv', '0..PEPsv', 'RV_mask', 'TrainIsTrue']].copy()

# df_ana.plot_ts_matric(df_corr, columns=columns, rename=rename, period='summer60days')
# fig_filename = os.path.join(working_folder, 'figures', 'cross_corr_summer60days')
# plt.savefig(fig_filename + '.pdf', bbox_inches='tight')

# df_ana.plot_ts_matric(df_corr, win=365, columns=columns, rename=rename, period='fullyear')
# fig_filename = os.path.join(working_folder, 'figures', 'cross_corr_fullyear')
# plt.savefig(fig_filename + '.pdf', bbox_inches='tight')

#%%
# ERA_data = user_dir + '/surfdrive/output_RGCPD/easternUS/1_ff393_12jun-11aug_lag15-15_from_imports/df_data_sst_CPPAs30_sm2_sm3_dt1_ff393.h5'
# fc_ = fcev(path_data=ERA_data, precur_aggr=1, 
#                     use_fold=None, start_end_TVdate=None,
#                     stat_model=logitCV, 
#                     kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
#                     dataset=f'CPPA vs PEP',
#                     keys_d=None,
#                     n_cpu=n_cpu)

# rename = {'1':'mx2t', '0..102..CPPAsv':'CPPAsp', '0..103..PEPsv':'PEP', '0..101..PDO':'PDO', '0..100..ENSO34':'ENSO34' }
# columns = list(rename.keys()) ; columns.append('TrainIsTrue') ; columns.append('RV_mask')
# df_corr_ = fc_.df_data.loc[:,columns].copy()
# df_ana.plot_ts_matric(df_corr_, columns=list(rename.keys()), rename=rename, period='summer60days')

# df_ana.plot_ts_matric(df_corr_, win=365, columns=list(rename.keys()), rename=rename, period='fullyear')
