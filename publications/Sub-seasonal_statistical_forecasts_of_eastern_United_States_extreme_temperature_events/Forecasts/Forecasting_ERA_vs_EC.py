#!/usr/bin/env python
# coding: utf-8

# # Forecasting
# Below done with test data, same format as df_data

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import os, inspect, sys
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-3])
data_dir = '/'.join(curr_dir.split('/')[:-1]) + '/data'
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



from class_fc import fcev


# Define statmodel:
logit = ('logit', None)

logitCV = ('logitCV',
          {'Cs':np.logspace(-4,1,10),
          'class_weight':{ 0:1, 1:1},
           'scoring':'neg_brier_score',
           'penalty':'l2',
           'solver':'lbfgs',
           'max_iter':100,
           'kfold':5})



# In[6]:
EC_data  = data_dir + '/CPPA_EC_21-03-20_16hr_lag_0_958dd.h5'
ERA_data = data_dir + '/CPPA_ERA5_21-03-20_12hr_lag_0_ff393.h5'
                    

kwrgs_events = {'event_percentile': 'std'}

kwrgs_events = kwrgs_events
precur_aggr = 15
use_fold = None
n_boot = 1000
lags_i = np.array([0, 10, 15, 20 , 25, 30])
start_end_TVdate = None # ('7-04', '8-22')


list_of_fc = [fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model=logitCV, 
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
                    dataset=f'ERA-5',
                    keys_d='PEP'),
              fcev(path_data=ERA_data, precur_aggr=precur_aggr, 
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model=logitCV, 
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
                    dataset=f'ERA-5',
                    keys_d='CPPA'),
              fcev(path_data=EC_data, precur_aggr=precur_aggr, 
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model=logitCV, 
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
                    dataset=f'EC-earth',
                    keys_d='PEP'),
              fcev(path_data=EC_data, precur_aggr=precur_aggr, 
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model=logitCV, 
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
                    dataset=f'EC-earth',
                    keys_d='CPPA')]

              
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
    dict_all = functions_pp.load_hdf5(filename+'.h5')
    


kwrgs = {'wspace':0.15, 'col_wrap':None, 'lags_relcurve':[10, 20], 
         'skip_redundant_title':True}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']


line_dim = 'exper'


fig = dfplots.valid_figures(dict_all, 
                          line_dim=line_dim,
                          group_line_by=None,
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

#%% Get absolute anomalies of HW events
            
ERA_data = data_dir + '/CPPA_ERA5_21-03-20_12hr_lag_0_ff393.h5'
tfreq = 1
start_end_TVdate = ('06-05','09-02')
kwrgs_events = {'event_percentile': 'std', 'window':'mean', 'min_dur':3, 'max_break': 1}
fc = fcev(path_data=ERA_data, 
                    use_fold=use_fold, start_end_TVdate=start_end_TVdate,
                    stat_model=None, precur_aggr=tfreq,
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'}, 
                    dataset=f'ERA-5',
                    keys_d='PEP')
fc.get_TV(kwrgs_events=kwrgs_events, detrend=False)

HW = fc.TV.RV_ts[fc.TV.RV_bin.astype(bool).values]
# HW.mean()
# print(fc.TV.threshold)
# HW.min()
            
# #%% load raw data and calculate T90m

# #%% get clustered from Teleconnections (rg.plot_df_clust())

# xr_maskEUS = rg.ds['xrclustered'] == 1
# import clustering_spatial as cl
# selbox = (232, 295, 25, 50)

# Tmean = cl.spatial_mean_clusters(rawmx2t,
#                           xr_maskEUS,
#                           selbox=selbox)['ts']
# Tmean = Tmean.sel(cluster=1)['ts']
#%%            
import core_pp, plot_maps
import matplotlib.pyplot as plt
rawmx2t = '/Users/semvijverberg/surfdrive/ERA5/input_raw/mx2t_US_1979-2018_1_12_daily_0.25deg.nc'
selbox = (232, 295, 25, 50)
HW2d = core_pp.import_ds_lazy(rawmx2t, selbox=selbox, seldates=HW.index)
xr_mask = core_pp.import_ds_lazy('/Users/semvijverberg/surfdrive/Scripts/rasterio/mask_North_America_0.25deg_orig.nc', 
                                  var='lsm', selbox=selbox)
#%%

HW_composite = (HW2d - 273.15).mean(dim='time')

# xr_mask.values = make_country_mask.binary_erosion(xr_mask.values)

plot_maps.plot_corr_maps(HW_composite.where(xr_mask), 
                         clevels=np.arange(17, 36, 2), 
                         cmap=plt.cm.hot_r,
                         aspect=2.5,
                         zoomregion=(232, 295, 25, 50),
                         cbar_vert=-.05,
                         size=4,
                         units='Temperature [degree Celsius]',
                         subtitles=np.array([['Composite mean of heatwaves']]))
#%%

import core_pp, plot_maps
import matplotlib.pyplot as plt
rawmx2t = '/Users/semvijverberg/surfdrive/ERA5/input_raw/preprocessed/mx2t_US_1979-2018_1jan_31dec_daily_0.25deg.nc'
selbox = (232, 295, 25, 50)
HW2d_ano = core_pp.import_ds_lazy(rawmx2t, selbox=selbox, seldates=HW.index)
xr_mask = core_pp.import_ds_lazy('/Users/semvijverberg/surfdrive/Scripts/rasterio/mask_North_America_0.25deg_orig.nc', 
                                  var='lsm', selbox=selbox)
#%%

HW_ano_composite = (HW2d_ano).median(dim='time')

# xr_mask.values = make_country_mask.binary_erosion(xr_mask.values)

plot_maps.plot_corr_maps(HW_ano_composite.where(xr_mask), 
                          clevels=np.arange(-5, 6, 1), 
                         cmap=plt.cm.hot_r,
                         aspect=2.5,
                         zoomregion=(232, 295, 25, 50),
                         cbar_vert=-.05,
                         size=4,
                         units='Temperature anomalies [degree Celsius]',
                         subtitles=np.array([['Median of heatwave events']]))

