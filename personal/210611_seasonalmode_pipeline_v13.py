# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:13:54 2021

@author: Van Ingen
"""

"""
Seasonal mode steps and pipeline
"""

#%% Load packages
import os, inspect, sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
    n_cpu = 5
else:
    n_cpu = 3

main_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
RGCPD_dir = '/'.join(main_dir.split('/')[:-1])
sys.path.append(RGCPD_dir)
user_dir = os.path.expanduser('~')
os.chdir(RGCPD_dir)
ERA5_data_dir = user_dir + '/surfdrive/ERA5/input_raw'
working_dir = user_dir + '/surfdrive/output_RGCPD/EU'
main_dir = working_dir
print(RGCPD_dir)
from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI, functions_pp
# from IPython.display import Image
import numpy as np
import pandas as pd
import plot_maps
import xarray as xr
import argparse
from datetime import datetime


#%% Define Analysis
def define(list_of_name_path, TV_targetperiod, n_lags, kwrgs_MI, subfolder):

    #create lag list
    days_dict = {'01':'31',
                   '02':'28',
                   '03':'31',
                   '04':'30',
                   '05':'31',
                   '06':'30',
                   '07':'31',
                   '08':'31',
                   '09':'30',
                   '10':'31',
                   '11':'30',
                   '12':'31'}

    target_month_str = TV_targetperiod[0][:2] #derive month number
    if target_month_str[0] == '0':
        target_month = int(target_month_str[1]) # 01 or 02 ..
    else:
        target_month = int(target_month_str[:]) #10,11,12
    print(target_month)

    if target_month - (n_lags) <= 0: #cross year?
        crossyr = True
        start_end_year = (1951,2020) #hardcoded
    else:
        crossyr = False
        start_end_year = None

    lags = [] #initialize empty lags list
    for i in range(n_lags):
        lag = [] #initialize empty lag list
        if not crossyr: #if not crossyear with lags, do not add years to lags
            for j in range(1): #start and end date
                if target_month-i-1 < 10:
                    lag_month_str_start = '0'+str(target_month-i-1) # 01 or 02 ..
                    lag_month_str_end = '0'+str(target_month-i-1) # 01 or 02 ..
                else:
                    lag_month_str_start = str(target_month-i-1) #10,11,12
                    lag_month_str_end = str(target_month-i-1) #10,11,12
        else: #if crossyear, do add years to lags (1950 and 2019)
            for j in range(1): #start and end date
                if target_month-i-1 <= 0: #crossyear, lagged months in the year before
                    if target_month+12-i-1 < 10:
                        lag_month_str_start = str(start_end_year[0]-1)+'-0'+str(target_month+12-i-1) #months in year before TV-targetperiod, 01, 02
                        lag_month_str_end = str(start_end_year[1]-1)+'-0'+str(target_month+12-i-1)
                    else:
                        lag_month_str_start = str(start_end_year[0]-1)+'-'+str(target_month+12-i-1) #months in year before TV-targetperiod, 10,11,12
                        lag_month_str_end = str(start_end_year[1]-1)+'-'+str(target_month+12-i-1)
                else: #crossyear, but lagged months not in the year before, for instance tv_month 02, lag month 01
                    if target_month-i-1 < 10:
                        lag_month_str_start = str(start_end_year[0])+'-0'+str(target_month-i-1) # 01 or 02 ..
                        lag_month_str_end = str(start_end_year[1])+'-0'+str(target_month-i-1) # 01 or 02 ..
                    else:
                        lag_month_str_start = str(start_end_year[0])+'-'+str(target_month-i-1) #10,11,12
                        lag_month_str_end = str(start_end_year[1])+'-'+str(target_month-i-1) #10,11,12
        lag.append(lag_month_str_start+'-01') #first day of month always 01
        lag_month_days_str_end = days_dict[lag_month_str_start[-2:]] #get last day of month from dict
        lag.append(lag_month_str_end+'-'+lag_month_days_str_end) #concatenate days and months
        lags.append(lag) #append to lags list
    print(lags)

    #list with input variables
    list_for_MI = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                               alpha=kwrgs_MI['alpha'], FDR_control=kwrgs_MI['FDR_control'],
                               lags=np.array(lags), # <- selecting time periods to aggregate
                               distance_eps=kwrgs_MI['distance_eps'],
                               min_area_in_degrees2=kwrgs_MI['min_area_in_degrees2']),
                  BivariateMI(name='swvl1_2', func=class_BivariateMI.corr_map,
                               alpha=kwrgs_MI['alpha'], FDR_control=kwrgs_MI['FDR_control'],
                               lags=np.array(lags), # <- selecting time periods to aggregate
                               distance_eps=kwrgs_MI['distance_eps'],
                               min_area_in_degrees2=kwrgs_MI['min_area_in_degrees2'])]

    #initialize RGCPD class
    rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               tfreq=None, # <- seasonal forecasting mode, set tfreq to None!
               start_end_TVdate=TV_targetperiod, # <- defining target period (whole year)
               path_outmain=os.path.join(main_dir,f'Results/{subfolder}/{list_of_name_path[0][0]}'))

    #preprocess TV
    rg.pp_TV(TVdates_aggr=True, kwrgs_core_pp_time = {'start_end_year':start_end_year}) # <- start_end_TVdate defineds aggregated over period

    return rg, list_for_MI, lags, crossyr


#%% Data inspection
def check(rg, list_of_name_path, cluster_nr):

    import matplotlib.pyplot as plt
    import core_pp

    t2m_path = list_of_name_path[0][1]
    t2m = core_pp.import_ds_lazy(t2m_path, format_lon = 'west_east')
    t2m_clus = t2m.sel(cluster=cluster_nr)

    sst_path = list_of_name_path[1][1]
    sst = core_pp.import_ds_lazy(sst_path, format_lon = 'west_east')

    swvl12_path = list_of_name_path[2][1]
    swvl12 = core_pp.import_ds_lazy(swvl12_path, format_lon = 'west_east')



    #example time series plot for first cluster
    plt.figure()
    t2m_clus.ts.plot()

    #check plot for sst
    plt.figure()
    sst[0].plot()

    #check plot for swvl
    plt.figure()
    swvl12[0].plot()

    # Check plot of clusters
    # if TVpath contains the xr.DataArray that is clustered beforehand, we can have a look at the spatial regions.
    ds = rg.get_clust(format_lon='west_east')
    fig = plot_maps.plot_labels(ds['xrclustered'],
                                kwrgs_plot={'col_dim':'n_clusters',
                                            'title':'Hierarchical Clustering',
                                            'cbar_tick_dict':{'labelsize':8},
                                            'add_cfeature':'BORDERS'})
#%% processing
def process(rg, lags, fold_method, crossyr):
    import find_precursors, plot_maps
    #Preprocess precursors
    rg.pp_precursors(detrend=True, anomaly=True, selbox=None, format_lon='west_east')

    #set any nan value in ts to 0
    # ds = rg.get_clust(format_lon='west_east')['ts'][:]
    # ds = ds[np.where(np.isnan(rg.get_clust(format_lon='west_east')['ts'][:]))]
    # rg.get_clust(format_lon='west_east')['ts'][np.where(np.isnan(rg.get_clust(format_lon='west_east')['ts'][:]))] = 0.0

    # ts plot
    rg.df_fullts.plot()

    # define train and test periods
    rg.traintest(method=fold_method, seed=1)
    testyrs = rg._get_testyrs()
    print(testyrs)

    # save target region plot
    target_cluster = int(rg.list_of_name_path[0][0])
    xrclustered = rg.get_clust(format_lon='west_east')['xrclustered']
    fig = plot_maps.plot_labels(find_precursors.view_or_replace_labels(xrclustered,
                                                                 regions=target_cluster))
    fig.savefig(os.path.join(rg.path_outsub1, 'target_cluster_{target_cluster}.jpeg'))
    # calculate correlation maps
    rg.calc_corr_maps()

    # show correlation maps
    rg.plot_maps_corr(kwrgs_plot={'clevels':np.arange(-.6,.61, 0.1)})

    #
    rg.cluster_list_MI()

    # define period names
    period_dict = {'01':'January',
                   '02':'February',
                   '03':'March',
                   '04':'April',
                   '05':'May',
                   '06':'June',
                   '07':'July',
                   '08':'August',
                   '09':'September',
                   '10':'October',
                   '11':'November',
                   '12':'December'}
    periodnames = []
    if crossyr:
        for i in lags:
            month_nr_str = i[0][i[0].find("-")+1:i[0].find("-")+1+2] #find first instace of "-" +2
            periodnames.append(period_dict[month_nr_str])
    else:
        for i in lags:
            month_nr_str = i[0][:2] #find first instace of "-" +2
            periodnames.append(period_dict[month_nr_str])

    for i in range(len(rg.list_for_MI)):
        rg.list_for_MI[i].prec_labels['lag'] = ('lag', periodnames)
        rg.list_for_MI[i].corr_xr['lag'] = ('lag', periodnames)


    # View correlation regions
    rg.quick_view_labels(mean=True, save=True)
    rg.plot_maps_corr(save=True)

    # Handle precursor regions
    rg.get_ts_prec()
    count = rg._df_count # how many times is each precursor regions found in the different training sets
    print(count)


    df_prec_regions = find_precursors.labels_to_df(rg.list_for_MI[0].prec_labels)
    df_prec_regions # center lat,lon coordinates and size (in number of gridcells)

    return rg

#%% Causal inference Tigramite

# import wrapper_PCMCI
# corr, pvals = wrapper_PCMCI.df_data_Parcorr(rg.df_data,
#                                             target='3ts',
#                                             keys=['June..2..sst'],
#                                             z_keys=['June..1..sst'])
# pvals

# df_trans = rg.transform_df_data()

# df_z_removed = wrapper_PCMCI.df_data_remove_z(df_trans,
#                                               keys=['3ts'],
#                                               z_keys=['June..2..sst'])



#%% Forecasting
def forecast(rg, crossyr):

    # Forecasting pipeline 1

    import func_models as fc_utils
    from stat_models_cont import ScikitModel
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LogisticRegressionCV


    # choose type prediciton (continuous or probabilistic) by making comment #
    prediction = 'continuous'
    # prediction = 'events' ; q = .66 # quantile threshold for event definition

    if prediction == 'continuous':
        model = ScikitModel(Ridge, verbosity=0)
        # You can also tune parameters by passing a list of values. Then GridSearchCV from sklearn will
        # find the set of parameters that give the best mean score on all kfold test sets.
        # below we pass a list of alpha's to tune the regularization.
        alphas = list(np.concatenate([[1E-20],np.logspace(-5,0, 6), np.logspace(.01, 2.5, num=25)]))
        kwrgs_model = {'scoringCV':'neg_mean_absolute_error',
                       'kfold':10,
                       'alpha':alphas} # large a, strong regul.
    elif prediction == 'events':
        model = ScikitModel(LogisticRegressionCV, verbosity=0)
        kwrgs_model = {'kfold':5,
                       'scoring':'neg_brier_score'}

    target_ts = rg.TV.RV_ts ;
    target_ts = (target_ts - target_ts.mean()) / target_ts.std()

    if prediction == 'events':
        q = 0.66
        if q >= 0.5:
            target_ts = (target_ts > target_ts.quantile(q)).astype(int)
        elif q < .5:
            target_ts = (target_ts < target_ts.quantile(q)).astype(int)
        BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
        score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

    elif prediction == 'continuous':
        RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE #RMSE ERROR SKILL SCORE
        MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE #MAE ERROR SKILL SCORE
        score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]


    keys = [k for k in rg.df_data.columns[1:-2]]
    out = rg.fit_df_data_ridge(target=target_ts,
                                keys=keys,
                                fcmodel=model,
                                kwrgs_model=kwrgs_model,
                                transformer=None,
                                tau_min=0, tau_max=0) # <- lag should be zero
    predict, weights, model_lags = out

    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                     rg.df_data.iloc[:,-2:],
                                                                     score_func_list,
                                                                     n_boot = 100, #intensive
                                                                     score_per_test=False,
                                                                     blocksize=1,
                                                                     rng_seed=1)
    lag = 0
    if prediction == 'events':
        print(model.scikitmodel.__name__, '\n', f'Test score at lag {lag}\n',
              'BSS {:.2f}\n'.format(df_test_m.loc[0].loc[0].loc['BSS']),
              'AUC {:.2f}'.format(df_test_m.loc[0].loc[0].loc['roc_auc_score']),
              '\nTrain score\n',
              'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['BSS']),
              'AUC {:.2f}'.format(df_train_m.mean(0).loc[0]['roc_auc_score']))
    elif prediction == 'continuous':
        print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'RMSE_SS {:.2f}\n'.format(df_test_m.loc[0][0]['RMSE']),
              'MAE_SS {:.2f}\n'.format(df_test_m.loc[0][0]['MAE']),
              'corrcoef {:.2f}'.format(df_test_m.loc[0][0]['corrcoef']),
              '\nTrain score\n',
              'RMSE_SS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['RMSE']),
              'MAE_SS {:.2f}\n'.format(df_train_m.mean(0).loc[0]['MAE']),
              'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[0]['corrcoef']))

    test_scores = [df_test_m.loc[0][0]['RMSE'],
                   df_test_m.loc[0][0]['MAE'],
                   df_test_m.loc[0][0]['corrcoef']]
    train_scores = [df_train_m.mean(0).loc[0]['RMSE'],
                    df_train_m.mean(0).loc[0]['MAE'],
                    df_train_m.mean(0).loc[0]['corrcoef']]

    return test_scores, train_scores, predict

#%% list of name path function

def get_list_of_name_path(agg_level, cl_number):

    # define input data:
    path_data = os.path.join(working_dir, 'Data') # path of data sets
    # format list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]

    agg_level = agg_level #low, medium or high aggregation level
    if agg_level == 'high':
        list_of_name_path = [(cl_number, os.path.join(path_data, '[20]_dendo_52baa.nc')), #for a single cluster!
                        ('sst', os.path.join(ERA5_data_dir,'sst_1950-2020_1_12_monthly_1.0deg.nc')), #sst = global
                        ('swvl1_2', os.path.join(ERA5_data_dir,'swvl_1950-2020_1_12_monthly_1.0deg_mask_0N80N.nc'))] #swvl = global, summed over layer 1 and 2
    elif agg_level == 'medium':
        list_of_name_path = [(cl_number, os.path.join(path_data, '[42]_dendo_fca84.nc')), #for a single cluster!
                        ('sst', os.path.join(ERA5_data_dir,'sst_1950-2020_1_12_monthly_1.0deg.nc')), #sst = global
                        ('swvl1_2', os.path.join(ERA5_data_dir,'swvl_1950-2020_1_12_monthly_1.0deg_mask_0N80N.nc'))] #swvl = global, summed over layer 1 and 2
    elif agg_level == 'low':
        list_of_name_path = [(cl_number, os.path.join(path_data, '[135]_dendo_1c7fe.nc')), #for a single cluster!
                        ('sst', os.path.join(ERA5_data_dir,'sst_1950-2020_1_12_monthly_1.0deg.nc')), #sst = global
                        ('swvl1_2', os.path.join(ERA5_data_dir,'swvl_1950-2020_1_12_monthly_1.0deg_mask_0N80N.nc'))] #swvl = global, summed over layer 1 and 2

    print(cl_number)
    return list_of_name_path


#%% loop analysis

def loop_analysis(agg_level, n_lags, kwrgs_MI, fold_method,
                  distinct_cl = None, distinct_targetperiods = None):
    #retrieve number of clusters with aggregation level
    if distinct_cl is None:
        ncl_dict = {'high': 20,
                    'medium': 42,
                    'low': 135}
        ncl = ncl_dict['{}'.format(agg_level)]
        cl_list = list(range(1,ncl+1))
    else:
        cl_list = distinct_cl

    subfolder = f'{agg_level}_{fold_method}'

    #target periods, all or given
    all_targetperiods = [('01-01','01-31'),('02-01','02-28'),('03-01','03-31'),('04-01','04-30'),
                     ('05-01','05-31'),('06-01','06-30'),('07-01','07-31'),('08-01','08-31'),
                     ('09-01','09-30'),('10-01','10-31'),('11-01','11-30'),('12-01','12-31')]

    if distinct_targetperiods is None:
        targetperiods = all_targetperiods
    else:
        targetperiods = distinct_targetperiods

    #create dictionary of periods and month names
    all_targetperiods_names_list = ['January','February','March','April','May',
                                    'June','July','August','September','October',
                                    'November','December']
    zip_iterator = zip(all_targetperiods, all_targetperiods_names_list)
    all_targetperiods_dict = dict(zip_iterator)

    #create indices for multi index result dataframe
    row_idx_1_arr = np.array([val for val in cl_list for _ in range(6)]) # 1 1 1 1 1 1 2 2 2 2 2 2 ...
    row_idx_2 = ['test','train']
    row_idx_2_list = [val for val in row_idx_2 for _ in range(3)] # test test test train train train
    if len(cl_list) > 1:
        row_idx_2_list += (len(cl_list)-1)*row_idx_2_list # test test test train train train test test test train ...
    row_idx_2_arr = np.array(row_idx_2_list)
    row_idx_3_list = ['RMSE_SS','MAE_SS','corrcoef', 'RMSE_SS','MAE_SS','corrcoef'] #RMSE MAE corrcoef RMSE MAE corrcoef
    if len(cl_list) > 1:
        row_idx_3_list += (len(cl_list)-1)*row_idx_3_list  #RMSE MAE corrcoef RMSE MAE corrcoef RMSE MAE ...
    row_idx_3_arr = np.array(row_idx_3_list)
    row_arrays = [row_idx_1_arr,
                  row_idx_2_arr,
                  row_idx_3_arr] #cluster, test/train, scores as row multi index
    column_array = [all_targetperiods_dict[x] for x in targetperiods] #month names as column index

    #initiate zeros ss_result dataframe, rows = months, columns = scores per cluster
    df_ss_result = pd.DataFrame(np.zeros((len(row_arrays[0]), len(column_array)), dtype=float),
                             index=row_arrays, columns=column_array)

    #initiate zeros prediction_result dataframe, rows = not yet known, columns = target time series & prediction
    df_prediction_result = pd.DataFrame()

    #loop over clusters and target months
    for cluster_counter, cluster in enumerate(cl_list):

        #get list_of_name_path
        list_of_name_path = get_list_of_name_path(agg_level, cluster)
        for month_counter, month in enumerate(targetperiods):
            #run define
            rg, list_for_MI, lags, crossyr = define(list_of_name_path, month, n_lags, kwrgs_MI, subfolder)
            #run check (possible, not necessary)
            check(rg, list_of_name_path, cluster)
            #run processing
            rg = process(rg, lags, fold_method, crossyr)
            #run forecast
            test_scores, train_scores, prediction = forecast(rg, crossyr)

            #store skill score results in df_ss_result dataframe
            for count, i in enumerate(row_idx_2_arr[:6]): #always loop over test test test train train train per cluster
                if count < 3:
                        df_ss_result.loc[(cluster,i,row_idx_3_arr[count]), all_targetperiods_dict[month]] = test_scores[count]
                else:
                        df_ss_result.loc[(cluster,i,row_idx_3_arr[count]), all_targetperiods_dict[month]] = train_scores[count-3]

            #get test df actual and predictions
            test_df_pred = functions_pp.get_df_test(prediction, df_splits = pd.DataFrame(rg.df_data.iloc[:,-2:]))

            #update dates
            delta = int(month[0][:2])-1
            date_list = test_df_pred.index.get_level_values(0).shift(delta, freq='MS')
            test_df_pred.set_index([date_list], inplace=True)

            #change column header of prediction to RV#ts_pred
            new_columns = test_df_pred.columns.values
            new_columns[1] = new_columns[0]+'_pred'
            test_df_pred.columns = new_columns

            #append to prediction_result dataframe
            if cluster_counter == 0:
                df_prediction_result = df_prediction_result.append(test_df_pred)
            elif cluster_counter > 0 and month_counter == 0:
                df_prediction_result = df_prediction_result.join(test_df_pred, how='left')
            else:
                df_prediction_result.update(test_df_pred, join='left')

        #save intermediate cluster csv
        results_path = os.path.join(main_dir, 'Results', 'intermediate_'+fold_method) #path of results
        os.makedirs(results_path, exist_ok=True) # make folder if it doesn't exist
        df_ss_result.to_csv(os.path.join(results_path, str(cluster)+'_ss_scores_'+agg_level+'.csv')) #intermediate save skillscores per cluster to csv

    #return df_ss_result dataframe and prediction
    return df_ss_result, df_prediction_result, rg


#%%
def plot_ss2(rg, skillscores, col_wrap, metric=None):
    import find_precursors

    #load cluster_ds
    ds = rg.get_clust(format_lon='west_east')
    cluster_labels_org = ds.coords['cluster']
    ds = ds['xrclustered']

    #create list of skill score names
    skillscores_multi_idx = skillscores.index.levels
    ss_list  = []
    for i in skillscores_multi_idx[1:][0]:
        for j in skillscores_multi_idx[1:][1]:
            ss_name = '{}_{}'.format(i,j)
            ss_list.append(ss_name)

    if metric is not None: #only apply single metric
        ss_list = [metric]

    #add dimensions and coordinates
    xr_score = ds.copy() ; xr_score.attrs = {}
    list_xr = [xr_score.copy().expand_dims('metric', axis=0) for m in ss_list]
    xr_score = xr.concat(list_xr, dim = 'metric')
    xr_score['metric'] = ('metric', ss_list)
    list_xr = [xr_score.copy().expand_dims('target_month', axis=0) for m in skillscores.columns]
    xr_score = xr.concat(list_xr, dim = 'target_month')
    xr_score['target_month'] = ('target_month', skillscores.columns)

    #replace labels with skillscores
    for metric_nr, metric in enumerate(xr_score.metric.values):
        test_or_train = metric[:metric.find("_")]
        ss = metric[metric.find("_")+1:]
        for month_nr, month in enumerate(xr_score.target_month.values):
            #slice over metric, month in skill score df
            metric_cluster_dict = skillscores[month].xs((test_or_train, ss), level=(1,2)).to_dict()
            #replace cluster_labels with their skill score
            cluster_labels_new = [metric_cluster_dict.get(x, x) for x in cluster_labels_org.values]
            #set all non replaced values of cluster labels to np.nan
            cluster_labels_new = [np.nan if isinstance(x,np.int32) else x for x in cluster_labels_new]

            #replace values
            xarr_labels_to_replace = ds
            xr_score[month_nr,metric_nr] = find_precursors.view_or_replace_labels(xarr_labels_to_replace,
                                                                          regions=list(cluster_labels_org.values),
                                                                          replacement_labels=cluster_labels_new)

    #set col wrap and subtitles
    col_wrap = col_wrap #int determines nr of cols
    import math
    subtitles = [[] for i in range(int(math.ceil(xr_score.target_month.values.size/col_wrap)))]
    total_nr_fields = col_wrap*len(subtitles)
    j=-1
    for i, month in enumerate(xr_score.target_month.values):
        if i%col_wrap == 0:
            j+=1
        subtitles[j].append('{}, {}'.format(month, metric))
        if i == max(list(enumerate(xr_score.target_month.values)))[0] and total_nr_fields > xr_score.target_month.values.size:
            for k in range(total_nr_fields - xr_score.target_month.values.size):
                subtitles[j].append('0')

    #plot
    fig = plot_maps.plot_corr_maps(xr_score, col_dim = 'target_month', row_dim='metric',
                                   size = 4, clevels = np.arange(-.5,0.51,.1),
                                   cbar_vert=-0.1, hspace=-0.2,
                                   subtitles=subtitles, col_wrap=col_wrap)

    return fig

#%% run with params
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-i", "--intexper", help="intexper", type=int,
                        default=i_default)
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    #--------------------------------------------------------------------------------------------------------------------#
    #PARAMS
    #--------------------------------------------------------------------------------------------------------------------#
    agg_level_list = ['low', 'medium', 'high'] # high, medium or low
    fold_method_list = ['random_20', 'leave_1']
    ncl_dict = {'high': 20,
                'medium': 42,
                'low': 135}

    combinations = np.array(np.meshgrid(agg_level_list,
                                        fold_method_list)).T.reshape(-1,2)
    i_default = 0


    args = parseArguments()
    out = combinations[args.intexper]
    agg_level = out[0]
    cluster_numbers = np.arange(1,ncl_dict[agg_level]) #list with ints, high=20, medium=42, low=135
    #cluster_numbers = cluster_numbers.tolist()
    #cluster_numbers = [x for x in cluster_numbers if x not in [7,9,14]] #skip HIGH: [7,9,14],
                                                                        #MEDIUM: [2,5,17,20,22,27,35,37,38], LOW: []
                                                                        #LOW: [2,6,14,15,18,27,28,30,31,34,37,39,42,43,45,47,48,52,55,58,60,61,62,63,64,65,67,72,74,78,83,85,86,88,95,96,103,105,107,111,114,115,118,121,123,125,126,127,132,133,135]
    TV_targetperiod = None # list with tuples [(mm-dd,mm-dd)] or if None, all months are targeted
    n_lags = 3 #int, max 12
    kwrgs_list_for_MI = {'alpha':0.01,
                         'FDR_control':True,
                         'distance_eps':500,
                         'min_area_in_degrees2':5} #some controls for bivariateMI
    fold_method = str(out[1])   #choose from:
                                # (1) random_{int}   :   with the int(ex['method'][6:8]) determining the amount of folds
                                # (2) ranstrat_{int}:   random stratified folds, stratified based upon events,
                                #                       requires kwrgs_events.
                                # (3) leave_{int}    :   chronologically split train and test years.
                                # (4) split_{int}    :   (should be updated) split dataset into single train and test set
                                # (5) no_train_test_split or False 'random_#'
    col_wrap = 3 #3 months next to each other in figure

    #--------------------------------------------------------------------------------------------------------------------#
    #LOOP
    #--------------------------------------------------------------------------------------------------------------------#
    df_ss_result, df_prediction_result, rg = loop_analysis(agg_level, n_lags, kwrgs_list_for_MI, fold_method,
                  distinct_cl = cluster_numbers, distinct_targetperiods = TV_targetperiod)
    print(df_ss_result, '\n' , df_prediction_result)

    #--------------------------------------------------------------------------------------------------------------------#
    #PLOT
    #--------------------------------------------------------------------------------------------------------------------#
    fig = plot_ss2(rg, df_ss_result, col_wrap, metric='test_RMSE_SS')

    #--------------------------------------------------------------------------------------------------------------------#
    #SAVE
    #--------------------------------------------------------------------------------------------------------------------#
    results_path = os.path.join(os.path.dirname(main_dir), 'Results') #path of results

    datetimestamp = datetime.now()
    datetimestamp_str = datetimestamp.strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(os.path.join(results_path, datetimestamp_str+ '_test_RMSE_SS_fig_'+agg_level+'.png')) #save skillscore figures

    df_ss_result.to_csv(os.path.join(results_path, datetimestamp_str+'_ss_scores_'+agg_level+'.csv')) #save skillscores to csv
    df_prediction_result.to_csv(os.path.join(results_path, datetimestamp_str+ '_predictions_'+agg_level+'.csv')) #save predictions to csv

#%% check ts of clusters
def check_ts(agg_level):
    ncl_dict = {'high': 20,
                    'medium': 42,
                    'low': 135}
    clusters = np.arange(1,ncl_dict[agg_level]+1)
    path_data = os.path.join(os.path.dirname(main_dir), 'Data') # path of data sets
    kwrgs_list_for_MI = {'alpha':0.01,
                         'FDR_control':True,
                         'distance_eps':500,
                         'min_area_in_degrees2':5} #some controls for bivariateMI
    kwrgs_MI = kwrgs_list_for_MI
    targetperiods = [('01-01','01-31'),('02-01','02-28'),('03-01','03-31'),('04-01','04-30'),
                         ('05-01','05-31'),('06-01','06-30'),('07-01','07-31'),('08-01','08-31'),
                         ('09-01','09-30'),('10-01','10-31'),('11-01','11-30'),('12-01','12-31')]
    n_lags = 3 #int, max 12

    allts = pd.DataFrame()
    for c in clusters:
        print(c)
        for t in targetperiods:
            if agg_level == 'high':
                list_of_name_path = [(c, os.path.join(path_data, '[20]_dendo_52baa.nc')), #for a single cluster!
                                ('sst', os.path.join(path_data,'sst_1950-2020_1_12_monthly_1.0deg.nc')), #sst = global
                                ('swvl1_2', os.path.join(path_data,'swvl_1950-2020_1_12_monthly_1.0deg_mask_0N80N.nc'))] #swvl = global, summed over layer 1 and 2
            elif agg_level == 'medium':
                list_of_name_path = [(c, os.path.join(path_data, '[42]_dendo_fca84.nc')), #for a single cluster!
                                ('sst', os.path.join(path_data,'sst_1950-2020_1_12_monthly_1.0deg.nc')), #sst = global
                                ('swvl1_2', os.path.join(path_data,'swvl_1950-2020_1_12_monthly_1.0deg_mask_0N80N.nc'))] #swvl = global, summed over layer 1 and 2
            elif agg_level == 'low':
                list_of_name_path = [(c, os.path.join(path_data, '[135]_dendo_1c7fe.nc')), #for a single cluster!
                                ('sst', os.path.join(path_data,'sst_1950-2020_1_12_monthly_1.0deg.nc')), #sst = global
                                ('swvl1_2', os.path.join(path_data,'swvl_1950-2020_1_12_monthly_1.0deg_mask_0N80N.nc'))] #swvl = global, summed over layer 1 and 2

                #create lag list
            days_dict = {'01':'31',
                           '02':'28',
                           '03':'31',
                           '04':'30',
                           '05':'31',
                           '06':'30',
                           '07':'31',
                           '08':'31',
                           '09':'30',
                           '10':'31',
                           '11':'30',
                           '12':'31'}

            target_month_str = t[0][:2] #derive month number
            if target_month_str[0] == '0':
                target_month = int(target_month_str[1]) # 01 or 02 ..
            else:
                target_month = int(target_month_str[:]) #10,11,12

            if target_month - (n_lags) <= 0: #cross year?
                crossyr = True
                start_end_year = (1951,2020) #hardcoded
            else:
                crossyr = False
                start_end_year = None

            lags = [] #initialize empty lags list
            for i in range(n_lags):
                lag = [] #initialize empty lag list
                if not crossyr: #if not crossyear with lags, do not add years to lags
                    for j in range(1): #start and end date
                        if target_month-i-1 < 10:
                            lag_month_str_start = '0'+str(target_month-i-1) # 01 or 02 ..
                            lag_month_str_end = '0'+str(target_month-i-1) # 01 or 02 ..
                        else:
                            lag_month_str_start = str(target_month-i-1) #10,11,12
                            lag_month_str_end = str(target_month-i-1) #10,11,12
                else: #if crossyear, do add years to lags (1950 and 2019)
                    for j in range(1): #start and end date
                        if target_month-i-1 <= 0: #crossyear, lagged months in the year before
                            if target_month+12-i-1 < 10:
                                lag_month_str_start = str(start_end_year[0]-1)+'-0'+str(target_month+12-i-1) #months in year before TV-targetperiod, 01, 02
                                lag_month_str_end = str(start_end_year[1]-1)+'-0'+str(target_month+12-i-1)
                            else:
                                lag_month_str_start = str(start_end_year[0]-1)+'-'+str(target_month+12-i-1) #months in year before TV-targetperiod, 10,11,12
                                lag_month_str_end = str(start_end_year[1]-1)+'-'+str(target_month+12-i-1)
                        else: #crossyear, but lagged months not in the year before, for instance tv_month 02, lag month 01
                            if target_month-i-1 < 10:
                                lag_month_str_start = str(start_end_year[0])+'-0'+str(target_month-i-1) # 01 or 02 ..
                                lag_month_str_end = str(start_end_year[1])+'-0'+str(target_month-i-1) # 01 or 02 ..
                            else:
                                lag_month_str_start = str(start_end_year[0])+'-'+str(target_month-i-1) #10,11,12
                                lag_month_str_end = str(start_end_year[1])+'-'+str(target_month-i-1) #10,11,12
                lag.append(lag_month_str_start+'-01') #first day of month always 01
                lag_month_days_str_end = days_dict[lag_month_str_start[-2:]] #get last day of month from dict
                lag.append(lag_month_str_end+'-'+lag_month_days_str_end) #concatenate days and months
                lags.append(lag) #append to lags list

            #list with input variables
            list_for_MI = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                                       alpha=kwrgs_MI['alpha'], FDR_control=kwrgs_MI['FDR_control'],
                                       lags=np.array(lags), # <- selecting time periods to aggregate
                                       distance_eps=kwrgs_MI['distance_eps'],
                                       min_area_in_degrees2=kwrgs_MI['min_area_in_degrees2']),
                          BivariateMI(name='swvl1_2', func=class_BivariateMI.corr_map,
                                       alpha=kwrgs_MI['alpha'], FDR_control=kwrgs_MI['FDR_control'],
                                       lags=np.array(lags), # <- selecting time periods to aggregate
                                       distance_eps=kwrgs_MI['distance_eps'],
                                       min_area_in_degrees2=kwrgs_MI['min_area_in_degrees2'])]

            #initialize RGCPD class
            rg = RGCPD(list_of_name_path=list_of_name_path,
                       list_for_MI=list_for_MI,
                       tfreq=None, # <- seasonal forecasting mode, set tfreq to None!
                       start_end_TVdate=t, # <- defining target period (whole year)
                       path_outmain=os.path.join(main_dir,'data'))

            #preprocess TV
            rg.pp_TV(TVdates_aggr=True, kwrgs_core_pp_time = {'start_end_year':start_end_year}) # <- start_end_TVdate defineds aggregated over period

            #update dates
            month = int(t[0][:2])
            delta = month-1
            df = rg.df_fullts[:]
            date_list =  df.index.get_level_values(0).shift(delta, freq='MS')
            df.set_index([date_list], inplace=True)

            #store
            if c-1 == 0:
                allts  = allts.append(df)
            elif c-1 > 0 and month-1 == 0:
                allts = allts.join(df, how='left')
            else:
                allts.update(df, join='left')

    datetimestamp = datetime.now()
    datetimestamp_str = datetimestamp.strftime("%Y-%m-%d_%H-%M-%S")
    path_data = os.path.join(os.path.dirname(main_dir), 'Data') # path of data sets
    allts.to_csv(os.path.join(path_data, datetimestamp_str+'_cl_ts_'+agg_level+'.csv')) #save skillscores to csv

    return allts

#check_ts('high')