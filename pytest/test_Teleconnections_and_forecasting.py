#!/usr/bin/env python
# coding: utf-8

# # Find Teleconnections (precursor regions) via correlation maps

# In[1]:

import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = sep.join(curr_dir.split(sep)[:-1])
print(main_dir)
if main_dir not in sys.path:
    sys.path.append(main_dir)
from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI, functions_pp
import numpy as np
import shutil


def check_dates_RV(df_splits, traintestgroups, start_end_TVdate):
    df_splits_gr = df_splits.loc[0][traintestgroups==1]
    yrs = np.unique(df_splits_gr.index.year)
    if yrs.size > 1:
        startyr, endyr = yrs
    else:
        startyr = yrs[0] ; endyr = yrs[0]
    startTVdate = df_splits_gr[df_splits_gr['RV_mask']].index[0]
    endTVdate   = df_splits_gr[df_splits_gr['RV_mask']].index[-1]
    sd = functions_pp.pd.to_datetime(f'{startyr}-'+start_end_TVdate[0])
    ed = functions_pp.pd.to_datetime(f'{endyr}-'+start_end_TVdate[-1])
    assert sd <= startTVdate, 'Selected date not in RV window'
    assert ed >= endTVdate, 'Selected date not in RV window'
    print(startTVdate, endTVdate)

def test_subseas_US_t2m_tigramite(alpha=0.05, tfreq=10, method='random_5',
                                  start_end_TVdate=('07-01', '08-31'),
                                  dailytomonths=False,
                                  TVdates_aggr=False,
                                  lags=np.array([1]),
                                  start_end_yr_precur=None,
                                  start_end_yr_target=None):
    #%%
    # define input: list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]
    # start_end_yr_target=None; start_end_yr_precur = None; lags = np.array([1]); TVdates_aggr=False; dailytomonths=False;
    # alpha=0.05; tfreq=10; method='random_5';start_end_TVdate=('07-01', '08-31');

    curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
    main_dir = sep.join(curr_dir.split(sep)[:-1])
    path_test = os.path.join(main_dir, 'data')

    list_of_name_path = [(3, os.path.join(path_test, 'tf5_nc5_dendo_80d77.nc')),
                        ('sst', os.path.join(path_test,'sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc'))]

    # define analysis:
    list_for_MI = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                              alpha=alpha, FDR_control=True, lags=lags,
                              distance_eps=700, min_area_in_degrees2=5,
                              dailytomonths=dailytomonths)]

    rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               start_end_TVdate=start_end_TVdate,
               tfreq=tfreq,
               path_outmain=os.path.join(main_dir,'data', 'test'),
               save=True)

    # if TVpath contains the xr.DataArray xrclustered, we can have a look at the spatial regions.
    rg.plot_df_clust()

    rg.pp_precursors(detrend=True, anomaly=True, selbox=None)


    # ### Post-processing Target Variable
    rg.pp_TV(TVdates_aggr=TVdates_aggr,
              kwrgs_core_pp_time={'dailytomonths':dailytomonths,
                                  'start_end_year':start_end_yr_target})


    rg.traintest(method=method)

    # check
    if TVdates_aggr==False:
        check_dates_RV(rg.df_splits, rg.traintestgroups, start_end_TVdate)

    rg.kwrgs_load['start_end_year'] = start_end_yr_precur

    rg.calc_corr_maps()
    precur = rg.list_for_MI[0]

    rg.plot_maps_corr()

    rg.cluster_list_MI()

    # rg.quick_view_labels(mean=False)

    rg.get_ts_prec()
    try:
        import wrapper_PCMCI as wPCMCI
        if rg.df_data.columns.size <= 3:
            print('Skipping causal inference step')
        else:
            rg.PCMCI_df_data()

            rg.PCMCI_get_links(var=rg.TV.name, alpha_level=.05)
            rg.df_links

            rg.store_df_PCMCI()
    except:
        # raise(ModuleNotFoundError)
        print('Not able to load in Tigramite modules, to enable causal inference '
          'features, install Tigramite from '
          'https://github.com/jakobrunge/tigramite/')
    #%%
    return rg

test = test_subseas_US_t2m_tigramite


#%%
# =============================================================================
# Subseasonal use-cases (daily and monhtly)
# =============================================================================

# Daily data aggregated to 10-dm, JA, random_5
rg = test_subseas_US_t2m_tigramite()

# Daily data aggregated to 10-dm, DJF, random_5
rg = test(alpha=.3, start_end_TVdate=('11-01', '02-28'))

# Daily data aggregated to 10-dm, DJFM, No Train Test Split
rg = test(alpha=.3, method=False,
          start_end_TVdate=('12-01', '03-31'))

# Daily data aggregated to 10-dm, NDJF, No Train Test Split
rg = test(alpha=.3, method=False,
          start_end_TVdate=('11-01', '02-28'))

# Daily data aggregated to 20-dm, JJA, random_5
rg = test(alpha=.1, tfreq=20,
          start_end_TVdate=('06-01', '08-31'))

# Daily-to-monthly data, 2-month mean JJA, random_5
rg = test(alpha=.1, dailytomonths=True, tfreq=2,
          start_end_TVdate=('06-01', '08-31'))

# =============================================================================
# Seasonal use-cases (from daily to monhtly)
# =============================================================================

# Daily to JJA mean, random_5, precursor JFMAM
rg = test(alpha=.2,
          tfreq=None,
          TVdates_aggr=True,
          start_end_TVdate=('06-01', '08-31'),
          lags=np.array([['01-01', '05-31']]))

# Daily to DJF mean, random_5, precursor SON
rg = test(alpha=.7,
          tfreq=None,
          TVdates_aggr=True,
          start_end_TVdate=('12-01', '02-28'),
          lags=np.array([['09-01', '11-30']]),
          start_end_yr_precur=(1979,2017))

# Daily to JJA mean, random_5, precursor oct-may
rg = test(alpha=.2,
          tfreq=None,
          TVdates_aggr=True,
          start_end_TVdate=('06-01', '08-31'),
          lags=np.array([['10-01', '05-31']]),
          start_end_yr_target=(1980,2018))



#%%



rg = test_subseas_US_t2m_tigramite()
# Forecasting pipeline 1
import func_models as fc_utils
from stat_models_cont import ScikitModel
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
RFmodel = ScikitModel(RandomForestClassifier, verbosity=0)
kwrgs_model={'n_estimators':200,
            'max_depth':2,
            'scoringCV':'neg_brier_score',
            'oob_score':True,
            # 'min_samples_leaf':None,
            'random_state':0,
            'max_samples':.3,
            'n_jobs':1}

# choose type prediciton (continuous or probabilistic) by making comment #
prediction = 'continuous' ; q = None
# prediction = 'events' ; q = .66 # quantile threshold for event definition

if prediction == 'continuous':
    model = ScikitModel(Ridge, verbosity=0)
    # You can also tune parameters by passing a list of values. Then GridSearchCV from sklearn will
    # find the set of parameters that give the best mean score on all kfold test sets.
    # below we pass a list of alpha's to tune the regularization.
    kwrgs_model = {'scoringCV':'neg_mean_absolute_error',
                    'alpha':list(np.concatenate([[1E-20],np.logspace(-5,0, 6),
                                              np.logspace(.01, 2.5, num=25)])), # large a, strong regul.
                    'normalize':False,
                    'fit_intercept':False,
                    'kfold':5}
elif prediction == 'events':
    model = ScikitModel(LogisticRegressionCV, verbosity=0)
    kwrgs_model = {'kfold':5,
                    'scoring':'neg_brier_score'}


# target
target_ts = rg.TV.RV_ts.copy()
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
if prediction == 'events':
    if q >= 0.5:
        target_ts = (target_ts > target_ts.quantile(q)).astype(int)
    elif q < .5:
        target_ts = (target_ts < target_ts.quantile(q)).astype(int)
    BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
    score_func_list = [BSS, fc_utils.metrics.roc_auc_score]

elif prediction == 'continuous':
    RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
    MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
    score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]


out = rg.fit_df_data_ridge(target=target_ts,
                            keys=None,
                            fcmodel=model,
                            kwrgs_model=kwrgs_model,
                            transformer=False,
                            tau_min=1, tau_max=2)
predict, weights, model_lags = out

df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                  rg.df_data.iloc[:,-2:],
                                                                  score_func_list,
                                                                  n_boot = 100,
                                                                  score_per_test=False,
                                                                  blocksize=1,
                                                                  rng_seed=1)
lag = 1
if prediction == 'events':

    print(model.scikitmodel.__name__, '\n', f'Test score at lag {lag}\n',
          'BSS {:.2f}\n'.format(df_test_m.loc[0].loc[lag].loc['BSS']),
          'AUC {:.2f}'.format(df_test_m.loc[0].loc[lag].loc['roc_auc_score']),
          '\nTrain score\n',
          'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[lag]['BSS']),
          'AUC {:.2f}'.format(df_train_m.mean(0).loc[lag]['roc_auc_score']))
elif prediction == 'continuous':
    print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'RMSE {:.2f}\n'.format(df_test_m.loc[0][lag]['RMSE']),
              'MAE {:.2f}\n'.format(df_test_m.loc[0][lag]['MAE']),
              'corrcoef {:.2f}'.format(df_test_m.loc[0][lag]['corrcoef']),
              '\nTrain score\n',
              'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[lag]['RMSE']),
              'MAE {:.2f}\n'.format(df_train_m.mean(0).loc[lag]['MAE']),
              'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[lag]['corrcoef']))


# # Forecasting pipeline 2
# Used for paper https://doi.org/10.1175/MWR-D-19-0409.1
#
# There is some multiprocessing based on Python's standard 'concurrent futures' module. This only works when run script is run in one go. Will not work another time. Has to do with the running the code as the __main__ file or something.. (don't know the details).

# Now we load in the data, including info on the causal links.
try: # check if tigramite is installed
    import wrapper_PCMCI as wPCMCI
except ImportError as e:
    print('Not able to load in Tigramite modules, to enable causal inference '
          'features, install Tigramite from '
          'https://github.com/jakobrunge/tigramite/')
 	# remove created output folders
    shutil.rmtree(rg.path_outsub1)
    shutil.rmtree(os.path.join(main_dir, 'data', 'preprocessed'))
    raise(e)

from class_fc import fcev
import valid_plots as dfplots

if __name__ == '__main__':

    #%% test parallizing pipeline

    try:
        from joblib import Parallel, delayed
    except:
        print('Not able to load in joblib module or test parallization failed')

    tfreq_list = [10, 20]
    futures = []
    for tfreq in tfreq_list:
         # pipeline(lags, periodnames)
         futures.append(delayed(test)(0.05, tfreq))

    with Parallel(n_jobs=2, backend="loky", timeout=25) as loky:
        out = loky(futures)



    fc = fcev(path_data=path_df_data, n_cpu=1, causal=True)
    fc.get_TV(kwrgs_events=None)
    fc.fit_models(lead_max=35)
    dict_experiments = {}
    fc.perform_validation(n_boot=100, blocksize='auto',
                                  threshold_pred=(1.5, 'times_clim'))
    dict_experiments['test'] = fc.dict_sum


    working_folder, filename = fc._print_sett(list_of_fc=[fc])
    store=True
    dict_all = dfplots.merge_valid_info([fc], store=store)
    if store:
        dict_merge_all = functions_pp.load_hdf5(filename)

    kwrgs = {'wspace':0.25, 'col_wrap':3} #, 'threshold_bin':fc.threshold_pred}
    met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision', 'Accuracy']
    expers = list(dict_experiments.keys())
    line_dim = 'model'


    fig = dfplots.valid_figures(dict_merge_all,
                              line_dim=line_dim,
                              group_line_by=None,
                              lines_legend=None,
                              met=met, **kwrgs)

    # remove created output folders
    shutil.rmtree(rg.path_outsub1)
    shutil.rmtree(os.path.join(main_dir, 'data', 'preprocessed'))


