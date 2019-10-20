# -*- coding: utf-8 -*-
#%%

import sys, os, io


from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import functions_RGCPD as rgcpd
import matplotlib as mpl
import itertools
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import functions_pp
import func_fc
import classes
#import plot_maps
flatten = lambda l: list(itertools.chain.from_iterable(l))

#%%
def RV_and_traintest(RV, ex, method=str, kwrgs_events=None, precursor_ts=None,
                     seed=int):
                        
    ex['time_cycle'] = RV.dates[RV.dates.year == RV.startyear].size # time-cycle of data. total timesteps in one year
    ex['time_range_all'] = [0, RV.dates.size]
    verbosity=ex['verbosity']
    #==================================================================================
    # Start of experiment
    #==================================================================================

    # Define traintest:
    df_RVfullts = pd.DataFrame(RV.RVfullts.values, 
                               index=pd.to_datetime(RV.RVfullts.time.values))
    df_RV_ts    = pd.DataFrame(RV.RV_ts.values,
                               index=pd.to_datetime(RV.RV_ts.time.values))
    if method[:9] == 'ran_strat':
        kwrgs_events = kwrgs_events
        RV = classes.RV_class(df_RVfullts, df_RV_ts, kwrgs_events)
    else:
        RV = classes.RV_class(df_RVfullts, df_RV_ts)
    
    if precursor_ts is not None:
        # Retrieve same train test split as imported ts
        path_data = ''.join(precursor_ts[0][1])
        df_splits = func_fc.load_hdf5(path_data)['df_data'].loc[:,['TrainIsTrue', 'RV_mask']]
        test_yrs_imp  = functions_pp.get_testyrs(df_splits)
        df_splits = functions_pp.rand_traintest_years(RV, method=method,
                                                          seed=seed, 
                                                          kwrgs_events=kwrgs_events, 
                                                          verb=verbosity)
        test_yrs_set  = functions_pp.get_testyrs(df_splits)
        assert (np.equal(test_yrs_imp, test_yrs_set)).all(), "Train test split not equal"
    else:
        df_splits = functions_pp.rand_traintest_years(RV, method=method,
                                                          seed=seed, 
                                                          kwrgs_events=kwrgs_events, 
                                                          verb=verbosity)
    return RV, df_splits

def calculate_corr_maps(RV, df_splits, ex, list_varclass=list, lags=[0], alpha=0.05,
                        FDR_control=True, plot=True):
                         
    #%%
    

    outdic_actors = dict()
    class act:
        def __init__(self, name, corr_xr, precur_arr):
            self.name = name
            self.corr_xr = corr_xr
            self.precur_arr = precur_arr
            self.lat_grid = precur_arr.latitude.values
            self.lon_grid = precur_arr.longitude.values
            self.area_grid = rgcpd.get_area(precur_arr)
            self.grid_res = abs(self.lon_grid[1] - self.lon_grid[0])


    for var_class in list_varclass: # loop over all variables
        #===========================================
        # 3c) Precursor field
        #===========================================
        file_path = os.path.join(var_class.path_pp, var_class.filename_pp)
        precur_arr = functions_pp.import_ds_timemeanbins(file_path, ex)
#        precur_arr = rgcpd.convert_longitude(precur_arr, 'only_east')
        # =============================================================================
        # Calculate correlation
        # =============================================================================
        corr_xr = rgcpd.calc_corr_coeffs_new(precur_arr, RV, df_splits, lags=lags,
                                             alpha=alpha, FDR_control=FDR_control)

        # =============================================================================
        # Cluster into precursor regions
        # =============================================================================
        actor = act(var_class.name, corr_xr, precur_arr)

        outdic_actors[actor.name] = actor

    return outdic_actors

def cluster_regions(outdic_actors, ex, plot=True, distance_eps=400, min_area_in_degrees2=3,
                    group_split='together'):
    
    for name, actor in outdic_actors.items():
        actor = rgcpd.cluster_DBSCAN_regions(actor, distance_eps=400, 
                                             min_area_in_degrees2=3, group_split='together')
        if plot and np.isnan(actor.prec_labels.values).all() == False:
            prec_labels = actor.prec_labels.copy()
            rgcpd.plot_regs_xarray(prec_labels, ex)
        outdic_actors[name] = actor

#%%
    return outdic_actors

def get_prec_ts(outdic_actors, ex):
    # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
    # stacked on top of each other (from lag_min to lag_max)

    ex['n_tot_regs'] = 0
    allvar = ex['vars'][0] # list of all variable names
    for var in allvar[:]: # loop over all variables
        actor = outdic_actors[var]
        splits = actor.corr_xr.split
        if np.isnan(actor.prec_labels.values).all():
            actor.ts_corr = np.array([]), []
            pass
        else:
            actor.ts_corr = rgcpd.spatial_mean_regions(actor, ex)
            outdic_actors[var] = actor
            ex['n_tot_regs'] += max([actor.ts_corr[s].shape[1] for s in range(splits.size)])
    return outdic_actors


def run_PCMCI_CV(ex, outdic_actors, df_splits, map_proj):
    #%%
    splits = df_splits.index.levels[0]
    
    df_sum_s = np.zeros( (splits.size) , dtype=object)
    df_data_s   = np.zeros( (splits.size) , dtype=object)
    for s in range(splits.size):
        progress = 100 * (s+1) / splits.size
        print(f"\rProgress causal inference - traintest set {progress}%", end="")
        df_sum_s[s], df_data_s[s] = run_PCMCI(ex, outdic_actors, s, df_splits, map_proj)

    print("\n")

    df_data  = pd.concat(list(df_data_s), keys= range(splits.size))
    df_sum = pd.concat(list(df_sum_s), keys= range(splits.size))


    #%%
    return df_sum, df_data

def store_ts(df_data, df_sum, dict_ds, outdic_actors, ex, add_spatcov=True):


    today = datetime.datetime.today().strftime('%Y-%m-%d')
    file_name = 'fulldata_{}_{}'.format(ex['params'], today)
    ex['path_data'] = os.path.join(ex['fig_subpath'], file_name+'.h5')

    if add_spatcov:
        df_sp_s   = np.zeros( (ex['n_spl']) , dtype=object)
        for s in range(ex['n_spl']):
            df_split = df_data.loc[s]
            df_sp_s[s] = rgcpd.get_spatcovs(dict_ds, df_split, s, outdic_actors, normalize=True)

        df_sp = pd.concat(list(df_sp_s), keys= range(ex['n_spl']))
        df_data_to_store = pd.merge(df_data, df_sp, left_index=True, right_index=True)
        df_sum_to_store = rgcpd.add_sp_info(df_sum, df_sp)
    else:
        df_data_to_store = df_data
        df_sum_to_store = df_sum

    dict_of_dfs = {'df_data':df_data_to_store, 'df_sum':df_sum_to_store}
    if ex['store_format'] == 'hdf5':
        functions_pp.store_hdf_df(dict_of_dfs, ex['path_data'])
        print('Data stored in \n{}'.format(ex['path_data']))
    return


def run_PCMCI(ex, outdic_actors, s, df_splits, map_proj):
    #=====================================================================================
    #
    # 4) PCMCI-algorithm
    #
    #=====================================================================================

    # save output
    if ex['SaveTF'] == True:
#        from contextlib import redirect_stdout
        orig_stdout = sys.stdout
        # buffer print statement output to f
        if sys.version[:1] == '3':
            sys.stdout = f = io.StringIO()
        elif sys.version[:1] == '2':
            sys.stdout = f = open(os.path.join(ex['fig_subpath'], 'old.txt'), 'w+')

#%%
    # amount of text printed:
    verbosity = 3

    # alpha level for independence test within the pc procedure (finding parents)
    pc_alpha = ex['pcA_sets'][ex['pcA_set']]
    # alpha level for multiple linear regression model while conditining on parents of
    # parents
    alpha_level = ex['alpha_level_tig']
    print('run tigramite 4, run.pcmci')
    print(('alpha level(s) for independence tests within the pc procedure'
          '(finding parents): {}'.format(pc_alpha)))
    print(('alpha level for multiple linear regression model while conditining on parents of '
          'parents: {}'.format(ex['alpha_level_tig'])))

    # Retrieve traintest info
    traintest = df_splits

    # load Response Variable class
    RV = ex[ex['RV_name']]
    # create list with all actors, these will be merged into the fulldata array
    allvar = ex['vars'][0]
    var_names_corr = [] ; actorlist = [] ; cols = [[RV.name]]

    for var in allvar[:]:
        print(var)
        actor = outdic_actors[var]
        if actor.ts_corr[s].size != 0:
            ts_train = actor.ts_corr[s].values
            actorlist.append(ts_train)
            # create array which numbers the regions
            var_idx = allvar.index(var) 
            n_regions = actor.ts_corr[s].shape[1]
            actor.var_info = [[i+1, actor.ts_corr[s].columns[i], var_idx] for i in range(n_regions)]
            # Array of corresponing regions with var_names_corr (first entry is RV)
            var_names_corr = var_names_corr + actor.var_info
        cols.append(list(actor.ts_corr[s].columns))
    var_names_corr.insert(0, RV.name)


    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)


    print(('There are {} regions in total'.format(fulldata.shape[1])))
    # add the full 1D time series of interest as first entry:

    fulldata = np.column_stack((RV.RVfullts, fulldata))
    df_data = pd.DataFrame(fulldata, columns=flatten(cols), index=actor.ts_corr[s].index)
    
    if ex['import_prec_ts'] == True:
        var_names_full = var_names_corr.copy()
        for d in ex['precursor_ts']:
            path_data = d[1]    
            if len(path_data) > 1:
                path_data = ''.join(list(path_data))
            # skip first col because it is the RV ts
            df_data_ext = func_fc.load_hdf5(path_data)['df_data'].iloc[:,1:].loc[s]
            cols_ts = np.logical_or(df_data_ext.dtypes == 'float64', df_data_ext.dtypes == 'float32')
            cols_ext = list(df_data_ext.columns[cols_ts])
            # cols_ext must be of format '{}_{int}_{}'
            lab_int = 100
            for i, c in enumerate(cols_ext):
                char = c.split('_')[1]
                if char.isdigit():
                    pass
                else:
                    cols_ext[i] = c.replace(char, str(lab_int)) + char
                    lab_int += 1
                    
            df_data_ext = df_data_ext[cols_ext]
            to_freq = ex['tfreq']
            if to_freq != 1:
                start_end_date = (ex['sstartdate'], ex['senddate'])
                start_end_year = (ex['startyear'], ex['endyear'])
            df_data_ext = functions_pp.time_mean_bins(df_data_ext, to_freq,
                                        start_end_date,
                                        start_end_year,
                                        seldays='part')[0]
#            df_data_ext = functions_pp.time_mean_bins(df_data_ext,
#                                                     ex, ex['tfreq'], 
#                                                     seldays='part')[0]
            # Expand var_names_corr
            n = var_names_full[-1][0] + 1 ; add_n = n + len(cols_ext)
            n_var_idx = var_names_full[-1][-1] + 1
            for i in range(n, add_n):
                var_names_full.append([i, cols_ext[i-n], n_var_idx])
            df_data = df_data.merge(df_data_ext, left_index=True, right_index=True)
    else:
        var_names_full = var_names_corr
            
    bool_train     = traintest.loc[s]['TrainIsTrue']  
    bool_RV_train  = np.logical_and(bool_train, traintest.loc[s]['RV_mask'])
    dates_train    = traintest.loc[s]['TrainIsTrue'][bool_train].index
    dates_RV_train = traintest.loc[s]['TrainIsTrue'][bool_RV_train].index
    
    RVfull_train = RV.RVfullts.sel(time=dates_train)
    datesfull_train = pd.to_datetime(RVfull_train.time.values)
    data = df_data.loc[datesfull_train].values
    print((data.shape))

    # get RV datamask (same shape als data)
    data_mask = [True if d in dates_RV_train else False for d in datesfull_train]
    data_mask = np.repeat(data_mask, data.shape[1]).reshape(data.shape)

    # add traintest mask to fulldata
#    dates_all = pd.to_datetime(RV.RVfullts.index)
#    dates_RV  = pd.to_datetime(RV.RV_ts.index)
    dates_all = pd.to_datetime(RV.RVfullts.time.values)
    dates_RV  = pd.to_datetime(RV.RV_ts.time.values)
    df_data['TrainIsTrue'] = [True if d in datesfull_train else False for d in dates_all]
    df_data['RV_mask'] = [True if d in dates_RV else False for d in dates_all]


    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================

    T, N = data.shape # Time, Regions
    # ======================================================================================================================
    # Initialize dataframe object (needed for tigramite functions)
    # ======================================================================================================================
    dataframe = pp.DataFrame(data=data, mask=data_mask, var_names=var_names_full)
    # ======================================================================================================================
    # pc algorithm: only parents for selected_variables are calculated
    # ======================================================================================================================

    parcorr = ParCorr(significance='analytic',
                      mask_type='y',
                      verbosity=verbosity)
    #==========================================================================
    # multiple testing problem:
    #==========================================================================
    pcmci   = PCMCI(dataframe=dataframe,
                    cond_ind_test=parcorr,
                    selected_variables=None,
                    verbosity=4)

    # selected_variables : list of integers, optional (default: range(N))
    #    Specify to estimate parents only for selected variables. If None is
    #    passed, parents are estimated for all variables.

    # ======================================================================================================================
    #selected_links = dictionary/None
    results = pcmci.run_pcmci(tau_max=ex['tigr_tau_max'], pc_alpha = pc_alpha, tau_min = 0,
                              max_combinations=ex['max_comb_actors'])

    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

    pcmci.print_significant_links(p_matrix = results['p_matrix'],
                                   q_matrix = q_matrix,
                                   val_matrix = results['val_matrix'],
                                   alpha_level = alpha_level)

    # returns all parents, not just causal precursors (of lag>0)
    sig = rgcpd.return_sign_parents(pcmci, pq_matrix=q_matrix,
                                            val_matrix=results['val_matrix'],
                                            alpha_level=alpha_level)


    all_parents = sig['parents']
#    link_matrix = sig['link_matrix']

    links_RV = all_parents[0]

    df = rgcpd.bookkeeping_precursors(links_RV, var_names_full)
    #%%

    rgcpd.print_particular_region_new(links_RV, var_names_corr, s, outdic_actors, map_proj, ex)


#%%
    if ex['SaveTF'] == True:
        if sys.version[:1] == '3':
            fname = f's{s}_' +ex['params']+'.txt'
            file = io.open(os.path.join(ex['fig_subpath'], fname), mode='w+')
            file.write(f.getvalue())
            file.close()
            f.close()
        elif sys.version[:1] == '2':
            f.close()
        sys.stdout = orig_stdout


    return df, df_data

#%%



def standard_settings_and_tests(ex, kwrgs_RV, kwrgs_corr):
    '''Some boring settings and Perform some test'''

    RV = ex[ex['RV_name']]
    # =============================================================================
    # Test if you're not have a lag that will precede the start date of the year
    # =============================================================================
    # first date of year to be analyzed:
    if ex['input_freq'] == 'daily':
        f = 'D'
    elif ex['input_freq'] == 'monthly':
        f = 'M'

    firstdoy = RV.dates_RV.min() - np.timedelta64(int(max(ex['lags'])), f)
    if firstdoy < RV.dates[0] and (RV.dates[0].month,RV.dates[0].day) != (1,1):
        tdelta = RV.dates_RV.min() - RV.dates.min()
        lag_max = int(tdelta / np.timedelta64(ex['tfreq'], 'D'))
        ex['lags'] = ex['lags'][ex['lags'] < lag_max]
        ex['lags_i'] = ex['lags_i'][ex['lags_i'] < lag_max]
        print(('Changing maximum lag to {}, so that you not skip part of the '
              'year.'.format(max(ex['lags'])) ) )

    # Some IO settings
    ex['store_format']   = 'hdf5'
    ex['file_type1'] = ".pdf"
    ex['file_type2'] = ".png"
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 600
    ex['SaveTF'] = True # if false, output will be printed in console
    ex['plotin1fig'] = False
    ex['showplot'] = False
    # output paths
    method_str = '_'.join([kwrgs_RV['method'], 's'+ str(kwrgs_RV['seed'])])
    ex['subfolder_exp'] = ex['path_exp_periodmask'] +'_'+ method_str
    ex['path_output'] = os.path.join(ex['subfolder_exp'])
    ex['fig_path'] = os.path.join(ex['subfolder_exp'])

    ex['params'] = '{}_ac{}_at{}'.format(ex['pcA_set'], kwrgs_corr['alpha'],
                                                      ex['alpha_level_tig'])
    if os.path.isdir(ex['fig_path']) != True : os.makedirs(ex['fig_path'])
    ex['fig_subpath'] = os.path.join(ex['fig_path'], '{}_subinfo'.format(ex['params']))
    if os.path.isdir(ex['fig_subpath']) != True : os.makedirs(ex['fig_subpath'])


    assert RV.startyear == ex['startyear'], ('Make sure the dates '
             'of the RV match with the actors')
    # =============================================================================
    # Save Experiment design
    # =============================================================================
    filename_exp_design = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
    np.save(filename_exp_design, ex)

    print('\n**\nBegin summary of main experiment settings\n**\n')
    print('Response variable is {} is correlated vs {}'.format(RV.name,
          ex['vars'][0][:]))
    start_day = '{}-{}'.format(RV.dates[0].day, RV.dates[0].month_name())
    end_day   = '{}-{}'.format(RV.dates[-1].day, RV.dates[-1].month_name())
    print('Part of year investigated: {} - {}'.format(start_day, end_day))
    print('Part of year predicted (RV period): {} '.format(RV.RV_name_range[:-1]))
    print('Temporal resolution: {} {}'.format(ex['tfreq'], ex['input_freq']))
    print('Lags: {}'.format(ex['lags']))
    if ex['import_prec_ts']:
        print('\nTraintest years adjusted to TrainTest split in the imported timeseries.')
        print('Dataframe of imported ts, should have columns TrainIsTrue and RV_mask\n')
    else:
        print('\nTraintest setting: {} seed {}\n'.format(method_str.split('_')[0], method_str.split('_s')[1]))
    one_year_RV_data = RV.dates_RV.where(RV.dates_RV.year==RV.startyear).dropna(how='all').values
    print('For example\nPredictant (only one year) is:\n{} at \n{}\n'.format(ex['RV_name'],
          one_year_RV_data))
    print('\tVS\n')
    shift_lag_days = one_year_RV_data - pd.Timedelta(min(ex['lags']), unit='d')
    print('Predictor (only one year) at is:\n{} at lag {} \n{}\n'.format(
            ex['vars'][0][-1], min(ex['lags']),
            shift_lag_days))
    print('\n**\nEnd of summary\n**\n')

    print('\nNext time, you\'re able to redo the experiment by loading in the dict '
          '\'filename_exp_design\'.\n')
    return ex


#        kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault',
#                       'steps' : ex['max_N_regs']+1, 'subtitles': None,
#                       'vmin' : 0, 'vmax' : ex['max_N_regs'],
#                       'cmap' : cmap, 'column' : 1,
#                       'cbar_vert' : adjust_vert_cbar, 'cbar_hght' : 0.0,
#                       'adj_fig_h' : adj_fig_h, 'adj_fig_w' : 1.,
#                       'hspace' : 0.0, 'wspace' : 0.08,
#                       'cticks_center' : False, 'title_h' : 0.95} )
#        filename = '{}_{}_vs_{}'.format(ex['params'], ex['RV_name'], for_plt.name) + ex['file_type2']
#        rgcpd.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)


