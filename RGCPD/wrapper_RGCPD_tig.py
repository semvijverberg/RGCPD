# -*- coding: utf-8 -*-
#%%

import sys, os, io


from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import functions_RGCPD as rgcpd
import itertools
import numpy as np
import xarray as xr
import datetime
import cartopy.crs as ccrs
import pandas as pd
import functions_pp
import plot_maps
flatten = lambda l: list(itertools.chain.from_iterable(l))

#%%
def calculate_corr_maps(ex, map_proj):
    #%%
    # =============================================================================
    # Load 'exp' dictionairy with information of pre-processed data (variables, paths, filenames, etcetera..)
    # and add RGCPD/Tigrimate experiment settings
    # =============================================================================
    # Response Variable is what we want to predict
    RV = ex[ex['RV_name']]
    ex['time_cycle'] = RV.dates[RV.dates.year == RV.startyear].size # time-cycle of data. total timesteps in one year
    #=====================================================================================
    # Information on period taken for response-variable, already decided in main_download_and_pp
    #=====================================================================================
    ex['time_range_all'] = [0, RV.dates.size]
    #==================================================================================
    # Start of experiment
    #==================================================================================

    # Define traintest:
    df_splits, ex = functions_pp.rand_traintest_years(RV, ex)
    # =============================================================================
    # 2) DEFINE PRECURSOS COMMUNITIES:
    # =============================================================================
    # - calculate and plot pattern correltion for differnt fields
    # - create time-series over these regions
    #=====================================================================================
    outdic_actors = dict()
    class act:
        def __init__(self, name, corr_xr, precur_arr):
            self.name = var
            self.corr_xr = corr_xr
            self.precur_arr = precur_arr
            self.lat_grid = precur_arr.latitude.values
            self.lon_grid = precur_arr.longitude.values
            self.area_grid = rgcpd.get_area(precur_arr)
            self.grid_res = abs(self.lon_grid[1] - self.lon_grid[0])

    allvar = ex['vars'][0] # list of all variable names
    for var in allvar[ex['excludeRV']:]: # loop over all variables
        actor = ex[var]
        #===========================================
        # 3c) Precursor field
        #===========================================  
        file_path = os.path.join(actor.path_pp, actor.filename_pp)
        precur_arr = functions_pp.import_ds_timemeanbins(file_path, ex)
#        precur_arr = rgcpd.convert_longitude(precur_arr, 'only_east') 
        # =============================================================================
        # Calculate correlation
        # =============================================================================
        corr_xr = rgcpd.calc_corr_coeffs_new(precur_arr, RV, ex)
        
        # =============================================================================
        # Convert regions in time series
        # =============================================================================
        actor = act(var, corr_xr, precur_arr)
        actor, ex = rgcpd.cluster_DBSCAN_regions(actor, ex)
        if np.isnan(actor.prec_labels.values).all() == False:
            rgcpd.plot_regs_xarray(actor.prec_labels.copy(), ex)
          
        # Order of regions: strongest to lowest correlation strength
        outdic_actors[var] = actor
        # =============================================================================
        # Plot
        # =============================================================================
        if ex['plotin1fig'] == False:
#            xrdata, xrmask = xrcorr_vars([var], outdic_actors, ex)
            plot_maps.plot_corr_maps(corr_xr, corr_xr['mask'], map_proj)

            fig_filename = '{}_corr_{}_vs_{}'.format(ex['params'], ex['RV_name'], var) + ex['file_type2']
            plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
            if ex['showplot'] == False:
                plt.close()


#    if ex['plotin1fig'] == True and ex['showplot'] == True:
#        variables = list(outdic_actors.keys())
##        xrdata, xrmask = xrcorr_vars(variables, outdic_actors, ex)
#        plot_corr_maps(xrdata, xrmask, map_proj)
#        fig_filename = '{}_corr_all'.format(ex['params'], allvar[0], var) + ex['file_type2']
#        plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
#        if ex['showplot'] == False:
#            plt.close()
#%%
    return ex, outdic_actors

def get_prec_ts(outdic_actors, ex):
    # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
    # stacked on top of each other (from lag_min to lag_max)
    
    ex['n_tot_regs'] = 0
    allvar = ex['vars'][0] # list of all variable names
    for var in allvar[ex['excludeRV']:]: # loop over all variables
        actor = outdic_actors[var]
        
        if np.isnan(actor.prec_labels.values).all():
            actor.ts_corr = np.array([]), []
            pass
        else:
            actor.ts_corr = rgcpd.spatial_mean_regions(actor, ex)
            outdic_actors[var] = actor
            ex['n_tot_regs'] += max([actor.ts_corr[s].shape[1] for s in range(ex['n_spl'])])
    return outdic_actors
        

def run_PCMCI_CV(ex, outdic_actors, map_proj):
    #%%
    df_splits = np.zeros( (ex['n_spl']) , dtype=object)
    df_data_s   = np.zeros( (ex['n_spl']) , dtype=object)
    for s in range(ex['n_spl']):
        progress = 100 * (s+1) / ex['n_spl']
        print(f"\rProgress causal inference - traintest set {progress}%", end="")
        df_splits[s], df_data_s[s] = run_PCMCI(ex, outdic_actors, s, map_proj)
        
    print("\n")
    
    df_data  = pd.concat(list(df_data_s), keys= range(ex['n_spl']))
    df_sum = pd.concat(list(df_splits), keys= range(ex['n_spl']))


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
        store_hdf_df(dict_of_dfs, ex['path_data'])
        
    return


def store_hdf_df(dict_of_dfs, file_path):
    import warnings
    import tables
    
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    with pd.HDFStore(file_path, 'w') as hdf:
        for key, item in  dict_of_dfs.items():
            hdf.put(key, item, format='table', data_columns=True)
        hdf.close()
    return

def run_PCMCI(ex, outdic_actors, s, map_proj):
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
    traintest = ex['traintest']

    # load Response Variable class
    RV = ex[ex['RV_name']]
    # create list with all actors, these will be merged into the fulldata array
    allvar = ex['vars'][0]
    var_names = [] ; actorlist = [] ; cols = [[RV.name]]
    
    for var in allvar[ex['excludeRV']:]:
        print(var)
        actor = outdic_actors[var]
        if actor.ts_corr[s].size != 0:
            ts_train = actor.ts_corr[s].values
            actorlist.append(ts_train)
            # create array which numbers the regions
            var_idx = allvar.index(var) - ex['excludeRV']
            n_regions = actor.ts_corr[s].shape[1]
            actor.var_info = [[i+1, actor.ts_corr[s].columns[i], var_idx] for i in range(n_regions)]
            # Array of corresponing regions with var_names (first entry is RV)
            var_names = var_names + actor.var_info
        cols.append(list(actor.ts_corr[s].columns))
    var_names.insert(0, RV.name)

        
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)   
    
        
    print(('There are {} regions in total'.format(fulldata.shape[1])))
    # add the full 1D time series of interest as first entry:

    fulldata = np.column_stack((RV.RVfullts, fulldata))
    
    
    df_data = pd.DataFrame(fulldata, columns=flatten(cols), index=actor.ts_corr[s].index)
    RVfull_train = RV.RVfullts.isel(time=traintest[s]['Prec_train_idx'])
    datesfull_train = pd.to_datetime(RVfull_train.time.values)
    data = df_data.loc[datesfull_train].values
    print((data.shape))

    
    # get RV datamask (same shape als data)
    datesRV_train   = pd.to_datetime(traintest[s]['RV_train'].time.values)
    data_mask = [True if d in datesRV_train else False for d in datesfull_train]
    data_mask = np.repeat(data_mask, data.shape[1]).reshape(data.shape)

    # add traintest mask to fulldata
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
    dataframe = pp.DataFrame(data=data, mask=data_mask, var_names=var_names)
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
    
    df = rgcpd.bookkeeping_precursors(links_RV, var_names)
    #%%

    rgcpd.print_particular_region_new(links_RV, var_names, s, outdic_actors, map_proj, ex)

    
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


    
def standard_settings_and_tests(ex):
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
    ex['png_dpi'] = 400
    ex['SaveTF'] = True # if false, output will be printed in console
    ex['plotin1fig'] = False
    ex['showplot'] = True
    # output paths
    method_str = '_'.join([ex['method'], 's'+ str(ex['seed'])])
    ex['subfolder_exp'] = ex['path_exp_periodmask'] +'_'+ method_str
    ex['path_output'] = os.path.join(ex['subfolder_exp'])
    ex['fig_path'] = os.path.join(ex['subfolder_exp'])
    
    ex['params'] = '{}_ac{}_at{}'.format(ex['pcA_set'], ex['alpha'],
                                                      ex['alpha_level_tig'])
    if os.path.isdir(ex['fig_path']) != True : os.makedirs(ex['fig_path'])
    ex['fig_subpath'] = os.path.join(ex['fig_path'], '{}_subinfo'.format(ex['params']))
    if os.path.isdir(ex['fig_subpath']) != True : os.makedirs(ex['fig_subpath'])

    
    assert RV.startyear == ex['startyear'], ('Make sure the dates '
             'of the RV match with the actors')
    assert ((ex['excludeRV'] == 1) and (ex['importRV_1dts'] == True))==False, ('Are you sure you want '
             'exclude first element of array ex[\'vars\']. You are importing a seperate '
             ' time series, so you probably do not need to skip the first variable in ex[\'vars\'] ')
    # =============================================================================
    # Save Experiment design
    # =============================================================================
    filename_exp_design = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
    np.save(filename_exp_design, ex)
    print('\n\t**\n\tOkay, end of Part 2!\n\t**' )

    print('\n**\nBegin summary of main experiment settings\n**\n')
    print('Response variable is {} is correlated vs {}'.format(RV.name,
          ex['vars'][0][:]))
    start_day = '{}-{}'.format(RV.dates[0].day, RV.dates[0].month_name())
    end_day   = '{}-{}'.format(RV.dates[-1].day, RV.dates[-1].month_name())
    print('Part of year investigated: {} - {}'.format(start_day, end_day))
    print('Part of year predicted (RV period): {} '.format(RV.RV_name_range[:-1]))
    print('Temporal resolution: {} {}'.format(ex['tfreq'], ex['input_freq']))
    print('Lags: {}'.format(ex['lags']))
    print('Traintest setting: {} seed {}'.format(method_str.split('_')[0], method_str.split('_s')[1]))
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
            

