# -*- coding: utf-8 -*-
import os, io, sys
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPDC, CMIknn, CMIsymb
import numpy as np
import pandas as pd
import itertools




def init_pcmci(df_data, significance='analytic', mask_type='y', 
               selected_variables=None, verbosity=4):
    '''
    First initializing pcmci object for each training set. This allows to plot
    lagged cross-correlations which help to identity a reasonably tau_max.

    Parameters
    ----------
    df_data : pandas DataFrame
        df_data is retrieved by running rg.get_ts_prec().
    significance : str, optional
        DESCRIPTION. The default is 'analytic'.
    mask_type : str, optional
        DESCRIPTION. The default is 'y'.
    verbosity : int, optional
        DESCRIPTION. The default is 4.
    selected_variables : list of integers, optional (default: None)
        Specify to estimate parents only for selected variables. If None is
        passed, parents are estimated for all variables.

    Returns
    -------
    dictionary of format {split:pcmci}.

    '''
    splits = df_data.index.levels[0]
    pcmci_dict = {}
    RV_mask = df_data['RV_mask']
    for s in range(splits.size):
        
        TrainIsTrue = df_data['TrainIsTrue'].loc[s]
        df_data_s = df_data.loc[s][TrainIsTrue.values]
        df_data_s = df_data_s.dropna(axis=1, how='all')
        if any(df_data_s.isna().values.flatten()):
            print('Warnning: nans detected')
#        print(np.unique(df_data_s.isna().values))
        var_names = list(df_data_s.columns[(df_data_s.dtypes != np.bool)])
        df_data_s = df_data_s.loc[:,var_names]
        data = df_data_s.values
        data_mask = RV_mask.loc[s][TrainIsTrue.values].values
        data_mask = np.repeat(data_mask, data.shape[1]).reshape(data.shape)
        
        # create dataframe in Tigramite format
        dataframe = pp.DataFrame(data=data, mask=data_mask, var_names=var_names)

        parcorr = ParCorr(significance=significance,
                          mask_type=mask_type,
                          verbosity=0)
        
        # ======================================================================================================================
        # pc algorithm: only parents for selected_variables are calculated
        # ======================================================================================================================
        pcmci   = PCMCI(dataframe=dataframe,
                        cond_ind_test=parcorr,
                        selected_variables=None,
                        verbosity=verbosity)
        pcmci_dict[s] = pcmci
    return pcmci_dict

def plot_lagged_dependences(pcmci, selected_links: dict=None, tau_max=5):
    if selected_links is None:
        # only focus on target variable links:
        TV_links = list(itertools.product(range(pcmci.N), -1* np.arange(tau_max)))
        selected_links = {0:TV_links}
        selected_links.update({k:[] for k in range(1,pcmci.N)})
    origverbosity= pcmci.verbosity ; pcmci.verbosity = 0
    correlations = pcmci.get_lagged_dependencies(selected_links=selected_links,
                                                 tau_max=tau_max)
    df_lagged = pd.DataFrame(correlations[:,0,:-1], index=pcmci.var_names, 
                             columns=range(tau_max))
    
    df_lagged.T.plot(figsize=(10,10))
    # pcmci.lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations[:,0], 
    #                                    setup_args={'var_names':pcmci.var_names, 
    #                                                'x_base':5, 'y_base':.5,
    #                                                'figsize':(10,10)})
    pcmci.verbosity = origverbosity
    return

def loop_train_test(pcmci_dict, path_txtoutput, tau_min=0, tau_max=1, pc_alpha=None, 
                    max_conds_dim=4, max_combinations=1, 
                    max_conds_py=None, max_conds_px=None, verbosity=4):
    '''
    pc_alpha (float, optional (default: 0.05)) 
        Significance level in algorithm.
    tau_min (int, default: 0) 
        Minimum time lag.
    tau_max (int, default: 1) 
        Maximum time lag. Must be larger or equal to tau_min.
    max_conds_dim (int, optional (default: None)) 
        Maximum number of conditions to test. If None is passed, this number is unrestricted.
    max_combinations (int, optional (default: 1)) 
        Maximum number of combinations of conditions of current cardinality 
        to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm a 
        larger number, such as 10, can be used.
    max_conds_py (int or None) 
        Maximum number of conditions from parents of Y to use. If None is passed, 
        this number is unrestricted.
    max_conds_px (int or None) 
        Maximum number of conditions from parents of X to use. If None is passed, 
        this number is unrestricted.

    '''
    #%%
#    df_data = rg.df_data
#    path_txtoutput=rg.path_outsub2; tau_min=0; tau_max=1; pc_alpha=0.05; 
#    alpha_level=0.05; max_conds_dim=2; max_combinations=1; 
#    max_conds_py=None; max_conds_px=None; verbosity=4
                    
    splits = np.array(list(pcmci_dict.keys()))

    pcmci_results_dict = {}
    for s in range(splits.size):
        progress = 100 * (s+1) / splits.size
        print(f"\rProgress causal inference - traintest set {progress}%", end="")
        results = run_pcmci(pcmci_dict[s], path_txtoutput, s,
                        tau_min, tau_max, pc_alpha, max_conds_dim, 
                        max_combinations, max_conds_py, max_conds_px,  
                        verbosity)
        pcmci_results_dict[s] = results
    #%%
    return pcmci_results_dict

    #%%
def run_pcmci(pcmci, path_outsub2, s, tau_min=0, tau_max=1, 
              pc_alpha=None, max_conds_dim=4, max_combinations=1, 
              max_conds_py=None, max_conds_px=None, verbosity=4):
    

    
    #%%
    if path_outsub2 is not False:
        txt_fname = os.path.join(path_outsub2, f'split_{s}_PCMCI_out.txt')
#        from contextlib import redirect_stdout
        orig_stdout = sys.stdout
        # buffer print statement output to f
        sys.stdout = f = io.StringIO()
    #%%            
    # ======================================================================================================================
    # tigramite 4
    # ======================================================================================================================
    pcmci.cond_ind_test.print_info()
    print(f'time {pcmci.T}, samples {pcmci.N}')
    
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha, tau_min=tau_min,
                              max_conds_dim=max_conds_dim, 
                              max_combinations=max_combinations,
                              max_conds_px=max_conds_px,
                              max_conds_py=max_conds_py)

    results['q_matrix'] = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], 
                                                      fdr_method='fdr_bh')

    # print @ 3 alpha level values:
    alphas = [.1, .05, .01]
    for a in alphas:
        pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                       q_matrix=results['q_matrix'],
                                       val_matrix=results['val_matrix'],
                                       alpha_level=a)

    #%%
    if path_outsub2 is not False:
        file = io.open(txt_fname, mode='w+')
        file.write(f.getvalue())
        file.close()
        f.close()

        sys.stdout = orig_stdout


    return results

def get_links_pcmci(pcmci_dict, pcmci_results_dict, alpha_level):
    #%%
    splits = np.array(list(pcmci_dict.keys()))
    
    parents_dict = {}
    for s in range(splits.size):
        
        pcmci = pcmci_dict[s]
        results = pcmci_results_dict[s]
        # returns all causal links, not just causal parents/precursors (of lag>0)
        sig = return_sign_links(pcmci, pq_matrix=results['q_matrix'],
                                            val_matrix=results['val_matrix'],
                                            alpha_level=alpha_level)

        all_parents = sig['parents']
    #    link_matrix = sig['link_matrix']
    
        links_RV = all_parents[0]
        parents_dict[s] = links_RV, pcmci.var_names
        
    #%%
    return parents_dict

def get_df_sum(parents_dict):
    splits = np.array(list(parents_dict.keys()))
    df_sum_s = np.zeros( (splits.size) , dtype=object)
    mapping_links_dict = {}
    for s in range(splits.size):
        links_RV, var_names = parents_dict[s] 
        df, mapping_links = bookkeeping_precursors(links_RV, var_names)
        df_sum_s[s] = df
        
        mapping_links_dict[s] = mapping_links
    df_sum = pd.concat(list(df_sum_s), keys= range(splits.size))
    count_causal = df_sum.loc[:,'causal'].sum(level=[1]).astype(int)
    df_sum['count'] = pd.concat([count_causal] * splits.size, 
                                   keys=range(splits.size))
    return df_sum, mapping_links_dict

def return_sign_links(pc_class, pq_matrix, val_matrix,
                            alpha_level=0.05):
      # Initialize the return value
    all_parents = dict()
    for j in pc_class.selected_variables:
        # Get the good links
        good_links = np.argwhere(pq_matrix[:, j, :] <= alpha_level)
        # Build a dictionary from these links to their values
        links = {(i, -tau): np.abs(val_matrix[i, j, abs(tau) ])
                 for i, tau in good_links}
        # Sort by value
        all_parents[j] = sorted(links, key=links.get, reverse=True)
    # Return the significant parents
    return {'parents': all_parents,
            'link_matrix': pq_matrix <= alpha_level}


def bookkeeping_precursors(links_RV, var_names):
    #%%
    # if links_RV[0][0] == 0: # index 0 should refer to RV_name
    RV_name = var_names[0]
    # else:
        # RV_name = None
    var_names_ = var_names.copy()
    mapping_links = [(var_names_[l[0]],l[1]) for l in links_RV]
    lin_RVsor = sorted(links_RV)
    index = [n.split('..')[1] for n in var_names_[1:]] ; 
    index.insert(0, var_names_[0])
    link_names = [var_names_[l[0]].split('..')[1] if l[0] !=0 else var_names_[l[0]] for l in lin_RVsor]

    # check if two lags of same region and are tigr significant
    idx_tigr = [l[0] for l in lin_RVsor] ;
    var_names_ext = var_names_.copy()
    index_ext = index.copy()
    for r in np.unique(idx_tigr):
        # counting double indices (but with different lags)
        if idx_tigr.count(r) != 1:
            if var_names_[r] == RV_name:
                double = var_names_[r].split('..')[0]
            else:
                double = var_names_[r].split('..')[1]
#            print(double)
            idx = int(np.argwhere(np.array(index)==double)[0])
            # append each double to index for the dataframe
            for i in range(idx_tigr.count(r)-1):
                index_ext.insert(idx+i, double)
                d = len(index) - len(var_names_)
#                var_names_ext.insert(idx+i+1, var_names_[idx+1-d])
                var_names_ext.insert(idx+i, var_names_[idx-d])            

    # retrieving only var name
    l = [n.split('..')[-1] for n in var_names_ext[1:]]
    l.insert(0, var_names_ext[0])
    var = np.array(l)
    # creating mask (True) of causal links
    mask_causal = np.array([True if i in link_names else False for i in index_ext])
    # retrieving lag of corr map
    
    TV_lag = np.repeat(0, var_names_ext.count(RV_name))
    prec_corr_map = np.array([int(n.split('..')[0]) for n in var_names_ext if n != RV_name]) ;
    lag_corr_map = np.insert(prec_corr_map, TV_lag, 0) # unofficial lag for TV

    # retrieving region number, corresponding to figures
    region_number = np.array([int(n.split('..')[1]) for n in var_names_ext if n != RV_name])
    # unofficial region  number for TV
    region_number = np.insert(region_number, TV_lag, 0) 
    # retrieving lag of tigramite link
    # all Tigr links, can include same region at multiple lags:
    # looping through all unique tigr var labels format {lag..var_name}
    lag_tigr_map = {str(lin_RVsor[i][1])+'..'+link_names[i]:lin_RVsor[i][1] for i in range(len(link_names))}
    sorted(lag_tigr_map, reverse=False)
    # fill in the nans where the var_name is not causal
    lag_tigr_ = index_ext.copy() ; track_idx = list(range(len(index_ext)))
    # looping through all unique tigr var labels and tracking the concomitant indices 
    for k in lag_tigr_map.keys():
        l = int(k.split('..')[0])
        var_temp = k.split('..')[1]
        idx = lag_tigr_.index(var_temp)
        track_idx.remove(idx)
        lag_tigr_[idx] = l
    for i in track_idx:
        lag_tigr_[i] = np.nan
    lag_tigr_ = np.array(lag_tigr_)

    mask_causal = ~np.isnan(lag_tigr_)

    # print(var.shape, lag_corr_map.shape, region_number.shape, mask_causal.shape, lag_tigr_.shape)

    data = np.concatenate([lag_corr_map[None,:], region_number[None,:], var[None,:],
                            mask_causal[None,:], lag_tigr_[None,:]], axis=0)
    df = pd.DataFrame(data=data.T, index=var_names_ext,
                      columns=['lag_corr', 'region_number', 'var', 'causal', 'lag_tig'])
    df['causal'] = df['causal'] == 'True'
    df = df.astype({'lag_corr':int,
                               'region_number':int, 'var':str, 'causal':bool, 'lag_tig':float})
    #%%
    return df, mapping_links


## =============================================================================
## Haversine function
## =============================================================================
#from math import radians, cos, sin, asin, sqrt
#from enum import Enum
#
#
## mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
#_AVG_EARTH_RADIUS_KM = 6371.0088
#
#
#class Unit(Enum):
#    """
#    Enumeration of supported units.
#    The full list can be checked by iterating over the class; e.g.
#    the expression `tuple(Unit)`.
#    """
#
#    KILOMETERS = 'km'
#    METERS = 'm'
#    MILES = 'mi'
#    NAUTICAL_MILES = 'nmi'
#    FEET = 'ft'
#    INCHES = 'in'
#
#
## Unit values taken from http://www.unitconversion.org/unit_converter/length.html
#_CONVERSIONS = {Unit.KILOMETERS:       1.0,
#                Unit.METERS:           1000.0,
#                Unit.MILES:            0.621371192,
#                Unit.NAUTICAL_MILES:   0.539956803,
#                Unit.FEET:             3280.839895013,
#                Unit.INCHES:           39370.078740158}
#
#
#def haversine(point1, point2, unit=Unit.KILOMETERS):
#    """ Calculate the great-circle distance between two points on the Earth surface.
#    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
#    and, optionally, a unit of length.
#    :param point1: first point; tuple of (latitude, longitude) in decimal degrees
#    :param point2: second point; tuple of (latitude, longitude) in decimal degrees
#    :param unit: a member of haversine.Unit, or, equivalently, a string containing the
#                 initials of its corresponding unit of measurement (i.e. miles = mi)
#                 default 'km' (kilometers).
#    Example: ``haversine((45.7597, 4.8422), (48.8567, 2.3508), unit=Unit.METERS)``
#    Precondition: ``unit`` is a supported unit (supported units are listed in the `Unit` enum)
#    :return: the distance between the two points in the requested unit, as a float.
#    The default returned unit is kilometers. The default unit can be changed by
#    setting the unit parameter to a member of ``haversine.Unit``
#    (e.g. ``haversine.Unit.INCHES``), or, equivalently, to a string containing the
#    corresponding abbreviation (e.g. 'in'). All available units can be found in the ``Unit`` enum.
#    """
#
#    # get earth radius in required units
#    unit = Unit(unit)
#    avg_earth_radius = _AVG_EARTH_RADIUS_KM * _CONVERSIONS[unit]
#
#    # unpack latitude/longitude
#    lat1, lng1 = point1
#    lat2, lng2 = point2
#
#    # convert all latitudes/longitudes from decimal degrees to radians
#    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))
#
#    # calculate haversine
#    lat = lat2 - lat1
#    lng = lng2 - lng1
#    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
#
#    return 2 * avg_earth_radius * asin(sqrt(d))


# def print_particular_region_new(links_RV, var_names, s, outdic_precur, map_proj, ex):

#     #%%
#     n_parents = len(links_RV)

#     for i in range(n_parents):
#         tigr_lag = links_RV[i][1] #-1 There was a minus, but is it really correct?
#         index_in_fulldata = links_RV[i][0]
#         print("\n\nunique_label_format: \n\'lag\'_\'regionlabel\'_\'var\'")
#         if index_in_fulldata>0 and index_in_fulldata < len(var_names):
#             uniq_label = var_names[index_in_fulldata][1]
#             var_name = uniq_label.split('_')[-1]
#             according_varname = uniq_label
#             according_number = int(float(uniq_label.split('_')[1]))
# #            according_var_idx = ex['vars'][0].index(var_name)
#             corr_lag = int(uniq_label.split('_')[0])
#             print('index in fulldata {}: region: {} at lag {}'.format(
#                     index_in_fulldata, uniq_label, tigr_lag))
#             # *********************************************************
#             # print and save only significant regions
#             # *********************************************************
#             according_fullname = '{} at lag {} - ts_index_{}'.format(according_varname,
#                                   tigr_lag, index_in_fulldata)



#             actor = outdic_precur[var_name]
#             prec_labels = actor.prec_labels.sel(split=s)

#             for_plt = prec_labels.where(prec_labels.values==according_number).sel(lag=corr_lag)

#             map_proj = map_proj
#             plt.figure(figsize=(6, 4))
#             ax = plt.axes(projection=map_proj)
#             im = for_plt.plot.pcolormesh(ax=ax, cmap=plt.cm.BuPu,
#                              transform=ccrs.PlateCarree(), add_colorbar=False)
#             plt.colorbar(im, ax=ax , orientation='horizontal')
#             ax.coastlines(color='grey', alpha=0.3)
#             ax.set_title(according_fullname)
#             fig_file = 's{}_{}{}'.format(s, according_fullname, ex['file_type2'])

#             plt.savefig(os.path.join(ex['fig_subpath'], fig_file), dpi=100)
# #            plt.show()
#             plt.close()
#             # =============================================================================
#             # Print to text file
#             # =============================================================================
#             print('                                        ')
#             # *********************************************************
#             # save data
#             # *********************************************************
#             according_fullname = str(according_number) + according_varname
#             name = ''.join([str(index_in_fulldata),'_',uniq_label])

# #            print((fulldata[:,index_in_fulldata].size))
#             print(name)
#         else :
#             print('Index itself is also causal parent -> skipped')
#             print('*******************              ***************************')

# #%%
#     return

def store_ts(df_data, df_sum, dict_ds, filename): # outdic_precur, add_spatcov=True
    import functions_pp
    
    
    # splits = df_data.index.levels[0]
    # if add_spatcov:
    #     df_sp_s   = np.zeros( (splits.size) , dtype=object)
    #     for s in splits:
    #         df_split = df_data.loc[s]
    #         df_sp_s[s] = find_precursors.get_spatcovs(dict_ds, df_split, s, outdic_precur, normalize=True)

    #     df_sp = pd.concat(list(df_sp_s), keys= range(splits.size))
    #     df_data_to_store = pd.merge(df_data, df_sp, left_index=True, right_index=True)
    #     df_sum_to_store = find_precursors.add_sp_info(df_sum, df_sp)
    # else:
    df_data_to_store = df_data
    df_sum_to_store = df_sum

    dict_of_dfs = {'df_data':df_data_to_store, 'df_sum':df_sum_to_store}

    functions_pp.store_hdf_df(dict_of_dfs, filename)
    print('Data stored in \n{}'.format(filename))
    return

