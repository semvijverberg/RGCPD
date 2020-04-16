# -*- coding: utf-8 -*-
import os, io, sys
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPDC, CMIknn, CMIsymb
import numpy as np
import pandas as pd
import itertools
flatten = lambda l: list(itertools.chain.from_iterable(l))



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
        # # returns all causal links, not just causal parents/precursors (of lag>0)
        # sig = return_sign_links(pcmci, pq_matrix=results['q_matrix'],
        #                                     val_matrix=results['val_matrix'],
        #                                     alpha_level=alpha_level)
        
        sig = pcmci.return_significant_parents(results['q_matrix'], 
                                               val_matrix=results['val_matrix'],
                                               alpha_level=alpha_level)

        all_parents = sig['parents']
        link_matrix = sig['link_matrix']
    
        links_RV = all_parents[0]
        parents_dict[s] = links_RV, pcmci.var_names, link_matrix
        
    #%%
    return parents_dict

def get_df_links(parents_dict):
    splits = np.array(list(parents_dict.keys()))
    df_links_s = np.zeros( (splits.size) , dtype=object)
    for s in range(splits.size):
        var_names, link_matrix = parents_dict[s][1:] 
        df = pd.DataFrame(link_matrix[:,0], index=var_names)
        df_links_s[s] = df
        
    df_links = pd.concat(list(df_links_s), keys= range(splits.size))

    return df_links

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



def get_df_MCI(pcmci_dict, pcmci_results_dict, lags, variable):
    splits = np.array(list(pcmci_dict.keys()))
    df_MCIc_s = np.zeros( (splits.size) , dtype=object)
    df_MCIa_a = np.zeros( (splits.size) , dtype=object)
    for s in range(splits.size):
        pcmci_class = pcmci_dict[s]
        results_dict = pcmci_results_dict[s]
        var_names = pcmci_class.var_names
        idx = var_names.index(variable)
        try:
            pvals = results_dict['q_matrix'][:,idx]
            c = 'qval'
        except:
            c = 'pval'
            pvals = results_dict['p_matrix'][:,idx]
        coeffs = results_dict['val_matrix'][:,idx]
        # data = np.concatenate([coeffs, pvals],  1)
        
        cols = [f'coeff l{l}' for l in lags]
        # cols.append([f'pval l{l}' for l in lags])
        df_coeff = pd.DataFrame(coeffs, index=var_names, 
                          columns=cols)
        df_MCIc_s[s] = df_coeff
        cols = [f'{c} l{l}' for l in lags]
        df_alphas = pd.DataFrame(pvals, index=var_names, 
                          columns=cols)
        df_MCIa_a[s] = df_alphas
    df_MCIc = pd.concat(list(df_MCIc_s), keys= range(splits.size))
    df_MCIa = pd.concat(list(df_MCIa_a), keys= range(splits.size))
    return df_MCIc, df_MCIa
    
    


def extract_ParCorr_info_from_text(filepath_txt=str, variable=str, pc_alpha='auto'):
    #%%    
    assert variable is not None, 'variable not given' # check if var is not None
        
    if pc_alpha == 'auto' or pc_alpha is None:
        pc_alpha = print_pc_alphas_summ_from_txt(filepath_txt, variable)

    start_variable_line = f'## Variable {variable}\n'
    get_pc_alpha_lines = f'# pc_alpha = {pc_alpha}'
    convergence_line = 'converged'
    
    # get max_conds_dim parameter
    with open (filepath_txt, 'rt') as myfile:
        for i, myline in enumerate(myfile):           
            if 'max_conds_dim = ' in myline:
                max_conds_dim = int(myline.split(' = ')[1])
                break
        
    lines = [] ; 
    start_var = False ; start_pc_alpha = False    
    
    var_kickedout = 'Non-significance detected.'
    with open (filepath_txt, 'rt') as myfile:
        for i, myline in enumerate(myfile):      
            if start_variable_line == myline :
                lines.append(myline)
                start_var = True
            if start_var and get_pc_alpha_lines in myline:
                start_pc_alpha = True
            if start_pc_alpha:
                lines.append(myline)
            if start_var and start_pc_alpha and convergence_line in myline:
                break
            
    # collect init OLR results
    tested_links = [] ; pvalues = [] ; coeffs = [] 
    track = False
    start_init = 'Testing condition sets of dimension 0:'
    end_init   = 'Updating parents:'
    init_OLR = 'No conditions of dimension 0 left.' 
    for i, myline in enumerate(lines):
        
        if start_init in myline:
            track = True
        if track:
            if 'Link' in myline:
                # print(subline)
                link = myline
                var = link.split('Link (')[1].split(')')[0]
                tested_links.append(var)
            if 'pval' in myline:
                OLR = myline
                p = float(OLR.split('pval = ')[1].split(' / ')[0])
                pvalues.append(p)
                c = float(OLR.split(' val = ')[1].replace('\n',''))
                coeffs.append(c)
        if end_init in myline:
            break
    
    OLR_data = np.concatenate([np.array(coeffs)[:,None], np.array(pvalues)[:,None]], 
                              axis=1)
    df_OLR = pd.DataFrame(data=OLR_data, index=tested_links, 
                          columns=['coeff', 'pval'])
    
    # Find by which (set of) var(s) the link was found C.I.
    tested_links = [] ; pvalues = [] ; coeffs = [] ; by = {}
    for i, myline in enumerate(lines):
        
        if init_OLR in myline:   
            link = lines[i-2] 
            var = link.split('Link (')[1].split(')')[0]
            tested_links.append(var)
            OLR = lines[i-1]
            p = float(OLR.split('pval = ')[1].split(' / ')[0])
            pvalues.append(p)
            c = float(OLR.split(' val = ')[1].replace('\n',''))
            coeffs.append(c)
            
        if var_kickedout in myline:
            # Last line of xy == Non-significance detected.
            # second last line is the test where precursor it was found C.I.
            xy = lines[i-max_conds_dim-1 : i+1]            
            # Search for first name tested link in lines above 'Non-significance detected.'
            for subline in xy[::-1]:
                if 'Link' in subline:
                    link = subline
                    var = link.split('Link (')[1].split(')')[0]
                    tested_links.append(var)
                    break
            OLR = xy[-2]
            p = float(OLR.split('pval = ')[1].split(' / ')[0])
            pvalues.append(p)
            c = float(OLR.split(' val = ')[1].replace('\n',''))
            coeffs.append(c)
                
            if '(' in OLR:
                zvar = OLR.split(': ')[1].split('  -->')[0]
                by[var] = zvar
            else:
                by[var] = '-'
                
    for k in df_OLR.index:
        # print(k)
        if k not in by.keys():
            by[k] = 'C.D.'
    df_OLR['ParCorr'] = df_OLR.index.map(by)
    #%%
    return df_OLR

def print_pc_alphas_summ_from_txt(filepath_txt=str, variable=str):
    # get pc_alpha parameter from text
    #%%
    init_pc_alpha = 'pc_alpha = '
    start_pc_alpha_sum = '# Condition selection results:'
    var_searched = f'Algorithm converged for variable {variable}'
    end_pc_alpha_sum   = f'--> optimal pc_alpha for variable {variable} is '
    detected = False ; reached_pc_alpha_sum=False
    with open (filepath_txt, 'rt') as myfile:
        for i, myline in enumerate(myfile):   
            if init_pc_alpha in myline and i < 20:
                pc_alpha = myline.split('pc_alpha = ')[1].split('\n')[0]
                if pc_alpha == 'None':
                    continue
                else:
                    pc_alpha = float(pc_alpha)
                    break
           
            if var_searched in myline:
                detected = True
            if detected and start_pc_alpha_sum in myline:
                reached_pc_alpha_sum = True
            if reached_pc_alpha_sum:
                print(myline)
                pc_alpha = float(myline.split(end_pc_alpha_sum)[1].split('\n')[0])
                if end_pc_alpha_sum in myline:
                    break
    return pc_alpha


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

