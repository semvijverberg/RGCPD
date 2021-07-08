# -*- coding: utf-8 -*-
import os, io, sys
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats import multicomp
from tigramite.independence_tests import ParCorr #, GPDC, CMIknn, CMIsymb
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
flatten = lambda l: list(itertools.chain.from_iterable(l))



def init_pcmci(df_data, significance='analytic', mask_type='y',
               selected_variables=None, verbosity=5):
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
        df_data_s = df_data.loc[s][TrainIsTrue==True]
        df_data_s = df_data_s.dropna(axis=1, how='all')
        if any(df_data_s.isna().values.flatten()):
            if verbosity > 0:
                print('Warnning: nans detected')
#        print(np.unique(df_data_s.isna().values))
        var_names = [k for k in df_data_s.columns if k not in ['TrainIsTrue', 'RV_mask']]
        df_data_s = df_data_s.loc[:,var_names]
        data = df_data_s.values
        data_mask = ~RV_mask.loc[s][TrainIsTrue==True].values
        # indices with mask == False are used (with mask_type 'y')
        data_mask = np.repeat(data_mask, data.shape[1]).reshape(data.shape)

        # create dataframe in Tigramite format
        dataframe = pp.DataFrame(data=data, mask=data_mask, var_names=var_names)

        parcorr = ParCorr(significance=significance,
                          mask_type=mask_type,
                          verbosity=0)
        parcorr.verbosity = verbosity # to avoid print init text each time

        # ======================================================================================================================
        # pc algorithm: only parents for selected_variables are calculated
        # ======================================================================================================================
        pcmci   = PCMCI(dataframe=dataframe,
                        cond_ind_test=parcorr,
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
    df_lagged = pd.DataFrame(correlations['val_matrix'][:,0,:-1],
                             index=pcmci.var_names,
                             columns=range(tau_max))

    df_lagged.T.plot(figsize=(10,10))
    # pcmci.lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations[:,0],
    #                                    setup_args={'var_names':pcmci.var_names,
    #                                                'x_base':5, 'y_base':.5,
    #                                                'figsize':(10,10)})
    pcmci.verbosity = origverbosity
    return

def loop_train_test(pcmci_dict, path_txtoutput, tigr_function_call='run_pcmci',
                    kwrgs_tigr={}, n_cpu=1):
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
    splits = np.array(list(pcmci_dict.keys()))

    pcmci_results_dict = {}
    def run_tigr_on_splits(splits, pcmci_dict, path_txtoutput, tigr_function_call='run_pcmci',
                    kwrgs_tigr={}):

        for s in splits:
            progress = int(100 * (s+1) / splits.size)
            print(f"\rProgress causal inference - traintest set {progress}%", end="")
            results = run_tigramite(pcmci_dict[s], path_txtoutput, s,
                                    tigr_function_call,
                                    kwrgs_tigr=kwrgs_tigr)
            pcmci_results_dict[s] = results
        return pcmci_results_dict
    if n_cpu==1:
        pcmci_results_dict = run_tigr_on_splits(splits, pcmci_dict, path_txtoutput,
                                                tigr_function_call, kwrgs_tigr)
    elif n_cpu>1:
        futures = []
        for _s in np.array_split(splits, n_cpu):
            futures.append(delayed(run_tigr_on_splits)(_s, pcmci_dict,
                                                       path_txtoutput,
                                                       tigr_function_call,
                                                       kwrgs_tigr))
        futures = Parallel(n_jobs=n_cpu, backend='loky')(futures)
        [pcmci_results_dict.update(d) for d in futures]
    #%%
    return pcmci_results_dict

    #%%
def run_tigramite(pcmci, path_outsub2, s, tigr_function_call='run_pcmci',
                  kwrgs_tigr={}):

    #%%
    if path_outsub2 is not False:
        txt_fname = os.path.join(path_outsub2, f'split_{s}_PCMCI_out.txt')
        orig_stdout = sys.stdout
        # buffer print statement output to f
        sys.stdout = f = io.StringIO()
    #%%
    # ======================================================================================================================
    # tigramite 4
    # ======================================================================================================================
    pcmci.cond_ind_test.print_info()
    print(f'run {tigr_function_call}')
    tigr_func = getattr(pcmci, tigr_function_call)
    print(f'time {pcmci.T}, samples {pcmci.N}')

    kwrgs_tigr = kwrgs_tigr.copy() # ensure copy

    if 'selected_links' in kwrgs_tigr.keys() and 'remove_links' not in kwrgs_tigr.keys(): # selected split
        kwrgs_tigr['selected_links'] = kwrgs_tigr['selected_links'][s]
    elif 'remove_links' in kwrgs_tigr.keys() and 'selected_links' not in kwrgs_tigr.keys():
        remove_links = kwrgs_tigr.pop('remove_links')
        selected_links = {}
        for i in range(len(pcmci.var_names)):
            sel_links_var = []
            for j in range(len(pcmci.var_names)):
                for l in range(kwrgs_tigr['tau_min'],kwrgs_tigr['tau_max']+1):
                    if (i == j and l == 0)==False:
                        sel_links_var.append((j, -l))
            sel_links_var = [l for l in sel_links_var if l not in remove_links]
            selected_links[i] = sel_links_var
        kwrgs_tigr['selected_links'] = selected_links


    results = tigr_func(**kwrgs_tigr)

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

        sig = pcmci.return_significant_links(results['q_matrix'],
                                               val_matrix=results['val_matrix'],
                                               alpha_level=alpha_level,
                                               include_lagzero_links=True)

        all_parents = sig['link_dict']
        link_matrix = sig['link_matrix']

        parents_dict[s] = all_parents, pcmci.var_names, link_matrix

    #%%
    return parents_dict

def get_df_links(parents_dict, variable: str=None):
    splits = np.array(list(parents_dict.keys()))
    df_links_s = np.zeros( (splits.size) , dtype=object)
    for s in range(splits.size):
        var_names, link_matrix = parents_dict[s][1:]
        if variable is None:
            var_idx = 0
        else:
            var_idx = var_names.index(variable)
        df = pd.DataFrame(link_matrix[:,var_idx], index=var_names)
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

def df_pval_to_keys_dict(df_pvals, alpha=.05):
    n_spl = df_pvals.index.levels[0]
    keys_dict = {s:[] for s in n_spl}
    for s in n_spl:
        keys = list(df_pvals.loc[s][df_pvals.loc[s].values<alpha].index)
        keys_dict[s].append(keys)
    return keys_dict

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

        cols = [f'coeff l{l}' for l in range(0,max(lags)+1)]
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

    OLS_data = np.concatenate([np.array(coeffs)[:,None], np.array(pvalues)[:,None]],
                              axis=1)
    df_OLS = pd.DataFrame(data=OLS_data, index=tested_links,
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

    for k in df_OLS.index:
        # print(k)
        if k not in by.keys():
            by[k] = 'C.D.'
    df_OLS['ParCorr'] = df_OLS.index.map(by)
    #%%
    return df_OLS

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
                if end_pc_alpha_sum in myline:
                    pc_alpha = float(myline.split(end_pc_alpha_sum)[1].split('\n')[0])
                    break
    #%%
    return pc_alpha

def store_ts(df_data, df_sum, dict_ds, filename): # outdic_precur, add_spatcov=True
    import functions_pp

    df_data_to_store = df_data
    df_sum_to_store = df_sum

    dict_of_dfs = {'df_data':df_data_to_store, 'df_sum':df_sum_to_store}

    functions_pp.store_hdf_df(dict_of_dfs, filename)
    print('Data stored in \n{}'.format(filename))
    return

def get_traintest_links(pcmci_dict:dict, parents_dict:dict,
                        pcmci_results_dict:dict,
                        variable: str=None, s: int=None,
                        min_link_robustness: int=1, alpha=0.05, FDR='fdr_bh'):
    '''
    Retrieves the links / MCI coefficients of a single variable or all.
    Does this for a single traintest split, or by calculating the mean over
    all traintest splits. If so, weights are calculated based on the robustness
    to enable modification (number of times link was found) of the link width
    of a network graph.

    Parameters
    ----------
    pcmci_dict : dict
        Dictionairy with keys as traintest split index, and items are the pcmci
        classes.
    parents_dict : dict
        Dictionairy with keys as traintest split index, and items is tuple
        containing the links_parent, var_names and the link_matrix belonging
        to the alpha_value that was used in the analysis.
    pcmci_results_dict : dict, optional
        Dictionairy with keys as traintest split index, and items is another
        dictionairy with the results dict with:
        dict_keys(['val_matrix', 'p_matrix', 'q_matrix', 'conf_matrix'])
    variable : str, optional
        return links of a single or of all variables.
    s : int, optional
        return links of a single traintest split or take the mean over them.
    min_link_robustness : int, optional
        Only when s=None: If a link is not present at least min_link_robustness
        times, it is masked.

    Returns
    -------
    links_plot : ndarray
        link_matrix, shape (len(var_names), len(var_names), len(lags))
    val_plot : ndarray
        val_matrix, shape (len(var_names), len(var_names), len(lags))
    weights : ndarray
        in shape of links_plot, number of times link was found.
    var_names : list

    '''

    splits = np.array(list(pcmci_dict.keys()))
    if s is None:
        todef_order_index = [len(pcmci_dict[s].var_names) for s in splits]
        links_s     = np.zeros( splits.size , dtype=object)
        MCIvals_s   = np.zeros( splits.size , dtype=object)
        qvals_s     = np.zeros( splits.size , dtype=object)
        for s in splits:
            links_plot = np.zeros_like(parents_dict[s][2])
            link_matrix_s = parents_dict[s][2]
            val_plot = np.zeros_like(pcmci_results_dict[s]['val_matrix'], dtype=float)
            qval_plot = np.ones_like(pcmci_results_dict[s]['q_matrix'], dtype=float)
            val_matrix_s = pcmci_results_dict[s]['val_matrix']
            qval_matrix_s = pcmci_results_dict[s]['q_matrix']
            var_names = pcmci_dict[s].var_names
            if variable is not None:
                idx = var_names.index(variable)
                links_plot[:,idx] = link_matrix_s[:,idx]
                val_plot[:,:] = val_matrix_s[:,:] # keep val_matrix complete
                qval_plot[:,:] = qval_matrix_s[:,idx] # only idx with vals < 1
            else:
                links_plot[:,:] = link_matrix_s[:,:]
                val_plot[:,:] = val_matrix_s[:,:]
                qval_plot[:,:] = qval_matrix_s[:,:] # keep val_matrix complete
            index = [p for p in itertools.product(var_names, var_names)]
            if len(var_names) == max(todef_order_index):
                fullindex = index
                fullvar_names = var_names
            nplinks = links_plot.reshape(len(var_names)**2, -1)
            links_s[s] = pd.DataFrame(nplinks, index=index)
            npMCIvals = val_plot.reshape(len(var_names)**2, -1)
            MCIvals_s[s] = pd.DataFrame(npMCIvals, index=index)
            npqvals = qval_plot.reshape(len(var_names)**2, -1)
            qvals_s[s] = pd.DataFrame(npqvals, index=index)

        df_links = pd.concat(links_s, keys=splits)
        # get links present at least min_link_robustness times
        # get all qvals
        if FDR is not None or FDR is not False:
            df_qvals_all = pd.concat(qvals_s, keys=splits)
            df_qvals = df_qvals_all ; npq = df_qvals.values
            npq = npq.reshape(splits.size, len(index), -1)
            for i, link in enumerate(index):
                for l in df_qvals.columns: # lags
                    _qvals = df_qvals_all.loc[pd.MultiIndex.from_product([splits, index[i:i+1]]),l]
                    adjusted_pvalues = multicomp.multipletests(_qvals, method='fdr_bh')
                    ad_p = adjusted_pvalues[1]
                    npq[:,i,l] = ad_p
            df_qvals_all = pd.DataFrame(npq.reshape(df_qvals_all.shape),
                                        index=df_qvals_all.index,
                                        columns=df_qvals_all.columns)
        else:
            pass

        df_links = df_qvals_all < alpha
        df_robustness = df_links.sum(axis=0, level=1)
        df_robustness = df_robustness.reindex(index=fullindex)
        weights = df_robustness.values
        weights = weights.reshape(len(var_names), len(var_names), -1)
        df_links = df_robustness >= min_link_robustness

        # ensure that missing links due to potential precursor step are not
        # appended to pandas df (auto behavior)
        df_links = df_links.reindex(index=fullindex)
        mergeindex = pd.MultiIndex.from_tuples(df_links.index)
        df_link_matrix = df_links.reindex(index=mergeindex)
        # calculate mean MCI over train test splits
        df_MCIvals = pd.concat(MCIvals_s, keys=splits).mean(axis=0, level=1)
        # ensure that missing links due to potential precursor step are not
        # appended to pandas df (auto behavior)
        df_MCIvals = df_MCIvals.reindex(index=fullindex)
        df_MCIval_matrix = df_MCIvals.reindex(index=mergeindex)
        # now we can safely reshape
        links_plot = df_link_matrix.values.reshape(len(var_names), len(var_names), -1)
        val_plot = df_MCIval_matrix.values.reshape(len(var_names), len(var_names), -1)
        # Commented error in val_plots, 02-11-2020
        # val_plot = pcmci_results_dict[s]['val_matrix'] # was giving vals of split s
        var_names = fullvar_names
    elif type(s) is int:
        var_names = pcmci_dict[s].var_names
        weights = None
        link_matrix_s = parents_dict[s][2]
        links_plot = np.zeros_like(parents_dict[s][2])
        val_matrix_s = pcmci_results_dict[s]['val_matrix']
        val_plot = np.zeros_like(links_plot, dtype=float)
        if variable is not None:
            idx = pcmci_dict[s].var_names.index(variable)
            links_plot[:,idx] = link_matrix_s[:,idx]
            val_plot[:,idx] = val_matrix_s[:,idx]

        else:
            links_plot[:,:] = link_matrix_s[:,:]
            val_plot = val_matrix_s
    return links_plot, val_plot, weights, var_names

def df_data_remove_z(df_data, z_keys=[str, list], lag_z : [int, list]=[0],
                     keys=None, standardize: bool=True, plot: bool=True):

    '''


    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame containing timeseries.
    z_keys : str, optional
        variable z, of which influence will be remove of columns in keys.
        The default is str.

    Returns
    -------
    None.

    '''
    method = ParCorr()

    if type(z_keys) is str:
        z_keys = [z_keys]
    if keys is None:
        discard = ['TrainIsTrue', 'RV_mask'] + z_keys
        keys = [k for k in df_data.columns if k not in discard]
    if hasattr(df_data.index, 'levels') == False: # create fake multi-index for standard data format
        df_data.index = pd.MultiIndex.from_product([[0], df_data.index])

    if type(lag_z) is int:
        lag_z = [lag_z]

    max_lag = max(lag_z);
    dates = df_data.index.levels[1]
    df_z = df_data[z_keys]
    zlist = []
    if 0 in lag_z:
        zlist = [df_z.loc[pd.IndexSlice[:, dates[max_lag:]],:]]
    [zlist.append(df_z.loc[pd.IndexSlice[:, dates[max_lag-l:-l]],:]) for l in lag_z if l != 0]
    # update df_data to account for lags (first dates have no lag):
    df_data = df_data.loc[pd.IndexSlice[:, dates[max_lag:]],:]
    # align index dates for easy merging
    for d_ in zlist:
        d_.index = df_data.index
    df_z = pd.concat(zlist, axis=1).loc[pd.IndexSlice[:, dates[max_lag:]],:]
    # update columns with lag indication
    df_z.columns = [f'{c[0]}_lag{c[1]}' for c in np.array(np.meshgrid(z_keys, lag_z)).T.reshape(-1,2)]

    npstore = np.zeros(shape=(len(keys),df_data.index.levels[0].size, dates[max_lag:].size))
    for i, orig in enumerate(keys):
        orig = keys[i]

        # create fake X, Y format, needed for function _get_single_residuals
        dfxy = df_data[[orig]].merge(df_data[[orig]].copy().rename({orig:'copy'},
                                                                  axis=1),
                                     left_index=True, right_index=True).copy()
        # Append Z timeseries
        dfxyz = dfxy.merge(df_z, left_index=True, right_index=True)

        for s in df_data.index.levels[0]:
            dfxyz_s = dfxyz.loc[s].copy()
            if all(dfxyz_s[orig].isna().values):
                npstore[i,s,:] = dfxyz_s[orig].values # fill in all nans
            else:
                npstore[i,s,:] = method._get_single_residuals(np.moveaxis(dfxyz_s.values, 0,1), 0,
                                                              standardize=standardize,
                                                              return_means=True)[0]

    df_new = pd.DataFrame(np.moveaxis(npstore, 0, 2).reshape(-1,len(keys)),
                          index=df_data.index, columns=keys)
    if plot:
        fig, axes = plt.subplots(len(keys),1, figsize=(10,2.5*len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]
        for i, k in enumerate(keys):
            df_data[k].loc[0].plot(ax=axes[i], label=f'{k} original',
                                      legend=False, color='green', lw=1, alpha=.8)
            cols = ', '.join(c.replace('_', ' ') for c in df_z.columns)
            df_new[k].loc[0].plot(ax=axes[i], label=cols+' regressed out',
                                     legend=False, color='blue', lw=1)
            axes[i].legend()
        out = (df_new, fig)
    else:
        out = (df_new)
    return out

def df_data_Parcorr(df_data, z_keys=[str, list], keys: list=None, target: str=None,
                    x_lag: int=0, z_lag: int=0):
    '''
    Wrapper function to test bivariate partial correlation between x and y
    conditioning on z (parcorr(x,y,z)). Only univariate x or z supported.

    Parameters
    ----------
    df_data : TYPE
        MultiIndex Dataframe (with index levels (splits, time) and columns
        indicated precursors. DataFrame should contain TrainIsTrue and RV_mask
        columns.
    z_keys : list or str
        key(s) you want to iteratively condition on. The default is [str, list].
    keys : list, optional
        keys of which you want to iteratively test link with target.
        The default will test all x_keysin df.
    target : str, optional
        target key (should be in df_data). The default is df_data's first column.
    x_lag : int, optional
        lag shift to apply to x keys. The default is 0.
    z_lag : int, optional
        lag shift to apply to z keys. The default is 0.

    Returns
    -------
    vals : pd.DataFrame
        corr. vals pd.DataFrame with index as keys and columns as splits.
    pvals : pd.DataFrame
        corr. vals pd.DataFrame with index as keys and columns as splits.

    '''

    if type(z_keys) is str:
        z_keys = [z_keys]
    if keys is None:
        discard = ['TrainIsTrue', 'RV_mask'] + z_keys
        keys = [k for k in df_data.columns if k not in discard]
    if target is None:
        target = df_data.columns[0]
    if keys == z_keys:
        n_zkeys = len(z_keys) - 1 # cannot regress out itself
    else:
        n_zkeys = len(z_keys)
    splits = df_data.index.levels[0]
    valnp = np.zeros(shape=(len(keys), n_zkeys, splits.size))
    pvalnp = np.zeros(shape=(len(keys), n_zkeys, splits.size))
    index = []
    for ix, x_key in enumerate(keys):
        subz_keys = [k for k in z_keys if k != x_key]
        for iz, z in enumerate(subz_keys):
            index.append((x_key, z))

            # create X, Y, Z format
            dfxyz = df_data[[target, x_key, z]]
            dfxyz = df_data[[x_key, target, z] + ['TrainIsTrue', 'RV_mask']]
            pcmci_dict = init_pcmci(dfxyz, significance='analytic', mask_type='y',
                                    selected_variables=None, verbosity=0)

            for s in df_data.index.levels[0]:
                pcmci = pcmci_dict[s]

                if np.isnan(pcmci.dataframe.values).any(): #NaNs in array
                    # break
                    val, pval = np.nan, np.nan
                elif x_key not in pcmci.var_names or z not in pcmci.var_names:
                    val, pval = np.nan, np.nan
                else:
                    X, Y, Z = [(0, -x_lag)],[(1, 0)], [(2, -z_lag)]

                    runtest = pcmci.cond_ind_test.run_test
                    val, pval = runtest(X, Y, Z,
                                        tau_max=max(x_lag, z_lag),
                                        cut_off='max_lag')

                valnp[ix,iz,s] = val
                pvalnp[ix,iz,s] = pval

    MultiIndex = pd.MultiIndex.from_tuples(index, names=['x', 'z'])
    vals = pd.DataFrame(valnp.reshape(-1,splits.size), columns=splits,
                        index=MultiIndex)
    pvals = pd.DataFrame(pvalnp.reshape(-1,splits.size), columns=splits,
                        index=MultiIndex)
    return vals, pvals


