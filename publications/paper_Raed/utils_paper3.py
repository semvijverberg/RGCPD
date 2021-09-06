#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:14:02 2021

@author: semvijverberg
"""



import os, sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd

# import sklearn.linear_model as scikitlinear
from matplotlib import gridspec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

import func_models as fc_utils
import functions_pp, find_precursors

nice_colors = ['#EE6666', '#3388BB', '#88BB44', '#9988DD', '#EECC55',
                '#FFBBBB']
line_styles = ['-', '--', '-.', ':', '']

cl_combs = np.array(np.meshgrid(line_styles, nice_colors),
                     dtype=object).T.reshape(-1,2)



def get_df_mean_SST(rg, mean_vars=['sst'], alpha_CI=.05,
                    n_strongest='all',
                    weights=True, labels=None,
                    fcmodel=None, kwrgs_model=None,
                    target_ts=None):


    periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)
    df_pvals = rg.df_pvals.copy()
    df_corr  = rg.df_corr.copy()
    unique_keys = np.unique(['..'.join(k.split('..')[1:]) for k in rg.df_data.columns[1:-2]])
    if labels is not None:
        unique_keys = [k for k in unique_keys if k in labels]

    # dict with strongest mean parcorr over growing season
    mean_SST_list = []
    # keys_dict = {s:[] for s in range(rg.n_spl)} ;
    keys_dict_meansst = {s:[] for s in range(rg.n_spl)} ;
    for s in range(rg.n_spl):
        mean_SST_list_s = []
        sign_s = df_pvals[s][df_pvals[s] <= alpha_CI].dropna(axis=0, how='all')
        for uniqk in unique_keys:
            # uniqk = '1..smi'
            # region label (R) for each month in split (s)
            keys_mon = [mon+ '..'+uniqk for mon in periodnames]
            # significant region label (R) for each month in split (s)
            keys_mon_sig = [k for k in keys_mon if k in sign_s.index] # check if sig.
            if uniqk.split('..')[-1] in mean_vars and len(keys_mon_sig)!=0:
                # mean over region if they have same correlation sign across months
                for sign in [1,-1]:
                    mask = np.sign(df_corr.loc[keys_mon_sig][[s]]) == sign
                    k_sign = np.array(keys_mon_sig)[mask.values.flatten()]
                    if len(k_sign)==0:
                        continue
                    # calculate mean over n strongest SST timeseries
                    if len(k_sign) > 1:
                        meanparcorr = df_corr.loc[k_sign][[s]].squeeze().sort_values()
                        if n_strongest == 'all':
                            keys_str = meanparcorr.index
                        else:
                            keys_str = meanparcorr.index[-n_strongest:]
                    else:
                        keys_str  = k_sign
                    if weights:
                        fit_masks = rg.df_data.loc[s].iloc[:,-2:]
                        df_d = rg.df_data.loc[s][keys_str].copy()
                        df_d = df_d.apply(fc_utils.standardize_on_train_and_RV,
                                          args=[fit_masks, 0])
                        df_d = df_d.merge(fit_masks, left_index=True,right_index=True)
                        # df_train = df_d[fit_masks['TrainIsTrue']]
                        df_mean, model = fcmodel.fit_wrapper({'ts':target_ts.loc[s]},
                                                          df_d, keys_str,
                                                          kwrgs_model)


                    else:
                        df_mean = rg.df_data.loc[s][keys_str].copy().mean(1)
                    month_strings = [k.split('..')[0] for k in sorted(keys_str)]
                    df_mean = df_mean.rename({0:''.join(month_strings) + '..'+uniqk},
                                             axis=1)
                    keys_dict_meansst[s].append( df_mean.columns[0] )
                    mean_SST_list_s.append(df_mean)
            elif uniqk.split('..')[-1] not in mean_vars and len(keys_mon_sig)!=0:
                # use all timeseries (for each month)
                mean_SST_list_s.append(rg.df_data.loc[s][keys_mon_sig].copy())
                keys_dict_meansst[s] = keys_dict_meansst[s] + keys_mon_sig
            # elif len(keys_mon_sig) == 0:
            #     data = np.zeros(rg.df_RV_ts.size) ; data[:] = np.nan
            #     pd.DataFrame(data, index=rg.df_RV_ts.index)
        # if len(keys_mon_sig) != 0:
        df_s = pd.concat(mean_SST_list_s, axis=1)
        mean_SST_list.append(df_s)
    df_mean_SST = pd.concat(mean_SST_list, keys=range(rg.n_spl))
    df_mean_SST = df_mean_SST.merge(rg.df_splits.copy(),
                                    left_index=True, right_index=True)
    return df_mean_SST, keys_dict_meansst



#%% Functions for plotting continuous forecast
def df_scores_for_plot(rg_list, name_object):
    df_scores = [] ; df_boot = [] ; df_tests = []
    for i, rg in enumerate(rg_list):
        verification_tuple = rg.__dict__[name_object]
        df_scores.append(verification_tuple[2])
        df_boot.append(verification_tuple[3])
        df_tests.append(verification_tuple[1])
    df_scores = pd.concat(df_scores, axis=1)
    df_boot = pd.concat(df_boot, axis=1)
    df_tests = pd.concat(df_tests, axis=1)
    return df_scores, df_boot, df_tests

def df_predictions_for_plot(rg_list, name_object='prediction_tuple'):
    df_preds = [] ; df_weights = []
    for i, rg in enumerate(rg_list):
        rg.df_fulltso.index.name = None
        if i == 0:
            prediction = rg.__dict__[name_object][0]
            prediction = rg.merge_df_on_df_data(rg.df_fulltso, prediction)
        else:
            prediction = rg.__dict__[name_object][0].iloc[:,[1]]
        df_preds.append(prediction)
        if i == 0:
            df_preds.append
        if i+1 == len(rg_list):
            df_preds.append(rg.df_splits)
        weights = rg.__dict__[name_object][1]
        df_weights.append(weights.rename({0:rg.fc_month}, axis=1))

    # add Target if Target*Signal was fitter
    if df_preds[0].columns[0] != 'Target':
        target_ts = rg.transform_df_data(\
                     rg.df_data.iloc[:,[0]].merge(rg.df_splits,
                                                  left_index=True,
                                                  right_index=True),
                     transformer=fc_utils.standardize_on_train)
        target_ts = target_ts.rename({target_ts.columns[0]:'Target'},axis=1)
        df_preds.insert(1, target_ts)
    df_preds  = pd.concat(df_preds, axis=1)
    df_weights = pd.concat(df_weights, axis=1)

    return df_preds, df_weights

def plot_scores_wrapper(df_scores_list, df_boot_list, labels=None,
                        metrics_plot=None, fig_ax=None):

    alpha = .1
    if metrics_plot is None:
        if 'BSS' in df_scores_list[0].columns.levels[1]:
            metrics_plot = ['BSS', 'roc_auc_score']
        else:
            metrics_plot = ['corrcoef', 'MAE', 'RMSE', 'r2_score']
    rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                'MAE':'MAE-SS', 'CRPSS':'CRPSS', 'r2_score':'$R^2$',
                'mean_absolute_percentage_error':'MAPE',
                'BSS': 'BSS', 'roc_auc_score':'ROC-AUC'}

    if fig_ax is None:
        f, ax = plt.subplots(1,len(metrics_plot), figsize=(6.5*len(metrics_plot), 5),
                             sharey=False) ;
    else:
        f, ax = fig_ax

    if type(df_scores_list) is not list:
        df_scores_list = [df_scores_list]
    if type(df_boot_list) is not list:
        df_boot_list = [df_boot_list]
    if labels is None:
        labels = ['Verification on all years']

    # for j, df_sc in enumerate(df_scores_list):
    for j, (df_sc,df_b) in enumerate(zip(df_scores_list, df_boot_list)):
        for i, m in enumerate(metrics_plot):
            # normal SST
            # columns.levels auto-sorts order of labels, to avoid:
            steps = df_sc.columns.levels[1].size
            months = [t[0] for t in df_sc.columns][::steps]
            ax[i].plot(months, df_sc.reorder_levels((1,0), axis=1).iloc[0][m].T,
                       label=labels[j],
                       color=cl_combs[j][1],
                       linestyle=cl_combs[j][0])
            ax[i].fill_between(months,
                                df_b.reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                                df_b.reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                                edgecolor=cl_combs[j][1], facecolor=cl_combs[j][1], alpha=0.3,
                                linestyle=cl_combs[j][0], linewidth=2)

            if m == 'corrcoef':
                ax[i].set_ylim(-.2,1)
            elif m == 'roc_auc_score':
                ax[i].set_ylim(0,1)
            else:
                ax[i].set_ylim(-.1,.6)
            ax[i].axhline(y=0, color='black', linewidth=1)
            ax[i].tick_params(labelsize=14, pad=6)


            if i == 0:
                ax[i].legend(loc='lower right', fontsize=14)
            # ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=0)
            ax[i].set_title(rename_m[m])
    f.subplots_adjust(hspace=.1)
    f.subplots_adjust(wspace=.25)

    return f

def plot_forecast_ts(df_test_m, df_test, target_ts=None, fig_ax=None, fs=12,
                     metrics_plot=None, name_model=None):
    #%%
    # fig_ax=None
    fontsize = fs
    if fig_ax is None:
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
    else:
        fig, axs = fig_ax
        if type(axs) is np.ndarray:
            ax0 = axs[0]
            ax1 = axs[1]

    if target_ts is not None:
        ax0.plot_date(df_test.index, target_ts.loc[df_test.index],
                      ls='--', c='grey', label='Observed', marker=None)
        ax0.plot_date(df_test.index, df_test.iloc[:,0], ls='-',
                  label='Potential Predictable Signal', c='black')
    else:
        ax0.plot_date(df_test.index, df_test.iloc[:,0], ls='-',
                  label='Observed', c='black')

    if name_model is None:
        name_model = 'Prediction'

    ax0.plot_date(df_test.index, df_test.iloc[:,1], ls='-', c='red',
                  label=name_model)

    # ax0.set_xticks()
    # ax0.set_xticklabels(df_test.index.year,
    # ax0.set_ylabel('Standardized Soy Yield', fontsize=fontsize)
    ax0.tick_params(labelsize=fontsize)
    ax0.axhline(y=0, color='black', lw=1)

    if type(df_test_m) is not list:
        df_test_m = [df_test_m]
    order_verif = ['vs. PPS', 'vs. Target']
    df_scores = []
    for i, df_test_skill in enumerate(df_test_m):
        r = {df_test_skill.columns[0][0]:order_verif[i]}
        df_scores.append( df_test_skill.rename(r, axis=1) )
    df_scores = pd.concat(df_scores, axis=1)
    df_scores = df_scores.T.reset_index().pivot_table(index='level_1',
                                                      columns='level_0')[0]

        # y_offset = 0
        # if i == 1:
        #     y_offset = 0.2
        # df_scores = df_test_skill.loc[0][df_test_skill.columns[0][0]]
        # Texts1 = [] ; Texts2 = [] ;
        # textprops = dict(color='black', fontsize=fontsize, family='serif')
    rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr.', 'MAE':'MAE-SS',
                  'BSS':'BSS', 'roc_auc_score':'ROC-AUC', 'r2_score':'$r^2$',
                  'mean_absolute_percentage_error':'MAPE'}
    if metrics_plot is None:
        metrics_plot = df_scores.index

    table = ax1.table(cellText=np.round(df_scores.loc[metrics_plot].values, 2),
                      cellLoc='center',
                      rowLabels=[rename_met[m] for m in metrics_plot],
                      colLabels=df_scores.columns, loc='center',
                      edges='closed')
    table.scale(1.1, 1.5)
    table.set_fontsize(fontsize+4)
    # ax0.secondary_xaxis('right', functions=(deg2rad, rad2deg))
    # plt.table(['{:.2f}'.format(t) for t in df_scores[metrics_plot].values])
        # for k in metrics_plot:
        #     label = rename_met[k]
        #     val = round(df_scores[k], 2)
        #     Texts1.append(TextArea(f'{label}',textprops=textprops))
        #     Texts2.append(TextArea(f'{val}',textprops=textprops))
        # texts_vbox1 = VPacker(children=Texts1,pad=0,sep=4)
        # texts_vbox2 = VPacker(children=Texts2,pad=0,sep=4)

        # ann1 = AnnotationBbox(texts_vbox1,(1.02+y_offset,0.5),xycoords=ax0.transAxes,
        #                             box_alignment=(0,.5),
        #                             bboxprops = dict(facecolor='grey', alpha=.5,
        #                                              boxstyle='round',edgecolor='white'))
        # ann2 = AnnotationBbox(texts_vbox2,(1.15+y_offset,0.5),xycoords=ax0.transAxes,
        #                             box_alignment=(0,.5),
        #                             bboxprops = dict(facecolor='grey', alpha=.5,
        #                                              boxstyle='round',edgecolor='white'))
        # ann1.set_figure(fig) ; ann2.set_figure(fig)
        # fig.artists.append(ann1) ; fig.artists.append(ann2)
    ax0.set_title(df_test.columns[1] + ' forecast', fontsize=fs, y=.95)
    ax1.axis('off')
    ax1.axis('tight')
    ax1.set_facecolor('white')
    #%%
    return


#%% Conditional continuous forecast

def get_df_forcing_cond_fc(rg_list, #target_ts, fcmodel, kwrgs_model, mean_vars=['sst', 'smi'],
                           region='only_Pacific',
                           name_object='df_CL_data'):
    for j, rg in enumerate(rg_list):


        # find west-sub-tropical Atlantic region
        df_labels = find_precursors.labels_to_df(rg.list_for_MI[0].prec_labels)
        if region == 'only_Pacific':
            PacAtl = [int(df_labels['n_gridcells'].idxmax())] # only Pacific
        elif region == 'Pacific+SM':
            PacAtl = [int(df_labels['n_gridcells'].idxmax()), 0]
        else:
            PacAtl = []
            dlat = df_labels['latitude'] - 29
            dlon = df_labels['longitude'] - 290
            zz = pd.concat([dlat.abs(),dlon.abs()], axis=1)
            Atlan = zz.query('latitude < 10 & longitude < 10')
            if Atlan.size > 0:
                PacAtl.append(int(Atlan.index[0]))
            PacAtl.append(int(df_labels['n_gridcells'].idxmax())) # Pacific SST



        if hasattr(rg, name_object):
            df_mean = rg.__dict__[name_object]
        else:
            pass
            # get keys with different lags to construct CL models
            # keys = [k for k in rg.df_data.columns[1:-2] if int(k.split('..')[1]) in PacAtl]
            # # keys = [k for k in keys if 'sst' in k] # only SST

            # labels = ['..'.join(k.split('..')[1:]) for k in keys]
            # if '0..smi_sp' not in labels: # add smi just because it almost
            #     # always in there, otherwise error
            #     labels += '0..smi_sp'
            # df_mean, keys_dict = get_df_mean_SST(rg, mean_vars=mean_vars,
            #                                      n_strongest='all',
            #                                      weights=True,
            #                                      fcmodel=fcmodel,
            #                                      kwrgs_model=kwrgs_model,
            #                                      target_ts=target_ts,
            #                                      labels=labels)

        if region != 'only_Pacific' and region != 'Pacific+SM':
            # weights_norm = rg.prediction_tuple[1]
            # # weights_norm = weights_norm.sort_values(ascending=False, by=0)
            # # apply weighted mean based on coefficients of precursor regions
            # weights_norm = weights_norm.loc[pd.IndexSlice[:,keys],:]
            # # weights_norm = weights_norm.div(weights_norm.max(axis=0))
            # weights_norm = weights_norm.div(weights_norm.max(axis=0, level=0), level=0)
            # weights_norm = weights_norm.reset_index().pivot(index='level_0', columns='level_1')[0]
            # weights_norm.index.name = 'fold' ; df_mean.index.name = ('fold', 'time')
            # PacAtl_ts = weights_norm.multiply(df_mean[keys], axis=1, level=0)
            pass
        else:
            keys = [k for k in df_mean.columns[1:-2] if int(k.split('..')[1]) in PacAtl]
            PacAtl_ts = df_mean[keys]

        rg.df_forcing = PacAtl_ts

def cond_forecast_table(rg_list, score_func_list, df_predictions,
                        nameTarget='Target', n_boot=0):

    quantiles = [.15, .25]
    metrics = [s.__name__ for s in score_func_list]
    if n_boot > 0:
        cond_df = np.zeros((len(metrics), len(rg_list), len(quantiles)*2, n_boot))
    else:
        cond_df = np.zeros((len(metrics), len(rg_list), len(quantiles)*2))
    for i, met in enumerate(metrics):
        for j, rg in enumerate(rg_list):

            df_forctest = functions_pp.get_df_test(rg.df_forcing.mean(axis=1),
                                                   df_splits=rg.df_splits)
            df_forctrain = functions_pp.get_df_train(rg.df_forcing.mean(axis=1),
                                             df_splits=rg.df_splits, s='mean')

            prediction = df_predictions[[nameTarget, rg.fc_month]]
            df_test = functions_pp.get_df_test(prediction,
                                               df_splits=rg.df_splits)

            # df_test_m = rg.verification_tuple[2]
            # cond_df[i, j, 0] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            for k, l in enumerate(range(0,4,2)):
                q = quantiles[k]
                low = df_forctest < float(df_forctrain.quantile(q))
                high = df_forctest > float(df_forctrain.quantile(1-q))
                mask_anomalous = np.logical_or(low, high)
                # anomalous Boundary forcing
                condfc = df_test[mask_anomalous.values]
                # condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
                cond_verif_tuple = fc_utils.get_scores(condfc,
                                                       score_func_list=score_func_list,
                                                       n_boot=n_boot,
                                                       score_per_test=False,
                                                       blocksize=1,
                                                       rng_seed=1)
                df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
                rg.cond_verif_tuple  = cond_verif_tuple
                if n_boot == 0:
                    cond_df[i, j, l] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
                else:
                    cond_df[i, j, l, :] = df_boot[df_boot.columns[0][0]][met]
                # mild boundary forcing
                higher_low = df_forctest > float(df_forctrain.quantile(.5-q))
                lower_high = df_forctest < float(df_forctrain.quantile(.5+q))
                mask_anomalous = np.logical_and(higher_low, lower_high) # changed 11-5-21

                condfc = df_test[mask_anomalous.values]
                # condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
                cond_verif_tuple = fc_utils.get_scores(condfc,
                                                       score_func_list=score_func_list,
                                                       n_boot=n_boot,
                                                       score_per_test=False,
                                                       blocksize=1,
                                                       rng_seed=1)
                df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
                if n_boot == 0:
                    cond_df[i, j, l+1] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
                else:
                    cond_df[i, j, l+1, :] = df_boot[df_boot.columns[0][0]][met]

    columns = [[f'strong {int(q*200)}%', f'weak {int(q*200)}%'] for q in quantiles]
    columns = functions_pp.flatten(columns)
    if n_boot > 0:
        columns = pd.MultiIndex.from_product([columns, list(range(n_boot))])

    df_cond_fc = pd.DataFrame(cond_df.reshape((len(metrics)*len(rg_list), -1)),
                              index=pd.MultiIndex.from_product([list(metrics), [rg.fc_month for rg in rg_list]]),
                              columns=columns)


    return df_cond_fc

def to_df_scores_format(d_dfscores, condition='50% strong'):
    df_scores = d_dfscores['df_cond'].T.loc[[condition]].swaplevel(0,1,1)
    df_boot = d_dfscores['df_cond_b'][condition].T.swaplevel(0,1,1)
    steps = d_dfscores['df_cond'].index.levels[1].size
    months = [t[1] for t in d_dfscores['df_cond'].index][:steps]
    steps = d_dfscores['df_cond'].index.levels[1].size
    metrics_ = [t[0] for t in d_dfscores['df_cond'].index][::steps]
    df_scores = pd.DataFrame(df_scores, columns=pd.MultiIndex.from_product([months, metrics_]))
    df_boot = pd.DataFrame(df_boot, columns=pd.MultiIndex.from_product([months, metrics_]))
    df_scores.index = [0]
    return df_scores, df_boot

def load_scores(list_labels, model_name_CL, model_name, n_boot,
                filepath_df_datas, condition='strong 50%'):

    df_scores_list = [] ; df_boot_list = []
    for label in list_labels:
        if 'fitPPS' in label:
            nameTargetfit = 'Target*Signal'
        else:
            nameTargetfit = 'Target'
        if '| PPS' not in label:
            df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                        f'{nameTargetfit}_Target_{n_boot}'
            filepath_dfs = os.path.join(filepath_df_datas, df_file)
            d_dfscores = functions_pp.load_hdf5(filepath_dfs+'.h5')
            df_scores_list.append(d_dfscores['df_scores'])
            df_boot_list.append(d_dfscores['df_boot'])
        elif '| PPS' in label:
            df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                        f'{nameTargetfit}_Target_{n_boot}_CF'
            filepath_dfs = os.path.join(filepath_df_datas, df_file)
            d_dfscores = functions_pp.load_hdf5(filepath_dfs+'.h5')
            df_scores, df_boot = to_df_scores_format(d_dfscores, condition)
            df_scores_list.append(df_scores)
            df_boot_list.append(df_boot)
    return df_scores_list, df_boot_list


def boxplot_cond_fc(df_cond, metrics: list=None, forcing_name: str='', composite = 30):
    '''


    Parameters
    ----------
    df_cond : pd.DataFrame
        should have pd.MultiIndex of (metric, lead_time) and pd.MultiIndex column
        of (composite, n_boot).
    metrics : list, optional
        DESCRIPTION. The default is None.
    forcing_name : str, optional
        DESCRIPTION. The default is ''.
    composite : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    '''
    #%%
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                'MAE':'MAE-SS', 'mean_absolute_error':'Mean Absolute Error',
                'r2_score':'$r^2$ score', 'BSS':'BSS', 'roc_auc_score':'AUC-ROC'}

    n_boot = df_cond.columns.levels[1].size
    columns = [c[0] for c in df_cond.columns[::n_boot]]
    plot_cols = [f'strong {composite}%', f'weak {composite}%']
    if metrics is None:
        indices = np.unique([t[0] for t in df_cond.index],return_index=True)[1]
        metrics = [n[0] for n in df_cond.index[sorted(indices)]] # preserves order

    df_cond = df_cond.loc[metrics]

    metric = metrics[0]
    indices = np.unique([t[1] for t in df_cond.index],return_index=True)[1]
    lead_times = [n[1] for n in df_cond.index[sorted(indices)]] # preserves order
    f, axes = plt.subplots(len(metrics),len(lead_times),
                           figsize=(len(lead_times)*3, 6*len(metrics)**0.5),
                           sharex=True)
    axes = axes.reshape(len(metrics), -1)


    for iax, index in enumerate(df_cond.index):
        metric = index[0]
        lead_time = index[1]
        row = metrics.index(metric) ; col = list(lead_times).index(lead_time)
        ax = axes[row, col]

        data = df_cond.loc[metric, lead_time].values.reshape(len(columns), -1)
        data = pd.DataFrame(data.T, columns=columns)[plot_cols]

        perc_incr = (data[plot_cols[0]].mean() - data[plot_cols[1]].mean()) / abs(data[plot_cols[1]].mean())

        nlabels = plot_cols.copy() ; widths=(.5,.5)
        nlabels = [l.split(' ')[0] for l in nlabels]
        nlabels = [l.capitalize() for l in nlabels]


        boxprops = dict(linewidth=2.0, color='black')
        whiskerprops = dict(linestyle='-',linewidth=2.0, color='black')
        medianprops = dict(linestyle='-', linewidth=2, color='red')
        ax.boxplot(data, labels=['', ''],
                   widths=widths, whis=[2.5, 97.5], boxprops=boxprops, whiskerprops=whiskerprops,
                   medianprops=medianprops, showmeans=True)

        text = f'{int(100*perc_incr)}%'
        if perc_incr > 0: text = '+'+text
        ax.text(0.98, 0.98,text,
                horizontalalignment='right',
                verticalalignment='top',
                transform = ax.transAxes,
                fontsize=15)

        if metric == 'corrcoef' or metric=='roc_auc_score':
            ax.set_ylim(0,1) ; steps = 1
            yticks = np.round(np.arange(0,1.01,.2), 2)
            ax.set_yticks(yticks[::steps])
            ax.set_yticks(yticks, minor=True)
            ax.tick_params(which='minor', length=0)
            ax.set_yticklabels(yticks[::steps])
            if metric=='roc_auc_score':
                ax.axhline(y=0.5, color='black', linewidth=1)
        elif metric == 'mean_absolute_error':
            yticks = np.round(np.arange(0,1.61,.4), 1)
            ax.set_ylim(0,1.6) ; steps = 1
            ax.set_yticks(yticks[::steps])
            ax.set_yticks(yticks, minor=True)
            ax.tick_params(which='minor', length=0)
            ax.set_yticklabels(yticks[::steps])
        else:
            yticks = np.round(np.arange(-.2,1.1,.2), 1)
            ax.set_ylim(-.3,1) ; steps = 2
            ax.set_yticks(yticks[::steps])
            ax.set_yticks(yticks, minor=True)
            ax.tick_params(which='minor', length=0)
            ax.set_yticklabels(yticks[::steps])
            ax.axhline(y=0, color='black', linewidth=1)

        ax.tick_params(which='both', grid_ls='-', grid_lw=1,width=1,
                       labelsize=16, pad=6, color='black')
        ax.grid(which='both', ls='--')
        if col == 0:
            ax.set_ylabel(rename_m[metric], fontsize=18, labelpad=2)
        if row == 0:
            ax.set_title(lead_time, fontsize=18)
        if row+1 == len(metrics):
            ax.set_xticks([1,2])
            ax.set_xticklabels(nlabels, fontsize=14)
            ax.set_xlabel(forcing_name, fontsize=15)
    f.subplots_adjust(wspace=.4)
    #%%
    return f

def lineplot_cond_fc(filepath_dfs, metrics: list=None, composites=[30],
                     alpha=0.05, fs=16):
    #%%
    rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                'MAE':'MAE-SS', 'mean_absolute_error':'Mean Absolute Error',
                'r2_score':'$r^2$ score', 'BSS':'BSS', 'roc_auc_score':'AUC-ROC'}

    df_cond_fc = functions_pp.load_hdf5(filepath_dfs)['df_cond_fc']

    # preserves order
    indices = np.unique([t[1] for t in df_cond_fc.index],return_index=True)[1]
    lead_times = [n[1] for n in df_cond_fc.index[sorted(indices)]]

    if metrics is None:

        indices = np.unique([t[0] for t in df_cond_fc.index],
                            return_index=True)[1]
        metrics = [n[0] for n in df_cond_fc.index[sorted(indices)]]



    f, axes = plt.subplots(len(metrics),1,
                           figsize=(10, 6*len(metrics)**0.5),
                           sharex=True)
    # plt.style.use('seaborn-talk')
    cols = [c for c in df_cond_fc.columns.levels[0] if \
            int(c.split('%')[0][-2:]) in composites]
    # lines = [t for t in itertools.product(metrics, cols)]
    for met, cond in itertools.product(metrics, cols):
        print(met, cond)
        df = df_cond_fc.loc[met, cond] ; ax = axes[metrics.index(met)]
        if metrics.index(met) == 0:
            label = cond.split(' ')[0] +' SST forcing ('+cond.split(' ')[1]+' composite)'
        else:
            label=None
        ax.plot(lead_times, df.mean(axis=1).values, label=label)
        lower = df.quantile(q=alpha/2, axis=1)
        upper = df.quantile(q=1-alpha/2, axis=1)
        ax.fill_between(lead_times, lower, upper, alpha=0.2)

        if met == 'corrcoef':
            ax.set_ylim(0,1) ; labelpad=11
        elif met == 'r2_score':
            ax.set_ylim(-.3,0.9) ; labelpad=-9
        else:
            ax.set_ylim(-0.3,0.6) ; labelpad=0
        ax.axhline(y=0, color='black', lw=.75)
        ax.set_ylabel(rename_m[met], fontsize=fs, labelpad=labelpad)
        ax.grid(b=True, axis='y')
        ax.tick_params(direction='in', length=9)
        ax.xaxis.set_tick_params(labelsize=fs-2)
        ax.yaxis.set_tick_params(labelsize=fs-2)
        if metrics.index(met) == 0:
            ax.legend(fontsize=fs-5)



