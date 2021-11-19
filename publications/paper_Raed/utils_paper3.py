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
import xarray as xr

# import sklearn.linear_model as scikitlinear
from matplotlib import gridspec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
import matplotlib.patches as mpatches

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
import functions_pp, find_precursors, plot_maps

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
                ax[i].set_ylim(-.1,1.)
            ax[i].axhline(y=0, color='black', linewidth=1)
            ax[i].tick_params(labelsize=14, pad=6)


            if i == 0:
                ax[i].legend(loc='lower left', fontsize=12)
            # ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=0)
            ax[i].set_title(rename_m[m])
    f.subplots_adjust(hspace=.1)
    f.subplots_adjust(wspace=.25)

    return f

def plot_forecast_ts(df_test_m, df_test, df_forcings=None, df_boots_list=None,
                     fig_ax=None, fs=12,
                     metrics_plot=None, name_model=None):
    #%%
    # fig_ax=None
    alpha = .1
    fontsize = fs
    if fig_ax is None and df_forcings is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,4),
                             gridspec_kw={'width_ratios':[3.5,1]},
                             sharex=True, sharey=True)
    if fig_ax is None and df_forcings is not None:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,4),
                             gridspec_kw={'width_ratios':[3.5,1],
                                          'height_ratios':[3,1]},
                             sharex=True, sharey=False)
    else:
        fig, axs = fig_ax


    if axs.size==4:
        ax0u, ax0b = axs.reshape(-1)[[0,2]]
        ax1u, ax1b = axs.reshape(-1)[[1,3]]
    else:
        ax0u = axs[0]
        ax1u = axs[1]


    if df_forcings is not None and axs.size==4:
        table_col = [d.index[0] for d in df_test_m][1:]
        col = [c for c in df_forcings.columns if df_test.columns[1] in c]

        dates_match = pd.MultiIndex.from_product([df_forcings.index.levels[0],
                                                  df_test.index])
        df_forcings = df_forcings.loc[dates_match]
        mean = df_forcings[col].mean(0,level=1)
        std = df_forcings[col].std(0,level=1) * 2



        # identify strong forcing dates
        qs = [int(t[-3:-1]) for t in table_col]
        sizes = [30, 75] #; ls = ['-','-.']
        colors = [['#e76f51', '#61a5c2'], ['#d00000', '#3f37c9']]
        markercolor = np.repeat(['black'], mean.size)
        markercolor = np.array(markercolor, dtype='object')
        markersize = np.repeat(10, mean.size)
        for i, q_th in enumerate(qs):
            low = float(mean.quantile(q_th/200))
            high = float(mean.quantile(1-q_th/200))

            mean_np = mean.values.ravel()
            for j, dp in enumerate(mean_np):
                if dp > high:
                    markercolor[j] = colors[i][0]
                    markersize [j] = sizes[i]
                elif dp < low:
                    markercolor[j] = colors[i][1]
                    markersize [j] = sizes[i]
            # markersize = np.round((np.abs(mean_np)+3)**3,0)
            markercolor_S = markercolor.copy()
            markercolor_S[markercolor_S == 'black'] = 'black'
            # ax0b.axhline(low, lw=0.5, c='grey', ls=ls[i])
            # ax0b.axhline(high, lw=0.5, c='grey', ls=ls[i])
            ax0b.axhline(0, lw=0.5, c='grey')

        ax0b.plot_date(df_test.index, mean.loc[df_test.index],
                      ls='--', c='black', alpha=.7, lw=1,
                      markeredgecolor='black',
                      markerfacecolor='black', zorder=1,
                      label=mean.columns[0]+' Signal', markersize=2,
                      )

        ax0b.scatter(df_test.index, mean.loc[df_test.index], ls='-',
                    label=None, c=markercolor_S, s=markersize, zorder=2)

        ax0b.fill_between(df_test.index, (mean-std).values.ravel(),
                          (mean+std).values.ravel(), fc='black', alpha=0.5)

        ax0b.set_title(col[0], fontsize=fs, y=0.95)

        y_title = .99
    else:
        markercolor = 'black'
        markersize = 1
        table_col = 'S>50%'
        y_title = .95

    if np.isclose(df_test.iloc[:,0].mean().round(2), 0.33, atol=0.03):
        label_obs = 'Low yield events'
    elif np.isclose(df_test.iloc[:,0].mean().round(2), 0.66, atol=0.03):
        label_obs = 'High yield events'
    else:
        label_obs = 'Observed'

    ax0u.plot_date(df_test.index, df_test.iloc[:,0], ls='-',
                   label=label_obs, c='black', marker='o',
                   markersize=5)
    ax0u.scatter(df_test.index, df_test.iloc[:,0], ls='-',
                   label=None, c=markercolor, s=markersize, zorder=3)
    if name_model is None:
        name_model = 'Prediction'

    ax0u.plot_date(df_test.index, df_test.iloc[:,1], ls='-', c='orange',
                  label=name_model, markersize=4)

    ax0u.tick_params(labelsize=fontsize)


    if type(df_test_m) is not list:
        df_test_m = [df_test_m]
    order_verif = ['All'] + ['Top '+t.split(' ')[1] for t in table_col]
    df_scores = []
    for i, df_test_skill in enumerate(df_test_m):
        r = {df_test_skill.index[0]:order_verif[i]}
        df_scores.append( df_test_skill.rename(r, axis=0) )
    df_scores = pd.concat(df_scores, axis=1)
    df_scores = df_scores.T.reset_index().pivot_table(index='index')
    df_scores = df_scores[order_verif]

    rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr.', 'MAE':'MAE-SS',
                  'BSS':'BSS', 'roc_auc_score':'AUC', 'r2_score':'$r^2$',
                  'mean_absolute_percentage_error':'MAPE', 'AUC_SS':'AUC-SS',
                  'precision':'Precision', 'accuracy':'Accuracy'}
    if metrics_plot is None:
        metrics_plot = df_scores.index


    if df_boots_list is not None:
        bootlow = [] ; boothigh = []
        for i, df_b in enumerate(df_boots_list):
            _l = pd.DataFrame(df_b.quantile(alpha)).T
            _h = pd.DataFrame(df_b.quantile(1-alpha)).T
            bootlow.append( _l.rename({_l.index[0]:order_verif[i]}, axis=0) )
            boothigh.append( _h.rename({_h.index[0]:order_verif[i]}, axis=0) )
        bootlow = pd.concat(bootlow, axis=1)
        bootlow = bootlow.T.reset_index().pivot_table(index='index')
        boothigh = pd.concat(boothigh, axis=1)
        boothigh = boothigh.T.reset_index().pivot_table(index='index')
        bootlow = bootlow[order_verif] ; boothigh = boothigh[order_verif]
        tbl = bootlow.loc[metrics_plot].values
        tbh = boothigh.loc[metrics_plot].values

    tsc = df_scores.loc[metrics_plot].values
    tscs = np.zeros(tsc.shape, dtype=object)
    for i, m in enumerate(metrics_plot):
        _sc_ = tsc[i]
        if 'prec' in m or 'acc' in m:
            sc_f = ['{:.0f}%'.format(round(s,0)) for s in _sc_.ravel()]
            if df_boots_list is not None:
                for j, s in enumerate(zip(tbl[i], tbh[i])):
                    sc_f[j] = sc_f[j].replace('%', r'$_{{{:.0f}}}^{{{:.0f}}}$%'.format(*s))
        else:
            sc_f = ['{:.2f}'.format(round(s,2)) for s in tsc[i].ravel()]

            if df_boots_list is not None:
                for j, s in enumerate(zip(tbl[i], tbh[i])):
                    sc_f[j] = sc_f[j] + r'$_{{{:.2f}}}^{{{:.2f}}}$'.format(*s)
        tscs[i] = sc_f


    table = ax1u.table(cellText=tscs,
                      cellLoc='center',
                      rowLabels=[rename_met[m] for m in metrics_plot],
                      colLabels=df_scores.columns, loc='center',
                      edges='closed')
    if len(df_test_m) == 2:
        table.scale(0.9, 1.7)
    elif len(df_test_m) == 3:
        table.scale(1.1, 2.8)

    # color text BSS
    shadesgr = ["90a955","4f772d","31572c","132a13"]
    lin = np.round(np.linspace(0,1,5),2) ; sBSS = (lin[1]-lin[0])/2
    ct_BSS = {lin[i]+sBSS:shadesgr[i] for i in range(4)}
    base_ACC = 100 * (0.66*0.66) + (0.33 * 0.33)
    lin = np.round(np.linspace(base_ACC, 100, 5), 1) ; sACC = (lin[1]-lin[0])/2
    ct_ACC = {lin[i]+sACC:shadesgr[i] for i in range(4)}
    lin = np.round(np.linspace(33, 100, 5), 1) ; sPrec = (lin[1]-lin[0])/2
    ct_Prec  = {lin[i]+sPrec:shadesgr[i] for i in range(4)}
    combs_table = np.array(np.meshgrid(range(len(df_test_m)),
                                        range(len(metrics_plot)))).T.reshape(-1,2)
    for r,c in combs_table:
        val = tsc[r,c] ; colt = []
        if metrics_plot[r] == 'BSS':
            colt = [c for v,c in ct_BSS.items() if np.isclose(val,v, atol=sBSS)]
        elif metrics_plot[r] == 'accuracy':
            colt = [c for v,c in ct_ACC.items() if np.isclose(val,v, atol=sACC)]
        elif metrics_plot[r] == 'precision':
            colt = [c for v,c in ct_Prec.items() if np.isclose(val,v, atol=sPrec)]
        if len(colt)==0:
            colt = "ae2012" # no positive skill: color red
        else:
            colt = colt[0]

        table[(r+1, c)].get_text().set_color('#'+colt)



    table.set_fontsize(fontsize+2)
    ax0u.set_title(r"$\bf{"+df_test.columns[1] + '\ forecast'+"}$",
                   fontsize=fs+1, y=y_title)
    ax1u.axis('off') ; ax1b.axis('off')
    ax1u.axis('tight') ; ax1b.axis('tight')
    ax1u.set_facecolor('white') ; ax1b.set_facecolor('white')
    if 'BSS' in metrics_plot:
        ax0u.set_ylim(-0.05,1.05)
        ax0u.set_ylabel('Prob. Forecast', labelpad=3, fontsize=fs)
    else:
        ax0u.axhline(y=0, color='black', lw=1)
        ax0u.set_ylim(-3.05,3.05)
        ax0u.set_ylabel('Forecast')
    if df_forcings is not None and axs.size==4:
        ax0b.set_ylim(-3.05,3.05)
        ax0b.set_yticks(np.arange(-3,3.1,3))
        ax0b.set_yticklabels(np.arange(-3,3.1,3), fontsize=fs)
    #%%
    return


#%% Conditional continuous forecast

def get_df_forcing_cond_fc(rg_list, #target_ts, fcmodel, kwrgs_model, mean_vars=['sst', 'smi'],
                           regions=['only_Pacific'],
                           name_object='df_CL_data'):
    if len(regions) == 1:
        regions = regions * len(rg_list)
    for j, rg in enumerate(rg_list):
        region = regions[j]

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

            prediction = df_predictions[[nameTarget, rg.fc_month]]
            df_test = functions_pp.get_df_test(prediction,
                                               df_splits=rg.df_splits)

            # df_test_m = rg.verification_tuple[2]
            # cond_df[i, j, 0] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            for k, l in enumerate(range(0,4,2)):
                q = quantiles[k]
                # extrapolate quantile values based on training data
                q_low = functions_pp.get_df_train(rg.df_forcing.mean(axis=1),
                                         df_splits=rg.df_splits, s='extrapolate',
                                         function='quantile', kwrgs={'q':q})
                # Extract out-of-sample quantile values
                q_low = functions_pp.get_df_test(q_low,
                                                   df_splits=rg.df_splits)

                q_high = functions_pp.get_df_train(rg.df_forcing.mean(axis=1),
                                         df_splits=rg.df_splits, s='extrapolate',
                                         function='quantile', kwrgs={'q':1-q})
                q_high = functions_pp.get_df_test(q_high,
                                                   df_splits=rg.df_splits)

                low = df_forctest < q_low.values.ravel()
                high = df_forctest > q_high.values.ravel()
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
                # =============================================================
                # mild boundary forcing
                # =============================================================
                q_higher_low = functions_pp.get_df_train(rg.df_forcing.mean(axis=1),
                                         df_splits=rg.df_splits, s='extrapolate',
                                         function='quantile', kwrgs={'q':.5-q})
                q_higher_low = functions_pp.get_df_test(q_higher_low,
                                                   df_splits=rg.df_splits)


                q_lower_high = functions_pp.get_df_train(rg.df_forcing.mean(axis=1),
                                         df_splits=rg.df_splits, s='extrapolate',
                                         function='quantile', kwrgs={'q':.5+q})
                q_lower_high = functions_pp.get_df_test(q_lower_high,
                                                   df_splits=rg.df_splits)

                higher_low = df_forctest > q_higher_low.values.ravel()
                lower_high = df_forctest < q_lower_high.values.ravel()
                # higher_low = df_forctest > float(df_forctrain.quantile(.5-q))
                # lower_high = df_forctest < float(df_forctrain.quantile(.5+q))
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
    df_scores.index = [condition]
    return df_scores, df_boot

def load_scores(list_labels, model_name_CL, model_name, n_boot,
                filepath_df_datas, condition=None):

    df_scores_list = [] ; df_boot_list = []; df_predictions = []
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
            f_name = f'predictions_cont_CL{model_name_CL}_{model_name}_'\
                                                        f'{nameTargetfit}.h5'
            filepath_dfs = os.path.join(filepath_df_datas, f_name)
            d_dfs = functions_pp.load_hdf5(filepath_dfs)
            df_predictions.append(d_dfs['df_predictions'])
        elif '| PPS' in label:
            df_file = f'scores_cont_CL{model_name_CL}_{model_name}_'\
                        f'{nameTargetfit}_Target_{n_boot}_CF'
            filepath_dfs = os.path.join(filepath_df_datas, df_file)
            d_dfscores = functions_pp.load_hdf5(filepath_dfs+'.h5')
            if condition is None:
                conditions = [c for c in d_dfscores['df_cond'].columns if 'str' in c]
            elif type(condition) is str:
                conditions  = [condition]
            else:
                conditions   = condition
            for con in conditions:
                df_scores, df_boot = to_df_scores_format(d_dfscores, con)
                df_scores_list.append(df_scores)
                df_boot_list.append(df_boot)

    return df_scores_list, df_boot_list, df_predictions


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


def plot_regions(rg_list, save, plot_parcorr=False, min_detect=0.5,
                 selection='all'):
    '''


    Parameters
    ----------
    min_detect : TYPE, optional
        DESCRIPTION. The default is 0.5.
    selection : TYPE, optional
        'all', 'CD' or 'ind'. The default is 'all'.

    '''
    #%%
    # labels plot
    dirpath = os.path.join(rg_list[0].path_outsub1, 'causal_maps')
    os.makedirs(dirpath, exist_ok=True)


    if plot_parcorr:
        units = 'Partial Correlation [-]'
    else:
        units = 'Correlation [-]'

    alpha_CI = 0.05

    def get_map_rg(rg, ip, ir=0, textinmap=[], min_detect=.5):

        month_d = {'AS':'Aug-Sep mean', 'JJ':'July-June mean',
                   'JA':'July-June mean','MJ':'May-June mean',
                   'AM':'Apr-May mean',
                   'MA':'Mar-Apr mean', 'FM':'Feb-Mar mean',
                   'JF':'Jan-Feb mean', 'DJ':'Dec-Jan mean',
                   'ND':'Nov-Dec mean', 'ON':'Oct-Nov mean',
                   'SO':'Sep-Oct mean'}

        kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4,
                              'hspace':.37, 'wspace':0., 'size':2, 'cbar_vert':0.05,
                              'map_proj':plot_maps.ccrs.PlateCarree(central_longitude=220),
                              'y_ticks':False, 'x_ticks':False, #np.arange(-10,61,20), #'x_ticks':np.arange(130, 280, 25),
                              'title':'',
                              'subtitle_fontdict':{'fontsize':20},
                              # 'clevels':np.arange(-.6,.7,.1),
                              # 'clabels':np.arange(-.6,.7,.2),
                               'cbar_tick_dict':{'labelsize':18},
                              'units':units,
                              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

        kwrgs_plotlabels_sst = kwrgs_plotcorr_sst.copy()
        # kwrgs_plotlabels_sst.pop('clevels'); kwrgs_plotlabels_sst.pop('clabels')
        # kwrgs_plotlabels_sst.pop('cbar_tick_dict')
        kwrgs_plotlabels_sst['units'] = 'DBSCAN clusters'
        kwrgs_plotlabels_sst['cbar_vert'] = 0.06


        kwrgs_plotcorr_SM = {'row_dim':'lag', 'col_dim':'split','aspect':2,
                             'hspace':0.25, 'wspace':-0.5, 'size':3, 'cbar_vert':0.04,
                              'map_proj':plot_maps.ccrs.PlateCarree(central_longitude=220),
                               # 'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                               'y_ticks':False, 'x_ticks':False,
                               'title':'',
                               'subtitle_fontdict':{'fontsize':15},
                               # 'clevels':np.arange(-.6,.7,.1),
                               # 'clabels':np.arange(-.6,.7,.4),
                                'cbar_tick_dict':{'labelsize':20},
                               'units':units,
                               'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}


        kwrgs_plotlabels_SM = kwrgs_plotcorr_SM.copy()
        # kwrgs_plotlabels_SM.pop('clevels'); kwrgs_plotlabels_SM.pop('clabels')
        # kwrgs_plotlabels_SM.pop('cbar_tick_dict')
        kwrgs_plotlabels_SM['units'] = 'DBSCAN clusters'
        kwrgs_plotlabels_SM['cbar_vert'] = 0.05

        # Get ConDepKeys
        df_pvals = rg.df_pvals.copy()
        df_corr  = rg.df_corr.copy()
        periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)

        CondDepKeys = {} ;
        for i, mon in enumerate(periodnames):
            list_mon = []
            _keys = [k for k in df_pvals.index if mon in k] # month
            df_sig = df_pvals[df_pvals.loc[_keys] <= alpha_CI].dropna(axis=0, how='all') # significant

            for k in df_sig.index:
                corr_val = df_corr.loc[k].mean()
                RB = (df_pvals.loc[k]<alpha_CI).sum()
                list_mon.append((k, corr_val, RB))
            CondDepKeys[mon] = list_mon

        # get number of time precursor extracted training splits:
        allkeys = [list(rg.df_data.loc[s].dropna(axis=1).columns[1:-2]) for s in range(rg.n_spl)]
        allkeys = functions_pp.flatten(allkeys)
        {k:allkeys.count(k) for k in allkeys}
        rg._df_count = pd.Series({k:allkeys.count(k) for k in allkeys},
                                   dtype=object)


        precur = rg.list_for_MI[ip]

        CDlabels = precur.prec_labels.copy()

        if precur.group_lag:
            CDlabels = xr.concat([CDlabels]*len(periodnames), dim='lag')
            CDlabels['lag'] = ('lag', periodnames)
            CDcorr = precur.corr_xr_.copy()
        else:
            CDcorr = precur.corr_xr.copy()

        MCIstr = CDlabels.copy()
        for i, month in enumerate(CondDepKeys):


            CDkeys = [k[0] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
            MCIv = [k[1] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
            RB = [k[2] for k in CondDepKeys[month] if precur.name in k[0].split('..')[-1]]
            region_labels = [int(l.split('..')[1]) for l in CDkeys if precur.name in l.split('..')[-1]]
            indkeys = [k for k in allkeys if precur.name in k.split('..')[-1]]
            indkeys = [k for k in indkeys if month in k]
            indkeys = list(dict.fromkeys(indkeys))
            indkeys = [k for k in indkeys if k not in CDkeys]
            f = find_precursors.view_or_replace_labels
            if len(CDkeys) != 0:
                if region_labels[0] == 0: # pattern cov
                    region_labels = np.unique(CDlabels[:,i].values[~np.isnan(CDlabels[:,i]).values])
                    region_labels = np.array(region_labels, dtype=int)
                    MCIv = np.repeat(MCIv, len(region_labels))
                    CDkeys = [CDkeys[0].replace('..0..', f'..{r}..') for r in region_labels]

            # minimal number of times region needs to be detected
            min_d = int(rg.n_spl * min_detect)
            indkeys = [k for k in indkeys if rg._df_count[k] > min_d]
            if selection == 'CD':
                indkeys = []
                selregions = f(CDlabels[:,i].copy(), region_labels)
            elif selection == 'ind':
                CDkeys = []
                selregions = f(CDlabels[:,i].copy(),
                               [int(k.split('..')[1]) for k in indkeys])
            else:
                selregions = CDlabels[:,i].copy()


            if plot_parcorr:
                MCIstr[:,i]   = f(selregions.copy(), region_labels,
                                  replacement_labels=MCIv)

            MCIstr[:,i]   = CDcorr[:,i].where(~np.isnan(selregions))
            # all keys
            df_labelloc = find_precursors.labels_to_df(CDlabels[:,i])
            # Conditionally independent


            if len(indkeys) != 0:
                temp = []
                for q, k in enumerate(indkeys):
                    l = int(k.split('..')[1])
                    if l == 0: # pattern cov
                        lat, lon = df_labelloc.mean(0)[:2]
                    else:
                        lat, lon = df_labelloc.loc[l].iloc[:2].values.round(1)
                    if lon > 180: lon-360
                    count = rg._df_count[k]
                    text = f'{count}'
                    temp.append([lon,max(-12,min(54,lat)),
                                     text, {'fontsize':8,
                                            'bbox':dict(facecolor='pink', alpha=0.1)}])
                # for reordering first lag first: 3-i
                textinmap.append([(3-i,ir), temp])
            # get text on robustness per month:

            if len(CDkeys) != 0:
                temp = []
                for q, k in enumerate(CDkeys):
                    l = int(k.split('..')[1])
                    if l == 0: # pattern cov
                        lat, lon = df_labelloc.mean(0)[:2]
                    else:
                        lat, lon = df_labelloc.loc[l].iloc[:2].values.round(1)
                    if lon > 180: lon-360
                    if precur.calc_ts != 'pattern cov':
                        count = rg._df_count[k]
                        if count < min_d:
                            continue
                        text = f'{int(RB[q])}/{count}'
                        temp.append([lon-12,min(54,lat+7),
                                     text, {'fontsize':15,
                                            'bbox':dict(facecolor='white', alpha=0.2)}])
                    elif precur.calc_ts == 'pattern cov' and q == 0:
                        count = rg._df_count[f'{month}..0..{precur.name}_sp']
                        if count < min_d:
                            continue
                        text = f'{int(RB[0])}/{count}'
                        lon = float(CDlabels[:,i].longitude.mean())
                        lat = float(CDlabels[:,i].latitude.mean())
                        temp.append([lon,lat, text, {'fontsize':15,
                                               'bbox':dict(facecolor='white', alpha=0.2)}])
                # for reordering first lag first: 3-i
                textinmap.append([(3-i,ir), temp])

        if ip == 0:
            kwrgs_plot = kwrgs_plotlabels_sst.copy()
        elif ip == 1:
            kwrgs_plot = kwrgs_plotlabels_SM.copy()


        lags = rg.list_for_MI[0].corr_xr.lag
        subtitles = np.array([month_d[l] for l in lags.values], dtype='object')[::-1]

        subtitles = [subtitles[i-1]+f' ({leadtime+i*2-1}-month lag)' for i in range(1,5)]
        # reorder first lag first
        CDlabels = CDlabels.sel(lag=periodnames[::-1])
        MCIstr = MCIstr.sel(lag=periodnames[::-1])
        textinmap = textinmap[::-1]
        return CDlabels, MCIstr, textinmap, subtitles, kwrgs_plot


    lagsize = rg_list[0].list_for_MI[0].prec_labels.lag.size
    intmon_d = {'August': 2, 'July':3, 'June':4,'May':5, 'April':6,
                'March':7, 'February':8}
    for ip in range(1):
        if ip == 0:
            rg_subs = [rg_list[:3], rg_list[3:]]
        else:
            rg_subs = [rg_list]

        if ip == 1 and selection =='ind':
            continue

        for i, rg_sub in enumerate(rg_subs):
            precur = rg_sub[0].list_for_MI[ip]
            list_l = [] ; list_v = [] ; list_m = [] ; textinmap = []
            subs = np.zeros((lagsize,len(rg_sub)), dtype='object')
            for ir, rg in enumerate(rg_sub):
                leadtime = intmon_d[rg.fc_month]
                out = get_map_rg(rg,
                                 ip,
                                 ir,
                                 textinmap,
                                 min_detect)
                CDlabels, MCIstr, textinmap, subtitles, kwrgs_plot = out

                fcmontitle = f'{rg.fc_month}\ Forecast'
                subtitles[0] = r'$\bf{'+fcmontitle+'}$' \
                                f'\n{leadtime}-month lead-time\n' + subtitles[0]
                subs[:,ir] = subtitles

                get_mask = rg.__class__._get_sign_splits_masked
                CDlabels, mask_xr = get_mask(CDlabels,
                                             min_detect)
                MCIstr, mask_xr = get_mask(MCIstr,
                                           min_detect)
                                           # mask=np.isnan(MCIstr))


                MCIstr['lag'] = ('lag', range(CDlabels.lag.size))
                mask_xr['lag'] = ('lag', range(CDlabels.lag.size))
                CDlabels['lag'] = ('lag', range(CDlabels.lag.size))

                list_l.append(CDlabels.where(mask_xr))
                list_m.append(mask_xr)
                list_v.append(MCIstr.where(mask_xr))


            CDlabels = xr.concat(list_l, dim='split')
            mask_xr = xr.concat(list_m, dim='split')
            MCIstr = xr.concat(list_v, dim='split')
            CDlabels['split'] = ('split', range(len(rg_sub)))
            mask_xr['split'] = ('split', range(len(rg_sub)))
            MCIstr['split'] = ('split', range(len(rg_sub)))

            kwrgs_plot['subtitles'] = subs

            # plot_maps.plot_labels(CDlabels, kwrgs_plot=kwrgs_plot)

            if plot_parcorr==False:
                plot_maps.plot_labels(CDlabels,
                                      kwrgs_plot=kwrgs_plot)
                if save:
                    plt.savefig(os.path.join(dirpath,
                                          f'{precur.name}_eps{precur.distance_eps}'
                                          f'minarea{precur.min_area_in_degrees2}_'
                                          f'aCI{alpha_CI}_labels_{i}_{min_detect}'\
                                          +rg.figext),
                                  bbox_inches='tight')

            # MCI values plot
            if plot_parcorr:
                kwrgs_plot.update({'clevels':np.arange(-0.8, 0.9, .1),
                                   'clabels':np.arange(-.8,.9,.2)})
            else:
                kwrgs_plot.update({'clevels':np.arange(-0.6, 0.7, .1),
                                   'clabels':np.arange(-.6,.7,.2)})
            kwrgs_plot.update({'textinmap':textinmap,
                                'units':units})
            fg = plot_maps.plot_corr_maps(MCIstr,
                                          mask_xr=mask_xr,
                                          **kwrgs_plot)
    #%%
            if save:
                fg.fig.savefig(os.path.join(dirpath,
                              f'{precur.name}_eps{precur.distance_eps}'
                              f'minarea{precur.min_area_in_degrees2}_aCI{alpha_CI}_MCI_'
                              f'_parcorr{plot_parcorr}_{i}_{selection}'+rg.figext),
                            bbox_inches='tight')


    #%%

def detrend_oos_3d(ds, min_length=None, df_splits: pd.DataFrame=None,
                   standardize=True, path=None):
    #%%
    from scipy import stats

    if min_length is not None:
        ds_avail = (70 - np.isnan(ds).sum(axis=0))
        ds   = ds.where(ds_avail.values >= min_length)

    if df_splits is None:
        splits = np.array([0])
    else:
        splits = df_splits.index.levels[0]
        # ensure same time-axis
        ds = ds.sel(time=df_splits.index.levels[1])

    xy = np.argwhere((~np.isnan(ds)).any(axis=0).values)
    n_plots = 40 ; icount = int(xy.shape[0] / n_plots)

    f1, ax1 = plt.subplots(figsize=(5,4)) ;
    f2, axes = plt.subplots(nrows=int(n_plots / 5), ncols=5, sharex=True) ;
    axes = axes.ravel()
    splits_newdata = []
    for s in splits:
        newdata = np.zeros_like(ds)
        newdata[:] = np.nan
        for i, (x, y) in enumerate(xy):

            ts = ds[:,x,y]
            timesteps = ts.time.dt.year.values
            # mask NaN
            mask = ~np.isnan(ts)
            if df_splits is not None:
                trainmask = df_splits.loc[s]['TrainIsTrue']==1
                mask = np.logical_and(mask.values, trainmask.values)
                if mask[mask].size < min_length:
                    tonan = np.zeros_like(ts) ; tonan[:] = np.nan
                    newdata[:,x,y] = tonan


            m, b, r_val, p_val, std_err = stats.linregress(timesteps[mask],
                                                           ts[mask])
            trend = (m*timesteps + b)
            detrend_ts = ts - trend
            if standardize:
                detrend_ts = (detrend_ts - detrend_ts[mask].mean()) / \
                                detrend_ts[mask].std()
            if mask[mask].size >= min_length:
                newdata[:,x,y] = detrend_ts
            if i % icount == 0 : #or i+1 == xy.shape[0] and s==0:
                progress = int(100*(i+1)/xy.shape[0])
                print(f"\rProgress {progress}%", end="")
                idx = int(i/icount)
                if idx < axes.size:
                    if s == splits[-1]:
                        axes[idx].plot(timesteps, ts, lw=1) # only plot raw for first split
                    axes[idx].plot(timesteps, trend, lw=0.5, c='black', alpha=.5)
                    axes[idx].tick_params(labelsize=7)
                    ax1.plot(timesteps, detrend_ts, lw=0.5)
                    ax1.tick_params(labelsize=12)
                    # if mask[mask].size < min_length and s==splits[-1]:
                    #     axes[idx].text(0.2,0.5, 'Too short',
                    #                    horizontalalignment='center',
                    #                    verticalalignment='center',
                    #                    transform=axes[idx].transAxes)


        newdata = xr.DataArray(newdata, coords=ds.coords, dims=ds.dims)
        splits_newdata.append(newdata)
    print('\n')
    f2.subplots_adjust(wspace=.35)

    if path is not None:
        f1.savefig(os.path.join(path,'standardized.jpg'), dpi=250,
                   bbox_inches='tight')
        f2.savefig(os.path.join(path,'detrend_plot.jpg'), dpi=250,
                   bbox_inches='tight')


    splits_newdata = xr.concat(splits_newdata, dim='split')
    splits_newdata['split'] = ('split', splits)
    if splits.size==1:
        splits_newdata = splits_newdata.squeeze().drop('split')
    #%%
    return splits_newdata


