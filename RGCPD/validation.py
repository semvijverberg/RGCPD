#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:53:03 2019

@author: semvijverberg
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from sklearn import metrics
import functions_pp
from sklearn.calibration import calibration_curve
import seaborn as sns
from itertools import chain
flatten = lambda l: list(chain.from_iterable(l))


from matplotlib import cycler
nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
colors_nice = cycler('color',
                nice_colors)
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors_nice)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='black')
plt.rc('ytick', direction='out', color='black')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

mpl.rcParams['figure.figsize'] = [7.0, 5.0]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600

mpl.rcParams['font.size'] = 13
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

def get_metrics_sklearn(RV, y_pred_all, y_pred_c, alpha=0.05, n_boot=5, blocksize=10):
    #%%
    
    y = RV.RV_bin.values
    lags = y_pred_all.columns
    cont_pred = np.unique(y_pred_all).size > 5
#    class_pred = np.unique(y_pred_all).size < 5
    
    if cont_pred:
        df_auc = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                              index=['AUC-ROC', 'con_low', 'con_high'])
        
        df_aucPR = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                              index=['AUC-PR', 'con_low', 'con_high'])
        
        df_brier = pd.DataFrame(data=np.zeros( (7, len(lags)) ), columns=[lags],
                              index=['BSS', 'con_low', 'con_high', 'Brier', 'br_con_low', 'br_con_high', 'Brier_clim'])
    

    df_KSS = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                          index=['KSS', 'con_low', 'con_high'])
    
    
    df_prec  = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                          index=['prec', 'con_low', 'con_high'])

    df_acc  = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                          index=['acc', 'con_low', 'con_high'])
    
    
    for lag in lags:
        y_pred = y_pred_all[lag].values

        metrics_dict = metrics_sklearn(
                    y, y_pred, y_pred_c.values,
                    alpha=alpha, n_boot=n_boot, blocksize=blocksize)
        if cont_pred:
            # AUC
            AUC_score, conf_lower, conf_upper, sorted_AUC = metrics_dict['AUC']
            df_auc[lag] = (AUC_score, conf_lower, conf_upper) 
            # AUC Precision-Recall 
            AUCPR_score, ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs = metrics_dict['AUCPR']
            df_aucPR[lag] = (AUCPR_score, ci_low_AUCPR, ci_high_AUCPR)
            # Brier score
            brier_score, brier_clim, ci_low_brier, ci_high_brier, sorted_briers = metrics_dict['brier']
            BSS = (brier_clim - brier_score) / brier_clim
            BSS_low = (brier_clim - ci_high_brier) / brier_clim    
            BSS_high = (brier_clim - ci_low_brier) / brier_clim    
            df_brier[lag] = (BSS, BSS_low, BSS_high, 
                            brier_score, ci_low_brier, ci_high_brier, brier_clim)
        # HKSS
        KSS_score, ci_low_KSS, ci_high_KSS, sorted_KSSs = metrics_dict['KSS']
        df_KSS[lag] = (KSS_score, ci_low_KSS, ci_high_KSS)
        # Precision
        prec, ci_low_prec, ci_high_prec, sorted_precs = metrics_dict['prec']
        df_prec[lag] = (prec, ci_low_prec, ci_high_prec)
        # Accuracy
        acc, ci_low_acc, ci_high_acc, sorted_accs = metrics_dict['acc']
        df_acc[lag] = (acc, ci_low_acc, ci_high_acc)
    
    if cont_pred:
        df_valid = pd.concat([df_brier, df_auc, df_aucPR, df_KSS, df_prec, df_acc], 
                         keys=['BSS', 'AUC-ROC', 'AUC-PR', 'KSS', 'prec', 'acc'])
        print("ROC area: {:0.3f}".format( float(df_auc.iloc[0][0]) ))
        print("P-R area: {:0.3f}".format( float(df_aucPR.iloc[0][0]) ))
        print("BSS     : {:0.3f}".format( float(df_brier.iloc[0][0]) ))
        
    else:
        df_valid = pd.concat([df_KSS, df_prec, df_acc], 
                         keys=['KSS', 'prec', 'acc'])
    print("prec    : {:0.3f}".format( float(df_prec.iloc[0][0]) ))
    print("acc    : {:0.3f}".format( float(df_acc.iloc[0][0]) ))
    

    #%%
    return df_valid, metrics_dict


def get_metrics_bin(y_true, y_pred, t=None):
    '''
    y_true should be classes
    if t == None, y_pred is already classes
    else t (threshold) is used to make binary ts
    '''
    if t != None:
        if t < 1:
            t = 100*t
        else:
            t = t
        y_pred_b = np.array(y_pred > np.percentile(y_pred, t),dtype=int)
    else:
        y_pred_b = y_pred
    prec = metrics.precision_score(y_true, y_pred_b)
    recall = metrics.recall_score(y_true, y_pred_b)
    #        cm = metrics.confusion_matrix(y_true,  y_pred_b_lags[l])
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_b).ravel()
    FPR = fp / (fp + tn)
    SP = tn / (tn + fp)
    Acc  = metrics.accuracy_score(y_true, y_pred_b)
    f1  = metrics.f1_score(y_true, y_pred_b)
    # Hansen Kuiper score (TPR - FPR): 
    tpr = tp / (tp+fn) ; fpr = fp / (fp+tn)
    KSS_score = tpr - fpr
    return prec, recall, FPR, SP, Acc, f1, KSS_score

def get_metrics_confusion_matrix(RV, y_pred_all, thr=['clim', 33, 66], n_shuffle=0):
    #%%                    
    lags = y_pred_all.columns
    y_true = RV.RV_bin
    if thr[0] == 'clim':
        clim_prev = np.round((1-y_true[y_true.values==1.].size / y_true.size),2)
        thresholds = [[clim_prev]]
    perc_thr = [t for t in thr if type(t) == int]
    thresholds.append([t/100. for t in perc_thr])
    thresholds = flatten(thresholds)
    

    list_dfs = []
    for lag in lags:
        metric_names = ['Precision', 'Specifity', 'Recall', 'FPR', 'F1_score', 'Accuracy']
        if n_shuffle == 0:
            stats = ['fc']
        else:
            stats = ['fc', 'fc shuf', 'best shuf', 'impr.'] 
        stats_keys = stats * len(thresholds)
        thres_keys = np.repeat(thresholds, len(stats))
        data = np.zeros( (len(metric_names), len(stats_keys) ) )
        df_lag = pd.DataFrame(data=data, dtype=str, 
                         columns=pd.MultiIndex.from_tuples(zip(thres_keys,stats_keys)),
                          index=metric_names)
    
        for t in thresholds:
            y_pred = y_pred_all[lag].values
            y_pred_b = np.array(y_pred > np.percentile(y_pred, 100*t),dtype=int)
           
            
            out = get_metrics_bin(y_true, y_pred_b, t=None)
            (prec_f, recall_f, FPR_f, SP_f, Acc_f, f1_f, KSS) = out
            # shuffle the predictions 
            prec = [] ; recall = [] ; FPR = [] ; SP = [] ; Acc = [] ; f1 = []
            for i in range(n_shuffle):
                np.random.shuffle(y_pred_b); 
                out = get_metrics_bin(y_true, y_pred_b, t=None)
                (prec_s, recall_s, FPR_s, SP_s, Acc_s, f1_s, KSS) = out
                prec.append(prec_s)
                recall.append(recall_s)
                FPR.append(FPR_s)
                SP.append(SP_s)
                Acc.append(Acc_s)
                f1.append(f1_s)
            
            if n_shuffle > 0:
                precision_  = [prec_f, np.mean(prec),
                                          np.percentile(prec, 97.5), prec_f/np.mean(prec)]
                recall_     = [recall_f, np.mean(recall),
                                                  np.percentile(recall, 97.5),
                                                     recall_f/np.mean(recall)]
                FPR_        = [FPR_f, np.mean(FPR), np.percentile(FPR, 2.5),
                                                     np.mean(FPR)/FPR_f]
                specif_     = [SP_f, np.mean(SP), np.percentile(SP, 97.5), 
                                                     SP_f/np.mean(SP)]
                acc_        = [Acc_f, np.mean(Acc), np.percentile(Acc, 97.5), 
                                                      Acc_f/np.mean(Acc)]
                F1_         = [f1_f, np.mean(f1), np.percentile(f1, 97.5),
                                                     f1_f/np.mean(f1)]
            else:
                precision_  = [prec_f]
                recall_     = [recall_f]
                FPR_        = [FPR_f]
                specif_     = [SP_f]
                acc_        = [Acc_f]
                F1_         = [f1_f]
            
            df_lag.loc['Precision'][t] = pd.Series(precision_,
                                            index=stats)
            df_lag.loc['Recall'][t]  = pd.Series(recall_,
                                                    index=stats)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_b).ravel()
            df_lag.loc['FPR'][t] = pd.Series(FPR_, index=stats)
                                                     
            df_lag.loc['Specifity'][t] = pd.Series(specif_, index=stats)
            df_lag.loc['Accuracy'][t] = pd.Series(acc_, index=stats)
            df_lag.loc['F1_score'][t] = pd.Series(F1_, index=stats)

        if n_shuffle > 0:
            df_lag['mean_impr'] = df_lag.iloc[:, df_lag.columns.get_level_values(1)=='impr.'].mean(axis=1)
            df_lag = df_lag.rename(columns={clim_prev: 'clim'})
        list_dfs.append(df_lag)

    df_cm = pd.concat(list_dfs, keys= lags)
    #%%
    return df_cm


def autocorr_sm(ts, max_lag=None, alpha=0.01):
    import statsmodels as sm
    if max_lag == None:
        max_lag = ts.size
    ac, con_int = sm.tsa.stattools.acf(ts.values, nlags=max_lag-1, 
                                unbiased=True, alpha=0.01, 
                                 fft=True)
    return ac, con_int

def get_bstrap_size(ts, max_lag=200, n=1, plot=True):
    max_lag = min(max_lag, ts.size)
    ac, con_int = autocorr_sm(ts, max_lag=max_lag, alpha=0.01)
    if plot == True:
        plt.figure()
        # con high
        plt.plot(con_int[:,1][:20], color='orange')
        # ac values
        plt.plot(range(20),ac[:20])
        # con low
        plt.plot(con_int[:,0][:20], color='orange')
    where = np.where(con_int[:,0] < 0 )[0]
    # has to be below 0 for n times (not necessarily consecutive):
    n_of_times = np.array([idx+1 - where[0] for idx in where])
    cutoff = where[np.where(n_of_times == n)[0][0] ]
    return cutoff


def metrics_sklearn(y_true, y_pred, y_pred_c, alpha=0.05, n_boot=5, blocksize=1):
#    y_true, y_pred, y_pred_c = y_true_c, ts_logit_c, y_pred_c_c
    #%%

    y_true = np.array(y_true).squeeze()
    cont_pred = np.unique(y_pred).size > 5
    metrics_dict = {}

    # binary metrics for clim prevailance
    clim_prev = 100 * np.round((1-y_true[y_true==1.].size / y_true.size),2)
    y_pred_b = y_pred_b = np.array(y_pred > np.percentile(y_pred, clim_prev),dtype=int)
    
    out = get_metrics_bin(y_true, y_pred, t=clim_prev)
    (prec, recall, FPR, SP, Acc, f1, KSS_score) = out
    prec = metrics.precision_score(y_true, y_pred_b)
    acc = metrics.accuracy_score(y_true, y_pred_b)

    if cont_pred:
        
        AUC_score = metrics.roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_b)
        # P : Precision at threshold, R : Recall at threshold, PRthresholds 
        P, R, PRthresholds = metrics.precision_recall_curve(y_true, y_pred)
        AUCPR_score = metrics.average_precision_score(y_true, y_pred)
        
        # convert y_pred to fake probabilities if spatcov is given
        if y_pred.max() > 1 or y_pred.min() < 0: 
            y_pred = (y_pred+abs(y_pred.min()))/( y_pred.max()+abs(y_pred.min()) )
        else:
            y_pred = y_pred
        
        brier_score = metrics.brier_score_loss(y_true, y_pred)
        brier_score_clim = metrics.brier_score_loss(y_true, y_pred_c)
    
    rng_seed = 42  # control reproducibility
    boots_AUC = []
    boots_AUCPR = []
    boots_KSS = []
    boots_brier = []   
    boots_prec = []
    boots_acc = []
    
    
    old_index = range(0,len(y_pred),1)
    n_bl = blocksize
    chunks = [old_index[n_bl*i:n_bl*(i+1)] for i in range(int(len(old_index)/n_bl))]

    rng = np.random.RandomState(rng_seed)
#    random.seed(rng_seed)
    for i in range(n_boot):
        # bootstrap by sampling with replacement on the prediction indices
        ran_ind = rng.randint(0, len(chunks) - 1, len(chunks))
        ran_blok = [chunks[i] for i in ran_ind]
#        indices = random.sample(chunks, len(chunks))
        indices = list(chain.from_iterable(ran_blok))
        
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        out = get_metrics_bin(y_true[indices], y_pred[indices], t=clim_prev)
        (score_prec, recall, FPR, SP, score_acc, f1, score_KSS) = out

        
        if cont_pred:
            score_AUC = metrics.roc_auc_score(y_true[indices], y_pred[indices])
            score_AUCPR = metrics.average_precision_score(y_true[indices], y_pred[indices])
            score_brier = metrics.brier_score_loss(y_true[indices], y_pred[indices])    

        if cont_pred:
            boots_AUC.append(score_AUC)
            boots_AUCPR.append(score_AUCPR)
            boots_brier.append(score_brier)
        boots_prec.append(score_prec)
        boots_acc.append(score_acc)
        boots_KSS.append(score_KSS)
#        print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    def get_ci(boots, alpha=alpha):
        sorted_scores = np.array(boots)
        sorted_scores.sort()
        ci_low = sorted_scores[int(alpha * len(sorted_scores))]
        ci_high = sorted_scores[int((1-alpha) * len(sorted_scores))]
        return ci_low, ci_high, sorted_scores
    
    if len(boots_AUC) != 0:
        if cont_pred:
            ci_low_AUC, ci_high_AUC, sorted_AUCs = get_ci(boots_AUC, alpha)
            
            ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs = get_ci(boots_AUCPR, alpha)
        
            ci_low_brier, ci_high_brier, sorted_briers = get_ci(boots_brier, alpha)

        ci_low_KSS, ci_high_KSS, sorted_KSSs = get_ci(boots_KSS, alpha)
    
        ci_low_prec, ci_high_prec, sorted_precs = get_ci(boots_prec, alpha)
        
        ci_low_acc, ci_high_acc, sorted_accs = get_ci(boots_acc, alpha)
        
        
    else:
        if cont_pred:
            ci_low_AUC, ci_high_AUC, sorted_AUCs = (AUC_score, AUC_score, [AUC_score])
            
            ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs = (AUCPR_score, AUCPR_score, [AUCPR_score])
        
            ci_low_brier, ci_high_brier, sorted_briers = (brier_score, brier_score, [brier_score])
        
        ci_low_KSS, ci_high_KSS, sorted_KSSs = (KSS_score, KSS_score, [KSS_score])
        
        ci_low_prec, ci_high_prec, sorted_precs = (prec, prec, [prec])
        
        ci_low_acc, ci_high_acc, sorted_accs = (acc, acc, [acc])
    
    if cont_pred:
        metrics_dict['AUC'] = (AUC_score, ci_low_AUC, ci_high_AUC, sorted_AUCs)
        metrics_dict['AUCPR'] = (AUCPR_score, ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs)
        metrics_dict['brier'] = (brier_score, brier_score_clim, ci_low_brier, ci_high_brier, sorted_briers)
        metrics_dict['fpr_tpr_thres'] = fpr, tpr, thresholds
        metrics_dict['P_R_thres'] = P, R, PRthresholds
    metrics_dict['KSS'] = (KSS_score, ci_low_KSS, ci_high_KSS, sorted_KSSs)
    metrics_dict['prec'] = (prec, ci_low_prec, ci_high_prec, sorted_precs)
    metrics_dict['acc'] = (acc, ci_low_acc, ci_high_acc, sorted_accs)
    
#    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
#        confidence_lower, confidence_upper))
    #%%
    return metrics_dict

def get_KSS_clim(y_true, y_pred, threshold_clim_events):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    idx_clim_events = np.argmin(abs(thresholds[::-1] - threshold_clim_events))
    KSS_score = tpr[idx_clim_events] - fpr[idx_clim_events]
    return KSS_score 


def corr_matrix_pval(df, alpha=0.05):
    from scipy import stats
    if type(df) == type(pd.DataFrame()):
        cross_corr = np.zeros( (df.columns.size, df.columns.size) )
        pval_matrix = np.zeros_like(cross_corr)
        for i1, col1 in enumerate(df.columns):
            for i2, col2 in enumerate(df.columns):
                pval = stats.pearsonr(df[col1].values, df[col2].values)
                pval_matrix[i1, i2] = pval[-1]
                cross_corr[i1, i2]  = pval[0]
        # recreate pandas cross corr
        cross_corr = pd.DataFrame(data=cross_corr, columns=df.columns, 
                                  index=df.columns)
                
    sig_mask = pval_matrix < alpha
    return cross_corr, sig_mask, pval_matrix

def build_ts_matric(df_init, win=20, lag=0, columns=list, rename=dict, period='fullyear'):
    #%%
    '''
    period = ['fullyear', 'summer60days', 'pre60days']
    
    splits = df_init.index.levels[0]
    dates_full_orig = df_init.loc[0].index
    dates_RV_orig   = df_init.loc[0].index[df_init.loc[0]['RV_mask']==True]
    
    
    if columns is None:
        columns = df_init.columns
        
    df_cols = df_init[columns]
    
    TrainIsTrue = df_init['TrainIsTrue']

    list_test = []
    for s in range(splits.size):
        TestIsTrue = TrainIsTrue[s]==False
        list_test.append(df_cols.loc[s][TestIsTrue])
    
    df_test = pd.concat(list_test).sort_index()


    # shift precursor vs. tmax 
    for c in df_test.columns[1:]:
        df_test[c] = df_test[c].shift(periods=-lag)
         
    # bin means
    df_test = df_test.resample(f'{win}D').mean()
    
    if period=='fullyear':
        dates_sel = dates_full_orig.strftime('%Y-%m-%d')
    elif period == 'summer60days':
        dates_sel = dates_RV_orig.strftime('%Y-%m-%d')
    elif period == 'pre60days':
        dates_sel = (dates_RV_orig - pd.Timedelta(60, unit='d')).strftime('%Y-%m-%d')
    
    # after resampling, not all dates are in their:
    dates_sel =  pd.to_datetime([d for d in dates_sel if d in df_test.index] )
    df_period = df_test.loc[dates_sel, :].dropna()

    if rename is not None:
        df_period = df_period.rename(rename, axis=1)
        
    corr, sig_mask, pvals = corr_matrix_pval(df_period, alpha=0.01)

    # Generate a mask for the upper triangle
    mask_tri = np.zeros_like(corr, dtype=np.bool)
    
    mask_tri[np.triu_indices_from(mask_tri)] = True
    mask_sig = mask_tri.copy()
    mask_sig[sig_mask==False] = True
  
    # removing meaningless row and column
    cols = corr.columns
    corr = corr.drop(cols[0], axis=0).drop(cols[-1], axis=1)
    mask_sig = mask_sig[1:, :-1]
    mask_tri = mask_tri[1:, :-1]
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, n=9, l=30, as_cmap=True)
    
    
    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1E99, center=0,
                square=True, linewidths=.5, 
                 annot=False, annot_kws={'size':30}, cbar=False)
    
    
    sig_bold_labels = sig_bold_annot(corr, mask_sig)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                 annot=sig_bold_labels, annot_kws={'size':30}, cbar=False, fmt='s')
    
    ax.tick_params(axis='both', labelsize=15, 
                   bottom=True, top=False, left=True, right=False,
                   labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    
    ax.set_xticklabels(corr.columns, fontdict={'fontweight':'bold', 
                                               'fontsize':25})
    ax.set_yticklabels(corr.index, fontdict={'fontweight':'bold',
                                               'fontsize':25}, rotation=0) 
    #%%
    return

def sig_bold_annot(corr, pvals):
    corr_str = np.zeros_like( corr, dtype=str ).tolist()
    for i1, r in enumerate(corr.values):
        for i2, c in enumerate(r):
            if pvals[i1, i2] <= 0.05 and pvals[i1, i2] > 0.01:
                corr_str[i1][i2] = '{:.2f}*'.format(c)
            if pvals[i1, i2] <= 0.01:
                corr_str[i1][i2]= '{:.2f}**'.format(c)        
            elif pvals[i1, i2] > 0.05: 
                corr_str[i1][i2]= '{:.2f}'.format(c)
    return np.array(corr_str)              

def plot_score_lags(df_metric, metric, color, lags_tf, clim=None,
                    cv_lines=False, col=0, ax=None):
    #%%

    if ax==None:
        print('ax == None')
        ax = plt.subplot(111)

    if metric == 'BSS':
        y_lim = (-0.4, 0.6)
    elif metric[:3] == 'AUC':
        y_lim = (0,1.0)
    elif metric == 'prec':
        y_lim = (0,1)
        y_b = clim
    y = np.array(df_metric.loc[metric])
    y_min = np.array(df_metric.loc['con_low'])
    y_max = np.array(df_metric.loc['con_high'])
    if cv_lines == True:
        y_cv  = [0]


    x = lags_tf
    if 0 in lags_tf:
        tfreq = 2 * (lags_tf[1] - lags_tf[0])
    else:
        tfreq = (lags_tf[1] - lags_tf[0])
#    tfreq = max([lags_tf[i+1] - lags_tf[i] for i in range(len(lags_tf)-1)])
    
    style = 'solid'
    ax.fill_between(x, y_min, y_max, linestyle='solid', 
                            edgecolor='black', facecolor=color, alpha=0.3)
    ax.plot(x, y, color=color, linestyle=style, 
                    linewidth=2, alpha=1 ) 
    ax.scatter(x, y, color=color, linestyle=style, 
                    linewidth=2, alpha=1 ) 
    if cv_lines == True:
        for f in range(y_cv.shape[1]):
            style = 'dashed'
            ax.plot(x, y_cv[f,:], color=color, linestyle=style, 
                         alpha=0.35 ) 
    ax.set_xlabel('Lead time [days]', fontsize=13, labelpad=0.1)
    ax.grid(b=True, which='major')
#    ax.set_title('{}-day mean'.format(col))
    if min(x) == 1:
        xmin = 0
        xticks = np.arange(min(x), max(x)+1E-9, 10) ; 
        xticks[0] = 1
    elif min(x) == 0:
        xmin = int(tfreq/2)
        xticks = np.arange(xmin, max(x)+1E-9, 10) ; 
        xticks = np.insert(xticks, 0, 0)
    else:
        xticks = np.arange(min(x), max(x)+1E-9, 10) ; 

        
    ax.set_xticks(xticks)
    ax.set_ylim(y_lim)
    ax.set_ylabel(metric)
    if metric == 'BSS':
        y_major = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
        ax.set_yticks(y_major, minor=False)
        ax.set_yticklabels(y_major)
        ax.set_yticks(np.arange(-0.6,0.6+1E-9, 0.1), minor=True)
        ax.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=1)
        ax.text(max(x), 0 - 0.05, 'Benchmark clim. pred.', 
                horizontalalignment='right', fontsize=12,
                verticalalignment='center', 
                rotation=0, rotation_mode='anchor', alpha=0.5)
    elif metric == 'AUC-ROC':
        y_b = 0.5
        ax.set_yticks(np.arange(0.5,1+1E-9, 0.1), minor=True)
        ax.hlines(y=y_b, xmin=min(x), xmax=max(x), linewidth=1) 
    elif metric == 'AUC-PR':
        ax.set_yticks(np.arange(0.5,1+1E-9, 0.1), minor=True)
        y_b = clim
        ax.hlines(y=y_b, xmin=min(x)-int(tfreq/2), xmax=max(x), linewidth=1) 

    if metric in ['AUC-ROC', 'AUC-PR', 'prec']:
        ax.text(max(x), y_b-0.05, 'Benchmark rand. pred.', 
                horizontalalignment='right', fontsize=12,
                verticalalignment='center', 
                rotation=0, rotation_mode='anchor', alpha=0.5)
    if col != 0 :
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)
    
    
        
#    str_freq = str(x).replace(' ' ,'')  
    #%%
    return ax


def rel_curve_base(RV, lags_tf, n_bins=5, col=0, ax=None):
    #%%



    if ax==None:    
        print('ax == None')
        fig, ax = plt.subplots(1, facecolor='white')
        
    ax.set_fc('white')

    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('0.5')  
    ax.grid(b=True, which = 'major', axis='both', color='black',
            linestyle='--', alpha=0.2)
    
    n_bins = 5

    # perfect forecast
    perfect = np.arange(0,1+1E-9,(1/n_bins))
    pos_text = np.array((0.5, 0.52))
    ax.plot(perfect,perfect, color='black', alpha=0.5)
    trans_angle = plt.gca().transData.transform_angles(np.array((45,)),
                                                       pos_text.reshape((1, 2)))[0]
    ax.text(pos_text[0], pos_text[1], 'perfect forecast', fontsize=14,
                   rotation=trans_angle, rotation_mode='anchor')
    obs_clim = RV.prob_clim.mean()[0]
    ax.text(obs_clim+0.2, obs_clim-0.05, 'Obs. clim', 
                horizontalalignment='center', fontsize=14,
         verticalalignment='center', rotation=0, rotation_mode='anchor')
    ax.hlines(y=obs_clim, xmin=0, xmax=1, label=None, color='grey',
              linestyle='dashed')
    ax.vlines(x=obs_clim, ymin=0, ymax=1, label=None, color='grey', 
              linestyle='dashed')
    
    # forecast clim
#    pred_clim = y_pred_all.mean().values
#    ax.vlines(x=np.mean(pred_clim), ymin=0, ymax=1, label=None)
#    ax.vlines(x=np.min(pred_clim), ymin=0, ymax=1, label=None, alpha=0.2)
#    ax.vlines(x=np.max(pred_clim), ymin=0, ymax=1, label=None, alpha=0.2)
    ax.text(np.min(obs_clim)-0.025, obs_clim.mean()+0.3, 'Obs. clim', 
            horizontalalignment='center', fontsize=14,
     verticalalignment='center', rotation=90, rotation_mode='anchor')
    # resolution = reliability line
    BSS_clim_ref = perfect - obs_clim
    dist_perf = (BSS_clim_ref / 2.) + obs_clim
    x = np.arange(0,1+1E-9,1/n_bins)
    ax.plot(x, dist_perf, c='grey')
    def get_angle_xy(x, y):
        import math
        dy = np.mean(dist_perf[1:] - dist_perf[:-1])
        dx = np.mean(x[1:] - x[:-1])
        angle = np.rad2deg(math.atan(dy/dx))
        return angle
    angle = get_angle_xy(x, dist_perf)
    pos_text = (x[int(4/n_bins)], dist_perf[int(2/n_bins)]+0.04)
    trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                      np.array(pos_text).reshape((1, 2)))[0]
#    ax.text(pos_text[0], pos_text[1], 'resolution=reliability', 
#            horizontalalignment='center', fontsize=14,
#     verticalalignment='center', rotation=trans_angle, rotation_mode='anchor')
    # BSS > 0 ares
    ax.fill_between(x, dist_perf, perfect, color='grey', alpha=0.5) 
    ax.fill_betweenx(perfect, x, np.repeat(obs_clim, x.size), 
                    color='grey', alpha=0.5) 
    # Better than random
    ax.fill_between(x, dist_perf, np.repeat(obs_clim, x.size), color='grey', alpha=0.2) 
    if col == 0:
        ax.set_ylabel('Fraction of Positives')
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel('Mean Predicted Value')
    #%%
    return ax, n_bins
    #%%
def rel_curve(RV, y_pred_all, color, lags_tf, n_bins, mean_lags=True, ax=None):
    #%%
    
    if ax==None:    
        print('ax == None')
        ax, n_bins = rel_curve_base(RV, lags_tf)
    
    strategy = 'uniform' # 'quantile' or 'uniform'
    fop = [] ; mpv = []
    for l, lag in enumerate(lags_tf):
        
        fraction_of_positives, mean_predicted_value = calibration_curve(RV.RV_bin, y_pred_all[lag], 
                                                                       n_bins=n_bins, strategy=strategy)
        fop.append(fraction_of_positives)
        mpv.append(mean_predicted_value)
    fop = np.array(fop)
    mpv = np.array(mpv)
    if len(fop.shape) == 2:
        # al bins are present, we can take a mean over lags
        # plot forecast
        mean_mpv = np.mean(mpv, 0) ; mean_fop = np.mean(fop, 0)
        fop_std = np.std(fop, 0)
    else:
        bins = np.arange(0,1+1E-9,1/n_bins)
        b_prev = 0
        dic_mpv = {}
        dic_fop = {}
        for i, b in enumerate(bins[1:]):
            
            list_mpv = []
            list_fop = []
            for i_lags, m_ in enumerate(mpv):
                m_ = list(m_)
                # if mpv falls in bin, it is added to the list, which will added to 
                # the dict_mpv
                
                list_mpv.append([val for val in m_ if (val < b and val > b_prev)])
                list_fop.append([fop[i_lags][m_.index(val)] for idx,val in enumerate(m_) if (val < b and val > b_prev)])
            dic_mpv[i] = flatten(list_mpv)
            dic_fop[i] = flatten(list_fop)
            b_prev = b
        mean_mpv = np.zeros( (n_bins) )
        mean_fop = np.zeros( (n_bins) )
        fop_std  = np.zeros( (n_bins) )
        for k, item in dic_mpv.items():
            mean_mpv[k] = np.mean(item)
            mean_fop[k] = np.mean(dic_fop[k])
            fop_std[k]  = np.std(dic_fop[k])
    
    ax.plot(mean_mpv, mean_fop, label=f'fc lag {lag}') ; 
        
    ax.fill_between(mean_mpv, mean_fop+fop_std, 
                    mean_fop-fop_std, label=None,
                    alpha=0.2) ; 
    color = ax.lines[-1].get_c() # get color
    # determine size freq
    freq = np.histogram(y_pred_all[lag], bins=n_bins)[0]
    n_freq = freq / RV.RV_ts.size
    ax.scatter(mean_mpv, mean_fop, s=n_freq*2000, 
               c=color, alpha=0.5)
        

    #%%        
    return ax

def plot_events(RV, color, n_yrs = 10, col=0, ax=None):
    #%%
#    ax=None

    if type(n_yrs) == int:
        years = []
        sorted_ = RV.freq.sort_values().index
        years.append(sorted_[:int(n_yrs/2)])
        years.append(sorted_[-int(n_yrs/2):])
        years = flatten(years)
        dates_ts = []
        for y in years:
            d = RV.dates_RV[RV.dates_RV.year == y]
            dates_ts.append(d)
        
        dates_ts = pd.to_datetime(flatten(dates_ts))
    else:
        dates_ts = RV.RV_bin.index
        
    if ax==None:    
        print('ax == None')
        fig, ax = plt.subplots(1, subplot_kw={'facecolor':'white'})
        
    else:
        ax.axes.set_facecolor('white')
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('0.5')  
    
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    y = RV.RV_bin.loc[dates_ts]

    years = np.array(years)
    x = np.linspace(0, years.size, dates_ts.size)
    ax.bar(x, y.values.ravel(), alpha=0.75, color='silver', width=0.1, label='events')
    clim_prob = RV.RV_bin.sum() / RV.RV_bin.size
    ax.hlines(clim_prob, x[0], x[-1], label='Clim prob.')
    
    
    means = []
    for chnk in list(chunks(x, int(dates_ts.size/2.))):
        means.append( chnk.mean() )
    ax.margins(0.)
    cold_hot = means
    labels =['Lowest \nn-event years', 'Highest \nn-event years']
    ax.set_xticks(cold_hot)    
    ax.set_xticklabels(labels)
    minor_ticks = np.linspace(0, x.max(), dates_ts.size)
    ax.set_xticks(minor_ticks, minor=True);   
    ax.grid(b=True,which='major', axis='y', color='grey', linestyle='dotted', 
            linewidth=1)
    if col == 0:
        ax.legend(facecolor='white', markerscale=2, handlelength=0.75)
        ax.set_ylabel('Probability', labelpad=-3)
        probs = [f"{int(i*100)}%" for i in np.arange(0,1+1E-9,0.2)]
        ax.set_yticklabels(probs)
    else:
        ax.tick_params(labelleft=False)
        
        
    #%%
    return ax, dates_ts 

def plot_ts(RV, y_pred_all, dates_ts, color, lag_i=1, ax=None):
    #%%

    if ax == None:
        ax, dates_ts = plot_events(RV, color, ax=None)
        
#    dates = y_pred_all.index
    n_yrs = np.unique(dates_ts.year).size
    x = np.linspace(0, n_yrs, dates_ts.size)
    y = y_pred_all.iloc[:,lag_i].loc[dates_ts]
    ax.plot(x, y.values.ravel() , linestyle='solid', marker=None,
            linewidth=1)
    #%%
    return ax

def plot_freq_per_yr(RV):
    #%%
    dates_RV = RV.RV_bin.index
    all_yrs = np.unique(dates_RV.year)
    freq = pd.DataFrame(data= np.zeros(all_yrs.size), 
                        index = all_yrs, columns=['freq'])
    for i, yr in enumerate(all_yrs):
        oneyr = RV.RV_bin.loc[functions_pp.get_oneyr(dates_RV, yr)]
        freq.loc[yr] = oneyr.sum().values
    plt.figure( figsize=(8,6) )
    plt.bar(freq.index, freq['freq'])
    plt.ylabel('Events p/y', fontdict={'fontsize':14})
    #%%

    
def valid_figures(dict_experiments, line_dim='models', group_line_by=None, 
                  met='default', wspace=0.08):
    #%%
    '''
    3 dims to plot: [metrics, experiments, stat_models]
    2 can be assigned to row or col, the third will be lines in the same axes.
    '''
    
    expers = list(dict_experiments.keys())
    models   = list(dict_experiments[expers[0]].keys())
    dims = ['exper', 'models', 'met']
    col_dim = [s for s in dims if s not in [line_dim, 'met']][0]
    if met == 'default':
        met = ['AUC-ROC', 'AUC-PR', 'BSS', 'prec', 'Rel. Curve', 'ts']
    
    
    
    if line_dim == 'models':
        lines = models
        cols  = expers
    elif line_dim == 'exper':
        lines = expers
        cols  = models
    assert line_dim in ['models', 'exper'], ('illegal key for line_dim, '
                           'choose \'exper\' or \'models\'')
        
    if len(cols) == 1 and group_line_by is not None:
        group_s = len(group_line_by)
        cols = group_line_by
        lines_grouped = []
        for i in range(0,len(lines),group_s):
            lines_grouped.append(lines[i:i+group_s])
        

    grid_data = np.zeros( (2, len(met)), dtype=str)
    grid_data = np.stack( [np.repeat(met, len(cols)), 
                           np.repeat(cols, len(met))])

    
    df = pd.DataFrame(grid_data.T, columns=['met', col_dim])
    g = sns.FacetGrid(df, row='met', col=col_dim, size=3, aspect=1.4,
                      sharex=False,  sharey=False)


    
    
    for col, c_label in enumerate(cols):
        
        
        g.axes[0,col].set_title(c_label)
        if len(models) == 1 and group_line_by is not None:
            lines = lines_grouped[col]
        
        
        for row, metric in enumerate(met):
            
            legend = []
            for l, line in enumerate(lines):
                
                if line_dim == 'models':
                    model = line
                    exper = c_label
                    
                elif line_dim == 'exper':
                    model = c_label
                    exper = line
                    if len(models) == 1 and group_line_by is not None:
                        exper = line
                        model = models[0]
                
                    
                
                df_valid, RV, y_pred_all = dict_experiments[exper][model]
                tfreq = (y_pred_all.iloc[1].name - y_pred_all.iloc[0].name).days
                lags_i     = list(dict_experiments[exper][model][2].columns.astype(int))
                lags_tf = [l*tfreq for l in lags_i]
                if tfreq != 1:
                    # the last day of the time mean bin is tfreq/2 later then the centerered day
                    lags_tf = [l_tf- int(tfreq/2) if l_tf!=0 else 0 for l_tf in lags_tf ]
                
                color = nice_colors[l]
                
                if metric in ['AUC-ROC', 'AUC-PR', 'BSS', 'prec']: 
                    df_metric = df_valid.loc[metric]
                    if metric == 'AUC-PR':
                        clim = RV.RV_bin.values[RV.RV_bin==1].size / RV.RV_bin.size
                    else:
                        clim = None
                    plot_score_lags(df_metric, metric, color, lags_tf,
                                    clim, cv_lines=False, col=col,
                                    ax=g.axes[row,col])
                if metric == 'Rel. Curve':
                    if l == 0:
                        ax, n_bins = rel_curve_base(RV, lags_tf, col=col, ax=g.axes[row,col])
                    print(l,line)
                    
                    rel_curve(RV, y_pred_all, color, lags_i, n_bins, 
                              mean_lags=True, 
                              ax=g.axes[row,col])
                    
                if metric == 'ts':
                    if l == 0:
                        ax, dates_ts = plot_events(RV, color=nice_colors[-1], n_yrs=6, 
                                         col=col, ax=g.axes[row,col])
                    plot_ts(RV, y_pred_all, dates_ts, color, lag_i=1, ax=g.axes[row,col])
                
                # legend conditions
                same_models = np.logical_and(row==0, col==0)
                grouped_lines = np.logical_and(row==0, group_line_by is not None)
                if same_models or grouped_lines:
                    legend.append(patches.Rectangle((0,0),0.5,0.5,facecolor=color))

                    g.axes[row,col].legend(tuple(legend), lines, 
                          loc = 'lower left', fancybox=True,
                          handletextpad = 0.2, markerscale=0.1,
                          borderaxespad = 0.1,
                          handlelength=1, handleheight=1, prop={'size': 12})
    
    #%%
    g.fig.subplots_adjust(wspace=wspace)
                    
    return g.fig
                
                
                
