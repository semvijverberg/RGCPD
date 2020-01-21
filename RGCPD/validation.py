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
from sklearn import metrics
import seaborn as sns
from itertools import chain
flatten = lambda l: list(chain.from_iterable(l))
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
max_cpu = multiprocessing.cpu_count()


from matplotlib import cycler
nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
colors_nice = cycler('color',
                nice_colors)
colors_datasets = sns.color_palette('deep')

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


            
            
def get_metrics_sklearn(y_true, y_pred_all, y_pred_c, alpha=0.05, n_boot=5, 
                        blocksize=10, threshold_pred='upper_clim'):
                        
    #%%

    clim_prob = y_true[y_true==1].size / y_true.size
    lags = y_pred_all.columns
    cont_pred = np.unique(y_pred_all).size > 5

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
                          index=['Precision', 'con_low', 'con_high'])

    df_acc  = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                          index=['Accuracy', 'con_low', 'con_high'])

    df_EDI  = pd.DataFrame(data=np.zeros( (3, len(lags)) ), columns=[lags],
                      index=['EDI', 'con_low', 'con_high'])

    for lag in lags:
        y_pred = y_pred_all[[lag]].values

        metrics_dict = metrics_sklearn(y_true, y_pred, y_pred_c.values,
                                       alpha=alpha, n_boot=n_boot, blocksize=blocksize,
                                       clim_prob=clim_prob,
                                       threshold_pred=threshold_pred)
        if cont_pred:
            # AUC
            AUC_score, conf_lower, conf_upper, sorted_AUC = metrics_dict['AUC']
            df_auc[[lag]] = (AUC_score, conf_lower, conf_upper)
            # AUC Precision-Recall
            AUCPR_score, ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs = metrics_dict['AUCPR']
            df_aucPR[[lag]] = (AUCPR_score, ci_low_AUCPR, ci_high_AUCPR)
            # Brier score
            brier_score, brier_clim, ci_low_brier, ci_high_brier, sorted_briers = metrics_dict['brier']
            BSS = (brier_clim - brier_score) / brier_clim
            BSS_low = (brier_clim - ci_high_brier) / brier_clim
            BSS_high = (brier_clim - ci_low_brier) / brier_clim
            df_brier[[lag]] = (BSS, BSS_low, BSS_high,
                            brier_score, ci_low_brier, ci_high_brier, brier_clim)
        # HKSS
        KSS_score, ci_low_KSS, ci_high_KSS, sorted_KSSs = metrics_dict['KSS']
        df_KSS[[lag]] = (KSS_score, ci_low_KSS, ci_high_KSS)
        # Precision
        prec, ci_low_prec, ci_high_prec, sorted_precs = metrics_dict['prec']
        df_prec[[lag]] = (prec, ci_low_prec, ci_high_prec)
        # Accuracy
        acc, ci_low_acc, ci_high_acc, sorted_accs = metrics_dict['acc']
        df_acc[[lag]] = (acc, ci_low_acc, ci_high_acc)
        # EDI
        EDI, ci_low_EDI, ci_high_EDI, sorted_EDIs = metrics_dict['EDI']
        df_EDI[[lag]] = EDI, ci_low_EDI, ci_high_EDI

    if cont_pred:
        df_valid = pd.concat([df_brier, df_auc, df_aucPR, df_KSS, df_prec, df_acc, df_EDI],
                         keys=['BSS', 'AUC-ROC', 'AUC-PR', 'KSS', 'Precision', 'Accuracy', 'EDI'])
#        print("ROC area\t: {:0.3f}".format( float(df_auc.iloc[0][0]) ))
#        print("P-R area\t: {:0.3f}".format( float(df_aucPR.iloc[0][0]) ))
#        print("BSS     \t: {:0.3f}".format( float(df_brier.iloc[0][0]) ))

    else:
        df_valid = pd.concat([df_KSS, df_prec, df_acc],
                         keys=['KSS', 'Precision', 'Accuracy'])
#    print("Precision       : {:0.3f}".format( float(df_prec.iloc[0][0]) ))
#    print("Accuracy        : {:0.3f}".format( float(df_acc.iloc[0][0]) ))


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
#    y_true = np.repeat(1, 100); y_pred_b = np.repeat(1, 100)
#    y_pred_b[-1] = 0
    try:
        if np.sum(y_true) != 0 or np.sum(y_pred_b) != 0:
            prec = metrics.precision_score(y_true, y_pred_b)
            recall = metrics.recall_score(y_true, y_pred_b) # recall is TPR
        else:
            prec = 0.
            recall = 0.
    except:
        print(y_true)
        print(y_pred_b)
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_b).ravel()
    if (fp + tn) != 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.
    if (tn + fp) != 0:
        SP = tn / (tn + fp) # specifity / true negative rate
    else:
        SP = 1.
    Acc  = metrics.accuracy_score(y_true, y_pred_b)
    if (tp + fp) == 0 or (tp + fn) == 0:
        f1 = 0
    else:
        f1  = metrics.f1_score(y_true, y_pred_b)
    # Hansen Kuiper score (TPR - FPR):
#     ; fpr = fp / (fp+tn)
    tpr = recall
    KSS_score = tpr - fpr
    # Extremal Dependence Index (EDI) from :
    # Higher Subseasonal Predictability of Extreme Hot European Summer Temperatures as Compared to Average Summers, 2019
    # EDI = log(FPR) - log(TPR) / ( log(FPR) + log(TPR) )
    if fpr != 1. and fpr != 0. and tpr != 1. and tpr != 0.:
        EDI = ( np.log(fpr) - np.log(tpr) ) / (np.log(fpr) + np.log(tpr) )
    else:
        EDI = 1.
    return prec, recall, fpr, SP, Acc, f1, KSS_score, EDI

def get_metrics_confusion_matrix(RV, y_pred_all, thr=['clim', 33, 66], n_shuffle=0):
    #%%
    lags = y_pred_all.columns
    y_true = RV.RV_bin.squeeze().values
    if thr[0] == 'clim':
        clim_prob = np.round((1-y_true[y_true.values==1.].size / y_true.size),2)
        thresholds = [[clim_prob]]
    else:
        thresholds = []
    perc_thr = [t/100. if type(t) == int else t for t in thr ]
    thresholds.append([t for t in perc_thr])
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
            y_pred = y_pred_all[[lag]].values
            y_pred_b = np.array(y_pred > np.percentile(y_pred, 100*t),dtype=int)


            out = get_metrics_bin(y_true, y_pred_b, t=None)
            (prec_f, recall_f, FPR_f, SP_f, Acc_f, f1_f, KSS, EDI) = out
            # shuffle the predictions
            prec = [] ; recall = [] ; FPR = [] ; SP = [] ; Acc = [] ; f1 = []
            for i in range(n_shuffle):
                np.random.shuffle(y_pred_b);
                out = get_metrics_bin(y_true, y_pred_b, t=None)
                (prec_s, recall_s, FPR_s, SP_s, Acc_s, f1_s, KSS, EDI) = out
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
            if 'clim' in thr:
                df_lag = df_lag.rename(columns={clim_prob: 'clim'})
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
        plot_ac(ac, con_int, s='auto', ax=None)

    where = np.where(con_int[:,0] < 0 )[0]
    # has to be below 0 for n times (not necessarily consecutive):
    n_of_times = np.array([idx+1 - where[0] for idx in where])
    cutoff = where[np.where(n_of_times == n)[0][0] ]
    return cutoff


def metrics_sklearn(y_true=np.ndarray, y_pred=np.ndarray, y_pred_c=np.ndarray,
                    alpha=0.05, n_boot=5, blocksize=1, clim_prob=None, threshold_pred='upper_clim'):
    '''
    threshold_pred  options: 'clim', 'upper_clim', 'int or float'
                    if 'clim' is passed then all positive prediction is forecasted
                    for all values of y_pred above clim_prob
                    if 'upper_clim' is passed, from all values that are above 
                    the clim_prob, only the upper 75% of the prediction is used
                    
    '''
    
    #    y_true, y_pred, y_pred_c = y_true_c, ts_logit_c, y_pred_c_c
    #%%

    y_true = np.array(y_true).squeeze()
    cont_pred = np.unique(y_pred).size > 5
    metrics_dict = {}
    
    if clim_prob is None:
        clim_prob = np.round((y_true[(y_true==1)].size / y_true.size),2)
    
    sorval = np.array(sorted(y_pred))
    # probability to percentile
    if threshold_pred == 'clim':
        # binary metrics calculated for clim prevailance
        quantile = 1 - y_pred[sorval > clim_prob].size / y_pred.size
        # old : quantile = 100 * clim_prob
    elif threshold_pred == 'upper_clim':
        # binary metrics calculated for top 75% of 'above clim prob'
        No_vals_above_clim = y_pred[sorval > clim_prob].size / y_pred.size
        upper_75 = 0.75 * No_vals_above_clim # 0.75 * percentage values above clim
        quantile = 1-upper_75
        # old: bin_threshold = 100 * (1 - 0.75*clim_prob)
        # old:  quantile = bin_threshold
    elif isinstance(threshold_pred, int) or isinstance(threshold_pred, float):
        if threshold_pred < 1:
            quantile = 1 - y_pred[sorval > threshold_pred].size / y_pred.size
        else:
            quantile = 1 - y_pred[sorval > threshold_pred/100.].size / y_pred.size
    elif isinstance(threshold_pred, tuple):
        times = threshold_pred[0]
        quantile = 1 - (y_pred[sorval > times*clim_prob].size / y_pred.size) 
    percentile_t = 100 * quantile 

    y_pred_b = np.array(y_pred > np.percentile(y_pred, percentile_t),dtype=int)


    out = get_metrics_bin(y_true, y_pred, t=percentile_t)
    (prec, recall, FPR, SP, Acc, f1, KSS_score, EDI) = out
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
 
      
    
    if n_boot > 0:
        old_index = range(0,len(y_pred),1)
        n_bl = blocksize
        chunks = [old_index[n_bl*i:n_bl*(i+1)] for i in range(int(len(old_index)/n_bl))]
        
        
        # divide subchunks to boostrap to all cpus
        n_boot_sub = int(round((n_boot / max_cpu) + 0.4, 0))
        try:
            with ProcessPoolExecutor(max_workers=max_cpu) as pool:
                futures = []
                unique_seed = 42    
                for i_cpu in range(max_cpu):
                    unique_seed += 1 # ensure that no same shuffleling is done
                    futures.append(pool.submit(_bootstrap, y_true, y_pred, n_boot_sub, 
                                               chunks, percentile_t, unique_seed))
                out = [future.result() for future in futures]
        except:
            print('parallel bootstrapping failed')
            unique_seed = 42  
            out = []
            for i_cpu in range(max_cpu):
                unique_seed += 1 # ensure that no same shuffleling is done
                out.append(_bootstrap(y_true, y_pred, n_boot_sub, 
                                           chunks, percentile_t, unique_seed))
    
        
    
    boots_AUC = []
    boots_AUCPR = []
    boots_brier = []
    boots_prec = []
    boots_acc = []
    boots_KSS = []
    boots_EDI = []     
    if n_boot > 0:
        for i_cpu in range(max_cpu):
            _AUC, _AUCPR, _brier, _prec, _acc, _KSS, _EDI = out[i_cpu]
            boots_AUC.append(_AUC)
            boots_AUCPR.append(_AUCPR)
            boots_brier.append(_brier)
            boots_prec.append(_prec)
            boots_acc.append(_acc)
            boots_KSS.append(_KSS)
            boots_EDI.append(_EDI)
        
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    def get_ci(boots, alpha=0.05):
        if len(np.array(boots).shape) == 2:
            boots = flatten(boots)
        sorted_scores = np.array(boots)
        sorted_scores.sort()
        ci_low = sorted_scores[int(alpha * len(sorted_scores))]
        ci_high = sorted_scores[int((1-alpha) * len(sorted_scores))]
        return ci_low, ci_high, sorted_scores

    if np.array(boots_AUC).ravel().size != 0:
        if cont_pred:
            ci_low_AUC, ci_high_AUC, sorted_AUCs = get_ci(boots_AUC, alpha)

            ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs = get_ci(boots_AUCPR, alpha)

            ci_low_brier, ci_high_brier, sorted_briers = get_ci(boots_brier, alpha)

        ci_low_KSS, ci_high_KSS, sorted_KSSs = get_ci(boots_KSS, alpha)

        ci_low_prec, ci_high_prec, sorted_precs = get_ci(boots_prec, alpha)

        ci_low_acc, ci_high_acc, sorted_accs = get_ci(boots_acc, alpha)

        ci_low_EDI, ci_high_EDI, sorted_EDIs = get_ci(boots_EDI, alpha)


    else:
        if cont_pred:
            ci_low_AUC, ci_high_AUC, sorted_AUCs = (AUC_score, AUC_score, [AUC_score])

            ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs = (AUCPR_score, AUCPR_score, [AUCPR_score])

            ci_low_brier, ci_high_brier, sorted_briers = (brier_score, brier_score, [brier_score])

        ci_low_KSS, ci_high_KSS, sorted_KSSs = (KSS_score, KSS_score, [KSS_score])

        ci_low_prec, ci_high_prec, sorted_precs = (prec, prec, [prec])

        ci_low_acc, ci_high_acc, sorted_accs = (acc, acc, [acc])

        ci_low_EDI, ci_high_EDI, sorted_EDIs = (EDI, EDI, [EDI])

    if cont_pred:
        metrics_dict['AUC'] = (AUC_score, ci_low_AUC, ci_high_AUC, sorted_AUCs)
        metrics_dict['AUCPR'] = (AUCPR_score, ci_low_AUCPR, ci_high_AUCPR, sorted_AUCPRs)
        metrics_dict['brier'] = (brier_score, brier_score_clim, ci_low_brier, ci_high_brier, sorted_briers)
        metrics_dict['fpr_tpr_thres'] = fpr, tpr, thresholds
        metrics_dict['P_R_thres'] = P, R, PRthresholds
    metrics_dict['KSS'] = (KSS_score, ci_low_KSS, ci_high_KSS, sorted_KSSs)
    metrics_dict['prec'] = (prec, ci_low_prec, ci_high_prec, sorted_precs)
    metrics_dict['acc'] = (acc, ci_low_acc, ci_high_acc, sorted_accs)
    metrics_dict['EDI'] = EDI, ci_low_EDI, ci_high_EDI, sorted_EDIs

#    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
#        confidence_lower, confidence_upper))
    #%%
    return metrics_dict

def _bootstrap(y_true, y_pred, n_boot_sub, chunks, percentile_t, rng_seed):  
    rng = np.random.RandomState(rng_seed)
    boots_AUC = []
    boots_AUCPR = []
    boots_brier = []        
    boots_prec = []
    boots_acc = []
    boots_KSS = []
    boots_EDI = [] 
    for i in range(n_boot_sub):        
        # bootstrap by sampling with replacement on the prediction indices
        ran_ind = rng.randint(0, len(chunks) - 1, len(chunks))
        ran_blok = [chunks[i] for i in ran_ind]
#        indices = random.sample(chunks, len(chunks))
        indices = list(chain.from_iterable(ran_blok))

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        cont_pred = np.unique(y_pred).size > 5
        if cont_pred:
            score_AUC = metrics.roc_auc_score(y_true[indices], y_pred[indices])
            score_AUCPR = metrics.average_precision_score(y_true[indices], y_pred[indices])
            score_brier = metrics.brier_score_loss(y_true[indices], y_pred[indices])

            boots_AUC.append(score_AUC)
            boots_AUCPR.append(score_AUCPR)
            boots_brier.append(score_brier)
        
        out = get_metrics_bin(y_true[indices], y_pred[indices], t=percentile_t)
        (score_prec, recall, FPR, SP, score_acc, f1, score_KSS, score_EDI) = out
        boots_prec.append(score_prec)
        boots_acc.append(score_acc)
        boots_KSS.append(score_KSS)
        boots_EDI.append(score_EDI)
    return (boots_AUC, boots_AUCPR, boots_brier, boots_prec, boots_acc, boots_KSS, boots_EDI)

def get_KSS_clim(y_true, y_pred, threshold_clim_events):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    idx_clim_events = np.argmin(abs(thresholds[::-1] - threshold_clim_events))
    KSS_score = tpr[idx_clim_events] - fpr[idx_clim_events]
    return KSS_score


def get_testyrs(df_splits):
    #%%
    traintest_yrs = []
    splits = df_splits.index.levels[0]
    for s in splits:
        df_split = df_splits.loc[s]
        test_yrs = np.unique(df_split[df_split['TrainIsTrue']==False].index.year)
        traintest_yrs.append(test_yrs)
    return traintest_yrs


    
