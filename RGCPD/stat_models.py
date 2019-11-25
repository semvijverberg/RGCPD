#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:01:23 2019

@author: semvijverberg
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import make_scorer

def logit(RV, df_norm, keys):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    if keys is None:
        no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
        keys = df_norm.columns
        keys = [k for k in keys if k not in no_data_col]
        
    X = df_norm[keys]
    X = add_constant(X)
    y = RV.RV_bin_fit
    # Get training years
    TrainIsTrue = df_norm['TrainIsTrue'] 
    # Get mask to make only prediction for RV_mask dates
    pred_mask   = df_norm['RV_mask']

    model_set = sm.Logit(y['RV_binary'][TrainIsTrue] , 
                         X[TrainIsTrue], disp=0)
    try:
        model = model_set.fit( disp=0, maxfun=60 )
        prediction = model.predict(X[pred_mask])
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            model = model_set.fit(method='bfgs', disp=0 )
            prediction = model.predict(X[pred_mask])
        else:
            raise
    except Exception as e:
        print(e)
        model = model_set.fit(method='bfgs', disp=0 )
        prediction = model.predict(X)
    #%%
    return prediction, model

def GBR(RV, df_norm, keys=None, kwrgs_GBR=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    
    if keys is None:
        no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
        keys = df_norm.columns
        keys = [k for k in keys if k not in no_data_col]
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':3,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_ts = RV.RV_ts_fit
    # Get training years
    TrainIsTrue = df_norm['TrainIsTrue'] 
    # Get mask to make only prediction for RV_mask dates
    pred_mask   = df_norm['RV_mask']
  
    X_train = X[TrainIsTrue]
    y_train = RV_ts[TrainIsTrue.values] 
    
    # add sample weight mannually
#    y_train[y_train > y_train.mean()] = 10 * y_train[y_train > y_train.mean()]
    
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    regressor = GradientBoostingRegressor(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
#        scoring   = 'r2'
        scoring   = 'neg_mean_squared_error'
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=5, refit=scoring, 
                  return_train_score=False)
        regressor.fit(X_train, y_train.values.ravel())
        results = regressor.cv_results_
        scores = results['mean_test_score'] 
        improv = int(100* (min(scores)-max(scores)) / max(scores))
        print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
              "best params {}".format(
                improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    

    prediction = pd.DataFrame(regressor.predict(X[pred_mask]),
                              index=X[pred_mask].index, columns=[0])
    prediction['TrainIsTrue'] = pd.Series(TrainIsTrue.values, index=X.index)
    prediction['RV_mask'] = pd.Series(pred_mask.values, index=pred_mask.index)
    
    logit_pred, model_logit = logit(RV, prediction, keys=None)
    #%%
    return logit_pred, (model_logit, regressor)


def GBR_logitCV(RV, df_norm, keys, kwrgs_GBR=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':1,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_ts = RV.RV_ts_fit
    # Get training years
    TrainIsTrue = df_norm['TrainIsTrue'] 
    # Get mask to make only prediction for RV_mask dates
    pred_mask   = df_norm['RV_mask']
  
    X_train = X[TrainIsTrue.values]
    y_train = RV_ts[TrainIsTrue.values] 
    # add sample weight mannually
#    y_train[y_train > y_train.mean()] = 10 * y_train[y_train > y_train.mean()]
    
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    regressor = GradientBoostingRegressor(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
#        scoring   = 'r2'
        scoring   = 'neg_mean_squared_error'
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=5, refit=scoring, 
                  return_train_score=False)
        regressor.fit(X_train, y_train.values.ravel())
        if verbosity == 1:
            results = regressor.cv_results_
            scores = results['mean_test_score'] 
            improv = int(100* (min(scores)-max(scores)) / max(scores))
            print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
                  "best params {}".format(
                    improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    
        
    prediction = pd.DataFrame(regressor.predict(X), 
                              index=X.index, columns=[0])
    prediction['TrainIsTrue'] = pd.Series(TrainIsTrue.values, index=X.index)
    prediction['RV_mask'] = pd.Series(pred_mask.values, index=X.index)
    logit_pred, model_logit = logit_skl(RV, prediction, keys=None)
    
    
    
#    logit_pred.plot() ; plt.plot(RV.RV_bin)
#    plt.figure()
#    prediction.plot() ; plt.plot(RV.RV_ts)
#    metrics_sklearn(RV.RV_bin, logit_pred.values, y_pred_c)
    #%%
    return logit_pred, (model_logit, regressor)


def logit_skl(RV, df_norm, keys, kwrgs_logit=None):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''

    if keys is None:
            no_data_col = ['TrainIsTrue', 'RV_mask', 'fit_model_mask']
            keys = df_norm.columns
            keys = [k for k in keys if k not in no_data_col]
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
    if kwrgs_logit == None:
        # use Bram settings
        kwrgs_logit = { 'class_weight':{ 0:1, 1:1},
                'scoring':'brier_score_loss',
                'penalty':'l2',
                'solver':'lbfgs'}
                             
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_logit.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_logit.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X.values)
    RV_bin = RV.RV_bin_fit
    # Get training years
    TrainIsTrue = df_norm['TrainIsTrue'] 
    # Get mask to make only prediction for RV_mask dates
    pred_mask   = df_norm['RV_mask'].values
  
    X_train = X[TrainIsTrue.values] 
    y_train = RV_bin[TrainIsTrue.values].squeeze().values 
#    RV_ts_train = RV.RV_ts[TrainIsTrue.values] 
#    high_ano = metrics.roc_auc_score(RV.RV_bin.squeeze().values, RV.RV_ts.values)
#    if high_ano == 1.0:
#        # add counter classes below mean to improve resolution (discriminative power)
#        prev = y_train[y_train==1].size / y_train.size
#        mask_bm = (RV_ts_train < prev).values
#        mask_bm = (RV_ts_train < RV_ts_train.mean()).values
#        y_train[mask_bm] = -1
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    strat_cv = StratifiedKFold(5, shuffle=False)
    model = LogisticRegressionCV(Cs=10, fit_intercept=True, 
                                 cv=strat_cv,
                                 n_jobs=2, 
                                 **kwrgs_logit)
                                 
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X[pred_mask])[:,1]
#    regressor = GradientBoostingClassifier(**kwrgs)
#
#    if len(kwrgs_gridsearch) != 0:
##        brier_loss= 'brier_score_loss'
#        loss  = metrics.brier_score_loss
#        scoring  = make_scorer(loss, greater_is_better=False, needs_proba=True)
#        regressor = GridSearchCV(regressor,
#                  param_grid=kwrgs_gridsearch,
#                  scoring=scoring, cv=5, iid=False,
#                  return_train_score=False)
#        regressor.fit(X_train, y_train)
#        results = regressor.cv_results_
#        scores = results['mean_test_score']
#        improv = int(100* (min(scores)-max(scores)) / max(scores))
#        print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
#              "best params {}".format(
#                improv, regressor.best_score_, regressor.best_params_))
#    else:
#        regressor.fit(X_train, y_train)
    
#    y_pred = regressor.predict(X)
#    y_pred[y_pred!=1] = 0
    prediction = pd.DataFrame(y_pred, index=df_norm[pred_mask].index, columns=[0])

    #%%
    return prediction, model


def GBR_logit(RV, df_norm, keys, kwrgs_GBR=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':1,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_ts = RV.RV_ts_fit
    # Get training years
    TrainIsTrue = df_norm['TrainIsTrue'] 
    # Get mask to make only prediction for RV_mask dates
    pred_mask   = df_norm['RV_mask']
  
    X_train = X[TrainIsTrue]
    y_train = RV_ts[TrainIsTrue.values] 
    events_mask = RV.RV_bin.astype(bool).values
    
    mean = RV_ts.mean()
    y_train[y_train <= mean] = 0
    y_train[y_train >= mean] = 1
    y_train[events_mask[TrainIsTrue].ravel()] = 2

    
    regressor = GradientBoostingRegressor(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
#        scoring   = 'r2'
        scoring   = 'neg_mean_squared_error'
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=5, refit=scoring, 
                  return_train_score=False)
        regressor.fit(X_train, y_train.values.ravel())
        if verbosity == 1:
            results = regressor.cv_results_
            scores = results['mean_test_score'] 
            improv = int(100* (min(scores)-max(scores)) / max(scores))
            print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
                  "best params {}".format(
                    improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train.values.ravel())
    

    prediction = pd.DataFrame(regressor.predict(X[pred_mask]), 
                              index=X[pred_mask].index, columns=[0])
    prediction['TrainIsTrue'] = pd.Series(TrainIsTrue.values, index=X.index)
    prediction['RV_mask'] = pd.Series(pred_mask.values, index=X.index)
    
    logit_pred, model_logit = logit(RV, prediction, keys=[None])
    #%%
    return logit_pred, (model_logit, regressor)

def GBC(RV, df_norm, keys, kwrgs_GBR=None, verbosity=0):
    #%%
    '''
    X contains all precursor data, incl train and test
    X_train, y_train are split up by TrainIsTrue
    Preciction is made for whole timeseries    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
    if kwrgs_GBR == None:
        # use Bram settings
        kwrgs_GBR = {'max_depth':1,
                 'learning_rate':0.001,
                 'n_estimators' : 1250,
                 'max_features':'sqrt',
                 'subsample' : 0.5}
    
    # find parameters for gridsearch optimization
    kwrgs_gridsearch = {k:i for k, i in kwrgs_GBR.items() if type(i) == list}
    # only the constant parameters are kept
    kwrgs = kwrgs_GBR.copy()
    [kwrgs.pop(k) for k in kwrgs_gridsearch.keys()]
    
    X = df_norm[keys]
    X = add_constant(X)
    RV_bin = RV.RV_bin_fit
    TrainIsTrue = df_norm['TrainIsTrue']
  
    X_train = X[TrainIsTrue]
    y_train = RV_bin[TrainIsTrue.values].squeeze().values 
#    RV_ts_train = RV.RV_ts[TrainIsTrue.values] 
#    high_ano = metrics.roc_auc_score(RV.RV_bin.squeeze().values, RV.RV_ts.values)
#    if high_ano == 1.0:
#        # add counter classes below mean to improve resolution (discriminative power)
#        prev = y_train[y_train==1].size / y_train.size
#        mask_bm = (RV_ts_train < prev).values
#        mask_bm = (RV_ts_train < RV_ts_train.mean()).values
#        y_train[mask_bm] = -1
    # sample weight not yet supported by GridSearchCV (august, 2019)
#    y_wghts = (RV.RV_bin[TrainIsTrue.values] + 1).squeeze().values
    regressor = GradientBoostingClassifier(**kwrgs)

    if len(kwrgs_gridsearch) != 0:
#        brier_loss= 'brier_score_loss'
        loss  = metrics.brier_score_loss
        scoring  = make_scorer(loss, greater_is_better=False, needs_proba=True)
        strat_cv = StratifiedKFold(5, shuffle=False)
        regressor = GridSearchCV(regressor,
                  param_grid=kwrgs_gridsearch,
                  scoring=scoring, cv=strat_cv, iid=False,
                  return_train_score=False)
        regressor.fit(X_train, y_train)
        if verbosity == 1:
            results = regressor.cv_results_
            scores = results['mean_test_score'] 
            improv = int(100* (min(scores)-max(scores)) / max(scores))
            print("Hyperparam tuning led to {:}% improvement, best {:.2f}, "
                  "best params {}".format(
                    improv, regressor.best_score_, regressor.best_params_))
    else:
        regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X)
    y_pred[y_pred!=1] = 0
    prediction = pd.DataFrame(y_pred, index=X.index, columns=[0])

    #%%
    return prediction, regressor