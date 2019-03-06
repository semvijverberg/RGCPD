#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:58:14 2019

@author: semvijverberg
"""
#%%
import os
import pandas as pd
import xarray as xr
import numpy as np
# load jetlat data and save to .npy as xarray format

path_pp  = os.path.join("/Users/semvijverberg/surfdrive/Data_ERAint/", 
                        'input_pp')
filename = 'ERA_I_jet_data_NEWversion_1979_2018.csv'
path = os.path.join(path_pp, 'RVts2.5', filename)
data = pd.read_csv(path, index_col='Date', sep=';')

# match dates of netcdfs of my own

var_class_dates = pd.to_datetime(var_class.datesstr_fit_tfreq)
netcdfdates = var_class_dates.strftime('%Y-%m-%d')
data = data.loc[netcdfdates]

dates = pd.to_datetime(data.index.values)
latdata = data['Lat'].values

xrdata = xr.DataArray(data=latdata, coords=[dates], dims=['time'])

s = dates[0]
e = dates[-1]
filename = 'jetlat_{}-{}_{}_{}'.format(s.year, e.year,
                   s.strftime('%m-%d'), e.strftime('%m-%d'))
to_dict = dict( {'RVfullts'     : xrdata } )
np.save(os.path.join(path_pp, 'RVts2.5', filename+'.npy'), to_dict)



