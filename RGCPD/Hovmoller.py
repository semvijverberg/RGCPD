#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:24:50 2020

@author: semvijverberg
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import plot_maps
import functions_pp
import find_precursors

class Hovmoller:

    def __init__(self, kwrgs_load: dict=None, slice_dates: tuple=(str, str),
                 event_dates: pd.DatetimeIndex=None, max_lag_vs_events: int=None,
                 name=None, n_cpu=1):
        '''
        selbox has format of (lon_min, lon_max, lat_min, lat_max)
        '''
        self.kwrgs_load = kwrgs_load
        self.slice_dates = slice_dates
        self.event_dates = event_dates
        self.kwrgs_load['sel_dates'] = self.event_dates
        self.max_lag_vs_events = max_lag_vs_events
        self.name = name
        self.n_cpu = n_cpu
        return

    def get_data(self, filepath):
        self.filepath = filepath
        ds = functions_pp.import_ds_timemeanbins(self.filepath, **self.kwrgs_load)