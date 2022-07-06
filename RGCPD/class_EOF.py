#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:16:56 2020

@author: semvijverberg
"""

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import find_precursors, functions_pp, plot_maps


class EOF:

    def __init__(self, tfreq_EOF='monthly', neofs=1, selbox=None,
                 name=None, start_end_date: tuple=None,
                 start_end_year: tuple=None, n_cpu=1):
        '''
        selbox has format of (lon_min, lon_max, lat_min, lat_max)
        '''
        self.tfreq_EOF = tfreq_EOF
        self.neofs = neofs
        self.name = name
        self.selbox = selbox
        self.n_cpu = n_cpu
        self.start_end_date = start_end_date
        self.start_end_year = start_end_year
        return

    def get_pattern(self, filepath, df_splits=None):
        # filepath = '/Users/semvijverberg/surfdrive/ERA5/input_raw/preprocessed/sst_1979-2018_1jan_31dec_daily_2.5deg.nc'
        self.filepath = filepath


        if self.tfreq_EOF == 'monthly':
            self.ds_EOF = functions_pp.import_ds_timemeanbins(self.filepath, tfreq=1,
                                                     selbox=self.selbox,
                                                     dailytomonths=True,
                                                     start_end_date=self.start_end_date,
                                                     start_end_year=self.start_end_year)
        elif self.tfreq_EOF == 'daily':
            self.ds_EOF = functions_pp.import_ds_timemeanbins(self.filepath,
                                                          tfreq=self.tfreq_EOF,
                                                          selbox=self.selbox,
                                                          start_end_date=self.start_end_date,
                                                          start_end_year=self.start_end_year,
                                                          closed_on_date=self.start_end_date[-1])
        else:
            self.ds_EOF = self.filepath

        if self.name is None:
            if hasattr(self.ds_EOF, 'name'):
                # take name of variable
                self.name = self.ds_EOF.name

        if df_splits is None:
            print('no train test splits for fitting EOF')
            data = np.zeros( (1, self.neofs, self.ds_EOF.latitude.size, self.ds_EOF.longitude.size) )
            coords = [[0], [f'EOF{n}_'+self.name for n in range(self.neofs)],
                      self.ds_EOF.latitude.values, self.ds_EOF.longitude.values]
            self.eofs = xr.DataArray(data,
                                coords=coords,
                                dims = ['split', 'eof', 'latitude', 'longitude'])
            solvers = []
            self.eofs[0,:], solver = self._get_EOF_xarray(self.ds_EOF, self.neofs)
            solvers.append(solver)
            self.solvers = solvers
        else:
            self.df_splits = df_splits
            splits = df_splits.index.levels[0]
            func = self._get_EOF_xarray
            if self.n_cpu > 1:
                try:
                    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                        futures = []
                        for s in range(splits.size):
                            progress = int(100 * (s+1) / splits.size)
                            print(f"\rProgress traintest set {progress}%", end="")
                            futures.append(pool.submit(self._single_split, func,
                                                      self.ds_EOF, s, df_splits,
                                                      self.neofs))
                            results = [future.result() for future in futures]
                        pool.shutdown()
                except:
                    results = [self._single_split(func, self.ds_EOF, s, df_splits, self.neofs) for s in range(splits.size)]
            else:
                results = [self._single_split(func, self.ds_EOF, s, df_splits, self.neofs) for s in range(splits.size)]
            # unpack results
            data = np.zeros( (splits.size, self.neofs, self.ds_EOF.latitude.size,
                              self.ds_EOF.longitude.size) )
            coords = [splits, [f'0..{n+1}..EOF_'+self.name for n in range(self.neofs)],
                      self.ds_EOF.latitude.values, self.ds_EOF.longitude.values]
            self.eofs = xr.DataArray(data,
                                    coords=coords,
                                    dims = ['split', 'eof', 'latitude', 'longitude'])
            solvers = []
            for s in splits:
                self.eofs[s,:] = results[s][0]
                solvers.append(results[s][1])
                # ensure same sign
                mask_pos = (self.eofs[0] > self.eofs[0].mean())
                sign = np.sign(self.eofs[s].where(mask_pos).mean(axis=(1,2)))
                self.eofs[s,:] = sign * self.eofs[s,:]


    def plot_eofs(self, mean=True, kwrgs: dict=None):
        kwrgs_plot = {'col_dim':'eof'}
        if mean:
            eof_patterns = self.eofs.mean(dim='split')
            if kwrgs is None:
                kwrgs_plot.update({'aspect':3})
        else:
            eof_patterns = self.eofs
        if kwrgs is not None:
           kwrgs_plot.update(kwrgs)
        plot_maps.plot_corr_maps(eof_patterns, **kwrgs_plot)

    def get_ts(self, tfreq_ts=1, df_splits=None):
        if df_splits is None:
            df_splits = self.df_splits
        else:
            df_splits = df_splits
        splits = self.eofs['split'].values
        neofs  = self.eofs['eof'].values
        if type(self.filepath) is str:
            ds = functions_pp.import_ds_timemeanbins(self.filepath,
                                                tfreq=tfreq_ts,
                                                selbox=self.selbox,
                                                start_end_date=self.start_end_date,
                                                start_end_year=self.start_end_year)
        elif type(self.filepath) is xr.DataArray:
            ds = self.filepath

        df_data_s   = np.zeros( (splits.size) , dtype=object)
        dates = pd.to_datetime(ds['time'].values)
        for s in splits:

            dfs = pd.DataFrame(columns=neofs, index=dates)
            for i, e in enumerate(neofs):

                pattern = self.eofs.sel(split=s, eof=e)
                data = find_precursors.calc_spatcov(ds.values, pattern.values)
                dfs[e] = pd.Series(data,
                                   index=dates)
                if i == neofs.size-1:
                    dfs = dfs.merge(df_splits.loc[s], left_index=True, right_index=True)
            df_data_s[s] = dfs
        self.df = pd.concat(list(df_data_s), keys=range(splits.size))


    def plot_ts(self):
        neofs = np.array(self.df.dtypes.index[self.df.dtypes != bool], dtype='object')
        fig, axes = plt.subplots(1, neofs.size, figsize=(20,13))
        if neofs.size == 1:
            axes = [axes]
        for i, e in enumerate(neofs):
            ax = axes[i]
            for s in self.eofs['split'].values:
                ts = self.df.loc[s][e]
                ax.plot_date(ts.index, ts.values, label = f'split {s}')
                ax.legend()
                ax.set_title(e)


    @staticmethod
    def _single_split(func, ds_EOF, s, df_splits, neofs):
        dates_train = functions_pp.dfsplits_to_dates(df_splits, s)[0]
        # convert Train test year from original time to monthly
        train_yrs = np.unique(dates_train.year)
        dates_monthly = pd.to_datetime(ds_EOF.time.values)
        dates_train_monthly = pd.to_datetime([d for d in dates_monthly if d.year in train_yrs])
        ds_EOF_train = ds_EOF.sel(time=dates_train_monthly)
        eofs, solver = func(ds_EOF_train, neofs)
        return (eofs, solver)

    @staticmethod
    def _get_EOF_xarray(ds_EOF, neofs):
        from eofs.xarray import Eof

        coslat = np.cos(np.deg2rad(ds_EOF.coords['latitude'].values)).clip(0., 1.)
        area_weights = np.tile(coslat[..., np.newaxis],(1,ds_EOF.longitude.size))
        area_weights = area_weights / area_weights.mean()
        solver = Eof(ds_EOF, area_weights)
        eofs = solver.eofsAsCovariance(neofs=neofs).squeeze()
        return eofs, solver
