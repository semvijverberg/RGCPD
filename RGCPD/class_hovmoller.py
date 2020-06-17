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
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry.polygon import LinearRing
# import metpy.calc as mpcalc
import plot_maps
import functions_pp
import find_precursors
from typing import List, Tuple, Union


class Hovmoller:

    def __init__(self, kwrgs_load: dict=None, slice_dates: tuple=None,
                 event_dates: pd.DatetimeIndex=None, lags_prior: int=None,
                 lags_posterior: int=None, standardize: bool=False,
                 seldates: tuple=None, rollingmeanwindow: int=None,
                 name=None, n_cpu=1):

        '''
        selbox has format of (lon_min, lon_max, lat_min, lat_max)
        '''
        self.kwrgs_load = kwrgs_load.copy()
        self.slice_dates = slice_dates
        self.event_dates = event_dates
        self.seldates = seldates
        self.standardize = standardize
        self.rollingmeanwindow = rollingmeanwindow

        if slice_dates is None and event_dates is None:
            raise ValueError('No dates to select or slice, please define '
                             'slice_dates or events dates')

        if standardize and seldates is None:
            raise ValueError('Give seldates over which the standard deviation'
                             ' is calculated.')

        if slice_dates is not None:
            print('slice dates not supported yet')

        if lags_prior is None:
            lags_prior = 10
        if lags_posterior is None:
            lags_posterior = 1
        self.lags_prior = lags_prior
        self.lags_posterior = lags_posterior
        self.lags = list(range(-abs(self.lags_prior), self.lags_posterior+1))
        if self.rollingmeanwindow is not None and self.seldates is None:
            raise Exception('You cannot do a rolling mean over only event dates, '
                            'specify over which dates you want to do a rolling mean, '
                            'after that you the dates will be selected from the smoothened array')

        self.event_lagged = np.array([event_dates + pd.Timedelta(f'{l}d') for l in self.lags])

        if np.unique(self.event_lagged).size != self.event_lagged.size:
            raise Exception('There are overlapping dates when shifting events '
                            'dates with lags')

        self.lag_axes = np.zeros_like(self.event_lagged, dtype=int)
        for i,l in enumerate(self.lags):
            self.lag_axes[i] = np.repeat(l, self.event_dates.size)



        self.name = name
        self.n_cpu = n_cpu

        self._check_dates()

        return

    def get_HM_data(self, filepath, dim='latitude'):
        self.filepath = filepath
        if self.seldates is not None:
            self.kwrgs_load['seldates'] = self.seldates_ext
            self.ds_seldates = functions_pp.import_ds_timemeanbins(self.filepath, **self.kwrgs_load)

            if self.rollingmeanwindow is not None:
            # apply rolling mean
                self.ds = self.ds_seldates.rolling(time=self.rollingmeanwindow).mean()
            else:
                self.ds = self.ds_seldates
            # calculating std based on seldates
            self.std = self.ds.sel(time=self.seldates).std(dim='time')
            # now that we have std over seldates, select dates for HM
            self.ds = self.ds.sel(time=np.concatenate(self.event_lagged))
        else:
            self.kwrgs_load['seldates'] = np.concatenate(self.event_lagged)
            self.ds = functions_pp.import_ds_timemeanbins(self.filepath, **self.kwrgs_load)

        if 'units' in list(self.ds.attrs.keys()):
            self.units = self.ds.attrs['units']

        if self.standardize:
            self.units = 'std [-]'
            self.ds = self.ds / self.std



        if self.event_dates is not None:
            self.xarray = self.ds.copy().rename({'time':'lag'})
            self.xarray = self.xarray.assign_coords(lag=np.concatenate(self.lag_axes))
        else:
            self.xarray = self.ds
        xarray_w = functions_pp.area_weighted(self.xarray)
        self.xr_HM = xarray_w.mean(dim=dim).groupby('lag').mean()

    def quick_HM_plot(self):
        if hasattr(self, 'xr_HM') == False:
            print('first run get_HM_data(filepath)')
        else:
            self.xr_HM.plot()

    def plot_HM(self, main_title_right: str=None, ytickstep=5, lattickstep: int=3,
                clevels: np.ndarray=None, clim: Union[str, tuple]='relaxed',
                cmap=None, drawbox: list=None, save: bool=False,
                fig_path: str=None):
        #%%
        # main_title_right=None; ytickstep=5;
        # clevels=None; clim='relaxed' ; cmap=None; drawbox=None

        # Get times and make array of datetime objects
        if self.event_dates is not None:
            vtimes = self.xr_HM.lag.values
        else:
            vtimes = self.xr_HM.time.values.astype('datetime64[ms]').astype('O')

        if cmap is None:
            cmap = plt.cm.RdBu_r
        # Start figure
        fig = plt.figure(figsize=(10, 13))

        # Use gridspec to help size elements of plot; small top plot and big bottom plot
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.03)

        # Tick labels
        x_tick_labels = [u'0\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                         u'180\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}W',
                         u'0\N{DEGREE SIGN}E']

        # Plot of chosen variable averaged over latitude and slightly smoothed
        if clevels is None:
            class MidpointNormalize(mcolors.Normalize):
                def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                    self.midpoint = midpoint
                    mcolors.Normalize.__init__(self, vmin, vmax, clip)

                def __call__(self, value, clip=None):
                    # I'm ignoring masked values and all kinds of edge cases to make a
                    # simple example...
                    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                    return np.ma.masked_array(np.interp(value, x, y))

            if clim == 'relaxed':
                vmin_ = np.nanpercentile(self.xr_HM, 1) ;
                vmax_ = np.nanpercentile(self.xr_HM, 99)
            elif type(clim) == tuple:
                vmin_, vmax_ = clim
            else:
                vmin_ = self.xr_HM.min()-0.01 ; vmax_ = self.xr_HM.max()+0.01

            vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
            clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
            norm = MidpointNormalize(midpoint=0, vmin=clevels[0],vmax=clevels[-1])
            ticksteps = 4
        else:
            vmin_ = np.nanpercentile(self.xr_HM, 1) ; vmax_ = np.nanpercentile(self.xr_HM, 99)
            vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
            clevels=clevels
            norm=None
            ticksteps = 1

        # Top plot for geographic reference (makes small map)
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
        selbox = list(self.kwrgs_load['selbox']) ; selbox[1] = selbox[1]-.1

        # Add geopolitical boundaries for map reference
        ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
        xr_events = self.ds.sel(time=self.event_dates).mean(dim='time')
        lon = xr_events.longitude
        if abs(lon[-1] - 360) <= (lon[1] - lon[0]):
            xr_events = plot_maps.extend_longitude(xr_events)
            # xr_events = core_pp.convert_longitude(xr_events, to_format='west_east')
        xr_events.plot.contourf(levels=clevels, cmap=cmap,
                                                transform=ccrs.PlateCarree(),
                                                ax=ax1,
                                                add_colorbar=False)
        ax1.set_extent(selbox, ccrs.PlateCarree(central_longitude=180))
        y_ticks = np.unique(np.round(xr_events.latitude, decimals=-1))[::lattickstep]
        ax1.set_yticks(y_ticks.astype(int))
        ax1.set_yticklabels([u'{:.0f}\N{DEGREE SIGN}N'.format(l) for l in y_ticks])
        ax1.set_ylabel('Latitude')
        # ax1.set_xticks([-180, -90, 0, 90, 180])
        # ax1.set_xticklabels(x_tick_labels)
        ax1.grid(linestyle='dotted', linewidth=2)

        # =============================================================================
        # Draw (rectangular) box
        # =============================================================================
        if drawbox is not None:
            def get_ring(coords):
                '''tuple in format: west_lon, east_lon, south_lat, north_lat '''
                west_lon, east_lon, south_lat, north_lat = coords
                lons_sq = [west_lon, west_lon, east_lon, east_lon]
                lats_sq = [north_lat, south_lat, south_lat, north_lat]
                ring = [LinearRing(list(zip(lons_sq , lats_sq )))]
                return ring

            ring = get_ring(drawbox)

            ax1.add_geometries(ring, ccrs.PlateCarree(), facecolor='none', edgecolor='green',
                              linewidth=2, linestyle='dashed')

        # ax1.set_xticks([0, 90, 180, 270, 357.5])
        # ax1.set_xticklabels(x_tick_labels)


        # Set some titles
        plt.title('Hovmoller Diagram', loc='left')
        if main_title_right is not None:
            plt.title(main_title_right, loc='right')

        # Bottom plot for Hovmoller diagram
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.invert_yaxis()  # Reverse the time order to do oldest first

        cf = self.xr_HM.plot.contourf(levels=clevels, cmap=cmap, ax=ax2,
                                      add_colorbar=False)
        self.xr_HM.plot.contour(clevels=clevels, colors='k', linewidths=1, ax=ax2)
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50,
                            extendrect=True, norm=norm, ticks=clevels[::ticksteps])
        if hasattr(self, 'units'):
            cbar.set_label(self.units)

        # Make some ticks and tick labels
        ax2.set_xticks([0, 90, 180, 270, 357.5])
        ax2.set_xticklabels(x_tick_labels)
        ax2.set_xlabel('')
        if self.event_dates is not None:
            y_ticks = list(vtimes[::ytickstep]) ; #y_ticks.append(vtimes[-1])
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels(y_ticks)

        # Set some titles
        if self.name is not None:
            plt.title(self.name, loc='left', fontsize=10)
        if self.slice_dates != None:
            plt.title('Time Range: {0:%Y%m%d %HZ} - {1:%Y%m%d %HZ}'.format(vtimes[0], vtimes[-1]),
                      loc='right', fontsize=10)
        if save or fig_path is not None:
            fname = '_'.join(np.array(self.kwrgs_load['selbox']).astype(str)) + \
                    f'_w{self.rollingmeanwindow}_std{self.standardize}'
            if fig_path is None:
                fig_path = os.path.join(functions_pp.get_download_path(), fname)
            plt.savefig(fig_path, bbox_inches='tight')


    def _check_dates(self):
        ev_lag = pd.to_datetime(self.event_lagged.flatten())
        mde = [int('{:02d}{:02d}'.format(d.month, d.day)) for d in ev_lag] # monthdayevents
        if type(self.seldates) is pd.DatetimeIndex:
            mds = [int('{:02d}{:02d}'.format(d.month, d.day)) for d in self.seldates] # monthdayselect
        if min(mde) < min(mds):
            print(f'An event date minus the max lag {min(mde)} is not in seldates '
                  f'{min(mds)}, adapting startdates of seldates')
            start_date = (f'{self.seldates[0].year}-'
                          f'{pd.to_datetime(ev_lag[np.argmin(mde)]).month}-'
                          f'{pd.to_datetime(ev_lag[np.argmin(mde)]).day}')
        else:
            start_date = (f'{self.seldates[0].year}-'
                          f'{self.seldates[0].month}-'
                          f'{self.seldates[0].day}')

        if max(mde) > max(mds):
            print(f'An event date plus the max lag {max(mde)} is not in seldates '
                  f'{max(mds)}, adapting enddate of seldates')
            end_date = (f'{self.seldates[0].year}-'
                f'{pd.to_datetime(ev_lag[np.argmax(mde)]).month}-'
                f'{pd.to_datetime(ev_lag[np.argmax(mde)]).day}')
        else:
            start_date = (f'{self.seldates[0].year}-'
                          f'{self.seldates[-1].month}-'
                          f'{self.seldates[-1].day}')
        self.seldates_ext = functions_pp.make_dates(pd.date_range(start_date, end_date),
                                                    np.unique(self.seldates.year))



