#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:31:11 2019

@author: semvijverberg
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import num2date
import itertools
# from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: list(set([item for sublist in l for item in sublist]))
flatten = lambda l: list(itertools.chain.from_iterable(l))
from typing import Union

def get_oneyr(pddatetime, *args):
    dates = []
    pddatetime = pd.to_datetime(pddatetime)
    year = pddatetime.year[0]

    for arg in args:
        year = arg
        dates.append(pddatetime.where(pddatetime.year==year).dropna())
    dates = pd.to_datetime(flatten(dates))
    if len(dates) == 0:
        dates = pddatetime.where(pddatetime.year==year).dropna()
    return dates



def import_ds_lazy(filename, loadleap=False,
                   seldates: Union[tuple, pd.core.indexes.datetimes.DatetimeIndex]=None,
                   selbox: Union[list, tuple]=None, format_lon='only_east', var=None,
                   verbosity=0):

    '''
    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    # in format only_east
    # selbox assumes [lowest_east_lon, highest_east_lon, south_lat, north_lat]
    '''

    ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)

    if len(ds.dims.keys()) > 1: # more then 1-d
        multi_dims = True

    variables = list(ds.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    common_fields = ' time time_bnds longitude latitude lev lon lat level mask lsm '

    if var is None:
        # try to auto select var which is not a coordinate
        var = [var for var in strvars if var not in common_fields]
    else:
        var = [var]

    if len(var) == 1:
        var = var[0].replace(' ', '')
        ds = ds[var]
    elif len(var) > 1:
        # load whole dataset
        ds = ds

    if multi_dims:
        if 'latitude' and 'longitude' not in ds.dims:
            ds = ds.rename({'lat':'latitude',
                            'lon':'longitude'})

        if format_lon is not None:
            if test_periodic(ds)==False and 0 not in ds.longitude:
                format_lon = 'only_east'
            if _check_format(ds) != format_lon:
                ds = convert_longitude(ds, format_lon)

        # ensure longitude in increasing order
        if np.where(ds.longitude == ds.longitude.min()) > np.where(ds.longitude == ds.longitude.max()):
            print('sorting longitude')
            ds = ds.sortby('longitude')

        # ensure latitude is in increasing order
        if np.where(ds.latitude == ds.latitude.min()) > np.where(ds.latitude == ds.latitude.max()):
            print('sorting latitude')
            ds = ds.sortby('latitude')

        if selbox is not None:
            ds = get_selbox(ds, selbox, verbosity)


    # get dates
    if 'time' in ds.squeeze().dims:
        numtime = ds['time']
        dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])

        timestep_days = (dates[1] - dates[0]).days
        if timestep_days == 1:
            input_freq = 'daily'
        elif timestep_days == 30 or timestep_days == 31:
            input_freq = 'monthly'
        elif timestep_days == 365 or timestep_days == 366:
            input_freq = 'annual'

        if numtime.attrs['calendar'] != 'gregorian':
            dates = [d.strftime('%Y-%m-%d') for d in dates]

        if input_freq == 'monthly' or input_freq == 'annual':
            dates = [d.replace(day=1,hour=0) for d in pd.to_datetime(dates)]
        else:
            dates = pd.to_datetime(dates)
            stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')
            test_if_fullyr = np.logical_and(dates[stepsyr.size-1].month == 12,
                                       dates[stepsyr.size-1].day == 31)
            assert test_if_fullyr, ('full is needed as raw data since rolling'
                               ' mean is applied across timesteps')

        dates = pd.to_datetime(dates)
        # set hour to 00
        if dates.hour[0] != 0:
            dates -= pd.Timedelta(dates.hour[0], unit='h')

        ds['time'] = dates

        if seldates is None:
            pass
        elif type(seldates) is tuple:
            pddates = get_subdates(dates=pd.to_datetime(ds.time.values),
                         start_end_date=seldates,
                         start_end_year=None, lpyr=loadleap
                         )
            ds = ds.sel(time=pddates)
        else:
            ds = ds.sel(time=seldates)

        if loadleap==False:
            # mask away leapdays
            dates_noleap = remove_leapdays(pd.to_datetime(ds.time.values))
            ds = ds.sel(time=dates_noleap)
    if type(ds) == type(xr.DataArray(data=[0])):
        ds.attrs['is_DataArray'] = 1
    else:
        ds.attrs['is_DataArray'] = 0
    return ds


def remove_leapdays(datetime):
    mask_lpyrfeb = np.logical_and((datetime.month == 2), (datetime.day == 29))

    dates_noleap = datetime[mask_lpyrfeb==False]
    return dates_noleap

def get_selbox(ds, selbox, verbosity=0):
    '''
    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    # test selbox assumes [west_lon, east_lon, south_lat, north_lat]
    '''

    except_cross180_westeast = test_periodic(ds)==False and 0 not in ds.longitude

    if except_cross180_westeast:
        # convert selbox to degrees east
        selbox = np.array(selbox)
        selbox[selbox < 0] += 360
        selbox = list(selbox)

    if ds.latitude[0] > ds.latitude[1]:
        slice_lat = slice(max(selbox[2:]), min(selbox[2:]))
    else:
        slice_lat = slice(min(selbox[2:]), max(selbox[2:]))
    ds = ds.sel(latitude=slice_lat)
    east_lon = selbox[0]
    west_lon = selbox[1]
    if (east_lon > west_lon and east_lon > 180) or east_lon < 0:
        if verbosity > 0:
            print('east lon > 180 and cross GW meridional, converting to west '
                  'east longitude format because lons must be sorted by value')
        zz = convert_longitude(ds, to_format='east_west')
        zz = zz.sortby('longitude')
        if east_lon <= 0:
            e_lon =east_lon
        elif east_lon > 180:
            e_lon = east_lon - 360
        ds = zz.sel(longitude=slice(e_lon, west_lon))
    else:
        ds = ds.sel(longitude=slice(east_lon, west_lon))
    return ds

def detrend_anom_ncdf3D(infile, outfile, loadleap=False,
                        seldates=None, selbox=None, format_lon='east_west',
                        detrend=True, anomaly=True, apply_fft=True,
                        n_harmonics=6, encoding=None):
    '''
    Function for preprocessing
    - Calculate anomalies (by removing seasonal cycle based on first
      three harmonics)
    - linear long-term detrend
    '''

    #%%
    ds = import_ds_lazy(infile, loadleap=loadleap,
                        seldates=seldates, selbox=selbox, format_lon=format_lon)

    # check if 3D data (lat, lat, lev) or 2D
    check_dim_level = any([level in ds.dims for level in ['lev', 'level']])

    if check_dim_level:
        key = ['lev', 'level'][any([level in ds.dims for level in ['lev', 'level']])]
        levels = ds[key]
        output = np.empty( (ds.time.size,  ds.level.size, ds.latitude.size, ds.longitude.size), dtype='float32' )
        output[:] = np.nan
        for lev_idx, lev in enumerate(levels.values):
            ds_2D = ds.sel(levels=lev)

            output[:,lev_idx,:,:] = detrend_xarray_ds_2D(ds_2D, detrend=detrend, anomaly=anomaly,
                                      apply_fft=apply_fft, n_harmonics=n_harmonics)

            # output[:,lev_idx,:,:] = detrend_xarray_ds_2D(ds_2D, detrend=detrend, anomaly=anomaly)
    else:
        output = detrend_xarray_ds_2D(ds, detrend=detrend, anomaly=anomaly,
                                      apply_fft=apply_fft, n_harmonics=n_harmonics)
        # output = deseasonalizefft_detrend_2D(ds, detrend, anomaly)

    print(f'\nwriting ncdf file to:\n{outfile}')
    output = xr.DataArray(output, name=ds.name, dims=ds.dims, coords=ds.coords)
    # copy original attributes to xarray
    output.attrs = ds.attrs

    # ensure mask
    output = output.where(output.values != 0.).fillna(-9999)
    encoding = ( {ds.name : {'_FillValue': -9999}} )
    mask =  (('latitude', 'longitude'), (output.values[0] != -9999) )
    output.coords['mask'] = mask
#    xarray_plot(output[0])

    # save netcdf
    output.to_netcdf(outfile, mode='w', encoding=encoding)
#    diff = output - abs(marray)
#    diff.to_netcdf(filename.replace('.nc', 'diff.nc'))
    #%%
    return

def reconstruct_fft_2D(ds, coefficients:list=None,
                    list_of_harm: list=[1, 1/2, 1/3],
                    add_constant: bool=True):

    dates = pd.to_datetime(ds.time.values)
    N = dates.size
    oneyr = get_oneyr(dates)
    time_axis = np.linspace(0, N-1, N)
    A_k = np.fft.fft(ds.values, axis=0) # positive & negative amplitude
    freqs_belongingto_A_k = np.fft.fftfreq(ds.shape[0])
    periods = np.zeros_like(freqs_belongingto_A_k)
    periods[1:] = 1/(freqs_belongingto_A_k[1:]*oneyr.size)
    def get_harmonics(periods, list_of_harm=list):
        harmonics = []
        for h in list_of_harm:
            harmonics.append(np.argmin((abs(periods - h))))
        harmonics = np.array(harmonics) - 1 # subtract 1 because loop below is adding 1
        return harmonics

    if coefficients is None:
        if list_of_harm is [1, 1/2, 1/3]:
            print('using default first 3 annual harmonics, expecting cycles of 365 days')
        coefficients = get_harmonics(periods, list_of_harm=list_of_harm)
    elif coefficients is not None:
        coefficients = coefficients

    reconstructed_signal = np.zeros_like(ds.values, dtype='c16')
    reconstructed_signal += A_k[0].real * np.zeros_like(ds.values, dtype='c16')
    # Adding the dc term explicitly makes the looping easier in the next step.


    for c, k in enumerate(coefficients):
        progress = int((100*(c+1)/coefficients.size))
        print(f"\rProcessing {progress}%", end="")
        k += 1  # Bump by one since we already took care of the dc term.
        if k == N-k:
            reconstructed_signal += A_k[k] * np.exp(
                1.0j*2 * np.pi * (k) * time_axis / N)[:,None,None] # add fake space dims
        # This catches the case where N is even and ensures we don't double-
        # count the frequency k=N/2.

        else:
            reconstructed_signal += A_k[k] * np.exp(
                1.0j*2 * np.pi * (k) * time_axis / N)[:,None,None]
            reconstructed_signal += A_k[N-k] * np.exp(
                1.0j*2 * np.pi * (N-k) * time_axis / N)[:,None,None]
        # In this case we're just adding a frequency component and it's
        # "partner" at minus the frequency

    reconstructed_signal = (reconstructed_signal / N)
    if add_constant:
        reconstructed_signal.real += ds.values.mean(0)
    return reconstructed_signal.real

def deseasonalizefft_detrend_2D(ds, detrend: bool=True, anomaly: bool=True,
                                n_harmonics=3):
    '''
    Remove long-term trend and/or remove seasonal cycle.

    Seasonal cycle is removed by the subtracting the sum of the first 3 annual
    harmonics (freq = 1/365, .5/365, .33/365).

    Parameters
    ----------
    ds : TYPE
        xr.DataArray() of dims (time, latitude, longitude).
    detrend : bool, optional
        Detrend (long-term trend along all datapoints). The default is True.
    anomaly : bool, optional
        Remove Seasonal cycle using FFT. The default is True.
    Returns
    -------
    xr.DataArray()

    '''
    import df_ana

    dates = pd.to_datetime(ds.time.values)
    if anomaly:
        list_of_harm= [1/h for h in range(1,n_harmonics+1)]
        reconstructed_signal = reconstruct_fft_2D(ds, list_of_harm=list_of_harm,
                                                 add_constant=False)

    ds = ds - ds.mean(dim='time')
    # plot gridpoints for visual check

    # try to find location above EU
    ts = ds.sel(longitude=30, method='nearest').sel(latitude=40, method='nearest')
    la1 = np.argwhere(ts.latitude.values ==ds.latitude.values)[0][0]
    lo1 = np.argwhere(ts.longitude.values ==ds.longitude.values)[0][0]
    ts = ds[:,la1,lo1]

    la2 = int(ds.shape[1]/3)
    lo2 = int(ds.shape[2]/3)

    tuples = [(la1, lo1), (la1+1, lo1), (la1, lo1+1),
              (la2, lo2), (la2+1, lo2), (la2, lo2+1)]
    fig, ax = plt.subplots(3,2, figsize=(16,8))
    ax = ax.flatten()
    for i, lalo in enumerate(tuples):
        lat = int(ds.latitude[lalo[0]])
        lon = int(ds.longitude[lalo[1]])
        print(f"\rVisual test latlon {lat} {lon}", end="")
        ts = ds[:,lalo[0],lalo[1]]
        ax[i].set_title(f'latlon coord {lat} {lon}')
        for yr in np.unique(dates.year):
            singleyeardates = get_oneyr(dates, yr)
            ax[i].plot(ts.sel(time=singleyeardates), alpha=.1, color='purple')
        if anomaly:
            ax[i].plot(reconstructed_signal[:365, lalo[0],lalo[1]].real,
                       label=f'FFT with {n_harmonics}h')
        ax[i].legend()
    plt.subplots_adjust(hspace=.3)
    ax[-1].text(.5,1.2, 'Visual analysis:',
            transform=ax[0].transAxes,
            ha='center', va='bottom')
    if anomaly:
        ds = ds - reconstructed_signal

    fig, ax = plt.subplots(1, figsize=(5,3))
    ds[:,lalo[0],lalo[1]].groupby('time.month').mean(dim='time').plot(ax=ax)
    ax.set_title(f'climatological monhtly means anomalies latlon coord {lat} {lon}')
    plt.figure(figsize=(4,2.5))
    summer = ds.sel(time=get_subdates(dates, start_end_date=('06-01', '08-31')))
    summer.name = f'std {summer.name}'
    ax = (summer.mean(dim='time') / summer.std(dim='time')).plot()

    try:
        ts = ds[:,la1,lo1]
        df_ana.plot_spectrum(pd.Series(ts.values, index=dates), year_max=.1)
    except:
        pass
    if detrend:
        ds = detrend_lin_longterm(ds)
    return ds

def detrend_lin_longterm(ds):
    no_nans = np.nan_to_num(ds)
    detrended = sp.signal.detrend(no_nans, axis=0, type='linear')
    nan_true = np.isnan(ds)
    detrended[nan_true.values] = np.nan
    trend = ds - detrended
    constant = np.repeat(np.mean(ds, 0).expand_dims('time'), ds.time.size, 0 )
    detrended += constant
    detrended = detrended.assign_coords(
                coords={'time':pd.to_datetime(ds.time.values)})
    # plot single gridpoint for visual check
    la = int(ds.shape[1]/2)
    lo = int(ds.shape[2]/2)
    tuples = [(la, lo), (la+1, lo), (la, lo+1)]
    fig, ax = plt.subplots(3, figsize=(8,8))
    for i, lalo in enumerate(tuples):
        lat = int(ds.latitude[lalo[0]])
        lon = int(ds.longitude[lalo[1]])
        ax[i].set_title(f'latlon coord {lat} {lon}')
        ax[i].plot(ds.values[:,lalo[0],lalo[1]])
        ax[i].plot(detrended[:,lalo[0],lalo[1]])
        trend1d = trend[:,lalo[0],lalo[1]]
        linregab = np.polyfit(np.arange(trend1d.size), trend1d, 1)
        ax[i].plot(trend1d)
        ax[i].text(.05, .05,
        'y = {:.2g}x + {:.2g}'.format(*linregab),
        transform=ax[i].transAxes)
    plt.subplots_adjust(hspace=.3)
    ax[-1].text(.5,1.2, 'Visual analysis: trends of nearby gridcells should be similar',
                transform=ax[0].transAxes,
                ha='center', va='bottom')
    return detrended


def detrend_xarray_ds_2D(ds, detrend, anomaly, apply_fft=True, n_harmonics=6):
    #%%


    if type(ds.time[0].values) != type(np.datetime64()):
        numtime = ds['time']
        dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
        if numtime.attrs['calendar'] != 'gregorian':
            dates = [d.strftime('%Y-%m-%d') for d in dates]
        dates = pd.to_datetime(dates)
    else:
        dates = pd.to_datetime(ds['time'].values)
    stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')
    ds['time'] = dates

    def _detrendfunc2d(arr_oneday, arr_oneday_smooth):

        # get trend of smoothened signal
        no_nans = np.nan_to_num(arr_oneday_smooth)
        detrended_sm = sp.signal.detrend(no_nans, axis=0, type='linear')
        nan_true = np.isnan(arr_oneday)
        detrended_sm[nan_true.values] = np.nan
        # subtract trend smoothened signal of arr_oneday values
        trend = (arr_oneday_smooth - detrended_sm)- np.mean(arr_oneday_smooth, 0)
        detrended = arr_oneday - trend
        return detrended, detrended_sm


    def detrendfunc2d(arr_oneday):
        return xr.apply_ufunc(_detrendfunc2d, arr_oneday,
                              dask='parallelized',
                              output_dtypes=[float])
    #        return xr.apply_ufunc(_detrendfunc2d, arr_oneday.compute(),
    #                              dask='parallelized',
    #                              output_dtypes=[float])

    if (stepsyr.day== 1).all() == True or int(ds.time.size / 365) >= 120:
        print('\nHandling time series longer then 120 day or monthly data, no smoothening applied')
        data_smooth = ds.values

    elif (stepsyr.day== 1).all() == False and int(ds.time.size / 365) < 120:
        window_s = max(min(25,int(stepsyr.size / 12)), 1)
        # print('Performing {} day rolling mean'
        #       ' to get better interannual statistics'.format(window_s))

        data_smooth =  rolling_mean_np(ds.values, window_s, win_type='boxcar')

    output_clim3d = np.empty((stepsyr.size, ds.latitude.size, ds.longitude.size),
                               dtype='float32')



    for i in range(stepsyr.size):

        sliceyr = np.arange(i, ds.time.size, stepsyr.size)
        arr_oneday_smooth = data_smooth[sliceyr]


        if anomaly:
            if i==0: print('using absolute anomalies w.r.t. climatology of '
                            'smoothed concurrent day accross years\n')
            output_clim2d = arr_oneday_smooth.mean(axis=0)
            # output[i::stepsyr.size] = arr_oneday - output_clim3d
            output_clim3d[i,:,:] = output_clim2d

        progress = int((100*(i+1)/stepsyr.size))
        print(f"\rProcessing {progress}%", end="")

    if apply_fft:
        # beware, mean by default 0, add constant = False
        list_of_harm= [1/h for h in range(1,n_harmonics+1)]
        clim_rec = reconstruct_fft_2D(xr.DataArray(data=output_clim3d,
                                                   coords=ds.sel(time=stepsyr).coords,
                                                   dims=ds.dims),
                                      list_of_harm=list_of_harm,
                                      add_constant=False)
        # Adding mean of origninal ds
        clim_rec += ds.values.mean(0)
        output = ds - np.tile(clim_rec, (int(dates.size/stepsyr.size), 1, 1))
    else:
        output = ds - np.tile(output_clim3d, (int(dates.size/stepsyr.size), 1, 1))



    # =============================================================================
    # test gridcells:
    # =============================================================================
    # # try to find location above EU
    # ts = ds.sel(longitude=30, method='nearest').sel(latitude=40, method='nearest')
    # la1 = np.argwhere(ts.latitude.values ==ds.latitude.values)[0][0]
    # lo1 = np.argwhere(ts.longitude.values ==ds.longitude.values)[0][0]
    la1 = int(ds.shape[1]/2)
    lo1 = int(ds.shape[2]/2)
    la2 = int(ds.shape[1]/3)
    lo2 = int(ds.shape[2]/3)

    tuples = [(la1, lo1), (la1+1, lo1),
              (la2, lo2), (la2+1, lo2)]
    if apply_fft:
        fig, ax = plt.subplots(4,2, figsize=(16,8))
    else:
        fig, ax = plt.subplots(2,2, figsize=(16,8))
    ax = ax.flatten()
    for i, lalo in enumerate(tuples):
        lat = int(ds.latitude[lalo[0]])
        lon = int(ds.longitude[lalo[1]])
        print(f"\rVisual test latlon {lat} {lon}", end="")
        ts = ds[:,lalo[0],lalo[1]]
        rawdayofyear = ts.groupby('time.dayofyear').mean('time').sel(dayofyear=np.arange(365)+1)

        ax[i].set_title(f'latlon coord {lat} {lon}')
        for yr in np.unique(dates.year):
            singleyeardates = get_oneyr(dates, yr)
            ax[i].plot(ts.sel(time=singleyeardates), alpha=.1, color='purple')
        ax[i].plot(output_clim3d[:,lalo[0],lalo[1]], color='green', linewidth=2,
             label=f'clim {window_s}-day rm')
        ax[i].plot(rawdayofyear, color='black', alpha=.6,
                   label='clim mean dayofyear')
        if apply_fft:
            ax[i].plot(clim_rec[:,lalo[0],lalo[1]][:365], 'r-',
                       label=f'fft {n_harmonics}h on clim {window_s}-day rm')
            diff = clim_rec[:,lalo[0],lalo[1]][:365] - output_clim3d[:,lalo[0],lalo[1]]
            diff = diff / ts.std(dim='time').values
            ax[i+len(tuples)].plot(diff)
            ax[i+len(tuples)].set_title(f'latlon coord {lat} {lon} diff/std(alldata)')
        ax[i].legend()
    ax[-1].text(.5,1.2, 'Visual analysis',
            transform=ax[0].transAxes,
            ha='center', va='bottom')
    plt.subplots_adjust(hspace=.4)
    fig, ax = plt.subplots(1, figsize=(5,3))
    std_all = output[:,lalo[0],lalo[1]].std(dim='time')
    monthlymean = output[:,lalo[0],lalo[1]].groupby('time.month').mean(dim='time')
    (monthlymean/std_all).plot(ax=ax)
    ax.set_ylabel('standardized anomaly [-]')
    ax.set_title(f'climatological monthly means anomalies latlon coord {lat} {lon}')
    fig, ax = plt.subplots(1, figsize=(5,3))
    summer = output.sel(time=get_subdates(dates, start_end_date=('06-01', '08-31')))
    summer.name = f'std {summer.name}'
    (summer.mean(dim='time') / summer.std(dim='time')).plot(ax=ax)
    ax.set_title('summer composite mean [in std]')


    if detrend:
        output = detrend_lin_longterm(output)

    #%%
    return output
#%%
def rolling_mean_np(arr, win, center=True, win_type='boxcar'):
    import scipy.signal.windows as spwin

    df = pd.DataFrame(data=arr.reshape( (arr.shape[0], arr[0].size)))

    if win_type == 'gaussian':
        w_std = win/3.
        print('Performing {} day rolling mean with gaussian window (std={})'
              ' to get better interannual statistics'.format(win,w_std))
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(range(-int(win/2),+round(win/2+.49)), spwin.gaussian(win, w_std))
        plt.title('window used for rolling mean')
        plt.xlabel('timesteps')
        rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='gaussian').mean(std=w_std)
    elif win_type == 'boxcar':
        fig, ax = plt.subplots(figsize=(3,3))
        plt.plot(spwin.boxcar(win))
        plt.title('window used for rolling mean')
        plt.xlabel('timesteps')
        rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='boxcar').mean()

    return rollmean.values.reshape( (arr.shape))



def test_periodic(ds):
    dlon = ds.longitude[1] - ds.longitude[0]
    return (360 / dlon == ds.longitude.size).values

def test_periodic_lat(ds):
    dlat = ds.latitude[1] - ds.latitude[0]
    return ((180/dlat)+1 == ds.latitude.size).values

def _check_format(ds):
    longitude = ds.longitude.values
    if longitude[longitude > 180.].size != 0:
        format_lon = 'only_east'
    else:
        format_lon = 'west_east'
    return format_lon

def convert_longitude(data, to_format='east_west'):
    if to_format == 'east_west':
        data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))
    elif to_format == 'only_east':
        data = data.assign_coords(longitude=((data.longitude + 360) % 360))
    return data




def get_subdates(dates, start_end_date, start_end_year=None, lpyr=False):
    #%%
    '''
    dates is type pandas.core.indexes.datetimes.DatetimeIndex
    start_end_date is tuple of start- and enddate in format ('mm-dd', 'mm-dd')
    lpyr is boolean if you want load the leap days yes or no.
    '''
    #%%
    import calendar

    def oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()

    if start_end_year is None:
        startyr = dates.year.min()
        endyr   = dates.year.max()
    else:
        startyr = start_end_year[0]
        endyr   = start_end_year[1]

    sstartdate = pd.to_datetime(str(startyr) + '-' + start_end_date[0])
    senddate_   = pd.to_datetime(str(startyr) + '-' + start_end_date[1])


    tfreq = (dates[1] - dates[0]).days
    oneyr_dates = pd.date_range(start=sstartdate, end=senddate_,
                            freq=pd.Timedelta(1, 'd'))
    daily_yr_fit = np.round(oneyr_dates.size / tfreq, 0)

    # dont get following
#    firstyr = oneyr(oneyr_dates)
    firstyr = oneyr(dates)
    #find closest senddate
    closest_enddate_idx = np.argmin(abs(firstyr - senddate_))
    senddate = firstyr[closest_enddate_idx]
    if senddate > senddate_ :
        senddate = firstyr[closest_enddate_idx-1]

    #update startdate of RV period to fit bins
    if tfreq == 1:
        sstartdate = senddate - pd.Timedelta(int(tfreq * daily_yr_fit), 'd') + \
                             np.timedelta64(1, 'D')
    else:
        sstartdate = senddate - pd.Timedelta(int(tfreq * daily_yr_fit), 'd')
        while sstartdate < pd.to_datetime(str(startyr) + '-' + start_end_date[0]):
            daily_yr_fit -=1
            sstartdate = senddate - pd.Timedelta(int(tfreq * daily_yr_fit), 'd')


    start_yr = pd.date_range(start=sstartdate, end=senddate,
                                freq=(dates[1] - dates[0]))
    if lpyr==True and calendar.isleap(startyr):
        start_yr -= pd.Timedelta( '1 days')
    elif lpyr==False and calendar.isleap(startyr):
        start_yr = remove_leapdays(start_yr)
    breakyr = endyr
    datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
    nyears = (endyr - startyr)+1
    startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
    endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
    firstyear = startday[:4]
    def plusyearnoleap(curr_yr, startday, endday, incr):
        startday = startday.replace(firstyear, str(curr_yr+incr))
        endday = endday.replace(firstyear, str(curr_yr+incr))

        next_yr = pd.date_range(start=startday, end=endday,
                        freq=(dates[1] - dates[0]))
        if lpyr==True and calendar.isleap(curr_yr+incr):
            next_yr -= pd.Timedelta( '1 days')
        elif lpyr == False:
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
        return next_yr


    for yr in range(0,nyears):
        curr_yr = yr+startyr
        next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr

        if next_yr.year[0] == breakyr:
            break
    datessubset = pd.to_datetime(datesstr)
    #%%
    return datessubset

#%%
def ensmean(outfile, weights=list, name=None, *args):
    outfile = '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm12_1979-2018_1_12_daily_1.0deg.nc'
    for i, arg in enumerate(args):
        ds = import_ds_lazy(arg)
        if i == 0:
            list_xr = [ds.expand_dims('extra_dim', axis=0) for i in range(len(args))]
            ds_e = xr.concat(list_xr, dim = 'extra_dim')
            ds_e['extra_dim'] = range(len(args))
        ds_e[i] = ds

    if weights is not None:
        weights = xr.DataArray(weights,
                               dims=['extra_dim'],
                               coords={'extra_dim':list(range(len(args)))})
        weights.name = "weights"
        ds_e.weighted(weights)
    ds_mean = ds_e.mean(dim='extra_dim')
    if name is not None:
        ds_mean.name = name

    ds_mean = ds_mean.where(ds_mean.values != 0.).fillna(-9999)
    encoding = ( {ds_mean.name : {'_FillValue': -9999}} )
    mask =  (('latitude', 'longitude'), (ds_mean.values[0] != -9999) )
    ds_mean.coords['mask'] = mask
    ds_mean.to_netcdf(outfile, mode='w', encoding=encoding)

if __name__ == '__main__':
    pass
    # ex = {}
    # ex['input_freq'] = 'daily'
    # DATAFOLDER = input('give path to datafolder where raw data in folder input_raw:\n')
    # infilename  = input('give input filename you want to preprocess:\n')
    # infile = os.path.join(DATAFOLDER, 'input_raw', infilename)
    # outfilename = infilename.split('.nc')[0] + '_pp.nc'
    # output_folder = os.path.join(DATAFOLDER, 'input_pp')
    # if os.path.isdir(output_folder) != True : os.makedirs(output_folder)
    # outfile = os.path.join(output_folder, outfilename)

    # kwargs = {'detrend':True, 'anomaly':True}
    # try:
    #     detrend_anom_ncdf3D(infile, outfile, ex, **kwargs)
    # except:
    #     print('just chilling bro, relax..')
