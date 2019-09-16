#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:31:11 2019

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd
from netCDF4 import num2date
import matplotlib.pyplot as plt
import xarray as xr
import itertools
from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: list(set([item for sublist in l for item in sublist]))
flatten = lambda l: list(itertools.chain.from_iterable(l))

def get_oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()



# def import_ds_lazy(filename, ex):
#     ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
#     variables = list(ds.variables.keys())
#     strvars = [' {} '.format(var) for var in variables]
#     common_fields = (' time time_bnds longitude latitude lev lon lat level mask '
#                      ' lon_bounds lat_bounds lon_bnds lat_bnds ')
#     var = [var for var in strvars if var not in common_fields][0]
#     var = var.replace(' ', '')
#
#     ds = ds[var].squeeze()
#     if 'latitude' and 'longitude' not in ds.dims:
#         ds = ds.rename({'lat':'latitude',
#                    'lon':'longitude'})
#     if 'la_max' in ex.keys() and 'la_min' in ex.keys():
#         if ds.latitude[0] > ds.latitude[1]:
#             slice_ = slice(ex['la_max'], ex['la_min'])
#         else:
#             slice_ = slice(ex['la_min'], ex['la_max'])
#         ds = ds.sel(latitude=slice_)
#     if 'lo_max' in ex.keys() and 'lo_min' in ex.keys():
#         ds = ds.sel(longitude=slice(ex['lo_min'], ex['lo_max']))
#
#     # get dates
#     numtime = ds['time']
#     dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
#
#     if numtime.attrs['calendar'] != 'gregorian':
#         dates = [d.strftime('%Y-%m-%d') for d in dates]
#     if ex['input_freq'] == 'monthly':
#         dates = [d.replace(day=1,hour=0) for d in pd.to_datetime(dates)]
#         ex['n_oneyr'] = np.unique(pd.to_datetime(dates).month).size
#     else:
#         dates = pd.to_datetime(dates)
#         stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')
#         test_if_fullyr = np.logical_and(dates[stepsyr.size-1].month == 12,
#                                     dates[stepsyr.size-1].day == 31)
#         assert test_if_fullyr, ('full is needed as raw data since rolling'
#                             ' mean is applied across timesteps')
#
#     dates = pd.to_datetime(dates)
#     # set hour to 00
#     if dates.hour[0] != 0:
#         dates -= pd.Timedelta(dates.hour[0], unit='h')
#
#
#
#     ds['time'] = dates
#
#     # mask away leapdays
#     dates_noleap = remove_leapdays(dates)
#     ds = ds.sel(time=dates_noleap)
#
#
#     return ds

def import_ds_lazy(filename, loadleap=False, seldates=None, selbox=None, format_lon='west_east'):
    ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ds.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    common_fields = ' time time_bnds longitude latitude lev lon lat level mask '
    var = [var for var in strvars if var not in common_fields][0]
    var = var.replace(' ', '')

    ds = ds[var].squeeze()

    if 'latitude' and 'longitude' not in ds.dims:
        ds = ds.rename({'lat':'latitude',
                  'lon':'longitude'})

    ds = convert_longitude(ds, format_lon)   
   

    if selbox is not None:
        if ds.latitude[0] > ds.latitude[1]:
            slice_ = slice(selbox['la_max'], selbox['la_min'])
        else:
            slice_ = slice(selbox['la_min'], selbox['la_max'])
        ds = ds.sel(latitude=slice_)
        min_lon = min([selbox['lo_min'], selbox['lo_max']])
        max_lon = max([selbox['lo_min'], selbox['lo_max']])
        ds = ds.sel(longitude=slice(min_lon, max_lon))

    # get dates
    numtime = ds['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])

    if (dates[1] - dates[0]).days == 1:
        input_freq = 'daily'
    elif (dates[1] - dates[0]).days == 30 or (dates[1] - dates[0]).days == 31:
        input_freq = 'monthly'
        
    if numtime.attrs['calendar'] != 'gregorian':
        dates = [d.strftime('%Y-%m-%d') for d in dates]
   
    if input_freq == 'monthly':
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
    else:
        ds = ds.sel(time=seldates)

    if loadleap==False:
        # mask away leapdays
        dates_noleap = remove_leapdays(pd.to_datetime(ds.time.values))
        ds = ds.sel(time=dates_noleap)
    return ds


def remove_leapdays(datetime):
    mask_lpyrfeb = np.logical_and((datetime.month == 2), (datetime.day == 29))

    dates_noleap = datetime[mask_lpyrfeb==False]
    return dates_noleap


def detrend_anom_ncdf3D(infile, outfile, loadleap=False, 
                        seldates=None, selbox=None, format_lon='west_east', 
                        detrend=True, anomaly=True, encoding=None):
    '''
    Function for preprocessing
    - Select time period of interest from daily mean time series
    - Calculate anomalies (w.r.t. multi year daily means)
    - linear detrend
    '''

    #%%
    import xarray as xr
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
            output[:,lev_idx,:,:] = detrend_xarray_ds_2D(ds_2D, detrend=detrend, anomaly=anomaly)
    else:
        output = detrend_xarray_ds_2D(ds, detrend=detrend, anomaly=anomaly)

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

def detrend_xarray_ds_2D(ds, detrend, anomaly):
    #%%
    import xarray as xr
    import numpy as np
#    marray = np.squeeze(ncdf.to_array(name=var))
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
        from scipy import signal
        # get trend of smoothened signal

        no_nans = np.nan_to_num(arr_oneday_smooth)
        detrended_sm = signal.detrend(no_nans, axis=0, type='linear')
        nan_true = np.isnan(arr_oneday)
        detrended_sm[nan_true] = np.nan
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
        window_s = min(25,int(stepsyr.size / 12))
        print('Performing {} day rolling mean with gaussian window (std={})'
              ' to get better interannual statistics'.format(window_s, window_s/2))

        print('using absolute anomalies w.r.t. climatology of '
              'smoothed concurrent day accross years')
        data_smooth =  rolling_mean_np(ds.values, window_s)



#    output_std = np.empty( (stepsyr.size,  ds.latitude.size, ds.longitude.size), dtype='float32' )
#    output_std[:] = np.nan
#    output_clim = np.empty( (stepsyr.size,  ds.latitude.size, ds.longitude.size), dtype='float32' )
#    output_clim[:] = np.nan
    output = np.empty( (ds.time.size,  ds.latitude.size, ds.longitude.size), dtype='float32' )
#    output[:] = np.nan


    for i in range(stepsyr.size):

        sliceyr = np.arange(i, ds.time.size, stepsyr.size)
        arr_oneday = ds.isel(time=sliceyr)
        arr_oneday_smooth = data_smooth[sliceyr]
        if detrend:
            if i==0: print('Detrending based on interannual trend of 25 day smoothened day of year\n')
            arr_oneday, detrended_sm = _detrendfunc2d(arr_oneday, arr_oneday_smooth)

#        output_std[i]  = arr_oneday.std(axis=0)
        if anomaly:
            if i==0: print('using absolute anomalies w.r.t. climatology of '
                           'smoothed concurrent day accross years\n')
            output_clim = arr_oneday_smooth.mean(axis=0)
            output[i::stepsyr.size] = arr_oneday - output_clim
        else:
            output[i::stepsyr.size] = arr_oneday

        progress = int((100*(i+1)/stepsyr.size))
        print(f"\rProcessing {progress}%", end="")

    print('writing ncdf file')

#    output_std_new = rolling_mean_np(output_std, 50)

#    plt.figure(figsize=(15,10)) ; plt.title('T2m at 66N, 24E. 1 day bins mean (39 years)');
#    plt.plot((output_clim[:,16,10]-output_clim[:,16,10].mean()))
#    plt.plot((output_clim_old[:,16,10]-output_clim_old[:,16,10].mean()))
#    plt.yticks(np.arange(-15,15,2.5)) ; plt.xticks(np.arange(0,366,25)) ; plt.grid(which='major') ;
#    plt.ylabel('Kelvin')

#    plt.figure(figsize=(15,10))
#    plt.plot(output_std[:,16,10], label='one day of year')
#    plt.plot(output_std_new[:,16,10], label='50 day smooth of blue line') ; plt.yticks(np.arange(3,7.5,0.25)) ; plt.xticks(np.arange(0,366,25)) ; plt.grid(which='major') ;
#    plt.legend()
#    plt.ylabel('Kelvin')


    #%%
    return output

def rolling_mean_np(arr, win, center=True):
    import scipy.signal.windows as spwin
    plt.plot(range(-int(win/2),+int(win/2)+1), spwin.gaussian(win, win/2))
    plt.title('window used for rolling mean')
    plt.xlabel('timesteps')
    df = pd.DataFrame(data=arr.reshape( (arr.shape[0], arr[0].size)))

    rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='gaussian').mean(std=win/2.)

    return rollmean.values.reshape( (arr.shape))


def convert_longitude(data, to_format='west_east'):
    '''
    to_format = 'only_east' or 'west_east'
    '''
    longitude = data.longitude
    if to_format == 'west_east':
        lon_above = longitude[np.where(longitude > 180)[0]]
        lon_normal = longitude[np.where(longitude <= 180)[0]]
        # roll all values to the right for len(lon_above amount of steps)
        data = data.roll(longitude=len(lon_above))
        # adapt longitude values above 180 to negative values
        substract = lambda x, y: (x - y)
        lon_above = xr.apply_ufunc(substract, lon_above, 360)
        if lon_normal.size != 0:
            if lon_normal[0] == 0.:
                convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')

            else:
                convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
        else:
            convert_lon = lon_above

    elif to_format == 'only_east':
        deg = float(abs(longitude[1] - longitude[0]))
        lon_above = longitude[np.where(longitude >= 0)[0]]
        lon_below = longitude[np.where(longitude < 0)[0]]
        lon_below += 360

        if min(lon_above) < deg:
            # crossing the meridional:
            data = data.roll(longitude=-len(lon_below))
            convert_lon = xr.concat([lon_below, lon_above], dim='longitude')
        else:
            # crossing - 180 line
            data = data.roll(longitude=len(lon_below))
            convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
    data['longitude'] = convert_lon
    return data

if __name__ == '__main__':
    ex = {}
    ex['input_freq'] = 'daily'
    DATAFOLDER = input('give path to datafolder where raw data in folder input_raw:\n')
    infilename  = input('give input filename you want to preprocess:\n')
    infile = os.path.join(DATAFOLDER, 'input_raw', infilename)
    outfilename = infilename.split('.nc')[0] + '_pp.nc'
    output_folder = os.path.join(DATAFOLDER, 'input_pp')
    if os.path.isdir(output_folder) != True : os.makedirs(output_folder)
    outfile = os.path.join(output_folder, outfilename)

    kwargs = {'detrend':True, 'anomaly':True}
    try:
        detrend_anom_ncdf3D(infile, outfile, ex, **kwargs)
    except:
        print('just chilling bro, relax..')
