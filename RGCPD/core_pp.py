#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:31:11 2019

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import itertools
from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: list(set([item for sublist in l for item in sublist]))
flatten = lambda l: list(itertools.chain.from_iterable(l))
from typing import Union

def get_oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()



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
            ds = ds.sortby('longitude')

        # ensure latitude is in increasing order
        if np.where(ds.latitude == ds.latitude.min()) > np.where(ds.latitude == ds.latitude.max()):
            ds = ds.sortby('latitude')

        if selbox is not None:
            ds = get_selbox(ds, selbox, verbosity)


    # get dates
    if 'time' in ds.squeeze().dims:
        from netCDF4 import num2date
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
                        detrend=True, anomaly=True, encoding=None):
    '''
    Function for preprocessing
    - Select time period of interest from daily mean time series
    - Calculate anomalies (w.r.t. multi year daily means)
    - linear detrend
    '''

    #%%
    import xarray as xr
    from netCDF4 import num2date
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
#%%
def rolling_mean_np(arr, win, center=True):
    import scipy.signal.windows as spwin
    plt.plot(range(-int(win/2),+round(win/2+.49)), spwin.gaussian(win, win/2))
    plt.title('window used for rolling mean')
    plt.xlabel('timesteps')
    df = pd.DataFrame(data=arr.reshape( (arr.shape[0], arr[0].size)))

    rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='gaussian').mean(std=win/2.)

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


# def convert_longitude(data, to_format='east_west'):
#     '''
#     to_format = 'only_east' or 'east_west'
#     '''
#     longitude = data.longitude
#     if to_format == 'east_west':
#         lon_above = longitude[np.where(longitude > 180)[0]]
#         lon_normal = longitude[np.where(longitude <= 180)[0]]
#         # roll all values to the right for len(lon_above amount of steps)
#         data = data.roll(longitude=len(lon_above), roll_coords=False)
#         # adapt longitude values above 180 to negative values
#         substract = lambda x, y: (x - y)
#         lon_above = xr.apply_ufunc(substract, lon_above, 360)
#         if lon_normal.size != 0:
#             if lon_normal[0] == 0.:
#                 convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')

#             else:
#                 convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
#         else:
#             convert_lon = lon_above

#     elif to_format == 'only_east':
#         deg = float(abs(longitude[1] - longitude[0]))
#         lon_above = longitude[np.where(longitude >= 0)[0]]
#         lon_below = longitude[np.where(longitude < 0)[0]]
#         lon_below = lon_below.assign_coords(longitude=lon_below.values +360)
#         lon_below.values
#         if lon_above.size != 0:
#             if min(lon_above) < deg:
#                 # crossing the meridional:
#                 data = data.roll(longitude=-len(lon_below), roll_coords=False)
#                 convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
#             else:
#                 # crossing - 180 line
#                 data = data.roll(longitude=len(lon_below), roll_coords=False)
#                 convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
#         else:
#             # crossing - 180 line
#             data = data.roll(longitude=len(lon_below), roll_coords=False)
#             convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
#     data['longitude'] = convert_lon
#     return data


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



    start_yr = pd.date_range(start=sstartdate, end=senddate,
                                freq=(dates[1] - dates[0]))
    if lpyr==True and calendar.isleap(startyr):
        start_yr -= pd.Timedelta( '1 days')
    else:
        pass
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
