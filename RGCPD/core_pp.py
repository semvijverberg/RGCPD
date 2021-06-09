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
from dateutil.relativedelta import relativedelta as date_dt
import itertools
import scipy.signal.windows as spwin
# from dateutil.relativedelta import relativedelta as date_dt
from collections import Counter
flatten = lambda l: list(set([item for sublist in l for item in sublist]))
flatten = lambda l: list(itertools.chain.from_iterable(l))
from typing import Union

def get_oneyr(dt_pdf_pds_xr, *args):
    if type(dt_pdf_pds_xr) == pd.DatetimeIndex:
        pddatetime = dt_pdf_pds_xr
    if type(dt_pdf_pds_xr) == pd.DataFrame or type(dt_pdf_pds_xr) == pd.Series:
        pddatetime = dt_pdf_pds_xr.index # assuming index of df is DatetimeIndex
    if type(dt_pdf_pds_xr) == xr.DataArray:
        pddatetime = pd.to_datetime(dt_pdf_pds_xr.time.values)


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



def import_ds_lazy(filepath: str, loadleap: bool=False,
                   seldates: Union[tuple, pd.DatetimeIndex]=None,
                   start_end_year: tuple=None, selbox: Union[list, tuple]=None,
                   format_lon='only_east', var=None, auto_detect_mask: bool=False,
                   dailytomonths: bool=False, verbosity=0):
    '''


    Parameters
    ----------
    filepath : str
        filepath to .nc file.
    seldates: tuple, pd.DatetimeIndex, optional
        default is None.
        if type is tuple: selecting data that fits within start- and enddate,
        format ('mm-dd', 'mm-dd'). default is ('01-01' - '12-31')
        if type is pd.DatetimeIndex: select that exact timeindex with
        xarray.sel(time=seldates)
        Default is None.
    start_end_year : tuple, optional
        default is to load all years
    loadleap : bool, optional
        If True also loads the 29-02 leapdays. The default is False.
    dailytomonths: bool, optional
        When True, the daily input data will be aggregated to monthly data.
        Default is False.
    selbox : Union[list, tuple], optional
        selbox assumes [lowest_east_lon, highest_east_lon, south_lat, north_lat].
        The default is None.
    format_lon : TYPE, optional
        'only_east' or 'west_east. The default is 'only_east'.
    var : str, optional
        variable name. The default is None.
    auto_detect_mask : bool, optional
        Detect mask based on NaNs. The default is False.
    verbosity : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    '''


    ds = xr.open_dataset(filepath, decode_cf=True, decode_coords=True, decode_times=False)

    lats = any([True if 'lat' in k else False for k in list(ds.dims.keys())])
    lons = any([True if 'lon' in k else False for k in list(ds.dims.keys())])
    if np.logical_and(lats, lons): # lat,lon coords in dataset
        multi_dims = True
    else:
        multi_dims = False

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

    # get dates
    if 'time' in ds.squeeze().dims:
        ds = ds_num2date(ds, loadleap=loadleap)

        ds = xr_core_pp_time(ds, seldates, start_end_year, loadleap,
                             dailytomonths)

    if multi_dims:
        if 'latitude' and 'longitude' not in ds.dims:
            ds = ds.rename({'lat':'latitude',
                            'lon':'longitude'})
            if 'time' in ds.squeeze().dims and len(ds.squeeze().dims) == 3:
                ds = ds.transpose('time', 'latitude', 'longitude')




        if auto_detect_mask:
            ds = detect_mask(ds)

        if format_lon is not None:
            if test_periodic(ds)==False and crossing0lon(ds)==False:
                format_lon = 'only_east'
            if _check_format(ds) != format_lon:
                ds = convert_longitude(ds, format_lon)

        # ensure longitude in increasing order
        minidx = np.where(ds.longitude == ds.longitude.min())[0]
        maxidx = np.where(ds.longitude == ds.longitude.max())[0]
        if bool(minidx > maxidx):
            print('sorting longitude')
            ds = ds.sortby('longitude')

        # ensure latitude is in increasing order
        minidx = np.where(ds.latitude == ds.latitude.min())[0]
        maxidx = np.where(ds.latitude == ds.latitude.max())[0]
        if bool(minidx > maxidx):
            print('sorting latitude')
            ds = ds.sortby('latitude')

        if selbox is not None:
            ds = get_selbox(ds, selbox, verbosity)


    # if type(ds) == type(xr.DataArray(data=[0])):
    #     ds.attrs['is_DataArray'] = 1
    # else:
    #     ds.attrs['is_DataArray'] = 0
    return ds

def xr_core_pp_time(ds, seldates: Union[tuple, pd.DatetimeIndex]=None,
                    start_end_year: tuple=None, loadleap: bool=False,
                    dailytomonths: bool=False):
    ''' Wrapper for some essentials for basic timeslicing and dailytomonthly
        aggregation

    ds : xr.DataArray or xr.Dataset
        input xarray with 'time' dimension
    seldates: tuple, pd.DatetimeIndex, optional
        default is None.
        if type is tuple: selecting data that fits within start- and enddate,
        format ('mm-dd', 'mm-dd'). default is ('01-01' - '12-31')
        if type is pd.DatetimeIndex: select that exact timeindex with
        xarray.sel(time=seldates)
    start_end_year : tuple, optional
        default is to load all years
    loadleap : TYPE, optional
        If True also loads the 29-02 leapdays. The default is False.
    dailytomonths:
        When True, the daily input data will be aggregated to monthly data.

    '''

    if type(seldates) is tuple or start_end_year is not None:
        pddates = get_subdates(dates=pd.to_datetime(ds.time.values),
                               start_end_date=seldates,
                               start_end_year=start_end_year,
                               lpyr=loadleap)
        ds = ds.sel(time=pddates)
    elif type(seldates) is pd.DatetimeIndex:
        # seldates are pd.DatetimeIndex
        ds = ds.sel(time=seldates)
    if dailytomonths:
        # resample annoying replaces datetimes between date gaps, dropna().
        ds = ds.resample(time='1M', skipna=True, closed='right',
                         label='right', restore_coord_dims=False
                         ).mean().dropna(dim='time', how='all')

        dtfirst = [s+'-01' for s in ds["time"].dt.strftime('%Y-%m').values]
        ds = ds.assign_coords({'time':pd.to_datetime(dtfirst)})
    return ds

def detect_mask(ds):
    '''
    Auto detect mask based on finding 20 percent of exactly equal values in
    first timestep
    '''
    if 'time' in ds.dims:
        firstyear = get_oneyr(ds).year[-1]
        secondyear = get_oneyr(ds, firstyear+1)[0]
        idx = list(ds.time.values).index(secondyear)
        field = ds[idx] # timestep 1, because of potential time-boundary affects,
        # e.g. when calculating SPI2
    else:
        field = ds
    fieldsize = field.size
    if np.unique(field).size < .8 * fieldsize:
        # more then 20% has exactly the same value
        val = [k for k,v in Counter(list(field.values.flatten())).items() if v>.2*fieldsize]
        assert len(val)!=0, f'No constant value found in field at timestep {idx}'
        mask = field.values == val
        if 'time' in ds.dims:
            ds = ds.where(np.repeat(mask[np.newaxis,:,:],ds.time.size,0)==False)
        else:
            ds = ds.where(mask)
    return ds

def remove_leapdays(datetime):
    mask_lpyrfeb = np.logical_and((datetime.month == 2), (datetime.day == 29))

    dates_noleap = datetime[mask_lpyrfeb==False]
    return dates_noleap

def ds_num2date(ds, loadleap=False):
    numtime = ds['time']
    dates = num2date(numtime, units=numtime.units,
                     calendar=numtime.attrs['calendar'],
                     only_use_cftime_datetimes=False)

    timestep_days = (dates[1] - dates[0]).days
    if timestep_days == 1:
        input_freq = 'daily'
    elif timestep_days in [28,29,30,31]:
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

    if loadleap==False:
        # mask away leapdays
        dates_noleap = remove_leapdays(pd.to_datetime(ds.time.values))
        ds = ds.sel(time=dates_noleap)
    return ds

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

    east_lon = selbox[0]
    west_lon = selbox[1]
    if (east_lon > west_lon and east_lon > 180) or (east_lon < 0 and east_lon!=-180):
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
    ds = ds.sel(latitude=slice_lat)
    return ds

def detrend_anom_ncdf3D(infile, outfile, loadleap=False,
                        seldates=None, selbox=None, format_lon='east_west',
                        auto_detect_mask=False, detrend=True, anomaly=True,
                        apply_fft=True, n_harmonics=6, encoding={}):
    '''
    Function for preprocessing
    - Calculate anomalies (by removing seasonal cycle based on first
      three harmonics)
    - linear long-term detrend
    '''
    # loadleap=False; seldates=None; selbox=None; format_lon='east_west';
    # auto_detect_mask=False; detrend=True; anomaly=True;
    # apply_fft=True; n_harmonics=6; encoding=None
    #%%
    ds = import_ds_lazy(infile, loadleap=loadleap,
                        seldates=seldates, selbox=selbox, format_lon=format_lon,
                        auto_detect_mask=auto_detect_mask)

    # check if 3D data (lat, lat, lev) or 2D
    check_dim_level = any([level in ds.dims for level in ['lev', 'level']])

    if check_dim_level:
        key = ['lev', 'level'][any([level in ds.dims for level in ['lev', 'level']])]
        levels = ds[key]
        output = np.empty( (ds.time.size,  ds.level.size, ds.latitude.size, ds.longitude.size), dtype='float32' )
        output[:] = np.nan
        for lev_idx, lev in enumerate(levels.values):
            ds_2D = ds.sel(level=lev)

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
    pp_dict = {'anomaly':str(anomaly), 'fft':str(apply_fft), 'n_harmonics':n_harmonics,
               'detrend':str(detrend)}
    output.attrs.update(pp_dict)
    # ensure mask
    output = output.where(output.values != 0.).fillna(-9999)
    encoding.update({'_FillValue': -9999})
    encoding_var = ( {ds.name : encoding} )
    mask =  (('latitude', 'longitude'), (output.values[0] != -9999) )
    output.coords['mask'] = mask
#    xarray_plot(output[0])

    # save netcdf
    output.to_netcdf(outfile, mode='w', encoding=encoding_var)
#    diff = output - abs(marray)
#    diff.to_netcdf(filename.replace('.nc', 'diff.nc'))
    #%%
    return

def detrend_lin_longterm(ds, plot=True, return_trend=False,
                         NaN_interpolate='spline', spline_order=5):
    offset_clim = np.mean(ds, 0)
    dates = pd.to_datetime(ds.time.values)
    detrended = sp.signal.detrend(np.nan_to_num(ds), axis=0, type='linear')
    detrended[np.repeat(np.isnan(offset_clim).expand_dims('t').values,
                        dates.size, 0 )] = np.nan # restore NaNs
    detrended += np.repeat(offset_clim.expand_dims('time'), dates.size, 0 )
    detrended = detrended.assign_coords(
                coords={'time':dates})
    if plot:
        _check_trend_plot(ds, detrended)

    if return_trend:
        out = ( detrended,  (ds - detrended)+offset_clim )
    else:
        out = detrended
    return out

def _check_trend_plot(ds, detrended):
    if len(ds.shape) > 2:
        # plot single gridpoint for visual check
        always_NaN_mask = np.isnan(ds).all(axis=0)
        lats, lons = np.where(~always_NaN_mask)
        tuples = np.stack([lats, lons]).T
        tuples = tuples[::max(1,int(len(tuples)/3))]
        # tuples = np.stack([lats, lons]).T
        fig, ax = plt.subplots(len(tuples), figsize=(8,8))
        for i, lalo in enumerate(tuples):
            ts = ds[:,lalo[0],lalo[1]]
            la = lalo[0]
            lo = lalo[1]
            # while bool(np.isnan(ts).all()):
            #     lo += 5
            #     try:
            #         ts = ds[:,la,lo]
            #     except:

            lat = int(ds.latitude[la])
            lon = int(ds.longitude[lo])
            print(f"\rVisual test latlon {lat} {lon}", end="")

            ax[i].set_title(f'latlon coord {lat} {lon}')
            ax[i].plot(ts)
            ax[i].plot(detrended[:,la,lo])
            trend1d = ts - detrended[:,la,lo]
            linregab = np.polyfit(np.arange(trend1d.size), trend1d, 1)
            linregab = np.insert(linregab, 2, float(trend1d[-1] - trend1d[0]))
            ax[i].plot(trend1d)#+offset_clim[la,lo])
            ax[i].text(.05, .05,
            'y = {:.2g}x + {:.2g}, max diff: {:.2g}'.format(*linregab),
            transform=ax[i].transAxes)
        plt.subplots_adjust(hspace=.5)
        ax[-1].text(.5,1.2, 'Visual analysis: trends of nearby gridcells should be similar',
                    transform=ax[0].transAxes,
                    ha='center', va='bottom')
    elif len(ds.shape) == 1:
        fig, ax = plt.subplots(1, figsize=(8,4))
        ax.set_title('detrend 1D ts')
        ax.plot(ds.values)
        ax.plot(detrended)
        trend1d = ds - detrended
        linregab = np.polyfit(np.arange(trend1d.size), trend1d, 1)
        linregab = np.insert(linregab, 2, float(trend1d[-1] - trend1d[0]))
        ax.plot(trend1d)#+offset_clim)
        ax.text(.05, .05,
        'y = {:.2g}x + {:.2g}, max diff: {:.2g}'.format(*linregab),
        transform=ax.transAxes)
    else:
        pass

def to_np(data):
    if type(data) is pd.DataFrame:
        kwrgs = {'columns':data.columns, 'index':data.index};
        input_dtype = pd.DataFrame
    elif type(data) is xr.DataArray:
        kwrgs= {'coords':data.coords, 'dims':data.dims, 'attrs':data.attrs,
                'name':data.name}
        input_dtype = xr.DataArray
    if type(data) is not np.ndarray:
        data = data.values # attempt to make np.ndarray (if xr.DA of pd.DF)
    else:
        input_dtype = np.ndarray ; kwrgs={}
    return data, kwrgs, input_dtype

def back_to_input_dtype(data, kwrgs, input_dtype):
    if input_dtype is pd.DataFrame:
        data = pd.DataFrame(data, **kwrgs)
    elif input_dtype is xr.DataArray:
        data = xr.DataArray(data, **kwrgs)
    return data

def NaN_handling(data, method: str='quadratic', NaN_limit: float=None,
                 missing_data_ts_to_nan: Union[bool, float, int]=False):
    '''
    Interpolate or mask NaNs in data

    Parameters
    ----------
    data : xr.DataArray, pd.DataFrame or np.ndarray
        input data.
    method : str, optional
        method used to interpolate NaNs, build upon
        pd.DataFrame().interpolate(method=method).

        Interpolation technique to use:
        ‘linear’: Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
        ‘time’: Works on daily and higher resolution data to interpolate given length of interval.
        ‘index’, ‘values’: use the actual numerical values of the index.
        ‘pad’: Fill in NaNs using existing values.
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’: Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
        ‘krogh’, ‘piecewise_polynomial’, ‘spline’, ‘pchip’, ‘akima’, ‘cubicspline’: Wrappers around the SciPy interpolation methods of similar names. See Notes.
        If str is None of the valid option, will raise an ValueError.
        The default is 'quadratic'.
    NaN_limit : float, optional
        Limit the amount of consecutive NaNs. The default is None (no limit)
        The default is False.
    missing_data_ts_to_nan : bool, float, int
        Will mask complete timeseries to np.nan if more then a percentage (if float)
        or more then integer (if int) of NaNs are present in timeseries.

    Raises
    ------
    ValueError
        If NaNs are not allowed (method=False).

    Returns
    -------
    data : xr.DataArray, pd.DataFrame or np.ndarray
        data with interpolated / masked NaNs.

    references:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html

    '''

    data, kwrgs_dtype, input_dtype = to_np(data)

    orig_shape = data.shape
    data = data.reshape(orig_shape[0], -1)

    if type(missing_data_ts_to_nan) is float: # not allowed more then % of NaNs
        NaN_mask = np.isnan(data).sum(0) >= missing_data_ts_to_nan * orig_shape[0]
    elif type(missing_data_ts_to_nan) is int: # not allowed more then int NaNs
        NaN_mask = np.isnan(data).sum(0) >= missing_data_ts_to_nan
    else:
        NaN_mask = np.isnan(data).all(axis=0) # Only mask NaNs every timestep
    data[:,NaN_mask] = np.nan

    # interpolating NaNs
    t, o = np.where(np.isnan(data[:,~NaN_mask])) # NaNs some timesteps
    if t.size > 0:
        n_sparse_nans = t.size
        print(f'Warning: {n_sparse_nans} NaNs found at {np.unique(o).size} location(s)')
        if method is not False:
            print(f'Sparse NaNs will be interpolated using {method} spline (pandas)\n')
            if NaN_limit is not None:
                limit = int(orig_shape[0]*NaN_limit)
            else:
                limit=None
            try:
                data[:,~NaN_mask] = pd.DataFrame(data[:,~NaN_mask]).interpolate(method=method, limit=limit).values
            except:
                print(f'{method} spline gave error, reverting to linear interpolation')
                data[:,~NaN_mask] = pd.DataFrame(data[:,~NaN_mask]).interpolate(method='linear', limit=limit).values
        else:
            raise ValueError('NaNs not allowed')

    data = data.reshape(orig_shape)

    data = back_to_input_dtype(data, kwrgs_dtype, input_dtype)
    return data


def detrend(data, method='linear', kwrgs_detrend: dict={}, return_trend: bool=False,
            plot: bool=True):
    '''
    Wrapper supporting linear and loess detrending on xr.DataArray, pd.DataFrame
    & np.ndarray. Note, linear detrending is way faster then loess detrending.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    method : str, optional
        Choose detrending method ['linear', 'loess']. The default is 'linear'.
    kwrgs_detrend : dict, optional
        Kwrgs for loess detrending. The default is {'alpha':.75, 'order':2}.
    return_trend : bool, optional
        Return trend timeseries. The default is False.
    NaN_interpolate : str, optional
        method used to interpolate NaNs, build upon
        pd.DataFrame().interpolate(method=NaN_interpolate).

        Interpolation technique to use. One of:
        ‘linear’: Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
        ‘time’: Works on daily and higher resolution data to interpolate given length of interval.
        ‘index’, ‘values’: use the actual numerical values of the index.
        ‘pad’: Fill in NaNs using existing values.
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’: Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
        ‘krogh’, ‘piecewise_polynomial’, ‘spline’, ‘pchip’, ‘akima’, ‘cubicspline’: Wrappers around the SciPy interpolation methods of similar names. See Notes.
        If str is None of the valid option, will raise an ValueError.
        The default is 'quadratic'.
    NaN_limit : float, optional
        Limit the amount of consecutive NaNs. The default is None (no limit)
    plot : bool, optional
        Plot only available for method='linear' and type(data)=xr.DA.
        The default is True.

    Raises
    ------
    ValueError
        if NaNs not allowed.

    Returns
    -------
    Returns same dtype as input (xr.DataArray, pd.DataFrame or np.ndarray)
    if return_trend=True:
        out = (detrended, trend)
    else:
        out = detrended
    '''
    # method='linear'; kwrgs_detrend={}; return_trend=False;NaN_interpolate='quadratic'
    # NaN_limit=None;plot=True
    #%%

    if method == 'loess':
        kwrgs_d = {'alpha':0.75, 'order':2} ; kwrgs_d.update(kwrgs_detrend)

    if type(data) is pd.DataFrame:
        columns = data.columns ; index = data.index ; to_df = True; to_DA=False
    elif type(data) is xr.DataArray:
        coords = data.coords ; dims = data.dims ; attrs = data.attrs ;
        name = data.name ; to_DA = True ; to_df = False
    if type(data) is not np.ndarray:
        data = data.values # attempt to make np.ndarray (if xr.DA of pd.DF)
    else:
        to_DA = False ; to_df = False

    orig_shape = data.shape
    data = data.reshape(orig_shape[0], -1)
    always_NaN_mask = np.isnan(data).all(axis=0) # NaN every timestep

    if return_trend:
        trend_ts = np.zeros( (data.shape), dtype=np.float16)
        trend_ts[:,always_NaN_mask] = np.nan

    # # dealing with NaNs
    # t, o = np.where(np.isnan(data[:,~always_NaN_mask])) # NaNs some timesteps
    # if t.size > 0:
    #     n_sparse_nans = t.size
    #     print(f'Warning: {n_sparse_nans} NaNs found at {np.unique(o).size} location(s)')
    #     if NaN_interpolate is not False:
    #         print(f'Sparse NaNs will be interpolated using the pandas {NaN_interpolate}\n')
    #         if NaN_limit is not None:
    #             limit = int(orig_shape[0]*NaN_limit)
    #         else:
    #             limit=None
    #         try:
    #             data[:,~always_NaN_mask] = pd.DataFrame(data[:,~always_NaN_mask]).interpolate(method=NaN_interpolate, limit=limit).values
    #         except:
    #             print(f'{NaN_interpolate} spline gave error, reverting to linear interpolation')
    #             data[:,~always_NaN_mask] = pd.DataFrame(data[:,~always_NaN_mask]).interpolate(method='linear', limit=limit).values

    #     else:
    #         raise ValueError('NaNs not allowed')

    print(f'Start {method} detrending ...\n', end="")
    if method == 'loess':
        not_always_NaN_idx = np.where(~always_NaN_mask)
        last_index = not_always_NaN_idx[0].max() ; #div = round(last_index/40)
        div = min(20,last_index)
        for i_ts in not_always_NaN_idx[0]:
            if i_ts % div==0: # print 40 steps
                progress = int((100*(i_ts)/last_index))
                print(f"\rProcessing {progress}%", end="")
            ts = data[:,i_ts]
            if ts[np.isnan(ts)].size != 0: # if still NaNs: fill with NaN
                data[:,i_ts] = np.nan
                continue
            else:
                ts = _fit_loess(ts, **kwrgs_d)
            if return_trend:
                trend_ts[:,i_ts] = ts
            data[:,i_ts] -= ts
    elif method == 'linear':
        offset_clim = np.mean(data, 0)
        if return_trend == False and plot == False:
            data[:,~always_NaN_mask] = sp.signal.detrend(data[:,~always_NaN_mask], axis=0, type='linear')
            data += np.repeat(np.isnan(offset_clim)[np.newaxis,:], orig_shape[0], 0 )
        elif return_trend or plot:
            detrended = data.copy()
            detrended[:,~always_NaN_mask] = sp.signal.detrend(detrended[:,~always_NaN_mask], axis=0, type='linear')
            detrended += np.repeat(np.isnan(offset_clim)[np.newaxis,:], orig_shape[0], 0 )
            trend_ts = (data - detrended)
        print('Done')
    data = data.reshape(orig_shape)
    if return_trend:
        trend_ts = trend_ts.reshape(orig_shape)


    if to_df:
        data = pd.DataFrame(data, index=index, columns=columns)
        if return_trend:
            trend_ts = pd.DataFrame(trend_ts, index=index, columns=columns)
    elif to_DA:
        data = xr.DataArray(data, coords=coords, dims=dims, attrs=attrs, name=name)
        if return_trend:
            trend_ts = xr.DataArray(trend_ts, coords=coords, dims=dims, attrs=attrs,
                                name=name)
        if plot and method=='linear':
            _check_trend_plot(data, detrended.reshape(orig_shape))

    if return_trend:
        out = (data, trend_ts)
    else:
        out = data
    #%%
    return out



def _fit_loess(y, X=None, alpha=0.75, order=2):
    """
    Local Polynomial Regression (LOESS)

    Performs a LOWESS (LOcally WEighted Scatter-plot Smoother) regression.


    Parameters
    ----------
    y : list, array or Series
        The response variable (the y axis).
    X : list, array or Series
        Explanatory variable (the x axis). If 'None', will treat y as a continuous signal (useful for smoothing).
    alpha : float
        The parameter which controls the degree of smoothing, which corresponds
        to the proportion of the samples to include in local regression.
    order : int
        Degree of the polynomial to fit. Can be 1 or 2 (default).

    Returns
    -------
    array
        Prediciton of the LOESS algorithm.

    See Also
    ----------
    signal_smooth, signal_detrend, fit_error

    Examples
    ---------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000))
    >>> distorted = nk.signal_distort(signal, noise_amplitude=[0.3, 0.2, 0.1], noise_frequency=[5, 10, 50])
    >>>
    >>> pd.DataFrame({ "Raw": distorted, "Loess_1": nk.fit_loess(distorted, order=1), "Loess_2": nk.fit_loess(distorted, order=2)}).plot() #doctest: +SKIP

    References
    ----------
    - copied from: https://neurokit2.readthedocs.io/en/master/_modules/neurokit2/stats/fit_loess.html#fit_loess
    - https://simplyor.netlify.com/loess-from-scratch-in-python-animation.en-us/

    """
    if X is None:
        X = np.linspace(0, 100, len(y))

    assert order in [1, 2], "Deg has to be 1 or 2"
    assert (alpha > 0) and (alpha <= 1), "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"

    X_domain = X

    n = len(X)
    span = int(np.ceil(alpha * n))

    y_predicted = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)

    for i, val in enumerate(X_domain):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)

        Nx = X[ind[:span]]
        Ny = y[ind[:span]]

        delx0 = sorted_dist[span - 1]

        u = distance[ind[:span]] / delx0
        w = (1 - u ** 3) ** 3

        W = np.diag(w)
        A = np.vander(Nx, N=1 + order)

        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = sp.linalg.qr(V)
        p = sp.linalg.solve_triangular(R, np.matmul(Q.T, Y))

        y_predicted[i] = np.polyval(p, val)
        x_space[i] = val

    return y_predicted

def detrend_xarray_ds_2D(ds, detrend, anomaly, apply_fft=False, n_harmonics=6):
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

    if anomaly:
        if (stepsyr.day== 1).all() == True or int(ds.time.size / 365) >= 120:
            print('\nHandling time series longer then 120 day or monthly data, no smoothening applied')
            data_smooth = ds.values
            if (stepsyr[1] - stepsyr[0]).days in [28,29,30,31]:
                window_s = False

        elif (stepsyr.day== 1).all() == False and int(ds.time.size / 365) < 120:
            window_s = max(min(25,int(stepsyr.size / 12)), 1)
            # print('Performing {} day rolling mean'
            #       ' to get better interannual statistics'.format(window_s))
            from time import time
            start = time()
            print('applying rolling mean, beware: memory intensive')
            data_smooth =  rolling_mean_np(ds.values, window_s, win_type='boxcar')
            # data_smooth_xr = ds.rolling(time=window_s, min_periods=1,
            #                             center=True).mean(skipna=False)
            passed = time() - start / 60

        output_clim3d = np.zeros((stepsyr.size, ds.latitude.size, ds.longitude.size),
                                   dtype='float32')

        for i in range(stepsyr.size):

            sliceyr = np.arange(i, ds.time.size, stepsyr.size)
            arr_oneday_smooth = data_smooth[sliceyr]

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
        elif apply_fft==False:
            output = ds - np.tile(output_clim3d, (int(dates.size/stepsyr.size), 1, 1))
    else:
        output = ds


    # =============================================================================
    # test gridcells:
    # =============================================================================
    # # try to find location above EU
    # ts = ds.sel(longitude=30, method='nearest').sel(latitude=40, method='nearest')
    # la1 = np.argwhere(ts.latitude.values ==ds.latitude.values)[0][0]
    # lo1 = np.argwhere(ts.longitude.values ==ds.longitude.values)[0][0]
    if anomaly:
        la1 = int(ds.shape[1]/2)
        lo1 = int(ds.shape[2]/2)
        la2 = int(ds.shape[1]/3)
        lo2 = int(ds.shape[2]/3)

        tuples = [[la1, lo1], [la1+1, lo1],
                  [la2, lo2], [la2+1, lo2]]
        if apply_fft:
            fig, ax = plt.subplots(4,2, figsize=(16,8))
        else:
            fig, ax = plt.subplots(2,2, figsize=(16,8))
        ax = ax.flatten()
        for i, lalo in enumerate(tuples):
            ts = ds[:,lalo[0],lalo[1]]
            while bool(np.isnan(ts).all()):
                lalo[1] += 5
                ts = ds[:,lalo[0],lalo[1]]
            lat = int(ds.latitude[lalo[0]])
            lon = int(ds.longitude[lalo[1]])
            print(f"\rVisual test latlon {lat} {lon}", end="")

            if window_s == False: # no daily data
                rawdayofyear = ts.groupby('time.month').mean('time')
            else:
                rawdayofyear = ts.groupby('time.dayofyear').mean('time')

            ax[i].set_title(f'latlon coord {lat} {lon}')
            for yr in np.unique(dates.year):
                singleyeardates = get_oneyr(dates, yr)
                ax[i].plot(ts.sel(time=singleyeardates), alpha=.1, color='purple')

            if window_s is not None:
                ax[i].plot(output_clim3d[:,lalo[0],lalo[1]], color='green', linewidth=2,
                     label=f'clim {window_s}-day rm')
            ax[i].plot(rawdayofyear, color='black', alpha=.6,
                       label='clim mean dayofyear')
            if apply_fft:
                ax[i].plot(clim_rec[:,lalo[0],lalo[1]][:365], 'r-',
                           label=f'fft {n_harmonics}h on (smoothened) data')
                diff = clim_rec[:,lalo[0],lalo[1]][:singleyeardates.size] - output_clim3d[:,lalo[0],lalo[1]]
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
        (summer.mean(dim='time') / summer.std(dim='time')).plot(ax=ax,
                                                                vmin=-3,vmax=3,
                                                                cmap=plt.cm.bwr)
        ax.set_title('summer composite mean [in std]')
    print('\n')


    if detrend:
        print('Detrending ...')
        output = detrend_lin_longterm(output)

    #%%
    return output
#%%
def rolling_mean_np(arr, win, center=True, win_type='boxcar'):


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

    options = np.array(np.meshgrid(ds.latitude[1:-1], ds.longitude[1:-1])).T.reshape(-1,2)
    la1, lo1 = options[int(options.shape[0]/3)]
    la2, lo2 = options[int(2*options.shape[0]/3)]
    tuples = [(la1, lo1), (la1+1, lo1), (la1, lo1+1),
              (la2, lo2), (la2+1, lo2), (la2, lo2+1)]
    fig, ax = plt.subplots(3,2, figsize=(16,8))
    ax = ax.flatten() ;
    for i, lalo in enumerate(tuples):
        lalo = (np.where(ds.latitude==lalo[0])[0][0], np.where(ds.longitude==lalo[1])[0][0])
        ts = ds[:,lalo[0],lalo[1]]
        while bool(np.isnan(ts).all()):
            lalo[1] += 5
            ts = ds[:,lalo[0],lalo[1]]
        lat = int(ds.latitude[lalo[0]])
        lon = int(ds.longitude[lalo[1]])
        print(f"\rVisual test latlon {lat} {lon}", end="")


        ax[i].set_title(f'latlon coord {lat} {lon}')
        for yr in np.unique(dates.year):
            singleyeardates = get_oneyr(dates, yr)
            ax[i].plot(ts.sel(time=singleyeardates), alpha=.1, color='purple')
        if anomaly:
            ax[i].plot(reconstructed_signal[:singleyeardates.size, lalo[0],lalo[1]].real,
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

def test_periodic(ds):
    dlon = ds.longitude[1] - ds.longitude[0]
    return (360 / dlon == ds.longitude.size).values

def test_periodic_lat(ds):
    dlat = abs(ds.latitude[1] - ds.latitude[0])
    return ((180/dlat)+1 == ds.latitude.size).values

def crossing0lon(ds):
    dlon = ds.longitude[1] - ds.longitude[0]
    return ds.sel(longitude=0, method='nearest').longitude < dlon

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

def make_dates(datetime, years):
    '''
    Extend same date period to other years
    datetime is start year
    start_yr is date period to 'copy'
    '''

    start_yr = datetime
    next_yr = start_yr
    for yr in years:
        delta_year = yr - start_yr[-1].year
        if delta_year >= 1:
            next_yr = pd.to_datetime([date + date_dt(years=delta_year) for date in next_yr])
            start_yr = start_yr.append(next_yr)

    return start_yr

def get_subdates(dates, start_end_date=None, start_end_year=None, lpyr=False,
                 returngroups=False, input_freq: str=None):
    #%%
    '''
    dates is type pandas.core.indexes.datetimes.DatetimeIndex
    start_end_date is tuple of start- and enddate in format ('mm-dd', 'mm-dd')
    lpyr is boolean if you want load the leap days yes or no.
    '''
    #%%
    if start_end_year is None:
        startyr = dates.year.min()
        endyr   = dates.year.max()
        start_end_year = (startyr, endyr)
    else:
        startyr = start_end_year[0]
        endyr   = start_end_year[-1]
    # n_yrs = np.arange(startyr, endyr+1).size
    firstyr = get_oneyr(dates, startyr)

    if start_end_date is None:
        start_end_date = (('{:02d}-{:02d}'.format(dates[0].month,
                                                    dates[0].day)),
                         ('{:02d}-{:02d}'.format(dates[-1].month,
                                                    dates[-1].day)))

    sstartdate = pd.to_datetime(str(startyr) + '-' + start_end_date[0])
    senddate_   = pd.to_datetime(str(startyr) + '-' + start_end_date[1])
    crossyr = sstartdate > senddate_
    if crossyr: senddate_+=date_dt(years=1) ; #startyr-=1

    #find closest senddate
    if crossyr:
        closedrightyr = get_oneyr(dates, startyr+1)
    else:
        closedrightyr = firstyr
    closest_enddate_idx = np.argmin(abs(closedrightyr - senddate_))
    senddate = closedrightyr[closest_enddate_idx]
    if senddate > senddate_ :
        senddate = closedrightyr[closest_enddate_idx-1]


    tfreq = (dates[1] - dates[0]).days
    if dates.is_leap_year[0]:
        leapday = pd.to_datetime([f'{dates[1].year}-02-29'])
        if dates[1] >= leapday and dates[0] < leapday:
            tfreq -= 1 # adjust leapday when calc difference in days

    oneyr_dates = pd.date_range(start=sstartdate, end=senddate_,
                                    freq=pd.Timedelta(1, 'd'))
    if input_freq is None: # educated guess on input freq
        if tfreq in [28,29,30,31]: # monthly timeseries
            input_freq = 'monthly'
        else:
            input_freq = 'daily_or_annual_or_yearly'


    if 'month' in input_freq: # monthly timeseries
        yr_mon = np.unique(np.stack([oneyr_dates.year.values,
                                     oneyr_dates.month.values]).T,
                                     axis=0)
        start_yr = pd.to_datetime([f'{ym[0]}-{ym[1]}-01' for ym in yr_mon])
    else:
        daily_yr_fit = np.round(oneyr_dates.size / tfreq, 0)

        sstartdate = senddate - pd.Timedelta(int(tfreq * daily_yr_fit), 'd')
        while sstartdate < pd.to_datetime(str(startyr) + '-' + start_end_date[0]):
            daily_yr_fit -=1
            sstartdate = senddate - pd.Timedelta(int(tfreq * daily_yr_fit), 'd')

        start_yr = remove_leapdays(pd.date_range(start=sstartdate, end=senddate,
                                    freq=pd.Timedelta(tfreq, unit='day')))

    datessubset = make_dates(start_yr, np.arange(startyr, endyr+1))
    if tfreq == 1: # only check for daily data
        datessubset = pd.to_datetime([d for d in datessubset if d in dates])
    if lpyr:
        datessubset = remove_leapdays(datessubset)
    if crossyr:
        periodgroups = np.repeat(np.arange(startyr+1, endyr+1), start_yr.size)
    else:
        periodgroups = np.repeat(np.arange(startyr, endyr+1), start_yr.size)

    # old crossyr handling
    # if crossyr: # crossyr such as DJF, not possible for 1st yr
    #     datessubset = datessubset[periodgroups!=periodgroups[0]] # Del group 1st yr
    #     periodgroups = periodgroups[periodgroups!=periodgroups[0]]
    if returngroups:
        out = (datessubset, periodgroups)
    else:
        out = datessubset
    #%%
    return out

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

