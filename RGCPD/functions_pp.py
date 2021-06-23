#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import itertools
import core_pp
import datetime

from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: list(set([item for sublist in l for item in sublist]))
flatten = lambda l: list(itertools.chain.from_iterable(l))

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

def perform_post_processing(list_of_name_path, kwrgs_pp=None, verbosity=1):
    '''
    if argument of kwrgs_pp is list, then the first item is assumed to be the
    default argument value, the second item should be a dict contaning the
    {varname : alternative_argument}.
    '''

    list_precur_pp = []
    for idx, (name, infile) in enumerate(list_of_name_path[1:]):
        # update from kwrgs_pp for variable {name}
        kwrgs = {}
        for key, value in kwrgs_pp.items():
            if type(value) is list and name in value[1].keys():
                kwrgs[key] = value[1][name]
            elif type(value) is list and name not in value[1].keys():
                kwrgs[key] = value[0] # plugging in default value
            else:
                kwrgs[key] = value

        outfile = check_pp_done(name, infile, kwrgs_load=kwrgs)
        list_precur_pp.append( (name, outfile) )
        if os.path.isfile(outfile) == True:
            if verbosity == 1:
                print('Loaded pre-processed data of {}\n'.format(name))
            pass
        else:
            print('\nPerforming pre-processing {}'.format(name))


            core_pp.detrend_anom_ncdf3D(infile, outfile, **kwrgs)
    return list_precur_pp

def load_TV(list_of_name_path, name_ds='ts'):
    '''
    function will load first item of list_of_name_path
    list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]

    if TVpath refers to .npy file:
        it tries to import the variable {TVname}
    if TVpath refers to .nc file:
        it tries to import the timeseries of cluster {TVname}.
    if TVpath is pd.DataFrame:
        fulltso becomes first colummn of DataFrame. Index should be correct
        pd.DatetimeIndex.



    returns:
    fulltso : xr.DataArray() 1-d full timeseries
    '''
    name = list_of_name_path[0][0]
    filename = list_of_name_path[0][1]
    if type(filename) is str:
        if filename.split('.')[-1] == 'npy':
            fulltso = load_npy(filename, name=name)
        elif filename.split('.')[-1] == 'nc':
            ds = core_pp.import_ds_lazy(filename)
            if len(ds.dims) > 1:
                fulltso = ds[name_ds].sel(cluster=name)
            else:
                if type(ds) is xr.Dataset:
                    fulltso = ds.to_array(name=name_ds)
                else:
                    fulltso = ds
                fulltso = fulltso.squeeze()
        elif filename.split('.')[-1] == 'h5':
            dict_df = load_hdf5(filename)
            df = dict_df[list(dict_df.keys())[0]]
            based_on_test = True
            if hasattr(df.index, 'levels'):
                splits = df.index.levels[0]
                if splits.size == 1:
                    based_on_test = False
                    df = df.loc[0]
                if based_on_test:
                    print('Get test timeseries of target pd.DataFrame')
                    df = get_df_test(df)
                else:
                    df = df.mean(axis=0, level=1)
                    print('calculate mean of different train-test folds')
            df = df[[name_ds]] ; df.index.name = 'time'
            fulltso = df.to_xarray().to_array(name=name_ds).squeeze()
        hashh = filename.split('_')[-1].split('.')[0]
    elif type(filename) is pd.DataFrame:
        df_fulltso = filename.iloc[:,[0]] ; name_ds = df_fulltso.columns[0] ;
        df_fulltso.index.name = 'time' ; hashh = None
        fulltso = df_fulltso.to_xarray().to_array(name=name_ds).squeeze()
    else:
        print('Not a valid datatype for TV path. See functions_pp.load_TV?')
    fulltso.name = str(list_of_name_path[0][0])+name_ds
    return fulltso, hashh

def process_TV(fullts, tfreq, start_end_TVdate, start_end_date=None,
               start_end_year=None, RV_detrend=False, RV_anomaly=False,
               ext_annual_to_mon=True, TVdates_aggr: bool=False,
               dailytomonths: bool=False, verbosity=1):
    # fullts=rg.fulltso.copy();RV_detrend=False;RV_anomaly=False;verbosity=1;
    # ext_annual_to_mon=False;TVdates_aggr=False; start_end_date=None; start_end_year=None,
    # dailytomonths=False

    # For some incredibly inexplicable reason, fullts was not pickle, even after
    # copy or deepcopying the object. So I recreate the object now:
    fullts = xr.DataArray(fullts.values, dims=fullts.dims, coords=fullts.coords,
                          name=fullts.name)


    # start_end_year selection done on fulltso in func above, but not when re-aggregating
    fullts = core_pp.xr_core_pp_time(fullts, start_end_year=start_end_year,
                                     dailytomonths=dailytomonths)

    if RV_detrend==True: # do detrending on all timesteps
        fullts = core_pp.detrend_wrapper(fullts) # default linear method
    elif type(RV_detrend) is dict:
        fullts = core_pp.detrend_wrapper(fullts, kwrgs_detrend=RV_detrend)
    if RV_anomaly: # do anomaly on complete timeseries (rolling mean applied!)
        fullts = anom1D(fullts)

    dates = pd.to_datetime(fullts.time.values)
    startyear = dates.year[0]
    endyear = dates.year[-1]
    n_timesteps = dates.size
    n_yrs       = (endyear - startyear) + 1

    # align fullts with precursor import_ds_lazy()
    fullts = core_pp.xr_core_pp_time(fullts, seldates=start_end_date)
    # fullts = fullts.sel(time=core_pp.get_subdates(dates=dates,
    #                                               start_end_date=start_end_date))

    timestep_days = (dates[1] - dates[0]).days
    # if type(tfreq) == int: # timemeanbins between start_end_date
    if timestep_days == 1:
        input_freq = 'daily'
        same_freq = (dates[1] - dates[0]).days == tfreq #same_freq true/False
    elif timestep_days >= 28 and timestep_days <= 31 and n_yrs != n_timesteps:
        input_freq = 'monthly'
        same_freq = (dates[1].month - dates[0].month) == tfreq #same_freq true/False
    elif tfreq == timestep_days:
        same_freq = True
    elif timestep_days == 365 or timestep_days == 366:
        input_freq = 'annual' # temporary to work with lag as int

        if verbosity == 1:
            print('Detected timeseries with annual mean values')
            if tfreq is None:
                print('tfreq is None, no common time aggregation used, '
                      'loading annual mean data')
                same_freq = None
        if tfreq is not None:
            same_freq=False
            fullts = extend_annual_ts(fullts,
                                      tfreq=1,
                                      start_end_TVdate=start_end_TVdate,
                                      start_end_date=start_end_date,
                                      ext_annual_to_mon=ext_annual_to_mon)
        if ext_annual_to_mon:
            print('Extending annual data of target to monthly data')
            input_freq = 'monthly'


    # time_mean_bins: (multiple datapoints per year)
    if same_freq == False and TVdates_aggr==False:
        if verbosity == 1:
            print('original tfreq of imported response variable is converted to '
                  'desired tfreq')
        out = time_mean_bins(fullts, tfreq,
                             start_end_date,
                             start_end_year,
                             start_end_TVdate=start_end_TVdate)
        fullts, dates_tobin, traintestgroups = out

    if same_freq == True and TVdates_aggr==False: # and start_end_date is not None and input_freq == 'daily':
        out = timeseries_tofit_bins(fullts, tfreq, start_end_date, start_end_year)
        fullts, dates, traintestgroups = out
        print('Selecting subset as defined by start_end_date')

    # time_mean_period: aggregate over start_end_TVdate, one datapoint p/y
    if TVdates_aggr:
        fullts = time_mean_single_period(fullts, start_end_TVdate, start_end_year)
        time_index = ('time', pd.to_datetime([f'{y}-01-01' for y in fullts['time'].values]))
        fullts['time'] = time_index
        input_freq = 'annual' # now one-value-per-year timeseries


    if input_freq == 'annual':
        RV_ts = fullts
        traintestgroups = pd.Series(np.arange(1, RV_ts.size+1),
                                    index=pd.to_datetime(fullts.time.values))
    else:
        if input_freq == 'daily':
            dates_RV = core_pp.get_subdates(pd.to_datetime(fullts.time.values),
                                            start_end_TVdate,
                                            start_end_year,
                                            input_freq=input_freq)
        elif input_freq == 'monthly':
            dates_RV = TVmonthrange(fullts, start_end_TVdate)
        # get indices of RVdates
        string_RV = list(dates_RV.strftime('%Y-%m-%d'))
        string_full = list(pd.to_datetime(fullts.time.values).strftime('%Y-%m-%d'))
        RV_period = [string_full.index(date) for date in string_full if date in string_RV]
        RV_ts = fullts[RV_period]

    # convert to dataframe
    df_fullts = pd.DataFrame(fullts.values,
                         index=pd.to_datetime(fullts.time.values),
                         columns=[fullts.name])
    df_RV_ts = pd.DataFrame(RV_ts.values,
                                 index=pd.to_datetime(RV_ts.time.values),
                                 columns=['RV'+fullts.name])
    return df_fullts, df_RV_ts, input_freq, traintestgroups

def get_df_test(df, cols: list=None, df_splits: pd.DataFrame=None):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        df with train-test splits on the multi-index.
    cols : list, optional
        return sub df based on columns. The default is None.
    df_splits : pd.DataFrame
        seperate df with TrainIsTrue column specifying the train-test data

    Returns
    -------
    Returns only the data at which TrainIsTrue==False.

    '''
    if df_splits is None:
        splits = df.index.levels[0]
        TrainIsTrue = df['TrainIsTrue']
    else:
        splits = df_splits.index.levels[0]
        TrainIsTrue = df_splits['TrainIsTrue']
    list_test = []
    for s in range(splits.size):
        TestIsTrue = TrainIsTrue[s]==False
        try: # normal
            list_test.append(df.loc[s][TestIsTrue.values])
        except: # only RV_mask (for predictions)
            TestIsTrue = TestIsTrue[df_splits.loc[s]['RV_mask']]
            list_test.append(df.loc[s][TestIsTrue.values])
    df = pd.concat(list_test).sort_index()
    if cols is not None:
        df = df[cols]
    return df

def get_df_train(df, cols: list=None, df_splits: pd.DataFrame=None, s=0):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        df with train-test splits on the multi-index.
    cols : list, optional
        return sub df based on columns. The default is None.
    df_splits : pd.DataFrame
        seperate df with TrainIsTrue column specifying the train-test data

    Returns
    -------
    Returns only the data at which TrainIsTrue.

    '''
    if df_splits is None:
        TrainIsTrue = df['TrainIsTrue']
    else:
        TrainIsTrue = df_splits['TrainIsTrue']
    df_train = df.loc[s][TrainIsTrue.loc[s].values==False]
    if cols is not None:
        df_train = df_train[cols]
    return df_train

def nc_xr_ts_to_df(filename, name_ds='ts', format_lon='only_east'):
    if filename.split('.')[-1] == 'nc':
        ds = core_pp.import_ds_lazy(filename, format_lon=format_lon)
    else:
        print('not a NetCDF file')
    return xrts_to_df(ds[name_ds]), ds

def xrts_to_df(xarray):

    dims = list(xarray.coords.keys())
    if len(dims) > len(xarray.dims):
        standard_dim = ['latitude', 'longitude', 'time', 'mask', 'cluster']
        dims = [d for d in dims if d not in standard_dim]
        if 'n_clusters' in dims:
            idx = dims.index('n_clusters')
            dims[idx] = 'ncl'
            xarray = xarray.rename({'n_clusters':dims[idx]}).copy()
        var1 = int(xarray[dims[0]])
        dim1 = dims[0]
        name = '{}{}'.format(dim1, var1)
        xarray = xarray.drop(dim1)
        if len(dims) == 2:
            var2 = int(xarray[dims[1]])
            dim2 = dims[1]
            name += '_{}{}'.format(dim2, var2)
            xarray = xarray.drop(dim2)
        df = xarray.T.to_dataframe(name=name).unstack(level=1)

        df = df.droplevel(0, axis=1)
    else:
        attr = {k:i for k,i in xarray.attrs.items() if k != 'is_DataArray'}
        name = '_'.join("{!s}{!r}".format(key,val) for (key,val) in attr.items())
        df = xarray.T.to_dataframe(name=name).unstack(level=1)
    df.index.name = name
    return df

def import_ds_timemeanbins(filepath, tfreq: int=None, start_end_date=None,
                           start_end_year=None, closed: str='right',
                           start_end_TVdate: tuple=None,
                           selbox=None,
                           loadleap=False,
                           to_xarr=True,
                           seldates=None,
                           dailytomonths=False,
                           format_lon='only_east'):

    if start_end_date is not None:
        seldates = start_end_date # only load subset of year

    ds = core_pp.import_ds_lazy(filepath, loadleap=loadleap,
                                seldates=seldates,
                                selbox=selbox,
                                dailytomonths=dailytomonths,
                                format_lon=format_lon,
                                start_end_year=start_end_year)

    if tfreq is not None and tfreq != 1:
        ds = time_mean_bins(ds, tfreq,
                            start_end_date,
                            start_end_year,
                            closed=closed,
                            start_end_TVdate=start_end_TVdate)[0]
    ds = ds.squeeze()
    # check if no axis has length 0:
    assert 0 not in ds.shape, ('loaded ds has a dimension of length 0'
                               f', shape {ds.shape}')

    # masks in xr.DataArray are not pickable, which will cause _thread.lock
    # error when parallizing an analysis pipeline.
    if 'mask' in ds.coords:
        ds = ds.drop('mask')
    return ds

def time_mean_bins(xr_or_df, tfreq=int, start_end_date=None, start_end_year=None,
                   closed: str='right', start_end_TVdate: tuple=None,
                   verbosity=0):
   #%%
    '''
    Supports daily and monthly data
    '''

    types = [type(xr.Dataset()), type(xr.DataArray([0])), type(pd.DataFrame([0]))]

    assert (type(xr_or_df) in types), ('{} given, should be in {}'.format(type(xr_or_df), types) )

    if type(xr_or_df) == types[-1]:
        return_df = True
        xr_init = xr_or_df.to_xarray().to_array()
        if len(xr_init.shape) > 2:
            dims = xr_init.dims.items()
            i_time = np.argmax([ z[1] for z in dims])
            old_name = [ z[0] for z in dims][i_time]
        else:
            old_name = 'index'
        xarray = xr_init.rename({old_name : 'time'})
    else:
        return_df = False
        xarray = xr_or_df

    date_time = pd.to_datetime(xarray['time'].values)
    date_time  = core_pp.get_subdates(date_time,
                                      start_end_year=start_end_year)
    # ensure to remove leapdays
    date_time = core_pp.remove_leapdays(date_time)
    # input_freq = (date_time[1] - date_time[0]).days
    xarray = xarray.sel(time=date_time)
    years = np.unique(date_time.year)
    one_yr = get_oneyr(date_time, years[1])
    possible = []
    for i in np.arange(1,20):
        if one_yr.size%i == 0:
            possible.append(i)

    # if one_yr.size % tfreq != 0: # removed if changed on 09-02-2021
    if verbosity == 1:
        print('Note: stepsize {} does not fit in one year\n '
                        ' supply an integer that fits {}'.format(
                            tfreq, one_yr.size))
        print('\n Stepsize that do fit are {}'.format(possible))
        print('\n Will shorten the \'subyear\', so that the temporal'
             ' frequency fits in one year')

    dates_tobin, traintestgroups = timeseries_tofit_bins(date_time, tfreq,
                                        start_end_date=start_end_date,
                                        start_end_year=start_end_year,
                                        closed=closed,
                                        start_end_TVdate=start_end_TVdate,
                                        verbosity=verbosity)

    dates_notpresent = pd.to_datetime([d for d in dates_tobin if d not in date_time])
    assert len(dates_notpresent)==0, ('dates not present in xr_or_df\n'
    f' {dates_notpresent[:5]} \n...\n {dates_notpresent[-5:]}')
    xarray = xarray.sel(time=dates_tobin)
    one_yr = get_oneyr(dates_tobin, years[1])


    # first year maybe of different size if data is aggregated cyclic,
    # see timeseries_to_fit_bins
    fp = dates_tobin[traintestgroups==1] # first period
    sp = dates_tobin[traintestgroups==2] # second period

    n_years = np.unique(dates_tobin.year).size
    crossyr = get_oneyr(dates_tobin, years[0]).size != get_oneyr(dates_tobin, years[1]).size
    if crossyr and fp.size == sp.size:
        # crossyr timemeanbins,
        n_years -= 1
    assert (dates_tobin.size-fp.size) % sp.size==0, 'check output timeseries_to_fit_bins'

    fit_steps_yr = int(fp.size) / tfreq
    bins = np.repeat(np.arange(0, fit_steps_yr), tfreq) # first period bins
    fit_steps_yr = int(sp.size) / tfreq # other years fit_steps_yr
    for y in np.arange(1, n_years):
        x = np.repeat(np.arange(0, fit_steps_yr), tfreq)
        x = bins[-1]+1 + x
        bins = np.insert(bins, bins.size, x)
    label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
    label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
    xarray['bins'] = label_bins
    xarray['time_dates'] = label_dates
    xarray = xarray.set_index(time=['bins','time_dates'])

    half_step = tfreq/2.
    newidx = np.arange(half_step, dates_tobin.size, tfreq, dtype=int)
    newdate = label_dates[newidx]

    # suppres warning for nans in field
    import warnings
    warnings.simplefilter("ignore", category=RuntimeWarning)
    group_bins = xarray.groupby('bins', restore_coord_dims=True).mean(dim='time',
                               skipna=True,
                               keep_attrs=True)


    group_bins['bins'] = newdate.values
    xarray = group_bins.rename({'bins' : 'time'})
    dates = pd.to_datetime(newdate.values)
    traintestgroups = traintestgroups[::tfreq]
    traintestgroups.index = dates
    assert traintestgroups.size == dates.size, 'Wrong time_mean_bins'

    if return_df:
        if len(xr_init.shape) == 3:
            iterables = [xarray.level_0.values, dates]
            index = pd.MultiIndex.from_product(iterables,
                                              names=['split', 'time'])
            xr_index = xarray.stack(index=['level_0', 'time'])
            return_obj = pd.DataFrame(xr_index.values.T,
                                      index=index,
                                      columns=list(xr_init.coords['variable'].values))
        elif len(xr_init.shape) == 2:
            return_obj = pd.DataFrame(xarray.values.T,
                                      index=dates,
                                      columns=list(xr_init.coords['variable'].values))
            return_obj = return_obj.astype(xr_or_df.dtypes)
    elif return_df == False:
        return_obj = xarray
   #%%
    return return_obj, dates_tobin, traintestgroups


def timeseries_tofit_bins(xr_or_dt, tfreq, start_end_date=None, start_end_year=None,
                          closed: str='right', start_end_TVdate: tuple=None,
                          verbosity=0):
    '''
    if tfreq is an even number, the centered date will be
    1 day to the right of the window.


    If start_end_date is fullyear (01-01, 12-31) and closed=='right', will
    create cycle of bins starting on closed_on_date moving backward untill back
    at closed_on_date. With closed on date defined as the last day of TV_period
    '''
    #%%
    # xr_or_dt = rg.fulltso.copy();verbosity=1; closed='right'
    # start_end_date=None; start_end_year=None; verbosity=0

    if type(xr_or_dt) == type(xr.DataArray([0])):
        datetime = pd.to_datetime(xr_or_dt['time'].values)
    else:
        datetime = xr_or_dt.copy()

    datetime = core_pp.remove_leapdays(datetime)
    if (datetime[1] - datetime[0]).days == 1:
        input_freq = 'day'
    elif (datetime[1] - datetime[0]).days in [28,29,30,31]:
        input_freq = 'month'
    # =============================================================================
    #   # select dates
    # =============================================================================
    if start_end_year is None:
        startyear, endyear = datetime[0].year, datetime[-1].year
        years = np.unique(datetime.year)
        assert (np.unique(datetime.year).size == len(years)), \
            (f'Range of years ({len(years)}) requested not in dataset,',
            f'only {np.unique(datetime.year).size} years present.')
    else:
        startyear, endyear = start_end_year
        years = range(startyear, endyear+1)

    if start_end_date is not None:
        sstartdate, senddate = start_end_date
    if start_end_date is None:
        d_s = datetime[0]
        d_e = datetime[-1]
        sstartdate = '{:02d}-{:02d}'.format(d_s.month, d_s.day)
        senddate   = '{:02d}-{:02d}'.format(d_e.month, d_e.day)
    if start_end_TVdate is None:
        start_end_TVdate = (sstartdate, senddate) # select all dates
    else:
        senddate = start_end_TVdate[-1] # over-rule end date of start_end_date

    # check if Target variable period is crossing Dec-Jan
    crossyr = int(start_end_TVdate[0].replace('-','')) > int(start_end_TVdate[1].replace('-',''))
    closed_on_date = start_end_TVdate[-1]

    sstartdate = '{}-{}'.format(startyear, sstartdate)
    if crossyr:
        senddate   = '{}-{}'.format(startyear+1, closed_on_date)
        # max one year back in time from closed_dates
        maxlag = pd.to_datetime(senddate) - date_dt(years=1)
        sstartdate = max(pd.to_datetime(sstartdate), maxlag)
        sstartdate = sstartdate.strftime('%Y-%m-%d') # keep same format
    else:
        senddate   = '{}-{}'.format(startyear, closed_on_date)

    # if start_end_date is None:

    # else:
    #     sstartdate = '{}-{}'.format(startyear, sstartdate)

    # remnant two lines of code , may be removed
    adjhrsstartdate = sstartdate + ' {:02d}:00:00'.format(datetime[0].hour)
    adjhrsenddate   = senddate + ' {:02d}:00:00'.format(datetime[0].hour)


    def getdaily_firstyear(adjhrsstartdate, adjhrsenddate, closed_on_date, tfreq):
        fit_bins = int(365/tfreq) + 1
        if closed_on_date is not None:
            _closed_on_date = adjhrsenddate.replace(adjhrsenddate[5:10], closed_on_date)
            dates_aggr =  pd.date_range(end=_closed_on_date, freq=f'{tfreq}d',
                                        closed=closed,
                                        periods=fit_bins)

            # Extend untill adjhrsenddate
            while dates_aggr[-1] + pd.Timedelta(f'{tfreq}d') < pd.to_datetime(adjhrsenddate):
                dates_aggr = pd.date_range(start=dates_aggr[0], freq=f'{tfreq}d',
                                           closed='left',
                                           periods=fit_bins)
                fit_bins += 1
        else:
            dates_aggr =  pd.date_range(end=adjhrsenddate, freq=f'{tfreq}d',
                                    closed=closed,
                                    periods=fit_bins)
        # adjust startdate such the bins are closed and that the startdate
        # is not prior to the requisted adjhrsstartdate
        fit_bins = int(365/tfreq)
        while dates_aggr[0] < pd.to_datetime(adjhrsstartdate):
            dates_aggr =  pd.date_range(end=dates_aggr[-1], freq=f'{tfreq}d',
                                    closed=closed,
                                    periods=fit_bins)
            fit_bins -= 1


        # if crossyr == False and senddate[1] == '12-31':
        #     # Extend untill end of the year, which might be handy in case one
        #     # want to study stuff like autocorrelation. Not wanted with cross-year.
        #     fit_bins = int(365/tfreq)
        #     dates_aggr = pd.date_range(start=dates_aggr[0], freq=f'{tfreq}d',
        #                                 closed='left',
        #                                 periods=fit_bins)
        #     while dates_aggr[0].year < dates_aggr[-1].year:
        #         fit_bins -= 1
        #         dates_aggr = pd.date_range(start=dates_aggr[0], freq=f'{tfreq}d',
        #                                     closed='left',
        #                                     periods=fit_bins)
        # if closed right, convert to daily date that fit bins
        # also take into account leapday adjustement
        if dates_aggr.size == 1:
            sd = dates_aggr[0] - pd.Timedelta(f'{tfreq}d')
        else:
            sd = dates_aggr[0]

        # single year, dates between 01-01 and > 03-01
        leap1yr = all([dates_aggr.year.unique().size==1,
                       dates_aggr.is_leap_year[0],
                       dates_aggr[0] < pd.to_datetime(f'{startyear}-03-01')])
        # cross-year, one yr with dates both prior and after 03-01
        yrs = np.unique(dates_aggr.year) ; leap2yr = []
        for yr in yrs:
            syr = core_pp.get_oneyr(dates_aggr, yr)
            leap2yr.append(all([syr.is_leap_year[0],
                           any(syr < pd.to_datetime(f'{yr}-03-01')),
                           syr[-1] > pd.to_datetime(f'{yr}-03-01')]))
        leap2yr = any(leap2yr)


        if leap1yr or leap2yr:
            start_yr = pd.date_range(start=sd,
                                 end=dates_aggr[-1])
        else:
            start_yr = pd.date_range(start=sd + pd.Timedelta(f'{1}d'),
                                 end=dates_aggr[-1])
        start_yr = core_pp.remove_leapdays(start_yr)
        return start_yr

    if input_freq == 'day' and tfreq == 1:
        one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
        start_yr = pd.date_range(start=one_yr[0], end=one_yr[-1],
                                freq=datetime[1] - datetime[0])
        start_yr = core_pp.remove_leapdays(start_yr)
        end_day = start_yr.max()
        start_day = start_yr.min()


    if input_freq == 'day' and tfreq != 1:
        if closed == 'right':
            if start_end_date == None:
                # make cyclic around closed_end_date
                if crossyr:
                    startyear += 1 # first year full period not possible
                start_yr = getdaily_firstyear(adjhrsstartdate,
                                             f'{startyear}-{closed_on_date} 0:00:00',
                                             closed_on_date, tfreq)

                otheryrs = getdaily_firstyear(f'{startyear}-{closed_on_date} 0:00:00',
                                              f'{startyear+1}-{closed_on_date} 0:00:00',
                                              closed_on_date, tfreq)
                start_day = otheryrs.min()
                end_day = otheryrs.max()
            else:
                start_yr = getdaily_firstyear(adjhrsstartdate, adjhrsenddate,
                                          closed_on_date, tfreq)
                start_day = start_yr.min()
                end_day = start_yr.max()



    if input_freq == 'month':
        dt = date_dt(months=tfreq)
        start_day = adjhrsstartdate.split(' ')[0]
        start_day = pd.to_datetime(start_day.replace(start_day[-2:], '01'))
        end_day = adjhrsenddate.split(' ')[0]
        end_day = pd.to_datetime(end_day.replace(end_day[-2:], '01'))
        if crossyr:
            fit_steps_yr = (12-start_day.month + end_day.month+ 1 ) / tfreq
        else:
            fit_steps_yr = (end_day.month - start_day.month + 1) / tfreq
        start_day = (end_day - (dt * int(fit_steps_yr))) \
                + date_dt(months=+1)
        days_back = end_day
        start_yr = [end_day.strftime('%Y-%m-%d %H:%M:%S')]
        while start_day < days_back:
            days_back -= date_dt(months=+1)
            start_yr.append(days_back.strftime('%Y-%m-%d %H:%M:%S'))
        start_yr.reverse()
        start_yr = pd.to_datetime(start_yr)


    #    n_oneyr = start_yr.size
    #    end_year = endyear

    if input_freq == 'day' and tfreq != 1 and start_end_date == None:
        # make cyclic around closed_end_date
        other_cyclic_yrs = core_pp.make_dates(otheryrs, years[1:])
        datesdt = start_yr.append(other_cyclic_yrs)
        groupfirstyr = np.repeat(1, start_yr.size)
        n_periods = int(other_cyclic_yrs.size / otheryrs.size)
        groupsotheryrs = np.repeat(range(2,n_periods+2), otheryrs.size)
        traintestgroups = pd.Series(np.concatenate([groupfirstyr, groupsotheryrs]),
                                 index=datesdt)
    else: # Copy start_yr to other years
        datesdt = core_pp.make_dates(start_yr, years)
        n_periods = int(datesdt.size / start_yr.size)
        traintestgroups = pd.Series(np.repeat(range(1,n_periods+1), start_yr.size),
                                 index=datesdt)

    #    n_yrs = datesdt.size / n_oneyr
    if verbosity==1:
        months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                             8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
        startdatestr = '{} {}'.format(start_day.day, months[start_day.month])
        enddatestr   = '{} {}'.format(end_day.day, months[end_day.month])
        if input_freq == 'day':
            print('Period of year selected: \n{} to {}, tfreq {} days'.format(
                    startdatestr, enddatestr, tfreq))
        if input_freq == 'month':
            print('Months of year selected: \n{} to {}, tfreq {} months'.format(
                    startdatestr.split(' ')[-1], enddatestr.split(' ')[-1], tfreq))

    if type(xr_or_dt) == type(xr.DataArray([0])):
        adj_xarray = xr_or_dt.sel(time=datesdt)
        out = (adj_xarray, datesdt, traintestgroups)
    else:
        out = (datesdt, traintestgroups)
    #%%
    return out

def time_mean_periods(xr_or_df, start_end_periods=np.ndarray,
                      start_end_year: tuple=None):
    '''


    Parameters
    ----------
    xr_or_df : xr.DataArray or xr.Dataset, or pd.DataFrame
        input timeseries with dimension 'time'.
    start_end_periods : tuple or np.ndarray
        tuple of start- and enddate for target variable in
        format ('mm-dd', 'mm-dd'). If the first entry date of a period is after
        the last entry date, then ('startyear-mm-dd', 'startyear+1-mm-dd').
    start_end_year : tuple, optional


    Returns
    -------
    return_obj : xr.DataArray or pd.DataFrame
        Time aggregated data

    '''

    if np.array(start_end_periods).shape == (2,):
        start_end_periods = np.array([start_end_periods])

    types = [type(xr.Dataset()), type(xr.DataArray([0])), type(pd.DataFrame([0]))]

    assert (type(xr_or_df) in types), ('{} given, should be in {}'.format(type(xr_or_df), types) )

    if type(xr_or_df) == types[-1]:
        return_df = True
        xr_init = xr_or_df.to_xarray().to_array()
        if len(xr_init.shape) > 2:
            dims = xr_init.dims.items()
            i_time = np.argmax([ z[1] for z in dims])
            old_name = [ z[0] for z in dims][i_time]
        else:
            old_name = 'index'
        xarray = xr_init.rename({old_name : 'time'}).squeeze()
    else:
        return_df = False
        xarray = xr_or_df

    date_time = pd.to_datetime(xarray['time'].values)
    if start_end_year is None:
        start_end_year = (date_time.year[0], date_time.year[-1])
    upd_start_end_years = check_crossyr_periods(start_end_periods, start_end_year)
    xrgr = np.zeros(start_end_periods.shape[0], dtype=object)
    for i,p in enumerate(start_end_periods):
        # check format for of offset yr is given '{offsetyr}-{mm}-{dd}'
        p = ( '-'.join(p[0].split('-')[-2:]), '-'.join(p[-1].split('-')[-2:]) )
        s_e_y = upd_start_end_years[i]
        print(s_e_y, p)
        xrgr[i] = time_mean_single_period(xarray, p, s_e_y)

    # use time-index of last lag given
    time_index = ('time', pd.to_datetime([f'{y}-01-01' for y in xrgr[i]['time'].values]))
    for i in range(len(xrgr)):
        xrgr[i]['time'] = time_index
        xrgr[i].expand_dims('lag', axis=1)

    xarray = xr.concat(xrgr, dim='lag')
    xarray['lag'] = ('lag', np.arange(start_end_periods.shape[0]))
    xarray = xarray.transpose('time', 'lag', 'latitude', 'longitude', # ensure order dims
                              transpose_coords=True)

    if return_df:
        if len(xr_init.shape) == 3:
            iterables = [xarray.level_0.values, date_time]
            index = pd.MultiIndex.from_product(iterables,
                                              names=['split', 'time'])
            xr_index = xarray.stack(index=['level_0', 'time'])
            return_obj = pd.DataFrame(xr_index.values.T,
                                      index=index,
                                      columns=list(xr_init.coords['variable'].values))
        elif len(xr_init.shape) == 2:
            return_obj = pd.DataFrame(xarray.values,
                                      index=time_index[1],
                                      columns=list(xr_init.coords['variable'].values))
            return_obj = return_obj.astype(xr_or_df.dtypes)
    elif return_df == False:
        return_obj = xarray
    return return_obj

def time_mean_single_period(xarray, period: tuple, start_end_year: tuple=None):
    date_time = pd.to_datetime(xarray['time'].values)
    dperiod, groups = core_pp.get_subdates(date_time, period,
                               start_end_year=start_end_year,
                               lpyr=False,
                               returngroups=True)
    if dperiod.size != np.unique(groups).size:
        xrgr = xarray.sel(time=dperiod).groupby(
                                        xr.DataArray(groups,
                                                     coords={'time':dperiod},
                                                     dims=['time'],
                                                     name='time'),
                                        restore_coord_dims=True).mean(dim='time')
    else:
        xrgr = xarray.sel(time=dperiod) # No groupby mean needed
        xrgr['time'] = ('time', groups)
    return xrgr

def check_crossyr_periods(start_end_periods, start_end_year):
    ''' If some start_end_periods are cross yr, start year is aligned such that
    all aggregations have same start year (startyr of data + 1) '''

    startyr, endyr = start_end_year

    dateyears = np.zeros( (start_end_periods.shape[0],2), dtype=int)
    upd_start_end_years = np.zeros_like(dateyears)
    if len(start_end_periods[0][0].split('-')) >= 3: # offset explicitly given

        for i, sep in enumerate(start_end_periods):
            syp = int(sep[0].split('-')[0]) # start year period
            eyp = int('-'.join(sep[-1].split('-')[:-2])) # end year period
            dateyears[i] = [syp, eyp]

        for i, d in enumerate(dateyears):
            if np.unique(dateyears).size > 1: # varying startyr
                if d[0] == startyr and d[-1] == startyr:
                    upd_start_end_years[i] = (startyr, endyr-1)
                elif d[0] == startyr and d[-1] == startyr+1:
                    upd_start_end_years[i] = (startyr, endyr)
                else:
                    upd_start_end_years[i] = (startyr+1, endyr)
            else:
                # use given start yr
                upd_start_end_years[i] = (np.unique(dateyears)[0],
                                          endyr)

    else: # check for crossyr within periods
        crossyrlags = np.zeros(start_end_periods.shape[0], dtype=bool)
        for i, p in enumerate(start_end_periods):
            sd = pd.to_datetime('2000-'+p[0])
            ed = pd.to_datetime('2000-'+p[-1])
            crossyrlags[i] = sd > ed
        if crossyrlags.all() and ~crossyrlags.any():
            pass
        if crossyrlags.any():
            for i, crossyr in enumerate(crossyrlags):
                if crossyr:
                    upd_start_end_years[i] = (startyr, endyr)
                else:
                    upd_start_end_years[i] = (startyr+1, endyr)
        else:
            # or all crossyr or no crossyr at all.
            for i, crossyr in enumerate(crossyrlags):
                upd_start_end_years[i] = (startyr, endyr)
    return upd_start_end_years


def extend_annual_ts(fullts, tfreq: int, start_end_TVdate: tuple,
                     start_end_date: tuple=None, ext_annual_to_mon=True):

    tfreq = 1
    firstyear = pd.to_datetime(fullts.time.values)[0].year
    endyear   = pd.to_datetime(fullts.time.values)[-1].year

    sdTV = pd.to_datetime(f'{firstyear}-{start_end_TVdate[0]}')
    edTV = pd.to_datetime(f'{firstyear}-{start_end_TVdate[1]}')
    if edTV < sdTV: # endate prior to startdate
        crossyr = True
    else:
        crossyr = False

    if start_end_date is None and crossyr==False:
        start_end_date = ('01-01', '12-31')
    elif start_end_date is None and crossyr==True:
        start_end_date = start_end_TVdate

    sd = pd.to_datetime(f'{firstyear}-{start_end_date[0]}')
    if crossyr:
        ed = pd.to_datetime(f'{firstyear+1}-{start_end_date[-1]}')
    else:
        ed = pd.to_datetime(f'{firstyear}-{start_end_date[-1]}')

    if ext_annual_to_mon == False:
        fakedates = pd.date_range(start=sd, end=ed)
        fakedates = core_pp.remove_leapdays(fakedates)
    else:
        fakedates = pd.date_range(start=sd, end=ed, freq='M')
        dtfirst = [s+'-01' for s in fakedates.strftime('%Y-%m')]
        fakedates = pd.to_datetime(dtfirst)


    fakedates = core_pp.make_dates(fakedates, range(firstyear, endyear+1))
    # if tfreq != 1:
    #     prec_dates = timeseries_tofit_bins(fakedates, tfreq=tfreq,
    #                                    start_end_date=start_end_date,
    #                                    start_end_year=(firstyear, endyear),
    #                                    closed_on_date=start_end_TVdate[-1])
    # else:
    #     prec_dates = fakedates
    prec_dates = fakedates
    if crossyr:
        npdata = np.repeat(fullts.values[1:], tfreq)
    else:
        steps_tfreq = int(get_oneyr(prec_dates).size / tfreq)
        # trying to add values of previous year in one year
        # RV_dates = core_pp.get_subdates(fakedates, start_end_date=start_end_TVdate)
        # l = []
        # for lagshift in range(0,steps_tfreq):
        #     _dates = pd.to_datetime([d-date_dt(months=lagshift) for d in RV_dates])
        #     fakedates = pd.date_range(start=sd, end=ed, freq='M')
        #     dtfirst = [s for s in RV_dates.strftime('%Y-%d')]
        #     fakedates = pd.to_datetime(dtfirst)
        #     _d_match = pd.to_datetime([d for d in _dates if d in prec_dates])
        #     vals = fullts.shift(time=lagshift, fill_value=fullts.mean())
        #     l.append(pd.Series(vals, _dates))
        # fullts_ext = pd.concat(l).sort_index()
        RV_dates = core_pp.get_subdates(fakedates, start_end_date=start_end_TVdate)
        npdata = np.zeros(fullts.size*steps_tfreq*tfreq)
        for i in range(steps_tfreq):
            if any([True for r in get_oneyr(RV_dates) if r == prec_dates[i]]):
                npdata[i::get_oneyr(prec_dates).size] = fullts
            else:
                fakets = np.random.choice(fullts, fullts.size)
                while np.corrcoef(fakets, fullts)[0][1] > .15: # resample if corr.
                    fakets = np.random.choice(fullts, fullts.size)
                npdata[i::get_oneyr(prec_dates).size] = fakets

    fullts_ext = xr.DataArray(npdata,
                              coords=[prec_dates],
                              dims=['time'],
                              name=fullts.name)
    return fullts_ext


def TVmonthrange(fullts, start_end_TVdate):
    '''
    fullts : 1-D xarray timeseries
    start_end_TVdate is tuple of start- and enddate in format ('mm-dd', 'mm-dd')
    '''
    want_months = [int(start_end_TVdate[0].split('-')[0]),
                  int(start_end_TVdate[1].split('-')[0])]
    if want_months[0] > want_months[1]:
        crossyr = True
    else:
        crossyr = False

    if crossyr:
        want_month = np.concatenate([np.arange(want_months[0], 13),
                                  np.arange(1, want_months[1]+1)])
    else:
        want_month = np.arange(want_months[0],
                               want_months[1]+1)
    months = fullts.time.dt.month
    months_pres = np.unique(months)
    selmon = [m for m in want_month if m in list(months_pres)]
    if len(selmon) == 0:
        print('The RV months are no longer in the time series, perhaps due to '
              'time mean bins, in which time axis is changed, i.e. new time axis'
              'takes the center month of the bin')
        new_want_m = []
        for want_m in want_month:
            idx_close = max(months_pres)
            diff = []
            for m in months_pres:
                diff.append(abs(m - want_m))
                # choosing month present closest to desired month in ex['startperiod']
                min_diff = min(diff[-1], idx_close)
            new_want_m.append(months_pres[diff.index(min_diff)])
        selmon = [m for m in new_want_m if m in list(months_pres)]
    mask = np.zeros(months.size, dtype=bool)
    idx = [i for i in range(months.size) if months[i] in selmon]
    mask[idx] = True
    xrdates = fullts.time.where(mask).dropna(dim='time')
    dates_RV = pd.to_datetime(xrdates.values)
    if crossyr:
        startyr = int(fullts.time.dt.year[0])
        endyr = int(fullts.time.dt.year[-1])
        mask_incompl_TVperiod1 = dates_RV > pd.to_datetime(f'{startyr}-'+start_end_TVdate[-1])
        mask_incompl_TVperiod2 = dates_RV < pd.to_datetime(f'{endyr}-'+start_end_TVdate[0])
        dates_RV = dates_RV[np.logical_and(mask_incompl_TVperiod1, mask_incompl_TVperiod2)]
    return dates_RV

#def make_TVdatestr(dates, ex):
#
#    startyr = dates[0].year
#    sstartdate = pd.to_datetime(str(startyr) + '-' + ex['startperiod'])
#    senddate   = pd.to_datetime(str(startyr) + '-' + ex['endperiod'])
#    first_d = sstartdate.dayofyear
#    last_d  = senddate.dayofyear
#    datesRV = pd.to_datetime([d for d in dates if d.dayofyear >= first_d and d.dayofyear <= last_d])
#    return datesRV


def area_weighted(xarray):
   # Area weighted, taking cos of latitude in radians
   coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
   area_weights = np.tile(coslat[..., np.newaxis],(1,xarray.longitude.size))
#   area_weights = area_weights / area_weights.mean()
   return xr.DataArray(xarray.values * area_weights, coords=xarray.coords,
                          dims=xarray.dims, name=xarray.name, attrs=xarray.attrs)


def anom1D(da):

    if da.dims[0] == 'index':
        # rename axes to 'time'
        da = da.rename({'index':'time'})
    dates = pd.to_datetime(da.time.values)
    stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')

    if (stepsyr.day== 1).all() == True or int(da.time.size / 365) >= 120:
        print('\nHandling time series longer then 120 day or monthly data, no smoothening applied')
        data_smooth = da.values
        window_s = None

    elif (stepsyr.day== 1).all() == False and int(da.time.size / 365) < 120:
        window_s = max(min(25,int(stepsyr.size / 12)), 1)
        print('using absolute anomalies w.r.t. climatology of '
              'smoothed concurrent day accross years')
        data_smooth =  rolling_mean_np(da.values, window_s)
    output_clim = np.empty( (stepsyr.size), dtype='float32' )
    for i in range(stepsyr.size):

        sliceyr = np.arange(i, da.time.size, stepsyr.size)
        arr_oneday_smooth = data_smooth[sliceyr]
        output_clim[i] = arr_oneday_smooth.mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(3,3))
    ts = da
    tfreq = (dates[1] - dates[0]).days
    if tfreq in [28,29,30,31]: # monthly timeseries
        input_freq = 'monthly'
    else:
        input_freq = 'daily_or_annual_or_yearly'
    if input_freq == 'monthly':
        rawdayofyear = ts.groupby('time.month').mean('time').sel(month=np.arange(12)+1)
    else:
        rawdayofyear = ts.groupby('time.dayofyear').mean('time').sel(dayofyear=np.arange(365)+1)

    ax.set_title('Raw and est. clim')
    for yr in np.unique(dates.year):
        singleyeardates = get_oneyr(dates, yr)
        ax.plot(ts.sel(time=singleyeardates), alpha=.1, color='purple')
    if window_s is None:
        label = 'clim based on raw data'
    else:
         label = f'clim {window_s}-day rm'
    ax.plot(output_clim, color='green', linewidth=2,
         label=label)
    ax.plot(rawdayofyear, color='black', alpha=.6,
               label='clim mean dayofyear')
    output = da - np.tile(output_clim, (int(dates.size/stepsyr.size)))
    dao = xr.DataArray(output,
                            dims=da.dims,
                            coords=da.coords)
    return dao


def rolling_mean_np(arr, win, center=True, plot=False):
    import scipy.signal.windows as spwin
    if plot == True:
        plt.plot(range(-int(win/2),+int(win/2)+1), spwin.gaussian(win, win/2))
        plt.title('window used for rolling mean')
        plt.xlabel('timesteps')
    df = pd.DataFrame(data=arr.reshape( (arr.shape[0], arr[0].size)))

    rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='gaussian').mean(std=win/2.)

    return rollmean.values.reshape( (arr.shape))


def regrid_xarray(xarray_in, to_grid_res, periodic=True):
    #%%
    '''
    Only supports 2 (lat, lon) or 3 (time, lat, lon) xr.DataArrays
    '''
    import xesmf as xe
    method_list = ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']
    method = method_list[0]


    ds = xr.Dataset({'data':xarray_in})
    ds = xarray_in

    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon',
                        'latitude' : 'lat'})

    lats = ds.lat
    lons = ds.lon
    orig_grid = float(abs(ds.lat[1] - ds.lat[0] ))

    if method == 'conservative':
        # add lon_b and lat_b


        lat_b = np.concatenate(([lats.max()+orig_grid/2.], (lats - orig_grid/2.).values))
        lon_b = np.concatenate(([lons.max()+orig_grid/2.], (lons - orig_grid/2.).values))
        ds['lat_b'] = xr.DataArray(lat_b, dims=['lat_b'], coords={'lat_b':lat_b})
        ds['lon_b'] = xr.DataArray(lon_b, dims=['lon_b'], coords={'lon_b':lon_b})

        lat0_b = lat_b.min()
        lat1_b = lat_b.max()
        lon0_b = lon_b.min()
        lon1_b = lon_b.max()
    else:
        lat0_b = lats.min()
        lat1_b = lats.max()
        lon0_b = lons.min()
        lon1_b = lons.max()
    to_grid = xe.util.grid_2d(lon0_b, lon1_b, to_grid_res, lat0_b, lat1_b, to_grid_res)
#    to_grid = xe.util.grid_global(2.5, 2.5)
    regridder = xe.Regridder(ds, to_grid, method, periodic=periodic, reuse_weights=True)
    try:
        xarray_out = regridder(ds)
    except:
        xarray_out  = regridder.regrid_dataarray(ds)
    regridder.clean_weight_file()
    xarray_out = xarray_out.rename({'lon':'longitude',
                                    'lat':'latitude'})
    if len(xarray_out.shape) == 2:
        xarray_out = xr.DataArray(xarray_out.values[::-1],
                                  dims=['latitude', 'longitude'],
                                  coords={'latitude':xarray_out.latitude[:,0].values[::-1],
                                  'longitude':xarray_out.longitude[0].values})
    elif len(xarray_out.shape) == 3:
        xarray_out = xr.DataArray(xarray_out.values[:,::-1],
                                  dims=['time','latitude', 'longitude'],
                                  coords={'time':xarray_out.time,
                                          'latitude':xarray_out.latitude[:,0].values[::-1],
                                          'longitude':xarray_out.longitude[0].values})
    xarray_out.attrs = xarray_in.attrs
    xarray_out.name = xarray_in.name
    if 'is_DataArray' in xarray_out.attrs:
        del xarray_out.attrs['is_DataArray']
    xarray_out.attrs['regridded'] = f'{method}_{orig_grid}d_to_{to_grid_res}d'
#    xarray_out['longitude'] -= xarray_out['longitude'][0] # changed 17-11-20
    #%%
    return xarray_out

def store_hdf_df(dict_of_dfs, file_path=None):
    import warnings
    import tables
    today = datetime.datetime.today().strftime("%d-%m-%y_%Hhr")
    if file_path is None:
        file_path = get_download_path()+ f'/pandas_dfs_{today}.h5'

    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    with pd.HDFStore(file_path, 'w') as hdf:
        for key, item in  dict_of_dfs.items():
            try:
                hdf.put(key, item, format='table', data_columns=True)
            except:
                hdf.put(key, item, data_columns=True)
            if item.index.name is not None:
                hdf.root._v_attrs[key] = str(item.index.name)
        hdf.close()
    return file_path

def load_hdf5(path_data):
    '''
    Loading hdf5 can not be done simultaneously:
    '''
    import h5py, time
    attempt = 'Fail'
    c = 0
    while attempt =='Fail':
        c += 1
        try:
            hdf = h5py.File(path_data,'r+')
            dict_of_dfs = {}
            for k in hdf.keys():
                dict_of_dfs[k] = pd.read_hdf(path_data, k)
            hdf.close()
            attempt = 1
        except:
            time.sleep(1)
        assert c!= 5, print('loading in hdf5 failed')
    return dict_of_dfs

def cross_validation(RV_ts, traintestgroups=None, test_yrs=None, method=str,
                     seed=None, gap_prior: int=None, gap_after: int=None):
    # RV_ts = rg.df_RV_ts ; traintestgroups=rg.traintestgroups
    # test_yrs = None ; seed=1 ; gap_prior=None ; gap_after=None

    from func_models import get_cv_accounting_for_years
    from sklearn.model_selection import KFold, TimeSeriesSplit, RepeatedKFold

    if test_yrs is not None:
        method = 'copied_from_import_ts'
        kfold  = test_yrs.shape[0]

    if traintestgroups is not None:
        groups = traintestgroups
        index = traintestgroups.index
    else:
        groups = RV_ts.index.year
        index = RV_ts.index

    uniqgroups = np.unique(groups)

    if method == 'no_train_test_split':
        kfold = 1
        testgroups = np.array([[]])
    else:
        if test_yrs is None:
            kfold = int(method.split('_')[-1])
            if method[:8] == 'ranstrat':
                TVgroups = groups.loc[RV_ts.index]
                cv = get_cv_accounting_for_years(RV_ts, kfold, seed, TVgroups)
                testgroups = cv.uniqgroups
            elif method[:5] == 'leave':
                kfold = int(uniqgroups.size / int(method.split('_')[-1]))
                cv = KFold(n_splits=kfold, shuffle=False)
                testgroups = [list(f[1]) for f in cv.split(uniqgroups)]
            elif method[:6] == 'random':
                cv = KFold(n_splits=kfold, shuffle=True, random_state=seed)
                testgroups = [list(f[1]) for f in cv.split(uniqgroups)]
            elif method[:15] == 'TimeSeriesSplit':
                cv = TimeSeriesSplit(max_train_size=None, n_splits=kfold,
                                     test_size=1)
                testgroups = [list(f[1]) for f in cv.split(uniqgroups)]
            elif method[:13] == 'RepeatedKFold':
                n_repeats = int(method.split('_')[1])
                cv = RepeatedKFold(n_splits=kfold, n_repeats=n_repeats,
                                   random_state=seed)
                testgroups = [list(f[1]) for f in cv.split(uniqgroups)]

        else:
            testgroups = test_yrs

    if method[:13] == 'RepeatedKFold':
        testsetidx = np.zeros(groups.size * n_repeats , dtype=int)
    else:
        testsetidx = np.zeros(groups.size , dtype=int)
    testsetidx[:] = -999

    for i, test_fold_idx in enumerate(testgroups):
        # convert idx to grouplabel (year or dateyrgroup)
        if test_yrs is None:
            test_fold = [uniqgroups[i] for i in test_fold_idx]
        else:
            test_fold = test_fold_idx
        for j, gr in enumerate(groups):
            if gr in list(test_fold):
                testsetidx[j] = i % uniqgroups.size

    def gap_traintest(testsetidx, groups, gap):
        ign = np.zeros((np.unique(testsetidx).size, testsetidx.size))
        for f, i in enumerate(np.unique(testsetidx)):
            test_fold = testsetidx==i
            roll_account_traintest_gr = gap*groups[groups==groups[-1]].size
            ign[f] = np.roll(test_fold, roll_account_traintest_gr).astype(float)
            ign[f] = (ign[f] - test_fold) == 1 # everything prior to test
            if np.sign(gap) == -1:
                ign[f][roll_account_traintest_gr:] = False
            elif np.sign(gap) == 1:
                ign[f][:roll_account_traintest_gr] = False
        return ign.astype(bool)

    if gap_prior is not None:
        ignprior = gap_traintest(testsetidx, groups, -gap_prior)
    if gap_after is not None:
        ignafter = gap_traintest(testsetidx, groups, gap_after)

    TrainIsTrue = []
    for f, i in enumerate(np.unique(testsetidx)):
        # if -999, No Train Test split, all True
        if method[:15] == 'TimeSeriesSplit':
            if i == -999:
                continue
            mask = np.array(testsetidx < i, dtype=int)
            mask[testsetidx>i] = -1
        else:
            mask = np.logical_or(testsetidx!=i, testsetidx==-999)
        if gap_prior is not None:
            # if gap_prior, mask values will become -1 for unused (training) data
            mask = np.array(mask, dtype=int) ; mask[ignprior[f]] = -1
        if gap_after is not None:
            # same as above for gap_after.
            mask = np.array(mask, dtype=int) ; mask[ignafter[f]] = -1

        TrainIsTrue.append(pd.DataFrame(data=mask.T,
                                        columns=['TrainIsTrue'],
                                        index=index))
    df_TrainIsTrue = pd.concat(TrainIsTrue , axis=0, keys=range(kfold))

    if traintestgroups is not None:
        # first group may be of different size then other groups
        fg = df_TrainIsTrue.loc[0][groups==groups[0]] # first group
        RV_maskfg = [True if d in RV_ts.index else False for d in list(fg.index)]
        og = df_TrainIsTrue.loc[0][groups==groups[-1]] # other group size
        RV_maskog = [True if d in RV_ts.index else False for d in list(og.index)]
        if fg.size != og.size:
            RVmaskog = np.stack([RV_maskog]*(uniqgroups.size-1), 0).flatten()
            RV_mask = np.concatenate([RV_maskfg, RVmaskog])
        else:
            RV_mask = np.stack([RV_maskfg]*uniqgroups.size, 0).flatten()
    else:
        RV_mask = np.ones(RV_ts.size, dtype=bool)
    RV_mask = pd.concat([pd.DataFrame(RV_mask,
                                   index=index,
                                   columns=['RV_mask'])]*kfold,
                        keys=range(kfold))
    # weird pandas bug due to non-unique indices
    RV_mask.index = df_TrainIsTrue.index
    df_splits = df_TrainIsTrue.merge(RV_mask,
                                     left_index=True, right_index=True)
    return df_splits

def get_testyrs(df_splits: pd.DataFrame):
    '''
    Extracts test years if both:
        - TrainIsTrue mask present
        - More then 1 level (then train-test split is not False)
    Takes into account adjecent groups of dates to define a training group.

    Parameters
    ----------
    df_splits : pd.DataFrame
        Dataframe with TrainIsTrue mask Response Variable mask.

    Returns
    -------
    out : np.ndarray or None
        shape (train-test index, list of test years).

    '''
    #%%
    split_by_TrainIsTrue = False # if True, do not account for adjecent groups
    # of dates (needed for accounting for auto-correlation for good seperate
    # train-test splits)
    out = None
    if hasattr(df_splits.index, 'levels'):
        dates = df_splits.loc[0].index ;
        if df_splits.index.levels[0].size > 1:
            levels=True
        else:
            levels=False
    RV_mask_ = 'RV_mask' in df_splits.columns
    # if full year daily, no traintest groups with a gap that needs to be
    # taken into account
    fullyear = dates.size%365 == 0
    # checking if not one-val-per-yr data
    multipletargetdatesperyr = df_splits.loc[0]['RV_mask'].all()==False
    if RV_mask_ and fullyear==False and multipletargetdatesperyr:
        dates_RV = df_splits.loc[0][df_splits.loc[0]['RV_mask']].index
        gapdays = (dates_RV[1:] - dates_RV[:-1]).days
        adjecent_dates = gapdays > (np.median(gapdays)+gapdays/2)
        RVgroupsize = np.argmax(adjecent_dates) + 1
        closed_right = dates_RV[RVgroupsize-1]
        firstcyclicgroup = dates[dates <= closed_right]
        # middle years, first year might be cut-off due to limiting dates
        closed_right_yr2 = closed_right + date_dt(years=1)
        secondcyclic = dates[np.logical_and(dates > closed_right,
                                            dates <= closed_right_yr2)]
        firstgroup = np.repeat(1, firstcyclicgroup.size)
        secgroup = np.arange(2, int((dates.size-firstgroup.size)/secondcyclic.size+2))
        traintestgroups = np.repeat(secgroup,
                                    secondcyclic.size)
        traintestgroups = np.concatenate([firstgroup, traintestgroups])
        uniqgroups = np.unique(traintestgroups)
        test_yrs = [] ; testgroups = []
        splits = df_splits.index.levels[0]
        for s in splits:
            df_split = df_splits.loc[s]
            TrainIsTrue_s = df_split[df_split['TrainIsTrue']==False].index
            groups_in_s = traintestgroups[(df_split['TrainIsTrue']==False).values]
            groupset = []
            for gr in np.unique(groups_in_s):
                yrs = TrainIsTrue_s[groups_in_s==gr]
                yrs = np.unique(yrs.year)
                groupset.append(list(yrs))
            test_yrs.append(flatten(groupset)) # changed to flatten() 20-05-21
            testgroups.append([list(uniqgroups).index(gr) for gr in np.unique(groups_in_s)])
        out = (np.array(test_yrs, dtype=object), testgroups)
    elif 'TrainIsTrue' in df_splits.columns:
        split_by_TrainIsTrue = True

    if split_by_TrainIsTrue and levels:
        traintest_yrs = []
        splits = df_splits.index.levels[0]
        for s in splits:
            df_split = df_splits.loc[s]
            test_yrs = np.unique(df_split[df_split['TrainIsTrue']==False].index.year)
            traintest_yrs.append(test_yrs)
        out = (np.array(traintest_yrs, dtype=object))
    elif split_by_TrainIsTrue==False and out is None:
        print('Note: No Train-test split found, could not extract test yrs')

    return out

def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'Downloads')

def load_npy(filename, name=None):
    '''
    expects dictionary with 1D xarray timeseries
    '''
    dicRV = np.load(filename, encoding='latin1',
                    allow_pickle=True).item()
    logical = ['RVfullts', 'RVfullts95']
    if name is not None:
        logical.insert(0, name)
    for name_ in logical:
        try:
            fullts = dicRV[name_]
        except:
            pass
    return fullts

def load_csv(path:str, sep=','):
   '''
   convert csv timeseries to hdf5 (.h5) format. Assumes column order:
    year, month, day, ts1, ts2, ..., ...,

    Parameters
    ----------
    path : str
        path to .csv file.
    sep : str, optional
        seperator to seperate columns. The default is ','.

    Returns
    -------
    df_data with datetime as index
        DESCRIPTION.

    '''
    #%%

   # load data from csv file and save to .h5

   # path = '/Users/semvijverberg/Downloads/OMI.csv'
   data = pd.read_csv(path, sep=sep, parse_dates=[[0,1,2]],
                       index_col='year_month_day')
   data.index.name='date'

   store_hdf_df(dict_of_dfs={'df_data':data},
                file_path=path.replace('.csv', '.h5'))
   return data

def dfsplits_to_dates(df_splits, s):
    dates_train = df_splits.loc[s]['TrainIsTrue'][df_splits.loc[s]['TrainIsTrue']].index
    dates_test  = df_splits.loc[s]['TrainIsTrue'][~df_splits.loc[s]['TrainIsTrue']].index
    return dates_train, dates_test

def func_dates_min_lag(dates, lag):
    if lag != 0:
        tfreq = dates[1] - dates[0]
        oneyr = get_oneyr(pd.to_datetime(dates.values))
        start_d_min_lag = oneyr[0] - pd.Timedelta(int(lag), unit='d')
        end_d_min_lag = oneyr[-1] - pd.Timedelta(int(lag), unit='d')
        if pd.Timestamp(f'{dates[0].year}-01-01') > start_d_min_lag:
            start_d_min_lag = pd.Timestamp(f'{dates[0].year}-01-01')

        startyr = pd.date_range(start_d_min_lag, end_d_min_lag, freq=tfreq)

        if startyr.is_leap_year[0]:
            # ensure that everything before the leap day is shifted one day back in time
            # years with leapdays now have a day less, thus everything before
            # the leapday should be extended back in time by 1 day.
            mask_lpyrfeb = np.logical_and(startyr.month == 2,
                                                 startyr.is_leap_year
                                                 )
            mask_lpyrjan = np.logical_and(startyr.month == 1,
                                                 startyr.is_leap_year
                                                 )
            mask_ = np.logical_or(mask_lpyrfeb, mask_lpyrjan)

            new_dates = np.array(startyr)
            if np.logical_and(startyr[0].month==1, startyr[0].day==1)==False:
                # compensate lag shift for removing leap day
                new_dates[mask_] = startyr[mask_] - tfreq
            else:
                startyr =core_pp.remove_leapdays(startyr)
    else:
        startyr = get_oneyr(pd.to_datetime(dates.values))

    dates_min_lag = core_pp.make_dates(startyr, np.unique(dates.year))


    # to be able to select date in pandas dataframe
    dates_min_lag_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_min_lag]
    return dates_min_lag_str, dates_min_lag

def apply_lsm(var_filepath, lsm_filepath, threshold_lsm=.8):
    from pathlib import Path
    path = Path(var_filepath)
    xarray = core_pp.import_ds_lazy(path.as_posix())
    lons = xarray.longitude.values
    lats = xarray.latitude.values
    selbox = (min(lons), max(lons)+1, min(lats), max(lats)+1)
    lsm = core_pp.import_ds_lazy(lsm_filepath, selbox=selbox)
    lsm = lsm.to_array().squeeze() > threshold_lsm
    xarray['mask'] = (('latitude', 'longitude'), lsm[::-1].values)
    xarray = xarray.where( xarray['mask'] )
    xarray[0].plot()
    xarray = xarray.where(xarray.values != 0.).fillna(-9999)
    xarray.attrs.pop('is_DataArray')
    encoding = ( {xarray.name : {'_FillValue': -9999}} )
    mask =  (('latitude', 'longitude'), (xarray.values[0] != -9999) )
    xarray.coords['mask'] = mask
    parts = list(path.parts)
    parts[5] = 'lsm_' +parts[5]
    outfile = Path(*parts)
    # save netcdf
    xarray.to_netcdf(outfile, mode='w', encoding=encoding)

def sort_d_by_vals(d, reverse=False):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}

def remove_duplicates_list(l):
    seen = set()
    uniq = []
    indices = []
    for i, x in enumerate(l):
        if x not in seen:
            uniq.append(x)
            seen.add(x)
            indices.append(i)
    return uniq, indices

def match_coords_xarrays(wanted_coords_arr, *to_match):
    dlon = float(wanted_coords_arr.longitude[:2].diff('longitude'))
    dlat = float(wanted_coords_arr.latitude[:2].diff('latitude'))
    lonmin = wanted_coords_arr.longitude.min()
    lonmax = wanted_coords_arr.longitude.max()
    latmin = wanted_coords_arr.latitude.min()
    latmax = wanted_coords_arr.latitude.max()
    return [tomatch.sel(longitude=np.arange(lonmin, lonmax+dlon,dlon),
                       latitude=np.arange(latmin, latmax+dlat,dlat),
                       method='nearest') for tomatch in to_match]

def kornshell_with_input(args, cls):
#    stopped working for cdo commands
    '''some kornshell with input '''
#    args = [anom]
    import os
    import subprocess
    cwd = os.getcwd()
    # Writing the bash script:
    new_bash_script = os.path.join(cwd,'bash_scripts', "bash_script.sh")
#    arg_5d_mean = 'cdo timselmean,5 {} {}'.format(infile, outfile)
    #arg1 = 'ncea -d latitude,59.0,84.0 -d longitude,-95,-10 {} {}'.format(infile, outfile)

    bash_and_args = [new_bash_script]
    [bash_and_args.append(arg) for arg in args]
    with open(new_bash_script, "w") as file:
        file.write("#!/bin/sh\n")
        file.write("echo bash script output\n")
        for cmd in range(len(args)):

            print(args[cmd].replace(cls.base_path, 'base_path/')[:300])
            file.write("${}\n".format(cmd+1))
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    out = p.communicate()
    print(out[0].decode())
    return

def check_pp_done(name, infile, kwrgs_load: dict=None, verbosity=1):
    #%%
    '''
    Check if pre processed ncdf already exists
    '''
    # =============================================================================
    # load dataset lazy
    # =============================================================================
#    infile = os.path.join(ex['path_raw'], cls.filename)
    # if kwrgs_load is None:
    #     kwrgs = {'loadleap':False, 'format_lon':None}
    # else:
    #     keep = ['loadleap', 'format_lon', 'selbox']
    #     kwrgs = {k: kwrgs_load[k] for k in keep}
    # ds = core_pp.import_ds_lazy(infile, **kwrgs)
    ds = xr.open_dataset(infile, decode_cf=True, decode_coords=True, decode_times=False)
    ds = core_pp.ds_num2date(ds)
    dates = pd.to_datetime(ds['time'].values)
    start_day = get_oneyr(dates)[0]
    end_day   = get_oneyr(dates)[-1]
    # degree = int(ds.longitude[1] - ds.longitude[0])
    # selbox = [int(ds.longitude.min()), int(ds.longitude.max()),
    #            int(ds.latitude.min()), int(ds.latitude.max())]
    selbox = kwrgs_load['selbox']

    # =============================================================================
    # give appropriate name to output file
    # =============================================================================
    outfilename = infile.split(sep)[-1];
#    outfilename = outfilename.replace('daily', 'dt-{}days'.format(1))
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )

    input_freq = (dates[1] - dates[0]).days
    if input_freq == 1: # daily data
        startdatestr = '_{}{}_'.format(start_day.day, months[start_day.month])
        enddatestr   = '_{}{}_'.format(end_day.day, months[end_day.month])
    elif input_freq > 27 and input_freq < 32: # monthly data
        startdatestr = '_{}_'.format(months[start_day.month])
        enddatestr   = '_{}_'.format(months[end_day.month])

    if selbox is not None:
        selboxstr = '_'+'_'.join(map(str, selbox))
    else:
        selboxstr = ''
    # if core_pp.test_periodic(ds) and core_pp.test_periodic_lat(ds):
    #     selboxstr = '' # if global, no selbox str
    selboxstr_startdate = selboxstr+startdatestr


    outfilename = outfilename.replace('_{}_'.format(1), selboxstr_startdate)
    outfilename = outfilename.replace('_{}_'.format(12), enddatestr)
#    filename_pp = outfilename
    path_raw = sep.join(infile.split(sep)[:-1])
    path_pp = os.path.join(path_raw, 'preprocessed')
    if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
    outfile = os.path.join(path_pp, outfilename)
#    dates_fit_tfreq = dates
    #%%
    return outfile
