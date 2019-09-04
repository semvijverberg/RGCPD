#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd
from netCDF4 import num2date
import matplotlib.pyplot as plt
import xarray as xr
import itertools
import core_pp
import func_fc
from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: list(set([item for sublist in l for item in sublist]))
flatten = lambda l: list(itertools.chain.from_iterable(l))

def get_oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()

def Variable(self, ex):
    self.startyear = ex['startyear']
    self.endyear = ex['endyear']
    self.startmonth = 1
    self.endmonth = 12
    self.grid = ex['grid_res']
    self.dataset = ex['dataset']
    self.base_path = ex['base_path']
    self.path_raw = ex['path_raw']
    self.path_pp = ex['path_pp']
    return self


class Var_import_RV_netcdf:
    def __init__(self, ex):
        vclass = Variable(self, ex)

        vclass.name = ex['RVnc_name'][0]
        vclass.filename = ex['RVnc_name'][1]
        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name,
               vclass.startyear, vclass.endyear, vclass.grid)))

class Var_import_precursor_netcdf:
    def __init__(self, ex, idx):
        vclass = Variable(self, ex)

        vclass.name = ex['precursor_ncdf'][idx][0]
        vclass.filename = ex['precursor_ncdf'][idx][1]
        ex['vars'][0].append(vclass.name)


def perform_post_processing(ex):
    print('\nPerforming the post processing steps on {}'.format(ex['vars'][0]))
    for var in ex['vars'][0]:
        var_class = ex[var]
        outfile, var_class, ex = check_pp_done(var_class, ex)

        if os.path.isfile(outfile) == True:
            print('\nlooks like pre-processing for {} is already done,\n'
                  'to save time let\'s not do it twice..\n'.format(var))
            pass
        else:
            infile = os.path.join(var_class.path_raw, var_class.filename)
            core_pp.detrend_anom_ncdf3D(infile, outfile, ex)
        # update the dates stored in var_class:
        var_class, ex = update_dates(var_class, ex)
        # store updates
        ex[var] = var_class


def check_pp_done(cls, ex):
    #%%
    '''
    Check if pre processed ncdf already exists
    '''
    # =============================================================================
    # load dataset lazy
    # =============================================================================

    import pandas as pd
    filename = os.path.join(ex['path_raw'], cls.filename)
    ds = core_pp.import_ds_lazy(filename, ex)
    dates = pd.to_datetime(ds['time'].values)

    # =============================================================================
    # get time series that you request
    # =============================================================================

#    dates = timeseries_tofit_bins(ds, ex, seldays='part')[1]

    start_day = get_oneyr(dates)[0]
    end_day   = get_oneyr(dates)[-1]

    # =============================================================================
    # give appropriate name to output file
    # =============================================================================
    outfilename = cls.filename[:-3]+'.nc'
#    outfilename = outfilename.replace('daily', 'dt-{}days'.format(1))
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )

    if ex['input_freq'] == 'daily':
        startdatestr = '_{}{}_'.format(start_day.day, months[start_day.month])
        enddatestr   = '_{}{}_'.format(end_day.day, months[end_day.month])
    elif ex['input_freq'] == 'monthly':
        startdatestr = '_{}_'.format(months[start_day.month])
        enddatestr   = '_{}_'.format(months[end_day.month])

    outfilename = outfilename.replace('_{}_'.format(1), startdatestr)
    outfilename = outfilename.replace('_{}_'.format(12), enddatestr)
    cls.filename_pp = outfilename
    cls.path_pp = ex['path_pp']
    outfile = os.path.join(ex['path_pp'], outfilename)
    cls.dates_fit_tfreq = dates
    print('output file of pp will be saved as: \n' + outfile)
    #%%
    return outfile, cls, ex


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

def update_dates(cls, ex):
    import os
    file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ds = core_pp.import_ds_lazy(file_path, ex)

    temporal_freq = pd.Timedelta((ds['time'][1] - ds['time'][0]).values)
    cls.dates = pd.to_datetime(ds['time'].values)
    cls.temporal_freq = '{}days'.format(temporal_freq.days)
    return cls, ex

def RV_spatial_temporal_mask(ex, RV, importRV_1dts):
    '''
    Select months of your Response Variable that you want to predict.
    RV = the RV class
    ex = experiment dictionary
    months = list of integers
    If you select [6,7] you will attempt to correlate precursor gridcells with
    lag x versus the response variable values in june and july.

    The second step is to insert a spatial mask -only if- you inserted a 3D field
    as your response variable (time, lats, lons).
    '''
    #%%
    if importRV_1dts == True:
        print('\nimportRV_1dts is true, so the 1D time serie given with filename\n'
              '{} is imported.\n'.format(ex['RVts_filename']))
        RV.name = ex['RV_name']
        if ex['RVts_filename'].split('.')[-1]  == 'csv':
            print('Assuming .csv, where rows are timesteps and 4 columns are\n'
                  'Year, Months, Day' )
            ex = csv_to_npy(ex)
        dicRV = np.load(os.path.join(ex['path_pp'], 'RVts', ex['RVts_filename']),
                        encoding='latin1', allow_pickle=True).item()
    #    dicRV = pickle.load( open(os.path.join(ex['path_pp'],ex['RVts_filename']+'.pkl'), "rb") )
        try:
            RV.RVfullts = dicRV['RVfullts']
        except:
            RV.RVfullts = dicRV['RVfullts95']
        RV.filename = ex['RVts_filename']




    elif importRV_1dts == False:
        RV.name = ex['vars'][0][0]
        # RV should always be the first variable of the vars list in ex
        RV = ex[RV.name]
        RVarray, RV = import_array(RV)
        print('The RV variable is the 0th index in ex[\'vars\'], '
              'i.e. {}'.format(RV.name))
        # =============================================================================
        # 3.2 Select spatial mask to create 1D timeseries (from .npy file)
        # =============================================================================
        # You can load a spatial mask here and use it to create your
        # full timeseries (of length equal to actor time series)
        if type(ex['spatial_mask_file']) == type(str()):
            try:
                mask_dic = np.load(ex['spatial_mask_file'], encoding='latin1').item()
                print('spatial mask loaded:')
                xarray_plot(mask_dic['RV_array'])
                resol_mask = mask_dic['RV_array'].longitude[1]-mask_dic['RV_array'].longitude[0]
                RV_array, RV = import_array(RV, path='pp')
                resol_ncdf = RV_array.longitude[1]-RV_array.longitude[0]
                # test if resolution matches
                assert (resol_mask - resol_ncdf).values == 0, ('resolution of '
                       'spatial mask not equal to resolution of precursor')
                RV_array.coords['mask'] = mask_dic['RV_array'].mask
                lats = RV_array.latitude.values
                cos_box = np.cos(np.deg2rad(lats))
                cos_box_array = np.tile(cos_box, (RVarray.longitude.size,1) )
                weights_box = np.swapaxes(cos_box_array, 1,0)
                weights_box = weights_box / np.mean(weights_box)
                RVarray_w = weights_box[None,:,:] * RVarray
                if RV_array.mask.dtype == 'float':
                    RV.mask = RV_array.mask == 1
                elif RV.mask.dtype == 'bool':
                    RV.mask = RV_array.mask
                print('spatial mask added to Response Variable:')
                xarray_plot(RV_array)
                RV.RVfullts = (RVarray_w).where(
                        RV.mask).mean(dim=['latitude','longitude']
                        ).squeeze()


            except IOError as e:
                print('\n\n**\nSpatial mask not found.\n \n {}'.format(
                        ex['spatial_mask_file']))
                raise(e)
        if type(ex['spatial_mask_file']) == type(list()):
            latlonbox = ex['spatial_mask_file']
            RV.RVfullts = selbox_to_1dts(RV, latlonbox)

    RV.dates = pd.to_datetime(RV.RVfullts.time.values)
    RV.startyear = RV.dates.year[0]
    RV.endyear = RV.dates.year[-1]
    RV.n_timesteps = RV.dates.size
    RV.n_yrs       = (RV.endyear - RV.startyear) + 1

    if ex['input_freq'] == 'daily':
        same_freq = (RV.dates[1] - RV.dates[0]).days == ex['tfreq']
    elif ex['input_freq'] == 'monthly' and RV.n_yrs != RV.n_timesteps:
        same_freq = (RV.dates[1].month - RV.dates[0].month) == ex['tfreq']
    else:
        same_freq = True
#    same_len_yr = RV.dates.size == ex[ex['vars'][0][0]].dates.size

    if same_freq == False:
        print('tfreq of imported 1d timeseries is unequal to the '
              'desired ex[tfreq]\nWill convert tfreq')
        to_freq = ex['tfreq']
        RV.RVfullts, RV.dates, = time_mean_bins(RV.RVfullts, ex, to_freq)


    if same_freq == True:

        RV.RVfullts, RV.dates = timeseries_tofit_bins(RV.RVfullts, ex, seldays='part')
        print('The amount of timesteps in the RV ts and the precursors'
                          ' do not match, selecting desired dates. ')


    if ex['RV_detrend'] == True:
        print('Detrending Respone Variable.')
        RV.RVfullts = detrend1D(RV.RVfullts)

    if ex['input_freq'] == 'daily':
        RV.dates_RV = make_RVdatestr(pd.to_datetime(RV.RVfullts.time.values), ex,
                              ex['startyear'], ex['endyear'], lpyr=False)
    elif ex['input_freq'] == 'monthly':

        want_month = np.arange(int(ex['startperiod'].split('-')[0]),
                           int(ex['endperiod'].split('-')[0])+1)
        months = RV.RVfullts.time.dt.month
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
        xrdates = RV.RVfullts.time.where(mask).dropna(dim='time')
        RV.dates_RV = pd.to_datetime(xrdates.values)

    # get indices of RVdates
    string_RV = list(RV.dates_RV.strftime('%Y-%m-%d'))
    string_full = list(RV.dates.strftime('%Y-%m-%d'))
    ex['RV_period'] = [string_full.index(date) for date in string_full if date in string_RV]

    RV.RV_ts = RV.RVfullts[ex['RV_period']] # extract specific months of MT index
    # Store added information in RV class to the exp dictionary
    ex['RV_name'] = RV.name

    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                    7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    RV_name_range = '{}{}-{}{}_'.format(RV.dates_RV[0].day, months[RV.dates_RV.month[0]],
                     RV.dates_RV[-1].day, months[RV.dates_RV.month[-1]] )

    info_lags = 'lag{}-{}'.format(min(ex['lags']), max(ex['lags']))

    # Creating a folder for the specific spatial mask, RV period and traintest set
    if importRV_1dts == True:
        ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], RV_name_range + \
                                          info_lags )


    elif importRV_1dts == False:
        ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], RV_name_range +
                                      ex['spatial_mask_naming'] + info_lags )

    RV.RV_name_range = RV_name_range
    #%%
    return RV, ex


def csv_to_npy(ex):
   #%%
   import os
   import pandas as pd
   import xarray as xr
   import numpy as np
   # load data from csv file and save to .npy as xarray format

   path = os.path.join(ex['path_pp'], 'RVts2.5', ex['RVts_filename'])
   table = pd.read_csv(path)
   data  = np.array(table)
   dates = pd.to_datetime(['{}-{}-{}'.format(A[0],A[1],A[2]) for A in data])

   y_val = data[:,-1]  # ATTENTION: This only works if values are in last column

   xrdata = xr.DataArray(data=y_val, coords=[dates], dims=['time'])

   ofile = ex['RVts_filename'].split('.')[0] + '.npy'
   to_dict = dict( {'RVfullts'     : xrdata } )
   np.save(os.path.join(ex['path_pp'], 'RVts2.5', ofile), to_dict)
   ex['RVts_filename'] = ofile

   #%%
   return ex


def time_mean_bins(xarray, ex, to_freq=int, verb=0):
   #%%
   datetime = pd.to_datetime(xarray['time'].values)
    # ensure to remove leapdays
   datetime = remove_leapdays(datetime)
   xarray = xarray.sel(time=datetime)
   one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')

   if one_yr.size % to_freq != 0:
       possible = []
       for i in np.arange(1,20):
           if one_yr.size%i == 0:
               possible.append(i)
       if verb == 1:
           print('Note: stepsize {} does not fit in one year\n '
                            ' supply an integer that fits {}'.format(
                                to_freq, one_yr.size))
           print('\n Stepsize that do fit are {}'.format(possible))
           print('\n Will shorten the \'subyear\', so that the temporal'
                 ' frequency fits in one year')
       datetime = pd.to_datetime(np.array(xarray['time'].values,
                                          dtype='datetime64[D]'))
       datetime = timeseries_tofit_bins(datetime, ex, to_freq, seldays='all', verb=0)
       xarray = xarray.sel(time=datetime)
       one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')

   else:
       pass
   fit_steps_yr = (one_yr.size )  / to_freq
   bins = list(np.repeat(np.arange(0, fit_steps_yr), to_freq))
   n_years = np.unique(datetime.year).size
   for y in np.arange(1, n_years):
       x = np.repeat(np.arange(0, fit_steps_yr), to_freq)
       x = x + fit_steps_yr * y
       [bins.append(i) for i in x]
   label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
   label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
   xarray['bins'] = label_bins
   xarray['time_dates'] = label_dates
   xarray = xarray.set_index(time=['bins','time_dates'])

   half_step = to_freq/2.
   newidx = np.arange(half_step, datetime.size, to_freq, dtype=int)
   newdate = label_dates[newidx]


   group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
   group_bins['bins'] = newdate.values
   dates = pd.to_datetime(newdate.values)
   #%%
   return group_bins.rename({'bins' : 'time'}), dates

# def time_mean_bins_old(xarray, ex, seldays = 'part'):
#     #%%
#     import xarray as xr
#     datetime_orig = pd.to_datetime(xarray['time'].values)
#     ex['n_oneyr'] = get_oneyr(datetime_orig).size
#
#     # does the amount of steps per year already fit the bins?
#     need_fit_bins = (ex['n_oneyr'] % ex['tfreq'] != 0)
#     if (need_fit_bins or seldays == 'part'):
#         possible = []
#         for i in np.arange(1,20):
#             if ex['n_oneyr']%i == 0:
#                 possible.append(i)
#         if ex['n_oneyr'] % ex['tfreq'] != 0:
#             if ex['verbosity'] > 1:
#                 print('Note: tfreq {} does not fit in the supplied (sub)year\n'
#                          'adjusting part of year to fit bins.'.format(
#                              ex['tfreq']))
# #            print('\n Stepsize that do fit are {}'.format(possible))
# #        print('\n Will shorten the \'subyear\', so that the temporal'
# #              ' frequency fits in one year')
#         xarray, datetime = timeseries_tofit_bins(xarray, ex, seldays='part')
#
#     else:
#         pass
#     fit_steps_yr = ex['n_oneyr']  / ex['tfreq']
#     assert fit_steps_yr >= 1, ('{} {} mean does not fit in the period '
#                               'you selected'.format(ex['tfreq'], ex['input_freq']))
#     bins = list(np.repeat(np.arange(0, fit_steps_yr), ex['tfreq']))
#     for y in np.arange(1, ex['n_yrs']):
#         x = np.repeat(np.arange(0, fit_steps_yr), ex['tfreq'])
#         x = x + fit_steps_yr * y
#         [bins.append(i) for i in x]
#     label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
#     label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
#     xarray['bins'] = label_bins
#     xarray['time_dates'] = label_dates
#     xarray = xarray.set_index(time=['bins','time_dates'])
#
#     half_step = ex['tfreq']/2.
#     newidx = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
#     newdate = label_dates[newidx]
#
#
#     group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
#     group_bins['bins'] = newdate.values
#     dates = pd.to_datetime(newdate.values)
#     #%%
#     return group_bins.rename({'bins' : 'time'}), dates, datetime_orig

def timeseries_tofit_bins(xarray, ex, seldays='part'):
    #%%
    datetime = pd.to_datetime(xarray['time'].values)

    leapdays = ((datetime.is_leap_year) & (datetime.month==2) & (datetime.day==29))==False
    datetime = datetime[leapdays].dropna(how='all')

# =============================================================================
#   # select dates
# =============================================================================
    # selday_pp is the period you aim to study
    if seldays == 'part':
        # add corresponding time information
        crossyr = int(ex['sstartdate'].replace('-','')) > int(ex['senddate'].replace('-',''))
        sstartdate = '{}-{}'.format(ex['startyear'], ex['sstartdate'])
        if crossyr:
            senddate   = '{}-{}'.format(ex['startyear']+1, ex['senddate'])
        else:
            senddate   = '{}-{}'.format(ex['startyear'], ex['senddate'])

        ex['adjhrsstartdate'] = sstartdate + ' {:}:00:00'.format(datetime[0].hour)
        ex['adjhrsenddate']   = senddate + ' {:}:00:00'.format(datetime[0].hour)
        sdate = pd.to_datetime(ex['adjhrsstartdate'])
        seldays_pp = pd.DatetimeIndex(start=ex['adjhrsstartdate'], end=ex['adjhrsenddate'],
                                freq=datetime[1] - datetime[0])


    if seldays == 'all':
        one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
        sdate = one_yr[0]
        seldays_pp = pd.DatetimeIndex(start=one_yr[0], end=one_yr[-1],
                                freq=datetime[1] - datetime[0])

    if ex['input_freq'] == 'daily':
        dt = np.timedelta64(ex['tfreq'], 'D')
        end_day = seldays_pp.max()
        start_day = seldays_pp.min()
        # after time averaging over 'tfreq' number of days, you want that each year
        # consists of the same day. For this to be true, you need to make sure that
        # the selday_pp period exactly fits in a integer multiple of 'tfreq'
        fit_steps_yr = (end_day - start_day + np.timedelta64(1, 'D'))  / dt
        # line below: The +1 = include day 1 in counting
        start_day = (end_day - (dt * np.round(fit_steps_yr, decimals=0))) \
                    + np.timedelta64(1, 'D')


        if start_day.dayofyear < sdate.dayofyear or start_day.year < sdate.year:
            # if startday is before the desired starting period, skip one bin forward in time
            start_day = (end_day - (dt * np.round(fit_steps_yr-1, decimals=0))) \
                    + np.timedelta64(1, 'D')

        start_yr = pd.DatetimeIndex(start=start_day, end=end_day,
                                    freq=(datetime[1] - datetime[0]))
        # exluding leap year from cdo select string
        noleapdays = (((start_yr.month==2) & (start_yr.day==29))==False)
        start_yr = start_yr[noleapdays].dropna(how='all')

    if ex['input_freq'] == 'monthly':
        dt = date_dt(months=ex['tfreq'])
        start_day = ex['adjhrsstartdate'].split(' ')[0]
        start_day = pd.to_datetime(start_day.replace(start_day[-2:], '01'))
        end_day = ex['adjhrsenddate'].split(' ')[0]
        end_day = pd.to_datetime(end_day.replace(end_day[-2:], '01'))
        fit_steps_yr = (end_day.month - start_day.month + 1) / ex['tfreq']
        start_day = (end_day - (dt * int(fit_steps_yr))) \
                + date_dt(months=+1)
        days_back = end_day
        start_yr = [end_day.strftime('%Y-%m-%d %H:%M:%S')]
        while start_day < days_back:
            days_back -= date_dt(months=+1)
            start_yr.append(days_back.strftime('%Y-%m-%d %H:%M:%S'))
        start_yr.reverse()
        start_yr = pd.to_datetime(start_yr)

        #%%
    def make_dates(datetime, start_yr):
        breakyr = datetime.year.max()
        nyears = (datetime.year[-1] - datetime.year[0])+1
        next_yr = start_yr
        for yr in range(0,nyears-1):
            next_yr = pd.to_datetime([date + date_dt(years=1) for date in next_yr])
            start_yr = start_yr.append(next_yr)
            if next_yr[-1].year == breakyr:
                break
        return start_yr


    ex['n_oneyr'] = start_yr.size

#    datesdt = make_datestr_2(datetime, start_yr)
    datesdt = make_dates(datetime, start_yr)

    ex['n_yrs'] = datesdt.size / ex['n_oneyr']
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    startdatestr = '{} {}'.format(start_day.day, months[start_day.month])
    enddatestr   = '{} {}'.format(end_day.day, months[end_day.month])

    if ex['input_freq'] == 'daily' and ex['verbosity'] > 0:
        print('Period of year selected: \n{} to {}, tfreq {} days'.format(
                startdatestr, enddatestr, ex['tfreq']))
    if ex['input_freq'] == 'monthly' and ex['verbosity'] > 0:
        print('Months of year selected: \n{} to {}, tfreq {} months'.format(
                startdatestr.split(' ')[-1], enddatestr.split(' ')[-1], ex['tfreq']))
    adj_xarray = xarray.sel(time=datesdt)
    #%%
    return adj_xarray, datesdt

def make_RVdatestr(dates, ex, startyr, endyr, lpyr=False):
    import calendar

    def oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()


    sstartdate = str(startyr) + '-' + ex['startperiod']
    senddate   = str(startyr) + '-' + ex['endperiod']

    daily_yr_fit = np.round(pd.DatetimeIndex(start=sstartdate, end=senddate,
                            freq=pd.Timedelta(1, 'd')).size / ex['tfreq'], 0)

    firstyr = oneyr(dates)
    #find closest senddate
    closest_enddate_idx = np.argmin(abs(firstyr - pd.to_datetime(senddate)))
    senddate = firstyr[closest_enddate_idx]

    #update startdate of RV period to fit bins
    sstartdate = senddate - pd.Timedelta(int(ex['tfreq'] * daily_yr_fit), 'd')

    start_yr = pd.DatetimeIndex(start=sstartdate, end=senddate,
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

        next_yr = pd.DatetimeIndex(start=startday, end=endday,
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
    datesmcK = pd.to_datetime(datesstr)
    return datesmcK

def import_array(cls, path='pp'):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    if path == 'raw':
        file_path = os.path.join(cls.path_raw, cls.filename)

    else:
        file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = np.squeeze(ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')})))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    if numtime.attrs['calendar'] != 'gregorian':
        dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates = pd.to_datetime(dates)
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    marray['time'] = dates
    cls.dates = dates
    return marray, cls

def import_ds_timemeanbins(file_path, ex, loadleap=True, to_xarr=True):

    ds = core_pp.import_ds_lazy(file_path, ex, loadleap=loadleap)
    to_freq = ex['tfreq']
    if to_freq != 1:
        ds, dates = time_mean_bins(ds, ex, to_freq=to_freq, seldays='part')
        ds['time'] = dates
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    if to_xarr:
        if type(ds) == type(xr.DataArray(data=[0])):
            ds = ds.squeeze()
        else:
            ds = ds.to_array().squeeze()

    return ds


def xarray_plot(data, path='default', name = 'default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    plt.figure()
    data = data.squeeze()
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        data = convert_longitude(data)
    else:
        pass
    if data.ndim != 2:
        print("number of dimension is {}, printing first element of first dimension".format(np.squeeze(data).ndim))
        data = data[0]
    else:
        pass
    if 'mask' in list(data.coords.keys()):
        cen_lon = data.where(data.mask==True, drop=True).longitude.mean()
        data = data.where(data.mask==True, drop=True)
    else:
        cen_lon = data.longitude.mean().values
    proj = ccrs.LambertCylindrical(central_longitude=cen_lon)
#    proj = ccrs.Orthographic(central_longitude=cen_lon, central_latitude=data.latitude.mean())
    ax = plt.axes(projection=proj)
    ax.coastlines()
    # ax.set_global()
    if 'mask' in list(data.coords.keys()):
        plot = data.where(data.mask==True).plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True)
    else:
        plot = data.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True)
    if saving == True:
        save_figure(data, path=path)
    plt.show()

def convert_longitude(data, to_format='west_east'):
    import numpy as np
    import xarray as xr
    if to_format == 'west_east':
        lon_above = data.longitude[np.where(data.longitude > 180)[0]]
        lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
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
        lon_above = data.longitude[np.where(data.longitude >= 0)[0]]
        lon_below = data.longitude[np.where(data.longitude < 0)[0]]
        lon_below += 360
        data = data.roll(longitude=len(lon_below))
        convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
    data['longitude'] = convert_lon
    return data

def find_region(data, region='EU'):
    import numpy as np

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)

    def find_nearest_coords(array, region_coords):
        for lon_value in region_coords[:2]:
            region_idx = region_coords.index(lon_value)
            idx = find_nearest(data['longitude'], lon_value)
            if region_coords[region_idx] != float(data['longitude'][idx].values):
                print('longitude value of latlonbox did not match, '
                      'updating to nearest value')
            region_coords[region_idx] = float(data['longitude'][idx].values)
        for lat_value in region_coords[2:]:
            region_idx = region_coords.index(lat_value)
            idx = find_nearest(data['latitude'], lat_value)
            if region_coords[region_idx] != float(data['latitude'][idx].values):
                print('latitude value of latlonbox did not match, '
                      'updating to nearest value')
            region_coords[region_idx] = float(data['latitude'][idx].values)
        return region_coords

    if region == 'EU':
        west_lon = -30; east_lon = 40; south_lat = 35; north_lat = 65

    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50

    if type(region) == list:
        west_lon = region[0]; east_lon = region[1];
        south_lat = region[2]; north_lat = region[3]
    region_coords = [west_lon, east_lon, south_lat, north_lat]

    # Update regions coords in case they do not exactly match
    region_coords = find_nearest_coords(data, region_coords)
    west_lon = region_coords[0]; east_lon = region_coords[1];
    south_lat = region_coords[2]; north_lat = region_coords[3]


    lonstep = abs(data.longitude[1] - data.longitude[0])
    latstep = abs(data.latitude[1] - data.latitude[0])
    # abs() enforces that all values are positve, if not the case, it will not meet
    # the conditions
    lons = abs(np.arange(data.longitude[0], data.longitude[-1]+lonstep, lonstep))



    if (lons == np.array(data.longitude.values)).all():

        lons = list(np.arange(west_lon, east_lon+lonstep, lonstep))
        lats = list(np.arange(south_lat, north_lat+latstep, latstep))

        all_values = data.sel(latitude=lats, longitude=lons)
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))
        lon_idx = np.concatenate(( np.arange(find_nearest(data['longitude'], 360 + west_lon), len(data['longitude'])),
                              np.arange(0,find_nearest(data['longitude'], east_lon), 1) ))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        all_values = data.sel(latitude=slice(north_lat, south_lat),
                              longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values, region_coords

def selbox_to_1dts(cls, latlonbox):
    marray, var_class = import_array(cls, path='pp')
    selboxmarray, region_coords = find_region(marray, latlonbox)
    print('spatial mean over latlonbox {}'.format(region_coords))
    lats = selboxmarray.latitude.values
    cos_box = np.cos(np.deg2rad(lats))
    cos_box_array = np.tile(cos_box, (selboxmarray.longitude.size,1) )
    weights_box = np.swapaxes(cos_box_array, 1,0)
    RV_fullts = (selboxmarray*weights_box).mean(dim=('latitude','longitude'))
    return RV_fullts



def detrend1D(da):
    import scipy.signal as sps
    import xarray as xr
    dates = pd.to_datetime(da.time.values)
    stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')

    if (stepsyr.day== 1).all() == True or int(da.time.size / 365) >= 120:
        print('\nHandling time series longer then 120 day or monthly data, no smoothening applied')
        data_smooth = da.values

    elif (stepsyr.day== 1).all() == False and int(da.time.size / 365) < 120:
        window_s = max(min(25,int(stepsyr.size / 12)), 1)
        print('Performing {} day rolling mean with gaussian window (std={})'
              ' to get better interannual statistics'.format(window_s, window_s/2))

        print('using absolute anomalies w.r.t. climatology of '
              'smoothed concurrent day accross years')
        data_smooth =  rolling_mean_np(da.values, window_s)
    output = np.empty( (da.time.size), dtype='float32' )
    for i in range(stepsyr.size):
        sliceyr = np.arange(i, da.time.size, stepsyr.size)
        arr_oneday = da.isel(time=sliceyr)
        arr_oneday_smooth = data_smooth[sliceyr]

        detrended_sm = xr.DataArray(sps.detrend(arr_oneday_smooth),
                            dims=arr_oneday.dims,
                            coords=arr_oneday.coords)
        # subtract trend smoothened signal of arr_oneday values
        trend = (arr_oneday_smooth - detrended_sm)- np.mean(arr_oneday_smooth, 0)
        detrended = arr_oneday - trend
        output[i::stepsyr.size] = detrended
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
    import xesmf as xe
    method_list = ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']
    method = method_list[0]


    ds = xr.Dataset({'data':xarray_in})

    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon',
                        'latitude' : 'lat'})

    lats = ds.lat
    lons = ds.lon

    if method == 'conservative':
        # add lon_b and lat_b
        orig_grid = float(abs(ds.lat[1] - ds.lat[0] ))

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
    regridder = xe.Regridder(ds, to_grid, method, periodic=periodic)
    xarray_out = regridder(ds['data'])
    regridder.clean_weight_file()
    xarray_out = xarray_out.rename({'lon':'longitude',
                                    'lat':'latitude'})
    xarray_out = xr.DataArray(xarray_out.values[::-1], dims=['latitude', 'longitude'],
                 coords={'latitude':xarray_out.latitude[:,0].values[::-1],
                         'longitude':xarray_out.longitude[0].values})
    xarray_out['longitude'] -= xarray_out['longitude'][0]
    #%%
    return xarray_out



def rand_traintest_years(RV, precur_arr, ex):
    #%%
    '''
    possible ex['method'] are:
    random{int} : with the int(ex['method'][6:8]) determining the amount of folds
    leave{int} : chronologically split train and test years
    split{int} : split dataset into single train and test set
    no_train_test_split.
    '''


    RV_ts = RV.RV_ts
    ex['tested_yrs'] = [] ; # ex['n_events'] = []
    ex['all_yrs'] = list(np.unique(RV_ts.time.dt.year))

    if ex['method'][:6] == 'random' or ex['method'][:9] == 'ran_strat':
        if 'seed' not in ex.keys():
            ex['seed'] = 30 # control reproducibility train/test split
        else:
            ex['seed'] = ex['seed']

        seed = ex['seed']

        if ex['method'][:6] == 'random':
            ex['n_spl'] = int(ex['method'][6:8])
        else:
             ex['n_spl'] = int(ex['method'][9:])


    elif ex['method'][:5] == 'leave':
        ex['n_spl'] = int(ex['n_yrs'] / int(ex['method'].split('_')[1]) )
        iterate = np.arange(0, ex['n_yrs']+1E-9,
                            int(ex['method'].split('_')[1]), dtype=int )
    elif ex['method'] == 'no_train_test_split': ex['n_spl'] = 1

    traintest = []
    for s in range(ex['n_spl']):

        # conditions failed initally assumed True
        a_conditions_failed = True
        count = 0

        while a_conditions_failed == True:
            count +=1
            a_conditions_failed = False


            if ex['method'][:6] == 'random' or ex['method'][:9] == 'ran_strat':


                rng = np.random.RandomState(seed)
                size_test  = int(np.round(ex['n_yrs'] / ex['n_spl']))
                size_train = int(ex['n_yrs'] - size_test)

                ex['leave_n_years_out'] = size_test
                yrs_to_draw_sample = [yr for yr in ex['all_yrs'] if yr not in flatten(ex['tested_yrs'])]
                if (len(yrs_to_draw_sample)) >= size_test:
                    rand_test_years = rng.choice(yrs_to_draw_sample, ex['leave_n_years_out'], replace=False)
                # if last test sample will be too small for next iteration, add test yrs to current test yrs
                if (len(yrs_to_draw_sample)) < size_test:
                    rand_test_years = yrs_to_draw_sample
                check_double_test = [yr for yr in rand_test_years if yr in flatten( ex['tested_yrs'] )]
                if len(check_double_test) != 0 :
                    a_conditions_failed = True
                    print('test year drawn twice, redoing sampling')


            elif ex['method'][:5] == 'leave':
                ex['leave_n_years_out'] = int(ex['method'].split('_')[1])
                t0 = iterate[s]
                t1 = iterate[s+1]
                rand_test_years = ex['all_yrs'][t0: t1]

            elif ex['method'][:5] == 'split':
                size_train = int(np.percentile(range(len(ex['all_yrs'])), int(ex['method'][5:])))
                size_test  = len(ex['all_yrs']) - size_train
                ex['leave_n_years_out'] = size_test
                print('Using {} years to train and {} to test'.format(size_train, size_test))
                rand_test_years = ex['all_yrs'][-size_test:]

            elif ex['method'] == 'no_train_test_split':
                size_train = len(ex['all_yrs'])
                size_test  = 0
                ex['leave_n_years_out'] = size_test
                print('No train test split'.format(size_train, size_test))
                rand_test_years = []



            # test duplicates
            a_conditions_failed = np.logical_and((len(set(rand_test_years)) != ex['leave_n_years_out']),
                                     s != ex['n_spl']-1)
            # Update random years to be selected as test years:
        #        initial_years = [yr for yr in initial_years if yr not in random_test_years]
            rand_train_years = [yr for yr in ex['all_yrs'] if yr not in rand_test_years]


            full_years  = list(precur_arr.time.dt.year.values)
            RV_years  = list(RV_ts.time.dt.year.values)

            Prec_train_idx = [i for i in range(len(full_years)) if full_years[i] in rand_train_years]
            RV_train_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_train_years]
            RV_train = RV_ts.isel(time=RV_train_idx)

            if ex['method'] != 'no_train_test_split':
                Prec_test_idx = [i for i in range(len(full_years)) if full_years[i] in rand_test_years]
                RV_test_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_test_years]
                RV_test = RV_ts.isel(time=RV_test_idx)
                test_years = np.unique(RV_test.time.dt.year)
                if ex['method'][:9] == 'ran_strat':
                    # check if representative sample
                    a_conditions_failed, count, seed = check_test_split(RV, RV_test, ex, a_conditions_failed,
                                                                        s, count, seed, ex['verbosity'])
            else:
                RV_test = [] ; test_years = [] ; Prec_test_idx = []


        ex['tested_yrs'].append(test_years)

        traintest_ = dict( { 'years'            : test_years,
                            'RV_train'          : RV_train,
                            'Prec_train_idx'    : Prec_train_idx,
                            'RV_test'           : RV_test,
                            'Prec_test_idx'     : Prec_test_idx} )
        traintest.append(traintest_)
        ex['traintest'] = traintest

    #%%
    return traintest, ex

def check_test_split(RV, RV_test, ex, a_conditions_failed, s, count, seed, verbosity=0):

    ex['event_thres'] = func_fc.Ev_threshold(RV.RV_ts, ex['kwrgs_events']['event_percentile'])
    tol_from_exp_events = 0.20

    if 'kwrgs_events' not in ex.keys():
        print('Stratified Train Test based on +1 tercile events\n')
        ex['kwrgs_events']  =  {'event_percentile': 66,
                    'min_dur' : 1,
                    'max_break' : 0,
                    'grouped' : False}

    if ex['kwrgs_events']['event_percentile'] == 'std':
        exp_events_r = 0.15
    elif type(ex['kwrgs_events']['event_percentile']) == int:
        exp_events_r = 1 - ex['kwrgs_events']['event_percentile']/100


    test_years = np.unique(RV_test.time.dt.year)
    exp_events = (exp_events_r * RV.RV_ts.size / ex['n_yrs']) * test_years.size
    tolerance      = tol_from_exp_events * exp_events
    event_test = func_fc.Ev_timeseries(RV_test, ex['event_thres'])[1]
    diff           = abs(len(event_test) - exp_events)


    if diff > tolerance:
        if verbosity > 1:
            print('not a representative sample drawn, drawing new sample')
        seed += 1 # next random sample
        a_conditions_failed = True
    else:
        if verbosity > 0:
            print('{}: test year is {}, with {} events'.format(s, test_years, len(event_test)))
    if count == 7:
        if verbosity > 1:
            print(f"{s}: {count+1} attempts made, lowering tolence threshold from {tol_from_exp_events} "
                "to 0.40 deviation from mean expected events" )
        tol_from_exp_events = 0.40
    if count == 10:
        if verbosity > 1:
            print(f"kept sample after {count+1} attempts")
            print('{}: test year is {}, with {} events'.format(s, test_years, len(event_test)))
        a_conditions_failed = False

    return a_conditions_failed, count, seed


