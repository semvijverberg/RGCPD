#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd

class Variable:
    from datetime import datetime, timedelta
    import pandas as pd
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper
    """
    ecmwf_website = 'http://apps.ecmwf.int/codes/grib/param-db'
    def __init__(self, ex, idx, ECMWFdown):
        import calendar
        import os
        # self is the instance of the employee class
        # below are listed the instance variables
        self.name = ex['vars'][0][idx]
        self.startyear = ex['startyear']
        self.endyear = ex['endyear']
        self.startmonth = 1
        self.endmonth = 12
        self.grid = ex['grid_res']
        self.dataset = ex['dataset']
        self.base_path = ex['base_path']
        self.path_raw = ex['path_raw']
        self.path_pp = ex['path_pp']
        if ECMWFdown == True:
            self.var_cf_code = ex['vars'][1][idx]
            self.levtype = ex['vars'][2][idx]
            self.lvllist = ex['vars'][3][idx]
            self.stream = 'oper'
#            if stream == 'oper':
            time_ana = "00:00:00/06:00:00/12:00:00/18:00:00"
#            else:
#                time_ana = "00:00:00"
            self.time_ana = time_ana 
            
            days_in_month = dict( {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} )
            days_in_month_leap = dict( {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} )
            start = Variable.datetime(self.startyear, self.startmonth, 1)
            
            # creating list of dates that we want to download given the startyear/startmonth to endyear/endmonth
            datelist_str = [start.strftime('%Y-%m-%d')]
            if self.stream == 'oper':
                end = Variable.datetime(self.endyear, self.endmonth, days_in_month[self.endmonth])
                while start < end:          
                    start += Variable.timedelta(days=1)
                    datelist_str.append(start.strftime('%Y-%m-%d'))
                    if start.month == end.month and start.day == days_in_month[self.endmonth] and start.year != self.endyear:
                        start = Variable.datetime(start.year+1, self.startmonth, 1)
                        datelist_str.append(start.strftime('%Y-%m-%d'))  
            elif self.stream == 'moda' or 'mnth':
                end = Variable.datetime(self.endyear, self.endmonth, 1)
                while start < end:          
                    days = days_in_month[start.month] if calendar.isleap(start.year)==False else days_in_month_leap[start.month]
                    start += Variable.timedelta(days=days)
                    datelist_str.append(start.strftime('%Y-%m-%d'))
                    if start.month == end.month and start.year != self.endyear:
                        start = Variable.datetime(start.year+1, self.startmonth, 1)
                        datelist_str.append(start.strftime('%Y-%m-%d'))             
            self.datelist_str = datelist_str

            # Convert to datetime datelist
#            self.dates_dt = [Variable.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_str]
            self.dates = Variable.pd.to_datetime(datelist_str)
    
            self.filename = '{}_{}-{}_{}_{}_{}_{}deg.nc'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, 'daily', self.grid).replace(' ', '_')
        elif ECMWFdown == False:
            if len(ex['own_actor_nc_names'][0]) != 0:
                self.name = ex['own_actor_nc_names'][idx][0]
                self.filename = ex['own_actor_nc_names'][idx][1]
                ex['vars'][0].append(self.name)
                print('t')
            if len(ex['own_RV_nc_name']) != 0:
                self.name = ex['own_RV_nc_name'][0]
                self.filename = ex['own_RV_nc_name'][1]
#                ex['vars'][0].insert(0, self.name)

        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(self.name, self.startyear, self.endyear, self.grid)))
#        print("Variable function selected {} \n".format(self.filename))

def retrieve_ERA_i_field(cls):
#    from functions_pp import kornshell_with_input
    from ecmwfapi import ECMWFDataServer
    import os
    server = ECMWFDataServer()
    file_path = os.path.join(cls.path_raw, cls.filename)
    file_path_raw = file_path.replace('daily','oper')
    datestring = "/".join(cls.datelist_str)
#    if cls.stream == "mnth" or cls.stream == "oper":
#        time = "00:00:00/06:00:00/12:00:00/18:00:00"
#    elif cls.stream == "moda":
#        time = "00:00:00"
#    else:
#        print("stream is not available")


    if os.path.isfile(path=file_path) == True:
        print("You have already download the variable")
        print(("to path: {} \n ".format(file_path)))
        pass
    else:
        print(("You WILL download variable {} \n stream is set to {} \n".format \
            (cls.name, cls.stream)))
        print(("to path: \n \n {} \n \n".format(file_path_raw)))
        # !/usr/bin/python
        if cls.levtype == 'sfc':
            server.retrieve({
                "dataset"   :   "interim",
                "class"     :   "ei",
                "expver"    :   "1",
                "date"      :   datestring,
                "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
                "levtype"   :   cls.levtype,
                # "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "stream"    :   cls.stream,
                 "time"      :  cls.time_ana,
                "type"      :   "an",
                "format"    :   "netcdf",
                "target"    :   file_path_raw,
                })
        elif cls.levtype == 'pl':
            server.retrieve({
                "dataset"   :   "interim",
                "class"     :   "ei",
                "expver"    :   "1",
                "date"      :   datestring,
                "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
                "levtype"   :   cls.levtype,
                "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "stream"    :   cls.stream,
                 "time"      :  cls.time_ana,
                "type"      :   "an",
                "format"    :   "netcdf",
                "target"    :   file_path_raw,
                })
        print("convert operational 6hrly data to daily means")
        args = ['cdo daymean {} {}'.format(file_path_raw, file_path)]
        kornshell_with_input(args, cls)
    return

def datestr_for_preproc(cls, ex):
    ''' 
    The cdo timselmean that is used in the preprocessing_ncdf() will keep calculating 
    a mean over 10 days and does not care about the date of the years (also logical). 
    This means that after 36 timesteps you have averaged 360 days into steps of 10. 
    The last 5/6 days of the year do not fit the 10 day mean. It will just continuing 
    doing timselmean operations, meaning that the second year the first timestep will 
    be e.g. the first of januari (instead of the fifth of the first year). To ensure 
    the months and days in each year correspond, we need to adjust the dates that were 
    given by ex['sstartdate'] - ex['senddate'].
    '''
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    datesnc = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    leapdays = ((datesnc.is_leap_year) & (datesnc.month==2) & (datesnc.day==29))==False
    datesnc = datesnc[leapdays].dropna(how='all')
#    if len(leapdays) != 0:

# =============================================================================
#   # select dates
# =============================================================================
    # selday_pp is the period you aim to study
    seldays_pp = pd.DatetimeIndex(start=ex['sstartdate'], end=ex['senddate'], 
                                freq=(datesnc[1] - datesnc[0]))
    end_day = seldays_pp.max() 
    # after time averaging over 'tfreq' number of days, you want that each year 
    # consists of the same day. For this to be true, you need to make sure that
    # the selday_pp period exactly fits in a integer multiple of 'tfreq'
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    fit_steps_yr = (end_day - seldays_pp.min())  / temporal_freq
    # line below: The +1 = include day 1 in counting
    start_day = (end_day - (temporal_freq * np.round(fit_steps_yr, decimals=0))) + 1 
    # create datestring that will be used for the cdo selectdate, 
    def make_datestr(dates, start_yr):
        breakyr = dates.year.max()
        datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
        nyears = (dates.year[-1] - dates.year[0])+1
        startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
        endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
        firstyear = startday[:4]
        def plusyearnoleap(curr_yr, startday, endday, incr):
            startday = startday.replace(firstyear, str(curr_yr+incr))
            endday = endday.replace(firstyear, str(curr_yr+incr))
            next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                            freq=(datesnc[1] - datesnc[0]))
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
            return next_yr
        

        for yr in range(0,nyears-1):
            curr_yr = yr+dates.year[0]
            next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
#            print(len(next_yr))
            nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
            datesstr = datesstr + nextstr
#            print(nextstr[0])
            
            upd_start_yr = plusyearnoleap(next_yr.year[0], startday, endday, 1)

            if next_yr.year[0] == breakyr:
                break
            
        return datesstr, upd_start_yr

# =============================================================================
#   # sel_dates string is too long for high # of timesteps, so slicing timeseries
#   # in 2. 
# =============================================================================
    dateofslice = datesnc[int(len(datesnc)/4.)]
    idxsd = np.argwhere(datesnc == dateofslice)[0][0]
    dates1 = datesnc[:idxsd*1]
    dates2 = datesnc[idxsd*1:idxsd*2]
    dates3 = datesnc[idxsd*2:idxsd*3]
    dates4 = datesnc[idxsd*3:]
    start_yr = pd.DatetimeIndex(start=start_day, end=end_day, 
                                freq=(datesnc[1] - datesnc[0]))
    # exluding leap year from cdo select string
    noleapdays = (((start_yr.month==2) & (start_yr.day==29))==False)
    start_yr = start_yr[noleapdays].dropna(how='all')
#    start_yr = start_yr[(datesnc.month==2) & (datesnc.day==29) == False]
    datesstr1, next_yr = make_datestr(dates1, start_yr)
    datesstr2, next_yr = make_datestr(dates2, next_yr)
    datesstr3, next_yr = make_datestr(dates3, next_yr)
    datesstr4, next_yr = make_datestr(dates4, next_yr)
    datesstr = [datesstr1, datesstr2, datesstr3, datesstr4]
#    datelist = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in list(dates)]
#    firsthalfts = convert_list_cdo_string(datelist[:idxsd])
#    seconhalfts = convert_list_cdo_string(datelist[idxsd:])
# =============================================================================
#   # give appropriate name to output file    
# =============================================================================
    outfilename = cls.filename[:-3]+'.nc'
    outfilename = outfilename.replace('daily', 'dt-{}days'.format(ex['tfreq']))
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )

    startdatestr = '_{}{}_'.format(start_day.day, months[start_day.month])
    enddatestr   = '_{}{}_'.format(end_day.day, months[end_day.month])
    outfilename = outfilename.replace('_{}_'.format(1), startdatestr)
    outfilename = outfilename.replace('_{}_'.format(12), enddatestr)
    cls.filename_pp = outfilename
    cls.path_pp = ex['path_pp']
    outfile = os.path.join(ex['path_pp'], outfilename)
    print('output file of pp will be saved as: \n' + outfile + '\n')
    return outfile, datesstr, cls

def preprocessing_ncdf(outfile, datesstr, cls, ex):
    ''' 
    This function does some python manipulation based on your experiment 
    to create the corresponding cdo commands. 
    A kornshell script is created in the folder bash_scripts. First time you
    run it, it will give execution rights error. Open terminal -> go to 
    bash_scrips folder -> type 'chmod 755 bash_script.sh' to give exec rights.
    - Select time period of interest from daily mean time series
    - Do timesel mean based on your ex['tfreq']
    - Make sure the calenders are the same, dates are used to select data by xarray
    - Gridpoint detrending
    - Calculate anomalies (w.r.t. multi year daily means)
    - deletes time bonds from the variables
    - stores new relative time axis converted to numpy.datetime64 format in var_class
    '''
    import os
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    # Final input and output files
    infile = os.path.join(cls.path_raw, cls.filename)
    # convert to inter annual daily mean to make this faster
    tmpfile = os.path.join(cls.path_raw, 'tmpfiles')
    if os.path.isdir(tmpfile) == False : os.makedirs(tmpfile)
    tmpfile = os.path.join(tmpfile, 'tmp')
    # check temporal frequency raw data
    file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    timesteps = int(np.timedelta64(ex['tfreq'], 'D')  / (dates[1] - dates[0]))
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    def cdostr(thelist):
        string = str(thelist)
        string = string.replace('[','').replace(']','').replace(' ' , '')
        return string.replace('"','').replace('\'','')
## =============================================================================
##   # Select days and temporal frequency 
## =============================================================================
#    sel_dates = 'cdo select,date={} {} {}'.format(datesstr, infile, tmpfile)
#    sel_dates = sel_dates.replace(', ',',').replace('\'','').replace('[','').replace(']','')
#    convert_temp_freq = 'cdo timselmean,{} {} {}'.format(timesteps, tmpfile, outfile)

    sel_dates1 = 'cdo -O select,date={} {} {}'.format(cdostr(datesstr[0]), infile, tmpfile+'1.nc')
    sel_dates2 = 'cdo -O select,date={} {} {}'.format(cdostr(datesstr[1]), infile, tmpfile+'2.nc')
    sel_dates3 = 'cdo -O select,date={} {} {}'.format(cdostr(datesstr[2]), infile, tmpfile+'3.nc')
    sel_dates4 = 'cdo -O select,date={} {} {}'.format(cdostr(datesstr[3]), infile, tmpfile+'4.nc')
    concat = 'cdo -O cat {} {} {} {} {}'.format(tmpfile+'1.nc', tmpfile+'2.nc', tmpfile+'3.nc',
                             tmpfile+'4.nc', tmpfile+'sd.nc')
    convert_time_axis = 'cdo -O setreftime,1900-01-01,00:00:00 -setcalendar,gregorian {} {}'.format(
            tmpfile+'sd.nc', tmpfile+'cal.nc')
    convert_temp_freq = 'cdo -O timselmean,{} {} {}'.format(timesteps, tmpfile+'cal.nc', tmpfile+'tf.nc')
    rm_timebnds = 'ncks -O -C -x -v time_bnds {} {}'.format(tmpfile+'tf.nc', tmpfile+'rmtime.nc')
    rm_res_timebnds = 'ncpdq -O {} {}'.format(tmpfile+'rmtime.nc', tmpfile+'rmtime.nc')
# =============================================================================
#    # problem with remapping, ruins the time coordinates
# =============================================================================
#    gridfile = os.path.join(cls.path_raw, 'grids', 'landseamask_{}deg.nc'.format(ex['grid_res']))
#    convert_grid = 'ncks -O --map={} {} {}'.format(gridfile, outfile, outfile)
#    cdo_gridfile = os.path.join(cls.path_raw, 'grids', 'lonlat_{}d_grid.txt'.format(grid_res))
#    convert_grid = 'cdo remapnn,{} {} {}'.format(cdo_gridfile, outfile, outfile)
# =============================================================================
#   # other unused cdo commands
# =============================================================================
#    del_days = 'cdo delete,month=12,11 -delete,month=10,day=31,30,29,28 -delete,month=2,day=29 {} {}'.format(infile, tmpfile)
#    selmon = 'cdo -selmon,{}/{} {} {}'.format(ex['sstartmonth'],ex['sendmonth'], outfile, outfile)#.replace('[','').replace(']','').replace(', ',',')
#    echo_selmon = 'echo '+selmon
#    overwrite_taxis =   'cdo settaxis,{},1month {} {}'.format(starttime.strftime('%Y-%m-%d,%H:%M'), tmpfile, tmpfile)
#    del_leapdays = 'cdo delete,month=2,day=29 {} {}'.format(infile, tmpfile)
# =============================================================================
#  # detrend
# =============================================================================

    # will detrend only annual mean values over years, no seasonality accounted for
    # will subtract a*t + b, leaving only anomaly around the linear trend + intercept
    detrend = 'cdo -b 32 detrend {} {}'.format(tmpfile+'rmtime.nc', #tmpfile+'hom.nc',
                                             tmpfile+'detrend.nc')
#    trend = 'cdo -b 32 trend {} {} {}'.format(tmpfile+'homrm.nc', tmpfile+'intercept.nc',
#                                             tmpfile+'trend.nc')
# =============================================================================
#  # calculate anomalies w.r.t. interannual daily mean
# =============================================================================
#    clim = 'cdo ydaymean {} {}'.format(outfile, tmpfile)
    anom = 'cdo -O -b 32 ydaysub {} -ydayavg {} {}'.format(tmpfile+'detrend.nc', 
                              tmpfile+'detrend.nc', outfile)
# =============================================================================
#   # commands to add some info 
# =============================================================================
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude '][0] 
    var = var.replace(' ', '')

    add_path_raw = 'ncatted -a path_raw,global,c,c,{} {}'.format(str(ncdf.filepath()), outfile) 
    add_units = 'ncatted -a units,global,c,c,{} {}'.format(ncdf.variables[var].units, outfile) 
 
    echo_end = ("echo data is being detrended w.r.t. global mean trend " 
                "and are anomaly versus muli-year daily mean\n")
    ncdf.close()
    # ORDER of commands, --> is important!

#    args = [detrend, anom] # splitting string because of a limit to length
    args = [sel_dates1, sel_dates2]
    kornshell_with_input(args, cls)
    args = [sel_dates3, sel_dates4, concat, convert_time_axis, convert_temp_freq, 
            rm_timebnds, rm_res_timebnds, detrend, anom, add_path_raw, add_units, echo_end] 
    kornshell_with_input(args, cls)
# =============================================================================
#     # update class (more a check if dates are indeed correct)
# =============================================================================
    cls, ex = update_dates(cls, ex)
    return cls, ex

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
    from netCDF4 import Dataset
    from netCDF4 import num2date
    import pandas as pd
    import numpy as np
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    file_path = os.path.join(cls.path_pp, cls.filename_pp)
    ncdf = Dataset(file_path)
    numtime = ncdf.variables['time']
    dates = pd.to_datetime(num2date(numtime[:], units=numtime.units, calendar=numtime.calendar))
    cls.dates = dates
    cls.temporal_freq = '{}days'.format(temporal_freq.astype('timedelta64[D]').astype(int))
    return cls, ex

def perform_post_processing(ex):
    print('\nPerforming the post processing steps on {}'.format(ex['vars'][0]))
    for var in ex['vars'][0]:
        var_class = ex[var]
        outfile, datesstr, var_class = datestr_for_preproc(var_class, ex)
#        var_class, ex = functions_pp.preprocessing_ncdf(outfile, datesstr, var_class, ex)
        if os.path.isfile(outfile) == True: 
            print('looks like you already have done the pre-processing,\n'
                  'to save time let\'s not do it twice..')
            # but we will update the dates stored in var_class:
            var_class, ex = update_dates(var_class, ex)
            pass
        else:    
            var_class, ex = preprocessing_ncdf(outfile, datesstr, var_class, ex)
    
def RV_spatial_temporal_mask(ex, RV, importRVts, months):
    '''Select months of your Response Variable that you want to predict.
    RV = the RV class
    ex = experiment dictionary 
    months = list of integers 
    If you select [6,7] you will attempt to correlate precursor gridcells with 
    lag x versus the response variable values in june and july.
    
    The second step is to insert a spatial mask -only if- you inserted a 3D field 
    as your response variable (time, lats, lons). '''
    
    if importRVts == True:
        print('\nImportRVts is true, so the 1D time serie given with name\n'
              ' {}\n is imported and inserted as the first entry of '
              'ex[\'vars\']'.format(ex['RVts_filename']))
        RV.name = 'RV_imp'
        dicRV = np.load(os.path.join(ex['path_pp'], 'RVts2.5', ex['RVts_filename']), 
                        encoding='latin1').item()
    #    dicRV = pickle.load( open(os.path.join(ex['path_pp'],ex['RVts_filename']+'.pkl'), "rb") ) 
           
        RV.RVfullts = dicRV['RVfullts']
        RV.dates = pd.to_datetime(dicRV['RVfullts'].time.values)
        RV.startyear = RV.dates.year[0]
        RV.endyear = RV.dates.year[-1]
        RV.filename = ex['RVts_filename']
#        ex['vars'][0].insert(0, RV.name)
    
    elif importRVts == False:
        RV.name = ex['vars'][0][0]
        # RV should always be the first variable of the vars list in ex
        RV = ex[RV.name]
        RVarray, RV = import_array(RV)
        print('The RV variable is the 0th index in ex[\'vars\'], '
              'i.e. {}'.format(RV.name))
    
#    one_year = RV.dates.where(RV.dates.year == RV.startyear+1).dropna()
     # Selecting the timesteps of 14 day mean ts that fall in juli and august
    RV_period = []
    for mon in months:
        # append the indices of each year corresponding to your RV period
        RV_period.insert(-1, np.where(RV.dates.month == mon)[0] )
    RV_period = [x for sublist in RV_period for x in sublist]
    RV_period.sort()
    ex['RV_period'] = RV_period
    RV.datesRV = RV.dates[RV_period]
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                    7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    RV_name_range = '{}{}-{}{}_'.format(RV.datesRV[0].day, months[RV.datesRV.month[0]], 
                     RV.datesRV[-1].day, months[RV.datesRV.month[-1]] )
    
    print('\nCreating a folder for the specific spatial mask and RV period')
    if importRVts == True:
        i = len(RV_name_range)
        ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], RV_name_range + 
                                      ex['RVts_filename'][i:])
                                              
    elif importRVts == False:
        ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], RV_name_range + 
                                      ex['maskname'] )
        # =============================================================================
        # 3.2 Select spatial mask to create 1D timeseries (e.g. a SREX region)
        # =============================================================================
        # You can load a spatial mask here and use it to create your
        # full timeseries (of length equal to actor time series)                                                        
        try:
            mask_dic = np.load(ex['path_masks'], encoding='latin1').item()
            RV_array = mask_dic['RV_array']
            xarray_plot(RV_array)
        except IOError as e:
            print('\n\n**\nSpatial mask not found.\n')
        #              'Place your spatial mask in folder: \n{}\n'
        #              'and rerun this section.\n**'.format(ex['path_pp'], 'grids'))
            raise(e)
        RVarray.coords['mask'] = RV_array.mask
        RV.RVfullts = RVarray.where(RVarray.mask==False).mean(dim=['latitude','longitude']).squeeze()
    
    RV.RV_ts = RV.RVfullts[ex['RV_period']] # extract specific months of MT index 
    # Store added information in RV class to the exp dictionary
    ex['RV_name'] = RV.name
    
    return RV, ex, RV_name_range 

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
    dates = pd.to_datetime(dates)
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    marray['time'] = dates
    cls.dates = dates
    return marray, cls

def find_region(data, region='EU'):
    if region == 'EU':
        west_lon = -30; east_lon = 40; south_lat = 35; north_lat = 65

    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50

    region_coords = [west_lon, east_lon, south_lat, north_lat]
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)
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

def xarray_plot(data, path='default', name = 'default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    plt.figure()
    data = np.squeeze(data)
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
    proj = ccrs.Orthographic(central_longitude=cen_lon.values, central_latitude=data.latitude.mean().values)
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

def convert_longitude(data):
    import numpy as np
    import xarray as xr
    lon_above = data.longitude[np.where(data.longitude > 180)[0]]
    lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
    # roll all values to the right for len(lon_above amount of steps)
    data = data.roll(longitude=len(lon_above))
    # adapt longitude values above 180 to negative values
    substract = lambda x, y: (x - y)
    lon_above = xr.apply_ufunc(substract, lon_above, 360)
    convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
    data['longitude'] = convert_lon
    return data

def detrendncdf3D(filename):
    import xarray as xr
    import pandas as pd
    import numpy as np
    import scipy
    filename = tmpfile+'an.nc'
    ncdf = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude '][0] 
    var = var.replace(' ', '')
    numtime = ncdf.variables['time'].values
    timeattr = ncdf.variables['time'].attrs
    dates = pd.to_datetime(num2date(numtime[:], units=timeattr['units'], calendar=timeattr['calendar']))
    stepsyr = dates.where(dates.year == 1980).dropna(how='all')
    marray = np.squeeze(ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')})))
    marray['time'] = dates
    
    def detrendfunc2d(arr_oneday):
#        outputarr = np.zeros((arr_oneday.latitude.size, arr_oneday.longitude.size))
        outputarr = np.zeros(arr_oneday.shape)
        output = xr.DataArray(outputarr, coords= arr_oneday.coords,
                              dims = arr_oneday.dims)
        lats = list(arr_oneday.latitude.values)
        lons = list(arr_oneday.longitude.values)
        slopesnp = np.zeros(arr_oneday[0].shape)
    
        for lat in lats:
            latidx = lats.index(lat)
            arr_onedaylats = arr_oneday.sel(latitude=lat)
            for lon in lons:
                lonidx = lons.index(lon)
                arr_oneday_gc = arr_onedaylats.sel(longitude=lon)
                slope = scipy.stats.linregress(range(arr_oneday.time.size), arr_oneday_gc)[0]
                slopesnp[latidx, lonidx] = slope
                y = slope * range(arr_oneday.time.size)
                output[:,latidx, lonidx] = arr_oneday_gc.values - y
                anom = output - output.mean(dim='time')
        slopes = xr.DataArray(slopesnp, coords= arr_oneday[0].coords,
                              dims = arr_oneday[0].dims)
        return output, anom, slopes
                
                
    for i in range(stepsyr.size):
        sd =(dates[i::stepsyr.size])
        arr_oneday = marray.sel(time=sd)
        output, anom, slopes = detrendfunc2d(arr_oneday)
    
    
    def trendtesting(anom):
        anom = output
        anomp = anom.sel(latitude=40, longitude=0)
#        slope, intercept = np.polyfit(range(39), anomp, deg=1)
        slope, intercept = scipy.stats.linregress(range(39), anomp)[:2]
        def trend(x, a, b):
            y = a*x + b
            return y
        tren = trend(range(39), slope, intercept)
        detrend = anomp - tren + intercept
        plt.figure()
        plt.title('slope is {}, doy is {}'.format(slope, int(da.time.dt.dayofyear[0])))
        plt.plot(tren)
        plt.plot(anomp.values)
        plt.plot(detrend)
    for i in range(0,stepsyr.size, 5):
        slopes = np.zeros(len(range(0,stepsyr.size, 5)))
        sd =(dates[i::stepsyr.size])
        print(sd)
        da = marray.sel(time=sd)
        trendtesting(da)



def detrend1D(da):
    import scipy.signal as sps
    import xarray as xr
    dao = xr.DataArray(sps.detrend(da),
                            dims=da.dims, coords=da.coords)
    return dao

