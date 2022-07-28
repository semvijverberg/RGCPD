#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:42:46 2019

@author: semvijverberg
"""
import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator


if sys.platform == 'linux': # on cluster
    try:
        os.system("module load cdo")
    except:
        print('Ensure CDO is loaded')

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
max_cpu = multiprocessing.cpu_count()
n_threads = min(10,2*max_cpu)

accumulated_vars = ['total_precipitation', 'potential_evaporation', 'Runoff']

# Common ERA5 variable names
# v_component_of_wind
# u_component_of_wind
# geopotential
mondict  =  {1: 'January',
             2: 'February',
             3: 'March',
             4: 'April',
             5: 'May',
             6: 'June',
             7: 'July',
             8: 'August',
             9: 'September',
             10: 'October',
             11: 'November',
             12: 'December'}

def Variable(self, ex):
    self.startyear = ex['startyear']
    self.endyear = ex['endyear']
    self.months = ex['months']
    self.startmonth = min(self.months)
    self.endmonth   = max(self.months)
    self.grid = ['{}'.format(ex['grid_res']),'{}'.format(ex['grid_res'])]
    self.dataset = ex['dataset']
    self.path_raw = ex['path_raw']
    self.base_path = ex['base_path']
    if 'CDO_command' in ex.keys():
        self.CDO_command = ex['CDO_command']
    return self

class Var_ECMWF_download():
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper
    Forecasts of accumulations  :   enda
    """


    def __init__(self, ex, idx):

#        from datetime import datetime, timedelta
#        import pandas as pd
#        import calendar
#        import os
        vclass = Variable(self, ex)
        # shared information of ECMWF downloaded variables
        # variables specific information
        if ex['vars'][2][idx] == 'sfc':
            vclass.dataset = '{}'.format('reanalysis-era5-single-levels')
        vclass.name = ex['vars'][0][idx]


        vclass.var_cf_code = ex['vars'][1][idx]
        vclass.levtype = ex['vars'][2][idx]
        vclass.lvllist = ex['vars'][3][idx]
        if 'area' in ex.keys():
            vclass.area    = ex['area']
            if vclass.area == 'global':
                vclass.area    = [90, -180, -90, 180] # global
        else:
            vclass.area    = [90, -180, -90, 180] # global
        vclass.stream = ex['stream']
        if vclass.stream == 'oper' or vclass.stream == 'enda':
            vclass.input_freq = 'daily'
        if vclass.stream == 'moda' or vclass.stream == 'mnth':
            vclass.input_freq = 'monthly'

        vclass.years    = [str(yr) for yr in np.arange(ex['startyear'],
                                               ex['endyear']+1E-9, dtype=int)]
        vclass.months   = [str(yr) for yr in ex['months']]
        vclass.days     = [str(yr) for yr in np.arange(1, 31+1E-9, dtype=int)]
        mon_as_int = (type(vclass.months[0]) is int or vclass.months[0].isnumeric())
        if mon_as_int and vclass.startyear < 1958:
            # for back extension ERA5, months should be written as text
            vclass.months = [mondict[int(monnum)] for monnum in vclass.months]



        vclass.time = list(ex['time'].strftime('%H:%M'))


        vclass.filename = '{}_{}-{}_{}_{}_{}_{}deg.nc'.format(vclass.name,
                           vclass.startyear, vclass.endyear, vclass.startmonth,
                           vclass.endmonth, vclass.input_freq, ex['grid_res']).replace(' ', '_')
        vclass.format = '{}'.format('netcdf')
        if idx == 0:
            print('(Down)loading precursors')
        print(('\n\t**\n\t{} {}-{} {} data on {} grid\n\t**\n'.format(vclass.name,
               vclass.startyear, vclass.endyear, vclass.input_freq, vclass.grid)))



def retrieve_field(cls):

    #%%
    file_path = os.path.join(cls.path_raw, cls.filename)
    if cls.stream == 'moda':
        file_path_raw = file_path
    else:
        file_path_raw = file_path.replace('daily',cls.stream)

    if os.path.isfile(path=file_path) == False:
        # create temporary folder
        cls.tmp_folder = os.path.join(cls.path_raw,
                  '{}_{}_{}_tmp'.format(cls.name, cls.stream, cls.grid[0]))
        if os.path.isdir(cls.tmp_folder) == False : os.makedirs(cls.tmp_folder)
        print(("You will download variable {} \n stream is set to {} \n".format \
            (cls.name, cls.stream)))
        if cls.stream == 'enda':
            # you will want data every 3 hours
            print('\nBecause stream is enda, time forced to 3hr interval')
            np_datetime = pd.date_range(start='00:00', end='23:00',
                                freq=(pd.Timedelta(3, unit='h')))
            cls.time = list(np_datetime.strftime('%H:%M'))
        print('Time: {}'.format(cls.time))
        print(("to path: \n \n {} \n \n".format(file_path_raw)))

        # =============================================================================
        #         daily data
        # =============================================================================
        if cls.stream == 'oper' or cls.stream == 'enda':
            download_targets = []
            for year in cls.years:
                # specifies the output file name
                target = os.path.join(cls.tmp_folder,
                          '{}_{}.nc'.format(cls.name, year))
                if os.path.isfile(target) == False:
                    print('Output file: ', target)
                    download_targets.append((year, target))

            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = [pool.submit(retrieval_yr, cls.var_cf_code, cls.time,
                           cls.months, cls.days,
                           cls.grid, cls.area, cls.lvllist,
                           cls.levtype, year, target) for year, target in download_targets]




            if cls.var_cf_code in accumulated_vars:
                acc_to_daysum = 'cdo -b 32 settime,00:00 -daysum -shifttime,-1hour -mergetime {}/*.nc {}'.format(cls.tmp_folder, file_path)

                args = [acc_to_daysum]

            else:
                print("convert operational oper data to daily means")
                ana_to_day = 'cdo -b 32 settime,00:00 -{} -mergetime {}{}*.nc {}'.format(
                                        cls.CDO_command,
                                        cls.tmp_folder,
                                        sep,
                                        file_path)
                args = [ana_to_day]

            kornshell_with_input(args, cls)

        # =============================================================================
        # monthly mean of daily means
        # =============================================================================
        if cls.stream == 'moda' or cls.stream == 'mnth':
            years = [int(yr) for yr in cls.years]
            decades = list(set([divmod(i, 10)[0] for i in years]))
            decades = [x * 10 for x in decades]
            decades.sort()
            print('Decades:', decades)

        # loop through decades and create a month list
            download_targets = []
            for d in decades:
                yr_in_d = []
                for y in years:
                    if ((divmod(y,10)[0])*10) == d:
                        yr_in_d.append(str(y))

                #             requestDates = requestDates+str(y)+m.zfill(2)+'01/'
                # requestDates = requestDates[:-1]
                print('Requesting dates: ', years, cls.months)
                target = os.path.join(cls.tmp_folder, '{}_{}s.nc'.format(cls.name,
                              d))
                if os.path.isfile(target) == False:
                    print('Output file: ', target)
                    download_targets.append((yr_in_d, d, target))

            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = [pool.submit(retrieval_moda, cls.var_cf_code, yr_in_d,
                                       cls.months, cls.stream,
                                       cls.grid, cls.area, cls.lvllist, cls.levtype,
                                       d, target) for yr_in_d, d, target in download_targets]
            mergetime = 'cdo -b 32 cat {}{}*.nc {}'.format(cls.tmp_folder, sep, file_path)
            kornshell_with_input([mergetime], cls)
#            retrieval_moda(cls, requestDates, d, target)
    #%%
    return

def retrieval_yr(var_cf_code, time, months, days, grid, area, lvllist,
                 levtype, year, target):
    import cdsapi
    server = cdsapi.Client()

    print('variable: {}'.format(var_cf_code))
    print(year)
    print('months: {}'.format(months))
    print('days {}'.format(days))


    # !/usr/bin/python
    if levtype == 'sfc':
        server.retrieve("reanalysis-era5-single-levels",
            {
            "product_type":  "reanalysis",
            "class"     :   "ei",
            "expver"    :   "1",
            "grid"      :   grid,
            "year"      :   year,
            "month"     :   months,
            "day"       :   days,
            'area'      :   area,
#                "levtype"   :   levtype,
            # "levelist"  :   lvllist,
            "variable"  :   var_cf_code,
            "time"      :  time,
            "format"    :   "netcdf",
            },
            target)
    elif levtype == 'pl':
        server.retrieve("reanalysis-era5-pressure-levels",
            {
            "product_type":  "reanalysis",
            "class"     :   "ei",
            "expver"    :   "1",
            "grid"      :   grid,
            "year"      :   year,
            "month"     :   months,
            "day"       :   days,
            'area'      :   area,
            "levelist"  :   lvllist,
            "variable"  :   var_cf_code,
             "time"     :  time,
            "format"    :   "netcdf",
            },
            target)
    return year

def retrieval_moda(var_cf_code, years, months, stream, grid, area, lvllist,
                 levtype, decade, target):
    import cdsapi
    server = cdsapi.Client()
    server_name = 'reanalysis-era5.1-complete'
    if levtype == 'sfc':
        if int(years[0]) < 1979:
            print('downloading from preliminary back extension')
        #     server_name  = 'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension'
        server.retrieve(server_name, {    # do not change this!
        'class'         :   'ea',
        'expver'        :   '1',
        'stream'        :   stream,
        'type'          :   'an',
        "grid"          :   grid,
        'param'         :   var_cf_code,
        'area'          :   area,
        'levtype'       :   levtype,
        'year'          :   years,
        'month'         :   months,
        'time'          :   '00:00',
        'decade'        :   decade,
        "format"        :   "netcdf",
        }, target)
    elif levtype == 'pl':
        if int(years[0]) < 1979: #first date prior to 1979
            server_name  = 'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension'
        server.retrieve('reanalysis-era5.1-complete', {    # do not change this!
        'class'         :   'ea',
        'expver'        :   '1',
        'stream'        :   stream,
        'type'          :   'an',
        "grid"          :   grid,
        'param'         :   var_cf_code,
        'area'          :   area,
        'levtype'       :   levtype,
        'levelist'      :   lvllist,
        'year'          :   years,
        'month'         :   months,
        'time'          :   '00:00',
        'decade'        :   decade,
        "format"        :   "netcdf",
        }, target)


def kornshell_with_input(args, cls):
    '''some kornshell with input '''
    import subprocess
    # cwd = os.getcwd()
    # Writing the bash script:
    new_bash_script = './bash_scripts/bash_script.sh'

    bash_and_args = [new_bash_script]
    [bash_and_args.append(arg) for arg in args]
    with open(new_bash_script, "w") as file:
        file.write("#!/bin/sh\n")
        file.write("echo bash script output\n")
        for cmd in range(len(args)):

            print(args[cmd].replace(cls.base_path, 'base_path')[:300])
            file.write("${}\n".format(cmd+1))
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    out = p.communicate()
    print(out[0].decode())
    return

if __name__ == '__main__':
    print('do nothing')
#ex = dict(
#     {'dataset'     :       'ERA-5',
#     'grid_res'     :       0.5,
#     'startyear'    :       1979, # download startyear
#     'endyear'      :       2018, # download endyear
#     'firstmonth'   :       1,
#     'lastmonth'    :       12,
#     'time'         :       pd.DatetimeIndex(start='00:00', end='23:00',
#                                freq=(pd.Timedelta(6, unit='h'))),
#     'format'       :       'netcdf',
#     'base_path'    :       base_path,
#     'path_raw'     :       path_raw,
#     'path_pp'     :        path_pp}
#     )
#
#ex['vars'] = [['2m_temperature'],['34.128'],['sfc'],['0']]
#
#vclass = Var_ECMWF_download(ex, idx=0)
#
#
#dict_retrieval =  ('\'{}\',\n\t{{\n'.format(vclass.dataset)+
#                     '\t\'variable\'         :   \'{}\',\n'.format(vclass.name)+
#                     '\t\'product_type\'     :   \'reanalysis\',\n'
#                     '\t\'year\'         :     {},\n'.format(vclass.years)+
#                     '\t\'month\'        :     {},\n'.format(vclass.months)+
#                     '\t\'day\'          :     {},\n'.format(vclass.days)+
#                     '\t\'grid\'         :     {},\n'.format(vclass.grid)+
#                     '\t\'time\'         :     {},\n'.format(vclass.time)+
#                     '\t\'format\'       :     \'{}\',\n'.format(vclass.format)+
#                     '\t}'
#                     )
#print(dict_retrieval)
