#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:27:05 2019

@author: semvijverberg
"""
import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator


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
#    def __init__(self, ex):
#    # self is the instance of the employee class
#    # below are listed the instance variables
#        self.startyear = ex['startyear']
#        self.endyear = ex['endyear']
#        self.startmonth = 1
#        self.endmonth = 12
#        self.grid = ex['grid_res']
#        self.dataset = ex['dataset']
#        self.base_path = ex['base_path']
#        self.path_raw = ex['path_raw']
#        self.path_pp = ex['path_pp']


class Var_ECMWF_download():
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper
    """

    def __init__(self, ex, idx):
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np
        import calendar
#        import os
        vclass = Variable(self, ex)
        # shared information of ECMWF downloaded variables
        # variables specific information
        if ex['dataset'] == 'ERAint':
            vclass.dataset = 'interim'
            vclass.dclass   = 'ei'
        elif ex['dataset'] == 'era20c':
            vclass.dataset = 'era20c'
            vclass.dclass   = 'e2'
        vclass.name = ex['vars'][0][idx]
        vclass.var_cf_code = ex['vars'][1][idx]
        vclass.levtype = ex['vars'][2][idx]
        vclass.lvllist = ex['vars'][3][idx]
        if ex['input_freq'] == 'daily':
            vclass.stream = 'oper'
            vclass.input_freq = 'daily'
        if ex['input_freq'] == 'monthly':
            vclass.stream = 'moda'
            vclass.input_freq = 'monthly'

        if vclass.stream == 'oper' and vclass.name != 'pr':

            vclass.time_ana = "00:00:00/06:00:00/12:00:00/18:00:00"
            vclass.type = 'an'
            vclass.step = "0"
        elif vclass.stream == 'oper' and vclass.name == 'pr':
            vclass.time_ana = "00:00:00/12:00:00"
            vclass.type = 'fc'
            vclass.step = "3/6/9/12"


        vclass.years    = [str(yr) for yr in np.arange(ex['startyear'],
                                               ex['endyear']+1E-9, dtype=int)]
        vclass.months   = [str(yr) for yr in ex['months']]
        vclass.days     = [str(yr) for yr in np.arange(1, 31+1E-9, dtype=int)]


        days_in_month = dict( {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31,
                               9:30, 10:31, 11:30, 12:31} )
        days_in_month_leap = dict( {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31,
                                    8:31, 9:30, 10:31, 11:30, 12:31} )
        start = datetime(vclass.startyear, vclass.startmonth, 1)

        # creating list of dates that we want to download given the startyear/
        # startmonth to endyear/endmonth
        datelist_str = [start.strftime('%Y-%m-%d')]
        if vclass.stream == 'oper':
            end = datetime(vclass.endyear, vclass.endmonth,
                                    days_in_month[vclass.endmonth])
            while start < end:
                start += timedelta(days=1)
                datelist_str.append(start.strftime('%Y-%m-%d'))
                if start.month == end.month and start.day == days_in_month[vclass.endmonth] and start.year != vclass.endyear:
                    start = datetime(start.year+1, vclass.startmonth, 1)
                    datelist_str.append(start.strftime('%Y-%m-%d'))
        elif vclass.stream == 'moda' or 'mnth':
            end = datetime(vclass.endyear, vclass.endmonth, 1)
            while start < end:
                days = days_in_month[start.month] if calendar.isleap(start.year)==False else days_in_month_leap[start.month]
                start += timedelta(days=days)
                datelist_str.append(start.strftime('%Y-%m-%d'))
                if start.month == end.month and start.year != vclass.endyear:
                    start = datetime(start.year+1, vclass.startmonth, 1)
                    datelist_str.append(start.strftime('%Y-%m-%d'))
        vclass.datelist_str = datelist_str
        # Convert to datetime datelist
        vclass.dates = pd.to_datetime(datelist_str)
        vclass.filename = '{}_{}-{}_{}_{}_{}_{}deg.nc'.format(vclass.name,
                           vclass.startyear, vclass.endyear, vclass.startmonth,
                           vclass.endmonth, ex['input_freq'], vclass.grid).replace(' ', '_')

        print(('\n\t**\n\t{} {}-{} {} data on {} grid\n\t**\n'.format(vclass.name,
               vclass.startyear, vclass.endyear, ex['input_freq'], vclass.grid)))

def retrieve_field(cls):

    import os

    downloaded, cls = check_downloaded(cls)

    file_path = os.path.join(cls.path_raw, cls.filename)

    if cls.stream == 'moda':
        file_path_raw = file_path
    else:
        file_path_raw = file_path.replace('daily','oper')


    if downloaded == True:
        print("You have already download the variable")
        print(("to path: {} \n ".format(file_path)))
        pass
    else:
        # create temporary folder
        cls.tmp_folder = os.path.join(cls.path_raw,
                  '{}_{}_{}_tmp'.format(cls.name, cls.stream, cls.grid))
        if os.path.isdir(cls.tmp_folder) == False : os.makedirs(cls.tmp_folder)
        print(("You WILL download variable {} \n stream is set to {} \n".format \
            (cls.name, cls.stream)))
        print(("to path: \n \n {} \n \n".format(file_path_raw)))
        # =============================================================================
        #         daily data
        # =============================================================================
        if cls.stream == 'oper':

            for year in cls.years:
                # specifies the output file name
                target = os.path.join(cls.tmp_folder,
                          '{}_{}.nc'.format(cls.name, year))
                if os.path.isfile(target) == False:
                    print('Output file: ', target)
                    retrieval_yr(cls, year, target)


            print("convert operational 6hrly data to daily means")
            cat  = 'cdo -O -b F64 mergetime {}{}*.nc {}'.format(cls.tmp_folder, sep, file_path_raw)
            daymean = 'cdo -b 32 daymean {} {}'.format(file_path_raw, file_path)
            args = [cat, daymean]
            kornshell_with_input(args, cls)


        # =============================================================================
        # monthly mean of daily means
        # =============================================================================
        if cls.stream == 'moda':
            years = [int(yr) for yr in cls.years]
            decades = list(set([divmod(i, 10)[0] for i in years]))
            decades = [x * 10 for x in decades]
            decades.sort()
            print('Decades:', decades)

        # loop through decades and create a month list
            for d in decades:
                requestDates=''
                for y in years:
                    if ((divmod(y,10)[0])*10) == d:
                        for m in cls.months:
                            requestDates = requestDates+str(y)+m.zfill(2)+'01/'
                requestDates = requestDates[:-1]
                print('Requesting dates: ', requestDates)
                target = os.path.join(cls.tmp_folder, '{}_{}.nc'.format(cls.name,
                              year))
                if os.path.isfile(target):
                    print('Output file: ', target)
                    retrieval_moda(cls, requestDates, d, target)


            cat  = 'cdo cat {}*.nc {}'.format(cls.tmp_folder, file_path_raw)
            args = [cat]
            kornshell_with_input(args, cls)

    return cls


def retrieval_yr(cls, year, target):
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()



    if cls.levtype == 'sfc':
        server.retrieve({
            "dataset"   :   cls.dataset,
            "class"     :   cls.dclass,
            "expver"    :   "1",
            "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
            "date"      :   '{}-01-01/TO/{}-12-31'.format(year, year),
            "levtype"   :   cls.levtype,
            "param"     :   cls.var_cf_code,
            "stream"    :   cls.stream,
            "time"      :  cls.time_ana,
            "type"      :   cls.type,
            "step"      :   cls.step,
            "format"    :   "netcdf",
            "target"    :   target
            })
    elif cls.levtype == 'pl':
        server.retrieve({
            "dataset"   :   cls.dataset,
            "class"     :   cls.dclass,
            "expver"    :   "1",
            "date"      :   '{}-01-01/TO/{}-12-31'.format(year, year),
            "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
            "levtype"   :   cls.levtype,
            "levelist"  :   cls.lvllist,
            "param"     :   cls.var_cf_code,
            "stream"    :   cls.stream,
            "time"      :   cls.time_ana,
            "type"      :   cls.type,
            "format"    :   "netcdf",
            "target"    :   target
            })


def retrieval_moda(cls, requestDates, decade, target):
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()


    if cls.levtype == 'sfc':
        server.retrieve({
        "dataset"       :   cls.dataset,
        "class"         :   cls.dclass,
        'expver'        :   '1',
        'stream'        :   'moda',
        'type'          :   'an',
        'param'         :   cls.var_cf_code,
        'levtype'       :   cls.levtype,
        'date'          :   requestDates,
        'decade'        :   decade,
        'target'        :   target
            })
    elif cls.levtype == 'pl':
        server.retrieve({
        "dataset"       :   cls.dataset,
        "class"         :   cls.dclass,
        'stream'        :   'moda',
        'type'          :   'an',
        'param'         :   cls.var_cf_code,
        'levtype'       :   cls.levtype,
        'levelist'      :   cls.lvllist,
        'date'          :   requestDates,
        'decade'        :   decade,
        'target'        :   target
            })
    return



def check_downloaded(cls):
    import os, re
    # assume not downloaded
    downloaded = False
    rootdir = cls.path_raw
    rx = re.compile(r'^{}_(\d\d\d\d)-(.*?)$'.format(cls.name))
    re.search(rx, cls.filename).groups()
    match = []

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            res = re.search(rx, file)
            if res:
                match.append([res.string, res.groups()])

    if len(match) != 0:
        # check if right gridres
        for file in match:
            file_path = file[0]
            reggroups = file[1]
            grid_res = float(file_path.split('_')[-1].split('deg')[0]) == cls.grid
            startyr  = cls.startyear >= int(reggroups[0])
            endyr    = cls.endyear <= int(reggroups[1][:4])
            input_freq = file_path.split('_')[-2] == cls.input_freq
            if grid_res and startyr and endyr and input_freq:
                downloaded = True
                # adapt filename that exists
                new = file_path.replace(str(cls.startyear), reggroups[0])
                new = new.replace(str(cls.endyear), reggroups[1][:4])
                cls.filename = new

    return downloaded, cls

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

            print(args[cmd].replace(cls.base_path, 'base_path')[:300])
            file.write("${}\n".format(cmd+1))
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    out = p.communicate()
    print(out[0].decode())
    return
