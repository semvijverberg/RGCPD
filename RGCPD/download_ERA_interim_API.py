#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:27:05 2019

@author: semvijverberg
"""

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
        import calendar
#        import os
        vclass = Variable(self, ex)
        # shared information of ECMWF downloaded variables
        # variables specific information
        vclass.dataset = ex['dataset']
        vclass.name = ex['vars'][0][idx]
        vclass.var_cf_code = ex['vars'][1][idx]
        vclass.levtype = ex['vars'][2][idx]
        vclass.lvllist = ex['vars'][3][idx]
        vclass.stream = 'oper'
        
        if vclass.name != 'pr':
            time_ana = "00:00:00/06:00:00/12:00:00/18:00:00"
            vclass.type = 'an'
            vclass.step = "0"
        else:
            time_ana = "00:00:00/12:00:00"
            vclass.type = 'fc'
            vclass.step = "3/6/9/12"
    #            else:
    #                time_ana = "00:00:00"
        vclass.time_ana = time_ana 
        
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
                           vclass.endmonth, 'daily', vclass.grid).replace(' ', '_')
        
        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name, 
               vclass.startyear, vclass.endyear, vclass.grid)))

def retrieve_field(cls):

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
                "dataset"   :   cls.dataset,
                "class"     :   "ei",
                "expver"    :   "1",
                "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
                "date"      :   datestring,
                "levtype"   :   cls.levtype,
                # "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "stream"    :   cls.stream,
                "time"      :  cls.time_ana,
                "type"      :   cls.type,
                "step"      :   cls.step,
                "format"    :   "netcdf",
                "target"    :   file_path_raw,
                })
        elif cls.levtype == 'pl':
            server.retrieve({
                "dataset"   :   cls.dataset,
                "class"     :   "ei",
                "expver"    :   "1",
                "date"      :   datestring,
                "grid"      :   '{}/{}'.format(cls.grid,cls.grid),
                "levtype"   :   cls.levtype,
                "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "stream"    :   cls.stream,
                 "time"      :  cls.time_ana,
                "type"      :   cls.type,
                "format"    :   "netcdf",
                "target"    :   file_path_raw,
                })
        print("convert operational 6hrly data to daily means")
        args = ['cdo daymean {} {}'.format(file_path_raw, file_path)]
        kornshell_with_input(args, cls)
    return

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