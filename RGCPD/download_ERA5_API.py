#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:42:46 2019

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd


## =============================================================================
## Data wil downloaded to path_raw
## =============================================================================
#base_path = "/Users/semvijverberg/surfdrive/RGCPD_jetlat/"
#path_raw = os.path.join("/Users/semvijverberg/surfdrive/Data_ERAint/", 
#                        'input_raw')
#path_pp  = os.path.join("/Users/semvijverberg/surfdrive/Data_ERAint/", 
#                        'input_pp')
#if os.path.isdir(path_raw) == False : os.makedirs(path_raw)
#if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
#
## *****************************************************************************
## Step 1 Create dictionary and variable class (and optionally download ncdfs)
## *****************************************************************************
## The dictionary is used as a container with all information for the experiment
## The dic is saved after the post-processes step, so you can continue the experiment
## from this point onward with different configurations. It also stored as a log
## in the final output.
##
#ex = dict(
#     {'dataset'     :       'era5',
#     'grid_res'     :       2.5,
#     'startyear'    :       1979, # download startyear
#     'endyear'      :       2018, # download endyear
#     'months'       :       list(range(1,12+1)), #downoad months
#     'time'         :       pd.DatetimeIndex(start='00:00', end='23:00', 
#                                freq=(pd.Timedelta(6, unit='h'))),
#     'base_path'    :       base_path,
#     'path_raw'     :       path_raw,
#     'path_pp'      :        path_pp}
#     )
#
#    
#
## Option 1111111111111111111111111111111111111111111111111111111111111111111111
## Download ncdf fields (in ex['vars']) through cds?
## 11111111111111111111111111111111111111111111111111111111111111111111111111111
## only reanalysis fields
#
## Info to download ncdf from ECMWF, atm only analytical fields (no forecasts)
## You need the cds-api-client package for this option.
#
## See https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
#
#ex['vars']     =   [
#                    ['t2m', 'u'],              # ['name_var1','name_var2', ...]
#                    ['167.128', '131.128'],    # ECMWF param ids
#                    ['sfc', 'pl'],             # Levtypes
#                    [0, 200],                  # Vertical levels
#                    ]
#
## assign first variables class
#var_class = Var_ECMWF_download(ex, 0) 
#retrieve_ERA5(var_class)



def Variable(self, ex):
    self.startyear = ex['startyear']
    self.endyear = ex['endyear']
    self.months = ex['months']
    self.startmonth = min(ex['months'])
    self.endmonth   = max(ex['months'])
    self.grid = ['{}'.format(ex['grid_res']),'{}'.format(ex['grid_res'])]
    self.dataset = ex['dataset']
    self.path_raw = ex['path_raw']
    self.path_pp = ex['path_pp']
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
        vclass.stream = 'oper'
        vclass.years    = [str(yr) for yr in np.arange(ex['startyear'], 
                                               ex['endyear']+1E-9, dtype=int)]
        vclass.months   = [str(yr) for yr in ex['months']] 
        vclass.days     = [str(yr) for yr in np.arange(1, 31+1E-9, dtype=int)] 
        

        vclass.time = list(ex['time'].strftime('%H:%M'))
        vclass.filename = '{}_{}-{}_{}_{}_{}_{}deg'.format(vclass.name, 
                           vclass.startyear, vclass.endyear, vclass.startmonth, 
                           vclass.endmonth, 'daily', ex['grid_res']).replace(' ', '_')
        vclass.format = '{}'.format('netcdf')
        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name, 
               vclass.startyear, vclass.endyear, vclass.grid)))
        
        
            
def retrieve_field(cls):

    import cdsapi
    import os
    server = cdsapi.Client()
    
    file_path = os.path.join(cls.path_raw, cls.filename)
    file_path_raw = file_path.replace('daily','oper')
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
            server.retrieve("reanalysis-era5-single-levels",
                {
                "producttype":  "reanalysis",
                "class"     :   "ei",
                "expver"    :   "1",
                "grid"      :   cls.grid,
                "year"      :   cls.years,
                "month"     :   cls.months,
                "day"       :   cls.days,
#                "levtype"   :   cls.levtype,
                # "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                "time"      :  cls.time,
                "format"    :   "netcdf",
#                "target"    :   ,
                }, 
                file_path_raw)
        elif cls.levtype == 'pl':
            server.retrieve("reanalysis-era5-pressure-levels",
                {
                "producttype":  "reanalysis",
                "class"     :   "ei",
                "expver"    :   "1",
                "grid"      :   cls.grid,
                "year"      :   cls.years,
                "month"     :   cls.months,
                "day"       :   cls.days,
#                "levtype"   :   cls.levtype,
                "levelist"  :   cls.lvllist,
                "param"     :   cls.var_cf_code,
                 "time"      :  cls.time,
                "format"    :   "netcdf",
                }, 
                file_path_raw)
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