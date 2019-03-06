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