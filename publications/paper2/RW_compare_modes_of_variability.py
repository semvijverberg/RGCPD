#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:44:42 2020

@author: semvijverberg
"""
import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import pandas as pd

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
data_dir = os.path.join(main_dir,'publications/paper2/data')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
from RGCPD import EOF
import class_BivariateMI
import climate_indices
import plot_maps, core_pp, df_ana, functions_pp




west_east = 'combine'
adapt_selbox = False
TV = 'USCAnew'

if west_east == 'east':
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/east/')
elif west_east == 'west':
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')

if TV == 'init':
    TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    if west_east == 'east':
        cluster_label = 2
    elif west_east == 'west':
        cluster_label = 1
elif TV == 'USCAnew':
    # mx2t 25N-70N
    TVpath = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tfreq15_nc7_dendo_57db0USCA.nc'
    # TVpath = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf15_nc8_dendo_57db0USCA.nc'
    if west_east == 'east':
        cluster_label = 4
    elif west_east == 'west':
        cluster_label = 1 # 8
    elif west_east == 'northwest':
        cluster_label = 7
elif TV == 'USCA':
    TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc5_dendo_5dbee_USCA.nc'
    # TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc8_dendo_5dbee_USCA.nc'
    if west_east == 'east':
        cluster_label = 1
    elif west_east == 'west':
        cluster_label = 5
        cluster_label = 4



name_ds='ts'
period = 'summer_center'
if period == 'summer_center':
    start_end_TVdate = ('06-01', '08-31')
start_end_date = None
method='ranstrat_10' ; seed = 1
tfreq = 15
min_detect_gc=.9



#%% Circulation vs temperature


list_of_name_path = [(cluster_label, TVpath),
                      ('v300', os.path.join(path_raw, 'v300_1979-2020_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500_1979-2020_1_12_daily_2.5deg.nc'))]


list_import_ts = None #[('W-RW', os.path.join(data_dir, f'westRW_{period}_s{seed}.h5')),
                   # ('E-RW', os.path.join(data_dir, f'eastRW_{period}_s{seed}.h5'))]

list_for_MI   = None # [BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                                # alpha=.05, FDR_control=True,
                                # distance_eps=600, min_area_in_degrees2=1,
                                # calc_ts='pattern cov', selbox=z500_green_bb,
                                # use_sign_pattern=True,
                                # lags=np.array([0]))]

# list_for_EOFS = [EOF(name='u', neofs=1, selbox=[120, 255, 0, 87.5],
#                      n_cpu=1, start_end_date=('01-12', '02-28'))]

list_for_EOFS = [EOF(name='v300', neofs=1, selbox=[-180, 360, 20, 90],
                    n_cpu=1, start_end_date=start_end_TVdate)]

rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            list_for_EOFS=list_for_EOFS,
            list_import_ts=list_import_ts,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=True)

rg.pp_precursors()

rg.traintest(method, seed=seed, subfoldername='comparison_indices_and_modes')

# rg.calc_corr_maps()
# rg.cluster_list_MI()

rg.get_EOFs()


v300EOF = rg.list_for_EOFS[0].eofs.copy()
z500EOFcl = EOF(name='z500', neofs=1,
              n_cpu=1, start_end_date=start_end_TVdate)
rg.list_for_EOFS.append(z500EOFcl)

# get original (winter) PNA
PNA = climate_indices.PNA_z500(rg.list_precur_pp[1][1])
PNA = PNA.rename({'PNA':'PNAliu'},axis=1)

# From Climate Explorer
# https://climexp.knmi.nl/getindices.cgi?WMO=NCEPData/cpc_pna_daily&STATION=PNA&TYPE=i&id=someone@somewhere&NPERYEAR=366
# on 20-07-2020
PNA_cpc = core_pp.import_ds_lazy(main_dir+'/publications/paper2/data/icpc_pna_daily_1980-2020.nc',
                                 start_end_year=(1979, 2020),
                                 seldates=start_end_TVdate).to_dataframe('PNAcpc')
PNA_cpc.index.name = None
#%%
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

matplotlib.rcParams['axes.labelweight']=100





namev300 = 'v-wind 300 hPa'
namez500 = 'z 500 hPa'
for west_east in ['west', 'east', 'combine']:
        # z500_green_bb = (140,260,20,73) #: Pacific box
    if west_east == 'east':
        z500_green_bb = (155,300,20,73) #: RW box
        v300_green_bb = z500_green_bb #(170,359,23,73)
    elif west_east == 'west':
        z500_green_bb = (145,325,20,62)
        v300_green_bb = z500_green_bb #(100,330,24,70)
    elif west_east == 'combine':
        z500_green_bb = (145,325,20,70)
        v300_green_bb = z500_green_bb
    z500EOFcl = rg.list_for_EOFS[1]
    z500EOFcl.selbox = z500_green_bb
    z500EOFcl.get_pattern(rg.list_precur_pp[1][1], df_splits=rg.df_splits)
    z500EOF = z500EOFcl.eofs
    kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':3.8, 'size':2.5,
                  'hspace':0.0, 'cbar_vert':-.02, 'units':'Corr. Coeff. [-]',
                  'drawbox':[(0,0), z500_green_bb], #'zoomregion':(-180,360,10,80),
                  'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6}

    for eof, name, selbox in [(z500EOF, namez500, z500_green_bb), (v300EOF, namev300, v300_green_bb)]:
        eofs = eof.mean(dim='split') # mean over training sets
        eofs[0] *= -1 # switch sign
        subtitles = np.array([[f'{name} - 1st EOF loading pattern']])
        kwrgs_plotEOF = kwrgs_plot.copy()
        kwrgs_plotEOF.update({'clim':None, 'units':None, 'subtitles':subtitles,
                              'y_ticks':np.array([10,30,50,70,90])})
        if name=='v-wind 300 hPa':
            kwrgs_plotEOF.update({'drawbox':None, 'clevels':np.arange(-3.5,3.6,.1),
                                  'clabels':np.arange(-3.5,3.6,1)})
        else:
            kwrgs_plotEOF.update({'clevels':np.arange(-350,360,25),
                                  'clabels':np.arange(-350,360,100)})
        plot_maps.plot_corr_maps(eofs[0] , **kwrgs_plotEOF)
        plt.savefig(os.path.join(rg.path_outsub1,
                                 eofs.eof.values[0].split('..')[-1]+'1.pdf'),
                    bbox_inches='tight')

        # subtitles = f'{name} hPa - 2nd EOF loading pattern'
        # kwrgs_plotEOF.update({'title':subtitles})
        # plot_maps.plot_corr_maps(eofs[1] , **kwrgs_plotEOF)
        # plt.savefig(os.path.join(rg.path_outsub1,
        #                          eofs.eof.values[0].split('..')[-1]+'2.pdf'),
        #             bbox_inches='tight')
        if adapt_selbox:
            # spatial correlation eof and v300 in between v300 green bb
            rg.list_for_EOFS[0].eofs = core_pp.get_selbox(v300EOF, selbox=v300_green_bb)
            rg.list_for_EOFS[0].selbox = v300_green_bb

    rg.get_ts_prec(precur_aggr=1)

    # import RW timeseries
    path_ext = os.path.join(main_dir,'publications/paper2/output/heatwave_circulation_v300_z500_SST/57db0USCA/z500_155-300-20-7357db0USCA.h5')
    df_ext = functions_pp.load_hdf5(path_ext)['df_data']
    df_ext = df_ext.mean(axis=0,level=1) ;
    df_ext.pop('TrainIsTrue') ; df_ext.pop('RV_mask')
    df_data = rg.df_data.copy().mean(axis=0,level=1) ;
    df_data = df_data.merge(df_ext.loc[df_data.index], left_index=True, right_index=True)
    # popkeys = [k for k in df_data.columns if k in ['1ts_y','x']]
    # [df_data.pop(pk) for pk in popkeys]
    df_data.pop('TrainIsTrue') ; df_data.pop('RV_mask')

    df_all = df_data.rename({'mx2teast':'$T^E$', 'mx2twest':'$T^W$',
                             'westRW':'$RW^W$', 'eastRW':'$RW^E$',
                             'EOF0_z500':'z500-EOF1', 'EOF1_z500':'z500-EOF2',
                             '0..1..EOF_v300':'$v300$-$EOF1$', '0..2..EOF_v300':'v300-EOF2'},axis=1)
    # PNA_cpc = PNA_cpc.rename({'PNAcpc':'$PNAcpc$'},axis=1)
    df_c = df_all.merge(PNA,left_index=True, right_index=True)
    df_c = df_c.merge(PNA_cpc.loc[df_all.index], left_index=True, right_index=True)
    if west_east == 'east':
        columns = ['E-RW', 'v300-EOF1', 'PNAliu', 'PNAcpc']
    elif west_east == 'west':
        columns = ['W-RW', 'v300-EOF1']
    elif west_east == 'combine':
        columns = ['$RW^E$', '$RW^W$', '$v300$-$EOF1$','PNAcpc']
    df_ana.plot_ts_matric(df_c[columns], win=15, plot_sign_stars=False, fontsizescaler=0)

    filename = os.path.join(rg.path_outsub1, f'cross_corr_{west_east}_15d_selbox{adapt_selbox}.pdf')
    plt.savefig(filename, bbox_inches='tight')