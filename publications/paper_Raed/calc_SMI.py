# -*- coding: utf-8 -*-
import os, inspect, sys

import matplotlib.pyplot as plt



user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
SPI_func_dir = '/Users/semvijverberg/surfdrive/Scripts/Drought_Risk_Kenya_CIP/Notebooks'
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)
    sys.path.append(SPI_func_dir)

path_raw = os.path.join(main_dir, 'data')

import core_pp, functions_pp
import func_SPI

SM_filepath = os.path.join(path_raw, 'swvl1_1950-2019_1_12_monthly_1.0deg.nc')
USBox = (225, 300, 20, 60)
ds = core_pp.import_ds_lazy(SM_filepath, selbox=USBox, auto_detect_mask=True)

aggr = 1
output = func_SPI.calc_SPI_from_monthly(ds, aggr)
output = core_pp.detect_mask(output)
output.to_netcdf(functions_pp.get_download_path() + f'/Own_SPI_{aggr}.nc')

#%% # comparison to package results
SMI_package_filepath = os.path.join(path_raw, '/Users/semvijverberg/surfdrive/ERA5/SM_spi_gamma_01_1950-2019_1_12_monthly_1.0deg.nc')
SMI_package = core_pp.import_ds_lazy(SMI_package_filepath)

ts_raw = ds.sel(latitude=40, longitude=240)
ts_stn = output.sel(latitude=40, longitude=240)
ts_pack = SMI_package.sel(latitude=40, longitude=240)

fig = plt.figure(figsize=(20,10) )

# plot observed versus corresponding Gamma probability
ax1 = plt.subplot(1, 2, 1)
vals = ax1.plot(ts_raw, '.k');
# ax1.set_ylabel('Cumulative probability (from Gamma distribution)')
# ax1.set_xlabel('Aggregated Precipitation [mm] - Original values')
# plot transformed standard normal from Gamma probability
ax1 = plt.subplot(1, 2, 2)
vals = ax1.plot(ts_stn, '.k');
ax1.plot(ts_pack, '.r')
# ax1.set_ylabel('Cumulative probability (from Gamma distribution)')
ax1.set_xlabel('SPI - Gamma prob. transformed to standard normal ')