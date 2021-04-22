
# RG-CPD (Response Guided - Causal Precursor Detection)
Introduction
=====

RG-CPD is a framework to process 3-dimensional climate data, such that relationships based on correlation can be tested for conditional dependence, i.e. causality. These causal teleconnections can be used to forecast a target variable of interest.


Causal inference frameworks have been proven valuable by going beyond defining a relationship based upon correlation. Autocorrelation, common drivers and indirect drivers are very common in the climate system, and they lead to spurious (significant) correlations. Tigramite has been successfully applied to 1 dimensional time series in climate science (Kretschmer et al. 2016 https://doi.org/10.1175/JCLI-D-15-0654.1), in order to filter out these spurious correlations using conditional indepence tests (Runge et al. 2017 http://arxiv.org/abs/1702.07007).

Within RG-CPD, the 1-d precursor time series are obtained by creating point-wise correlation maps and subsequently grouping adjacent significantly correlating gridcells together into precursor regions. These precursor regions are then converted into 1-d time series by taking a spatial mean (Kretschmer et al. 2017 https://doi.org/10.1002/2017GL074696).

The final step is the same, where the 1-d time series are processed by Tigramite to extract the causal relationships. This requires thorough understanding of the method, see Runge et al. 2017 http://arxiv.org/abs/1702.07007). These 1d time series contain more information since they are spatially aggregated. The 1d time series of different precursor regions are subsequently tested for causality using the Tigramite package (https://github.com/jakobrunge/tigramite). One has to have good knowledge about the assumptions needed for causal inference, https://doi.org/10.1063/1.5025050.

# Features
- basic pre-processing steps (removing climatology and linear detrending)
- time-aggregation handling (for subseasonal and seasonal user-case)
- set of cross-validations types (random k-fold, stratified k-fold, leave-n-out)
- extracting precursors (from netcdf4):
	- Correlation maps (corr. maps -> spatial clustering -> precursor timeseries).
	- Correlation maps while regressing out (1) influence of third timeseries or autocorrelation of (2) target and/or (3) precursor.
	- Empirical Orthogonal Functions (EOFs, or PCA)
	- some climate indices and/or directly loading precursor timeseries
- Tigramite (Causal discovery and inference)
- Scikit-learn models + optional GridSearch for tuning
- flexible forecast verification metrics
- (basic) plotting functions with Cartopy
- ECMWF_retrieval with a download python wrapper to get data from the Climate Data Store ERA-5 dataset
\
Have a look at **subseasonal.ipynb** and **seasonal.ipynb** for an overview of the core functionality.

Installation
===========
If depencies are correct, then all scripts should work. Please use the .yml file to create a new conda environment. Afterwards, Tigramite is installed from github.


### Create conda environment:
conda env create -f RGCPD_no_build.yml \
conda activate RGCPD \
Git clone https://github.com/jakobrunge/tigramite.git (you can clone this repo into any folder, e.g. your Download folder)\
python setup.py install



## Optional:
### ECWMF API (for MARS system and Climate Data Store)
----------------
If you did not have ecmwfapi installed before, you need to create an ecmwf account and copy your key into the file .ecmwfapirc in your home directory. See https://confluence-test.ecmwf.int/display/WEBAPI/Access+MARS#AccessMARS-downloadmars. This will look like this:
 \
{
\
    "url"   : "https://api.ecmwf.int/v1",
\
    "key"   : <your key>,\
    "email" : <your emailadress>\
}


### Installing CDO (only needed when you want to download from ECWMF)
----------------
cdo -V \
Climate Data Operators version 1.9.4 (http://mpimet.mpg.de/cdo) \
System: x86_64-apple-darwin17.6.0 \
CXX Compiler: /usr/bin/clang++ -std=gnu++11 -pipe -Os -stdlib=libc++ -isysroot/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -arch x86_64  -D_THREAD_SAFE -pthread \
CXX version : unknown \
C Compiler: /usr/bin/clang -pipe -Os -isysroot/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -arch x86_64  -D_THREAD_SAFE -pthread -D_THREAD_SAFE -D_THREAD_SAFE -pthread \
C version : unknown \
F77 Compiler:  -pipe -Os \
Features: 16GB C++11 DATA PTHREADS HDF5 NC4/HDF5 OPeNDAP UDUNITS2 PROJ.4 CURL FFTW3 SSE4_1 \
Libraries: HDF5/1.10.2 proj/5.1 curl/7.60.0 \
Filetypes: srv ext ieg grb1 grb2 nc1 nc2 nc4 nc4c nc5  \
     CDI library version : 1.9.4 \
GRIB_API library version : 2.7.0 \
  NetCDF library version : 4.4.1.1 of Jun  8 2018 03:07:16 $ \
    HDF5 library version : 1.10.2 \
    EXSE library version : 1.4.0 \
    FILE library version : 1.8.3 \



**************


User Agreement
----------------

You commit to cite RG-CPD in your reports or publications if used:

Dr. Marlene Kretschmer, who developed the method and used it for studying Polar Vortex dynamics. Please cite: 

Kretschmer, M., Runge, J., & Coumou, D. (2017). Early prediction of extreme stratospheric polar vortex states based on causal precursors. Geophysical Research Letters, 44(16), 8592–8600. https://doi.org/10.1002/2017GL074696

PhD. Sem Vijverberg, who expanded Kretschmer's original python code into a python code that can be applied in a versatile manner. 

Vijverberg, S.P., Kretschmer, M. (2018). Python code for applying the Response Guided - Causal Precursor Detection scheme. https://doi.org/10.5281/zenodo.1486739

Dr. Jakob Runge, who developed the causal inference python package Tigramite (https://github.com/jakobrunge/tigramite). Depending on what you implement from Tigramite, please cite accordingly.


License
------------

Copyright (c) 2018, VU Amsterdam

GNU General Public License v3
