
# RG-CPD (Response Guided - Causal Precursor Detection)
## Introduction


RG-CPD is a framework to process 3-dimensional climate data, such that relationships based on correlation can be tested for conditional dependence, i.e. causality. These causal teleconnections can be used to forecast a target variable of interest.


Causal inference frameworks have been proven valuable by going beyond defining a relationship based upon correlation. Autocorrelation, common drivers and indirect drivers are very common in the climate system, and they lead to spurious (significant) correlations. Tigramite has been successfully applied to 1 dimensional time series in climate science ([Kretschmer et al. 2016](https://doi.org/10.1175/JCLI-D-15-0654.1)), in order to filter out these spurious correlations using conditional indepence tests ([Runge et al. 2017](http://arxiv.org/abs/1702.07007)).

Within RG-CPD, the 1-d precursor time series are obtained by creating point-wise correlation maps and subsequently grouping adjacent significantly correlating gridcells together into precursor regions. These precursor regions are then converted into 1-d time series by taking a spatial mean ([Kretschmer et al. 2017](https://doi.org/10.1002/2017GL074696)).

The final step is the same, where the 1-d time series are processed by Tigramite to extract the causal relationships. This requires thorough understanding of the method, see [Runge et al. 2017](http://arxiv.org/abs/1702.07007)). These 1d time series contain more information since they are spatially aggregated. The 1d time series of different precursor regions are subsequently tested for causality using the [Tigramite package](https://github.com/jakobrunge/tigramite). One has to have good knowledge about the assumptions needed for causal inference ([Runge et al., 2018](https://doi.org/10.1063/1.5025050)).

This code has been used for my Phd (see publications), however, the coding and documentation quality is poor. 

With the [AI4S2S](https://ai4s2s.readthedocs.io/en/latest/index.html) project, we are currently working on a (follow-up) Python Package with a complete redesign of the code (in collaboration with professional research software engineers). The goal of s2spy is to make the construction of ML-based pipelines much more efficient, transparent, and scalable to any HPC system and climate data platform. If you are interested in doing these types of analysis (RGCPD/ML) I strongly recommend building your pipeline with the [s2spy](https://github.com/AI4S2S/s2spy) software. 

----------------
## Features
- basic pre-processing steps (removing climatology and linear detrending)
- time-aggregation handling (for subseasonal and seasonal user-case)
- set of cross-validation methods
- extracting precursors (from netcdf4):
	- Correlation maps (corr. maps -> spatial clustering -> precursor timeseries).
	- Correlation maps while regressing out (1) influence of third timeseries or autocorrelation of (2) target and/or (3) precursor.
	- Empirical Orthogonal Functions (EOFs)
	- Directly loading precursor timeseries
- Tigramite (Causal discovery and inference)
- Scikit-learn models + optional GridSearch for tuning
- flexible forecast verification metrics
- (basic) plotting functions with Cartopy
- ECMWF_retrieval with a download python wrapper to get data from the Climate Data Store ERA-5 dataset


Have a look at [subseasonal.ipynb](https://github.com/semvijverberg/RGCPD/blob/master/seasonal_mode.ipynb) and [seasonal.ipynb](https://github.com/semvijverberg/RGCPD/blob/master/subseasonal_mode.ipynb) for an overview of the core functionality.

----------------
## Installation

Requires Python version between 3.8 & 3.9.

To install requirements, create a new conda environment:

Mamba greatly speeds up the installation of conda environments, if mamba has not been installed in your conda base environment, then please run:
conda install mamba -n base -c conda-forge

mamba env create -f environment.yml

Additional dependencies:\
Tigramite and environment.yml do not install networkx (needed for plotting Causal Effect Networks), if needed:\
mamba install -c anaconda networkx\
To install spyder:\
mamba install -c conda-forge spyder\
If working on a Windows, you need to have installed the visual c++ build tools\
https://visualstudio.microsoft.com/visual-cpp-build-tools/


----------------

## User Agreement

Please adhere to the License, which states that any modifications to the code must also be published under the same open source license.

Depending on what you implement, please cite the work by dr. Jakob Runge accordingly:
Tigramite (https://github.com/jakobrunge/tigramite). 

----------------
## License

Copyright (c) 2018, VU Amsterdam

GNU General Public License v3

----------------
## Optional:
### ECWMF API (for MARS system and Climate Data Store)
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

