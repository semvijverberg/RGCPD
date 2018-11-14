

# RG-CPD
Features
=====

The basic idea behind RG-CPD was to create a python package which can process 3-dimensional data such that relationships based on correlation can be tested for causality.

Causal inference metrics has been proven a valuable to go beyond defining a relationship only by looking at e.g. correlation. Tigramite has been applied to 1 dimensional time series in climate science (See Kretschmer et al. 2016 10.1175/JCLI-D-15-0654.1).

Within RG-CPD, the 1-d precursor time series are obtained by creating point-wise correlation maps and subsequently grouping adjacent significantly correlating gridcells together into precursor regions. These precursor regions are then converted into 1-d time series by taking a spatial mean (see Kretschmer et al. 2017 10.1002/2017GL074696).

The final step is the same, where the 1-d time series are processed by Tigramite to extract the causal relationships. This requires thorough understanding of the method, see Runge et al. 2017 http://arxiv.org/abs/1702.07007) These 1d time series contain more information since they are spatially aggregated. The 1d time series of different precursor regions and subsequently tested for causality using the Tigramite package.  


Installation
===========


Installation Anaconda3, python 3.6.1 environment

Installing in conda base_root 
----------------------
conda install spyder=3.3.1\
conda config --add channels conda-forge\
conda config --append channels bioconda\
conda install numpy pandas matplotlib cartopy xarray netCDF4 ecmwfapi scipy seaborn netcdftime cyordereddict\
conda install -c conda-forge nco


If you want to install this package in an environment this will not work since it will automatically install Tigramite to your root site-package e.g. ~anaconda/lib/python2.7/site-packages/. And running Spyder from source activate <env> will point to the site-packages in your env folder ~/anaconda/envs/<env_name>/lib/python3.6/site-packages, not your root (thus it will not find the tigramite module). 
To solve this you could do:

Installing in conda env 
----------------------
Easy way:\
conda create --name <env_name> --file conda_env.txt \
More typing way, but probably cleaner: \
conda config --add channels conda-forge \
conda config --append channels bioconda\
conda create --name <env_name> python=3.6 numpy pandas matplotlib cartopy xarray netCDF4 ecmwfapi scipy seaborn netcdftime cyordereddict pip 


ECWMF MARS API
----------------
If you did not have ecmwfapi installed before, you need to create an ecmwf account and copy your key into the file .ecmwfapirc in your home directory. This will look like this: \
{\
    "url"   : "https://api.ecmwf.int/v1",\
    "key"   : <your key>,\
    "email" : <your emailadress>\
}


Installing Tigramite
----------------
To install Tigramite, see https://github.com/jakobrunge/tigramite \
If you install Tigramite on your base_root in conda, it will satisfy to download the tigramite-master folder and run \
python setup.py install .

Installing Tigramite in conda env:
----------------
Source activate <env_name> \
Git clone https://github.com/jakobrunge/tigramite.git \
pip install ./tigramite 

Installing CDO 
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

Vijverberg, S.P., Kretschmer, M. (2018). Python code for applying the Response Guided - Causal Precursor Detection scheme. https://doi.org/10.5281/zenodo.1478819


Dr. Jakob Runge, who developed the causal inference python package Tigramite (https://github.com/jakobrunge/tigramite).

1. J. Runge, S. Flaxman, and D. Sejdinovic (2017): Detecting causal associations in large nonlinear time series datasets. https://arxiv.org/abs/1702.07007

2. J. Runge et al. (2015): Identifying causal gateways and mediators in complex spatio-temporal systems. Nature Communications, 6, 8502. http://doi.org/10.1038/ncomms9502

3. J. Runge (2015): Quantifying information transfer and mediation along causal pathways in complex systems. Phys. Rev. E, 92(6), 62829. http://doi.org/10.1103/PhysRevE.92.062829

4. J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths (2012): Escaping the Curse of Dimensionality in Estimating Multivariate Transfer Entropy. Physical Review Letters, 108(25), 258701. http://doi.org/10.1103/PhysRevLett.108.258701





# Autogenerated text by Cookiecutter:

Python versions
---------------

This repository is set up with Python versions:
* 3.4
* 3.5
* 3.6

Installation
------------

To install RGCPD, do:

.. code-block:: console

  git clone https://github.com//RGCPD.git
  cd RGCPD
  pip install .



License
------------

Copyright (c) 2018, VU Amsterdam

GNU General Public License v3
