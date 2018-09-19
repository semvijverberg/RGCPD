#!/bin/ksh
echo starting bash script
cdo -b 32 ydaysub /Users/semvijverberg/surfdrive/Data_ERAint/input_raw/tmpfiles/tmpdet.nc -ydayavg /Users/semvijverberg/surfdrive/Data_ERAint/input_raw/tmpfiles/tmpdet.nc /Users/semvijverberg/surfdrive/Data_ERAint/input_raw/tmpfiles/tmpan.nc
