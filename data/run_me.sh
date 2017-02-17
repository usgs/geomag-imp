#!/bin/sh

#
# Generate Interpolated Magnetic Perturbations (G-IMPs).
#
# NOTES:
# - most configuration options are in the Python scripts called below
# - eventually, step 1 below will not be required because the USGS will provide
#   SV, SQ, and Dist(urbance) data for all observatories as a regular near-real
#   time data product; until then:
#   - if run in real time mode (i.e., starttime and endtime are NONE), only the
#     first pass will be slow due to 90 days of data being pulled from USGS to
#     initialize secular variation (SV) baseline and solar quiet (SQ) daily
#     variation; after first run, only data since last run is pulled.
#   - if NOT run in real time mode (i.e., starttime and endtime are specified),
#     step 1 will be slow as it downloads 90 days of data for each observatory.
#

# 1) create disturbance time series for list of observatories
python ../bin/make_svsqdist.py BOU BRW BSL CMO DED FRD FRN GUA HON NEW SHU SJG TUC

# 2) generate interpolated/gridded North American magnetic disturbance maps
python ../bin/make_imp_secs.py BOU BRW BSL CMO DED FRD FRN GUA HON NEW SHU SJG TUC

# 3) generate diagnostic plots of North America magnetic disturbance
python ../bin/make_impmaps.py *.zip
