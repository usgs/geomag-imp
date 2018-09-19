#!/usr/bin/env python

"""Command-line wrapper to geomag_imp module

This Python script requires that a separate companion procedure populate its
data_dir with IAGA2002-formatted data files that it can read to interpolate
magnetic perturbations (IMPs). At some point the USGS will provide real-time
access to the necessary magnetic disturbance data sets via its public waveform
server (cwbpub.cr.usgs.gov), and we can simply use the USGS' geomag-algorithms
software to pull data from it. For now...

This script does:
- read XYZF magnetic field data from a local IAGA2002 file-based data store
  (all other configurations are hard-coded below);
- interpolate magnetic fields onto a pre-defined output grid;
- save results to a file that can be read by imp_io.py;

This script does NOT:
- ensure that the input data exists (this is handled by make_svsqdist.py);
- generate diagnostic magnetic field vector maps of North America (this
  is handled by make_impmaps.py)

"""
import numpy as np
from glob import glob

from obspy.core import UTCDateTime,Stream,Trace,Stats
#from geomagio.edge import EdgeFactory
from geomagio.iaga2002 import IAGA2002Factory

from geomag_imp import imp_io
from geomag_imp import secs, secsRegressor

# import sys module, mostly for argv
import sys


if __name__ == "__main__":

   #
   # hard-coded configuration parameters
   #

   # Earth's radius in meters
   Re = 6370e3
   Hi = 110e3 # height of ionosphere

   # grid for SECs
   secs_lat = (15,85,2) # min,max,delta tuple for SEC latitudes in degrees
   secs_lon = (-175,-25,2) # min,max,delta tuple for SEC longitudes in degrees

   # epsilon for truncated SVD
   epsilon = 0.05

   # grid for predictions
   pred_lat = (20,80,2) # min/max pair for prediction latitudes
   pred_lon = (-170,-30,2) # min/max  pair for prediction longitudes

   # directory for IAGA2002 files
   data_dir = './'
   dist_dir = 'Disturbance/'
   sq_dir = 'SolarQuiet/'

   # input filename bases for building urlTemplate
   iaga_file = '{obs}{date:%Y%m%d}{t}'
   dist_file = "_dt_"
   sq_file = "_sq_"

   # urlInterval, combined with urlTemplate, define the expected filename
   # structure of our local file store
   urlInterval = 86400

   # The following define an allowed minimum age for requested data, a set of
   # discrete times (within a day) at which endtime may fall, and the width of
   # the data window to be requested and processed.
   min_obs_age = 600 # in seconds
   every_nth = 300 # in seconds
   win_size = 1200 # in seconds

   # if custom interval is required, modify the following lines to override the
   # realtime interval calcualted from min_obs_age and every_nth, otherwise set
   # starttime and endtime equal to None
   # starttime = UTCDateTime(2018,8,1,0,0,0)
   # endtime = UTCDateTime(2018,8,1,1,0,0)
   starttime = None
   endtime = None

   # output formatting
   write_CDF = False
   write_JSON = False
   write_ASCII = True


   #
   #
   # No more configuration parameters below this point
   #
   #

   # IAGA observatory codes are the only allowed command-line inputs, so no
   # need for fancy argument parsing
   iagaCodes = sys.argv[1:]


   # construct SECS grid
   lat_tmp, lon_tmp, r_tmp = np.meshgrid(
      np.linspace(secs_lat[0], secs_lat[1],
                  (secs_lat[1] - secs_lat[0]) / secs_lat[2] + 1),
      np.linspace(secs_lon[0], secs_lon[1],
                  (secs_lon[1] - secs_lon[0]) / secs_lon[2] + 1),
      Re+Hi,
      indexing='ij'
   )
   secs_lat_lon_r = np.hstack(
      (lat_tmp.reshape(-1,1),
       lon_tmp.reshape(-1,1),
       r_tmp.reshape(-1,1))
   )

   # construct prediction grid
   lat_tmp, lon_tmp, r_tmp = np.meshgrid(
      np.linspace(pred_lat[0], pred_lat[1],
                  (pred_lat[1] - pred_lat[0]) / pred_lat[2] + 1),
      np.linspace(pred_lon[0], pred_lon[1],
                  (pred_lon[1] - pred_lon[0]) / pred_lon[2] + 1),
      Re,
      indexing='ij'
   )
   pred_lat_lon_r = np.hstack(
      (lat_tmp.reshape(-1,1),
       lon_tmp.reshape(-1,1),
       r_tmp.reshape(-1,1))
   )

   # calculate nominal data window for output files (these get modified based
   # on state files later)
   now = UTCDateTime.now() - min_obs_age
   nsec = now.hour*3600 + now.minute*60 + now.second + now.microsecond*1e-6
   delta = (nsec // every_nth) * every_nth - nsec
   out_end = now + delta
   out_start = out_end - win_size

   # override with custom starttime and endtime
   if (starttime is not None and endtime is not None):
      out_start = starttime
      out_end = endtime
   elif starttime is not None:
      out_start = starttime
      out_end = starttime
   elif endtime is not None:
      out_start = endtime
      out_end = endtime

   # NOTE:
   # The `urlTemplate` below corresponds to filenames that comply with IAGA2002
   # (https://www.ngdc.noaa.gov/IAGA/vdat/IAGA2002/iaga2002format.html)
   # recommendations. In short, the USGS' geomag-algorithms software treats the
   # local file system as a data store whose atomic units are IAGA2002 ascii
   # files named according to `urlTemplate`. This is incredibly inefficient
   # compared to a USGS Edge server, so once these data are avaialable there,
   # this script should be modified to access that.

   # create factory to read in _Dist data
   urlTemplate = \
     'file://' + data_dir + dist_dir + iaga_file + dist_file + '{i}.{i}'
   dist_factory = IAGA2002Factory(
      urlTemplate = urlTemplate,
      urlInterval = urlInterval
   )

   # create factory to read in _SQ data
   urlTemplate = \
     'file://' + data_dir + sq_dir + iaga_file + sq_file + '{i}.{i}'
   sq_factory = IAGA2002Factory(
      urlTemplate = urlTemplate,
      urlInterval = urlInterval
   )

   # loop over observatories, reading in data, and building up data arrays
   dist_stream = Stream()
   sq_stream = Stream()
   obs_lat_lon_r = []
   obs_Btheta_Bphi_Br = []
   obs_sigma_Btheta_Bphi_Br = []
   badObs = []
   for ob in iagaCodes:

      dist_stream = dist_factory.get_timeseries(
            observatory = ob,
            starttime = out_start,
            endtime = out_end,
            channels = ['Xdt','Ydt','Zdt','Fdt']
      )

      sq_stream = sq_factory.get_timeseries(
            observatory = ob,
            starttime = out_start,
            endtime = out_end,
            channels = ['Xsq','Ysq','Zsq','Fsq']
      )

      if dist_stream.count() == 0 or sq_stream.count() == 0:
         print ob, 'data could not be read; skipping...'
         badObs.append(ob) # remove bad iagaCodes after for-loop
         continue
      else:
         print ob, 'data read in successfully'

      dist_X = dist_stream.select(channel="Xdt")[0]
      dist_Y = dist_stream.select(channel="Ydt")[0]
      dist_Z = dist_stream.select(channel="Zdt")[0]

      sq_X = sq_stream.select(channel="Xsq")[0]
      sq_Y = sq_stream.select(channel="Ysq")[0]
      sq_Z = sq_stream.select(channel="Zsq")[0]

      #
      # exract and append data to lists
      #

      # geodetic coordinates
      obs_lat_lon_r.append(
         [
            float(dist_X.stats['geodetic_latitude']),
            float(dist_X.stats['geodetic_longitude']),
            Re
         ]
      )

      # magnetic field vector components (add dist+sq)
      # (convert from nanoTesla to Tesla)
      obs_Btheta_Bphi_Br.append(
         [
            -(dist_X.data + sq_X.data) * 1e-9,
             (dist_Y.data + sq_Y.data) * 1e-9,
            -(dist_Z.data + sq_Z.data) * 1e-9
         ]
      )

      # magnetic field vector sigmas
      obs_sigma_Btheta_Bphi_Br.append(
         [
            [1 if good else np.inf for good in np.isfinite(dist_X.data)],
            [1 if good else np.inf for good in np.isfinite(dist_Y.data)],
            [np.inf if good else np.inf for good in np.isfinite(dist_Z.data)]
         ]
      )

   # end 'for ob in stations:'

   # necessary to remove these outside the loop due to implicit loop indexing
   for bo in badObs:
      iagaCodes.remove(bo)

   # convert lists to numpy arrays with time varying along axis 0
   obs_lat_lon_r = np.transpose(obs_lat_lon_r, (0,1)) # time-invariant
   obs_Btheta_Bphi_Br = np.transpose(obs_Btheta_Bphi_Br, (2,0,1))
   obs_sigma_Btheta_Bphi_Br = np.transpose(obs_sigma_Btheta_Bphi_Br, (2,0,1))


   # initialize the secs object and secsRegressor
   imp = secsRegressor(
      secs(secs_lat_lon_r),
      epsilon
   )


   # Finally, generate a map for each time step
   X = []
   Y = []
   Z = []
   for tidx in np.arange(len(obs_Btheta_Bphi_Br)):

      # these element-wise comparisons help avoid re-calculation of various
      # imp attributes that would be much more computationally intensive
      if np.array_equal(obs_Btheta_Bphi_Br[tidx],
                        imp.obs_Btheta_Bphi_Br_):
         obs_Bt_Bp_Br = imp.obs_Btheta_Bphi_Br_
      else:
         obs_Bt_Bp_Br = obs_Btheta_Bphi_Br[tidx]
      if np.array_equal(obs_sigma_Btheta_Bphi_Br[tidx],
                        imp.obs_sigma_Btheta_Bphi_Br_):
         obs_sigma_Bt_Bp_Br = imp.obs_sigma_Btheta_Bphi_Br_
      else:
         obs_sigma_Bt_Bp_Br = obs_sigma_Btheta_Bphi_Br[tidx]


      # fit the observed data (NaNs are converted to zero, but these should
      #  be ignored because the corresponding sigmas are infinite)
      imp.fit(obs_lat_lon_r, np.nan_to_num(obs_Bt_Bp_Br),
              sigma_Btheta_Bphi_Br = obs_sigma_Bt_Bp_Br)

      # interpolate to prediction grid
      pred_Btheta_Bphi_Br = imp.predict(pred_lat_lon_r)

      # accumulate interpolated output
      # (convert from Tesla to nanoTesla)
      X.append(-pred_Btheta_Bphi_Br[:,0] * 1e9)
      Y.append(pred_Btheta_Bphi_Br[:,1] * 1e9)
      Z.append(-pred_Btheta_Bphi_Br[:,2] * 1e9)


   # create/convert/format outputs so imp_io can handle them
   Epoch = np.array([(dist_X.stats['starttime'] + delta).datetime
                     for delta in dist_X.times()])

   X = np.array(X)
   Y = np.array(Y)
   Z = np.array(Z)
   Label = np.array(
      ["" for site in np.arange(len(pred_lat_lon_r))] # empty labels for now
   )

   # convert observations from Tesla to nanoTesla
   ObsX = np.array(-obs_Btheta_Bphi_Br[:,:,0] * 1e9)
   ObsY = np.array(obs_Btheta_Bphi_Br[:,:,1] * 1e9)
   ObsZ = np.array(-obs_Btheta_Bphi_Br[:,:,2] * 1e9)
   ObsFit = np.array(np.sum(np.isfinite(obs_sigma_Btheta_Bphi_Br), axis=2))
   ObsName = np.array(iagaCodes)


   if write_CDF:
      # write out to a CDF file
      imp_io.write_imp_CDF(
         Epoch,
         (coord for coord in pred_lat_lon_r.T), X, Y, Z, Label,
         (coord for coord in obs_lat_lon_r.T), ObsX, ObsY, ObsZ, ObsFit, ObsName,
         filename = "impSECS_%s--%s.cdf"%(Epoch[0].isoformat(),
                                      Epoch[-1].isoformat())
      )

   if write_JSON:
      # write out to a JSON file
      imp_io.write_imp_JSON(
         Epoch,
         (coord for coord in pred_lat_lon_r.T), X, Y, Z, Label,
         (coord for coord in obs_lat_lon_r.T), ObsX, ObsY, ObsZ, ObsFit, ObsName,
         filename = "impSECS_%s--%s.json"%(Epoch[0].isoformat(),
                                      Epoch[-1].isoformat())
      )

   if write_ASCII:
      # write out using Antti Pulkkinen's zipped multi-file ascii format
      imp_io.write_imp_ASCII(
         Epoch,
         (coord for coord in pred_lat_lon_r.T), X, Y, Z, Label,
         (coord for coord in obs_lat_lon_r.T), ObsX, ObsY, ObsZ, ObsFit, ObsName,
         filename = "impSECS_%s--%s.zip"%(Epoch[0].isoformat(),
                                      Epoch[-1].isoformat())
      )
