#!/usr/bin/env python

"""Command-line wrapper to geomagio.algorithm.SqDistAlgorithm()

This Python script is considerably less robust than what will eventually run in
the USGS GeoHazards' production environment, which actually queries its own
database to determine when the last valid data were available, and makes smart
decisions on how to proceed from there. We just rely on the state file from
SqDistAlgorithm to help us make slightly-less-than-dumb decisions. Eventually
this script will be unnecessary, as we will have transitioned SV+SQ+Dist into
operations, and users can simply retrieve these data via a web service.

For now...

This script does:
- pull near real-time XYZF magnetic field data from a USGS Edge server for each
  IAGA code specified at the command line (other configurations are hard-coded
  below);
- apply the SqDistAlgorithm to XYZF;
- write out similar IAGA2002-formatted XYZF files for each of:
   - Dist - magnetic disturbance
   - SQ - solar quiet variation
   - SV - secular variation
   - Sigma - standard deviation of Dist

This script does NOT:
- back-fill, even for the real-time, end-of-series data gap...this turns out
  to be difficult for several reasons, and we want to keep this script (a
  *temporary* solution) as simple as possible.

NOTES:

"""
import numpy as np
from glob import glob
import sys

from obspy.core import UTCDateTime,Stream,Trace,Stats
from geomagio.edge import EdgeFactory
from geomagio.iaga2002 import IAGA2002Factory
from geomagio.algorithm import SqDistAlgorithm
from geomagio.TimeseriesUtility import merge_streams
from geomagio.Util import get_intervals

from geomag_imp import imp_io
from geomag_imp.secs import secs, secsRegressor


if __name__ == "__main__":

   #
   # input related configuration parameters
   #

   # IAGA 3-letter station codes allowed to be read in
   # NOTE: non-USGS stations can be obtained from internal USGS Edge servers,
   #       but only by partners for whom a hole in the USGS firewall has been
   #       created. Otherwise, the public waveform buffer cwbpub.cr.usgs.gov
   #       must be specified as the `edge_url`, and it does not currently allow
   #       access to non-USGS data.
   valid_stations = [
      'BOU', 'BRW', 'BSL', 'CMO', 'DED', 'FRD', 'FRN',
      'GUA', 'HON', 'NEW', 'SHU', 'SIT', 'SJG', 'TUC'
   ]
   # valid_stations += [
   #    'BLC', 'BRD', 'HAD', 'IQA', 'MEA',
   #    'OTT', 'RES', 'STJ', 'VIC', 'YKC'
   # ]

   # Edge url/IP number and port
   edge_url = "cwbpub.cr.usgs.gov"
   edge_port = 2060

   #
   # algorithm related configurations
   #

   # - "seasonal" SQ interval is 1440 minutes
   m = 1440

   # - average age of baseline data is 30 days (43200 minutes)
   alpha = 1./1440./30

   # - average age of slope is 30 days (43200) minutes
   beta = 1./1440./30

   # - average age of "seasonal" SQ data is 30 days
   gamma = 1./30

   # - dampening factor for slope (this is the default; i.e., no dampening)
   phi = 1

   # z-score threshold (standard deviations) is 3 (this is not the default)
   zthresh = 3

   #
   # output related configurations
   #

   # channels to pull and process
   channels = ['X', 'Y', 'Z', 'F']

   # output data directories
   data_dir = './'
   obs_dir = 'Observatory/'
   dist_dir = 'Disturbance/'
   sq_dir = 'SolarQuiet/'
   sv_dir = 'SecularVariation/'
   sd_dir = 'StandardDeviation/'

   # output filename bases used to build urlTemplates
   iaga_file = '{obs}{date:%Y%m%d}{t}'
   dist_file = "_dt_"
   sq_file = "_sq_"
   sv_file = "_sv_"
   sd_file = "_sd_"

   # urlInterval and urlTemplate define a standard filename convention that will
   # establish a local file-based data store that geomag-algorithms understands.
   # The default value is supposed to be 86400, but we set it explicitly because
   # there seems to be a bug in geomag-algorithms where the default does NOT get
   # set in certain circumstances.
   urlInterval = 86400

   # state file base (Observatory code and channel will be appended to this)
   state_dir = 'PriorState/'
   state_file = 'SvSqDistState'

   # The following define an allowed minimum age for requested data, and a set
   # of discrete times (within a day) at which endtime may fall. Actual endtime
   # and startime values will be calculated in the code.
   min_obs_age = 600 # in seconds
   every_nth_sec = 300 # in seconds

   # if custom interval is required, modify the following lines to override the
   # realtime interval calcualted from min_obs_age and every_nth_sec, otherwise
   # set starttime and endtime equal to None
   starttime = UTCDateTime(2017,1,1,0,0,0)
   endtime = UTCDateTime(2017,1,1,1,0,0)
   # starttime = None
   # endtime = None

   #
   #
   # No more configuration parameters below this point
   #
   #

   # IAGA observatory codes are the only allowed command-line inputs, so no
   # need for fancy argument parsing
   iagaCodes = sys.argv[1:]
   for ob in iagaCodes:
      if not ob in valid_stations:
         raise Exception("Invalid IAGA code '%s' passed on command line"%ob)

   # calculate the end of the data window to be processed
   now = UTCDateTime.now() - min_obs_age
   nsec = now.hour*3600 + now.minute*60 + now.second + now.microsecond*1e-6
   delta = (nsec // every_nth_sec) * every_nth_sec - nsec
   out_end = now + delta

   # loop over iagaCodes to process each observatory
   for ob in iagaCodes:

      # initialize dictionary of SqDistAlgorithm objects
      svsqdist = {}

      # initialize input stream to hold all channels
      in_stream = Stream()

      # initialize output stream to hold all channels and SqDist components
      out_stream = Stream()

      # loop over channels to retrieve input data for SqDistAlgorithm
      for ch in channels:

         sf = state_dir + state_file + '_' + ob + ch

         # create SqDistAlgorithm object
         svsqdist[ch] = SqDistAlgorithm(
            # non-default configuration parameters
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            m = m,
            zthresh = zthresh,
            statefile = sf
         )

         # load statefile if it exists
         svsqdist[ch].load_state()

         # set out_start to the next_starttime if it exists
         if svsqdist[ch].next_starttime is not None:
            out_start = svsqdist[ch].next_starttime
         else:
            out_start = out_end
            # override with custom starttime and endtime
            if (starttime is not None and endtime is not None):
               out_start = starttime
               out_end = endtime
            elif starttime is not None:
               out_start = starttime

         # possibly re-initialize with previous 90 days of data
         in_start, in_end = svsqdist[ch].get_input_interval(
            out_start,
            out_end,
            observatory = ob,
            channels = ch
         )

         if in_start <= in_end:
            # create factory and pull data from USGS Edge
            in_factory = EdgeFactory(
               host = edge_url,
               port = edge_port,
               interval = 'minute',
               type = 'variation'
            )
            in_stream += in_factory.get_timeseries(
               starttime = in_start,
               endtime = in_end,
               observatory = ob,
               channels = ch
            )
            print 'Retreived from Edge: %s-%s'%(ob,ch),
            print 'from', in_start, 'to', in_end
         else:
            print "Non-monotonic interval requested (",
            print in_start, 'to', in_end, ")",
            print "skipping %s-%s..."%(ob,ch)


      if in_stream.count() is not len(channels):
         # if any channel was not read in, STOP PROCESSING
         print "No inputs processed or written..."
         pass

      else:
         # channels are processed separately from input retrieval in order to
         # guarantee and maintain synchronization
         chan_dt = []
         chan_sq = []
         chan_sv = []
         chan_sd = []
         for ch in channels:

            # process time series with SqDistAlgorithm
            out_stream += svsqdist[ch].process(in_stream.select(channel=ch))

            # rename sqdist channels to something more IAGA2002-friendly (i.e.,
            # channel names should be 3 characters or less)
            for trace in out_stream.select(channel = ch + "_Dist"):
               trace.stats.channel = ch + 'dt'
               chan_dt.append(ch + 'dt')
            for trace in out_stream.select(channel = ch + "_SV"):
               trace.stats.channel = ch + 'sv'
               chan_sv.append(ch + 'sv')
            for trace in out_stream.select(channel = ch + "_SQ"):
               trace.stats.channel = ch + 'sq'
               chan_sq.append(ch + 'sq')
            for trace in out_stream.select(channel = ch + "_Sigma"):
               trace.stats.channel = ch + 'sd'
               chan_sd.append(ch + 'sd')

         #
         # write data out to familiar IAGA2002 formatted files
         #

         # first, trim data to desired output interval...
         # this gets rid of initialization data
         out_stream.trim(starttime=out_start, endtime=out_end)

         # then, pad to starttime of first urlInterval, and endtime of last...
         # this creates familiar padded "daily" files if urlInterval=86400
         urlIntervals = get_intervals(
            starttime=out_start,
            endtime=out_end,
            size=urlInterval
         )
         urlIntervals.sort() # sort in place...probably not necessary
         out_stream.trim(
            starttime=urlIntervals[0]['start'],
            endtime=urlIntervals[-1]['end'],
            pad=True,
            fill_value=np.nan)

         # NOTE:
         # The `urlTemplate` below produces filenames that comply with IAGA2002
         # (https://www.ngdc.noaa.gov/IAGA/vdat/IAGA2002/iaga2002format.html)
         # recommendations. It appends files as time increments, starting a new
         # file every `urlInterval`.

         # create output factory and write out _Dist data
         urlTemplate = \
           'file://' + data_dir + dist_dir + iaga_file + dist_file + '{i}.{i}'
         out_factory = IAGA2002Factory(
            urlTemplate = urlTemplate,
            urlInterval = urlInterval
         )
         out_factory.put_timeseries(
               out_stream,
               starttime = out_start,
               endtime = out_end,
               channels = chan_dt
         )

         # create output factory and write out _SQ data
         urlTemplate = \
           'file://' + data_dir + sq_dir + iaga_file + sq_file + '{i}.{i}'
         out_factory = IAGA2002Factory(
            urlTemplate = urlTemplate,
            urlInterval = urlInterval
         )
         out_factory.put_timeseries(
               out_stream,
               starttime = out_start,
               endtime = out_end,
               channels = chan_sq
         )

         # create output factory and write out _SV data
         urlTemplate = \
           'file://' + data_dir + sv_dir+ iaga_file + sv_file + '{i}.{i}'
         out_factory = IAGA2002Factory(
            urlTemplate = urlTemplate,
            urlInterval = urlInterval
         )
         out_factory.put_timeseries(
               out_stream,
               starttime = out_start,
               endtime = out_end,
               channels = chan_sv
         )

         # create output factory and write out _Sigma data
         urlTemplate = \
           'file://' + data_dir + sd_dir + iaga_file + sd_file + '{i}.{i}'
         out_factory = IAGA2002Factory(
            urlTemplate = urlTemplate,
            urlInterval = urlInterval
         )
         out_factory.put_timeseries(
               out_stream,
               starttime = out_start,
               endtime = out_end,
               channels = chan_sd
         )
