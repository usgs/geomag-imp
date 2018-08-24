#!/usr/bin/env python

"""Command-line script to generate IMP maps

This Python script requires that a separate companion procedure populate its
data_dir with properly formatted data files that it can read, and use to map
interpolated magnetic perturbations (IMPS). For now...

This script does:
- read from a list of IMP-formatted files;
- generate PNG vector maps over North America;

"""

# import Pyplot module
# NOTE: matplotlib.use('Agg') is necessary to plot when no X environment is
#       available, like when running as a cron job
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import numpy module
import numpy as np

# import scipy's interpolate sub-package
from scipy import interpolate as sInterp

# import Basemap to initialze maps
from mpl_toolkits.basemap import Basemap

# import sys module, mostly for argv
import sys

# import os module to query and modify environment
import os

# import r[egular] e[xpression] module
import re

# import UTCDateTime module
from obspy.core import UTCDateTime

# import imp_io from geomag_imp module
from geomag_imp import imp_io


if __name__ == "__main__":

   #
   # map related configuration parameters
   #

   # set up a Lambert Azimuthal equal area basemap for North America
   mapH = 6000000 # encompasses most of N.A., South to North, in ~meters
   mapW = 10000000 # encompasses all of N.A., East to West, and HON, in ~meters
   lon_0 = 250. # central longitude
   lat_0 = 50.  # central latitude
   lat_ts = lat_0 # latitude of "true scale"

   # set nx and ny to None if you want the original Lat/Lon grid
   nx = 40 # number of interpolated points in map's X direction
   ny = 24 # number of interpolated points in map's Y direction

   # configure some map drawing preferences
   land_color = 'lightgray'
   water_color = 'white'
   arrow_color = 'darkblue'
   arrow_scale = 67
   arrow_width = .003

   # configure output map file
   data_dir = './'
   plot_dir = data_dir + 'Plots/'
   map_ext = '.png'
   map_dpi = 150

   #
   #
   # No more configuration parameters below this point
   #
   #

   # assume any command line arguments are files to read; skip the script name
   files = sys.argv[1:]

   # if nx or ny is None, and the other is not, make nx=ny, or ny=nx
   if nx is None and ny is not None:
      nx = ny
   if ny is None and nx is not None:
      ny = nx

   # loop over files, plot all maps in each file,
   for fname in files:
      try:
         # attempt to read CDF file
         (
            UTs,
            (Lats, Lons, Rads),
            Xs,
            Ys,
            Zs,
            Labels,
            (ObsLats, ObsLons, ObsRads),
            ObsXs,
            ObsYs,
            ObsZs,
            ObsFits,
            ObsNames
         ) = imp_io.read_imp_CDF(fname)
      except:
         try:
            # attempt to read JSON file
            (
               UTs,
               (Lats, Lons, Rads),
               Xs,
               Ys,
               Zs,
               Labels,
               (ObsLats, ObsLons, ObsRads),
               ObsXs,
               ObsYs,
               ObsZs,
               ObsFits,
               ObsNames
            ) = imp_io.read_imp_JSON(fname)
         except:
            try:
               # attempt to read in Antti Pulkkinen ASCII data
               (
                  UTs,
                  (Lats, Lons, Rads),
                  Xs,
                  Ys,
                  Zs,
                  Labels,
                  (ObsLats, ObsLons, ObsRads),
                  ObsXs,
                  ObsYs,
                  ObsZs,
                  ObsFits,
                  ObsNames
               ) = imp_io.read_imp_ASCII(fname)
            except:
               raise Exception('Unreadable input file: ' + fname)

      # calculate magnitudes, then convert Xs and Ys into unit vectors
      Ms = [np.sqrt(X**2 + Y**2) for X,Y in zip(Xs,Ys)]
      Xs = [X / M for X,M in zip(Xs,Ms)]
      Ys = [Y / M for Y,M in zip(Ys,Ms)]

      # convert Ms to Log10 scale; set negative log values to 0 (i.e., 1 nT)
      Ms = [np.clip(np.log10(M), 0, np.Inf) for M in Ms]



      # calculate magnitudes, then convert ObsXs and ObsYs into unit vectors
      ObsMs = [np.sqrt(X**2 + Y**2) for X,Y in zip(ObsXs,ObsYs)]
      ObsXs = [X / M for X,M in zip(ObsXs,ObsMs)]
      ObsYs = [Y / M for Y,M in zip(ObsYs,ObsMs)]

      # convert ObsMs to Log10 scale; set negative log values to 0 (i.e., 1 nT)
      ObsMs = [np.clip(np.log10(M), 0, np.Inf) for M in ObsMs]


      # roughly keep default size, but force new aspect ratio to match map
      w,h = plt.figaspect(float(mapH)/float(mapW))
      plt.figure(figsize = (w,h))

      # create a basemap
      bm = Basemap(width = mapW, height = mapH,
                   resolution = 'l', projection = 'laea',
                   lat_ts = lat_ts, lat_0 = lat_0, lon_0 = lon_0)
      bm.drawcoastlines()
      bm.fillcontinents(color = land_color, lake_color = water_color)
      # maybe parallels and meridians should be configurable (?)
      bm.drawparallels(np.arange(-80.,81.,20.), labels = [1,1,0,0])
      bm.drawmeridians(np.arange(-180.,181.,20.), labels = [0,0,0,1])
      bm.drawmapboundary(fill_color = water_color)


      for i,ut in enumerate(UTs):

         print 'Plotting '+ut.isoformat()
         sys.stdout.flush()


         if nx==None and ny==None:
            # vectors remain on input grid
            u,v,x,y = bm.rotate_vector(Ys[i] * Ms[i], Xs[i] * Ms[i],
                                       Lons, Lats,
                                       returnxy = True)
         else:
            # Basemap.transform_vector() does not generate results consistent
            # with Basemap.rotate_vector(). We re-implment transform_vector()
            # here, but call a different 2d interpolator.
            uin,vin,xin,yin = bm.rotate_vector(Ys[i] * Ms[i], Xs[i] * Ms[i],
                                               Lons, Lats,
                                               returnxy = True)
            longs, lats, x, y = bm.makegrid(nx, ny, returnxy = True)


            u = sInterp.griddata((xin.flatten(), yin.flatten()), uin.flatten(),
                                 (x, y), method = 'linear')
            v = sInterp.griddata((xin.flatten(), yin.flatten()), vin.flatten(),
                                 (x, y), method = 'linear')

         if i == 0:

             # plot vector field
             q_pred = bm.quiver(x, y, u, v, 10 ** np.sqrt(u**2 + v**2),
                           scale = arrow_scale,
                           width = arrow_width,
                           clim = [1,1000],
                           zorder = 10)

             # NOTE: the following seems broken in MPL-1.4.2, but fixed by MPL-1.4.3
             plt.quiverkey(q_pred, .09, .98, 3,
                           ('%4.0f '+'%s') % (10**3, 'nT'),
                           coordinates='axes', labelpos='W',
                           color=q_pred.get_cmap()((10.**3 - q_pred.get_clim()[0]) /
                                              np.float(q_pred.get_clim()[1] - q_pred.get_clim()[0]) ),
                           fontproperties={'size':10})
             plt.quiverkey(q_pred, .09, .95, 2,
                           ('%4.0f '+'%s') % (10**2, 'nT'),
                           coordinates='axes', labelpos='W',
                           color=q_pred.get_cmap()((10**2 - q_pred.get_clim()[0]) /
                                              np.float(q_pred.get_clim()[1] - q_pred.get_clim()[0]) ),
                           fontproperties={'size':10})
             plt.quiverkey(q_pred, .09, .92, 1,
                           ('%4.0f '+'%s') % (10**1, 'nT'),
                           coordinates='axes', labelpos='W',
                           color=q_pred.get_cmap()((10**1 - q_pred.get_clim()[0]) /
                                              np.float(q_pred.get_clim()[1] - q_pred.get_clim()[0]) ),
                           fontproperties={'size':10})

             # place green dots at observatory locations included in this map solution
             s_avail = bm.scatter(ObsLons[ObsFits[i].astype(bool)],
                        ObsLats[ObsFits[i].astype(bool)],
                        latlon=True, zorder=10, color='green', s=50, alpha=0.75)

             # place red dots at observatory locations NOT included in this map solution
             s_miss = bm.scatter(ObsLons[~ObsFits[i].astype(bool)],
                        ObsLats[~ObsFits[i].astype(bool)],
                        latlon=True, zorder=10, color='red', s=50, alpha=0.75)



             # place observation vectors over top green dots
             u,v,x,y = bm.rotate_vector(ObsYs[i] * ObsMs[i], ObsXs[i] * ObsMs[i],
                                        ObsLons, ObsLats,
                                        returnxy = True)
             q_obs = bm.quiver(x, y, u, v, 10 ** np.sqrt(u**2 + v**2),
                               scale = arrow_scale,
                               width = arrow_width,
                               clim = [1,1000],
                               zorder = 10)

         else:
             # change only the predicted vector data, don't update plot elements
             q_pred.set_UVC(u, v, 10 ** np.sqrt(u**2 + v**2))

             # change only the offsets, after transforming to the map projection
             x,y = bm(ObsLons, ObsLats)
             s_avail.set_offsets(zip(x[ObsFits[i].astype(bool)],
                                     y[ObsFits[i].astype(bool)]))
             s_miss.set_offsets(zip(x[~ObsFits[i].astype(bool)],
                                    y[~ObsFits[i].astype(bool)]))

             # place observation vectors over top green dots
             u,v,x,y = bm.rotate_vector(ObsYs[i] * ObsMs[i], ObsXs[i] * ObsMs[i],
                                        ObsLons, ObsLats,
                                        returnxy = True)
             q_obs.set_UVC(u, v, 10 ** np.sqrt(u**2 + v**2))


         # labels
         plt.title(ut.isoformat())

         # save to file
         baseName = re.split('_.', fname)[0] # gets basename up to first '_', excluding suffixes
         plotFilename = (plot_dir+baseName+"_%04d%02d%02dT%02d%02d%02d.png"%
                         (ut.year,ut.month,ut.day,ut.hour,ut.minute,ut.second))

         try:
            plt.savefig(plotFilename, dpi=150)
         except IOError:
            # if plot_dir didn't exist, create and try again
            os.mkdir(plot_dir)
            plt.savefig(plotFilename, dpi=150)
