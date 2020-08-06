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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import numpy module
import numpy as np

# import Cartopy for maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.vector_transform as cvt

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


#
# map configuration parameters
#

# map center and bounds
lon_bounds = (-165, -55)
lat_bounds = (20, 70)
lon_0 = 250. # central longitude
lat_0 = 60.  # central latitude

# set regrid to None if you want the original Lat/Lon grid
regrid = (40, 24) # number of interpolated points per dimension
# regrid = None

# map drawing preferences
land_color = 'lightgray'
water_color = 'white'
arrow_color = 'darkblue'
arrow_scale = 67
arrow_width = .003

# output map file
data_dir = './'
plot_dir = data_dir + 'Plots/'
map_ext = 'png'
map_dpi = 150


if __name__ == "__main__":

   # assume any command line arguments are files to read; skip the script name
   files = sys.argv[1:]

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


      # create a figure
      fig = plt.figure(figsize = (10,6))

      # set up maps

      # define colormap
      cmap = get_cmap('plasma')

      # map bounds
      plot_lat_bounds = (lat_bounds[0], lat_bounds[1])
      plot_lon_bounds = (lon_bounds[0] + 5, lon_bounds[1] - 5)

      # map projection
      proj_data = ccrs.PlateCarree()
      projection = ccrs.LambertConformal(
         central_latitude=lat_0, 
         central_longitude=lon_0
      )


      # create a GeoAxes
      gs = fig.add_gridspec(ncols=1, nrows=1, height_ratios=[1])
      ax = fig.add_subplot(gs[0], projection=projection)

      # draw map
      ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
      land_alpha = 0.7
      scale = '110m'
      # 10m oceans are super slow...
      ax.add_feature(cfeature.OCEAN.with_scale(scale),
                     facecolor='slategrey', alpha=0.65, zorder=-1)
      ax.add_feature(cfeature.LAND.with_scale(scale),
                     facecolor='k', alpha=land_alpha, zorder=0)
      # ax.add_feature(cfeature.STATES.with_scale('50m'),
      #                edgecolor='w', linewidth=0.3, alpha=land_alpha, zorder=0)
      ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                     edgecolor='w', linewidth=0.5, alpha=land_alpha, zorder=0)
      ax.add_feature(cfeature.LAKES.with_scale(scale),
                     facecolor='slategrey', alpha=0.25, zorder=0)
      
      # this only draws lines
      gl1 = ax.gridlines(
         proj_data, draw_labels=False,
         linewidth=1, linestyle='--',
         color='gray', alpha=0.5,
         x_inline=False, y_inline=False,
         zorder=0
      )
      gl1.bottom_labels = False # toggle labels on bottom
      gl1.top_labels = False # toggle labels on top
      gl1.left_labels = False # toggle labels on left
      gl1.right_labels = False # toggle labels on right
      gl1.xlocator = mticker.FixedLocator([-150, -120, -90, -60, -30])
      gl1.ylocator = mticker.FixedLocator([20, 40, 60, 80])
      gl1.rotate_labels = False

      # this only draws labels
      gl2 = ax.gridlines(
         proj_data, draw_labels=True,
         x_inline=False, y_inline=True,
         zorder=0
      )
      gl2.xlines = False
      gl2.ylines = False
      gl2.bottom_labels = True # toggle labels on bottom
      gl2.top_labels = False # toggle labels on top
      gl2.left_labels = False # toggle labels on left
      gl2.right_labels = False # toggle labels on right
      gl2.xlocator = mticker.FixedLocator([-150, -120, -90])
      gl2.ylocator = mticker.FixedLocator([40, 60, 80])
      gl2.rotate_labels = False

      
      # loop over universal times
      for i,ut in enumerate(UTs):

         print('Plotting '+ut.isoformat())
         sys.stdout.flush()


         if i == 0:
            # plot vectors and dots normally first time; after that, just
            # update data values and leave other map elements alone (faster!)
            
            # plot (re)gridded vector field
            norm = mpl.colors.Normalize(vmin=0, vmax=1000)
            Q = ax.quiver(
               Lons, Lats,
               Ys[i] * Ms[i], Xs[i] * Ms[i],
               10**Ms[i], cmap=cmap,
               transform=proj_data,
               norm=norm,
               scale=arrow_scale,
               width=arrow_width,
               regrid_shape=regrid
            )
            
            # plot station locations with data
            avail = ObsFits[i].astype(bool)
            S_avail = ax.scatter(
               ObsLons[avail], ObsLats[avail],
               color='green', alpha=0.5, edgecolors='none',
               transform=proj_data
            )
            # plot station locations without data
            S_miss = ax.scatter(
               ObsLons[~avail], ObsLats[~avail],
               color='red', alpha=0.5, edgecolors='none',
               transform=proj_data
            )

            # plot observatory vector field
            Q_obs = ax.quiver(
               ObsLons, ObsLats,
               ObsYs[i] * ObsMs[i], ObsXs[i] * ObsMs[i],
               10**ObsMs[i], cmap=cmap,
               transform=proj_data,
               norm=norm,
               scale=arrow_scale,
               width=arrow_width
            )

            # generate quiver keys
            # - length is logarithmic (base 10)
            # - color is linear
            cmap = Q.get_cmap()
            color = cmap(norm(1e3))
            ax.quiverkey(Q, .09, .98, 3,
                         ('%4.0f '+'%s') % (1e3, 'nT'),
                         coordinates='axes', labelpos='W',
                         color=color,
                         fontproperties={'size':8})
            color = cmap(norm(1e2))
            ax.quiverkey(Q, .09, .95, 2,
                         ('%4.0f '+'%s') % (1e2, 'nT'),
                         coordinates='axes', labelpos='W',
                         color=color,
                         fontproperties={'size':8})
            color = cmap(norm(1e1))
            ax.quiverkey(Q, .09, .92, 1,
                         ('%4.0f '+'%s') % (1e1, 'nT'),
                         coordinates='axes', labelpos='W',
                         color=color,
                         fontproperties={'size':8})

            # generate colorbar
            cbar_ax = inset_axes(ax, width='3%', height='86%', loc='center right',
                         bbox_to_anchor=(0., 0., .90, 1.0),
                         bbox_transform=fig.transFigure, borderpad=0)
            cbar = fig.colorbar(Q, cax=cbar_ax, orientation='vertical',
                                 use_gridspec=True, fraction=1., aspect=35.)
            cbar.set_label('$B_h$ [nT]', fontsize=12,
                           labelpad=4, rotation=0.)
            cbar.ax.tick_params(labelsize=12)            

         else:
            # change the gridded vector data, don't update other plot elements;
            # this speeds things up immensely, especially with high-res maps
            
            if regrid:
               # manually perform calculations normally done by ax.quiver()
               target_extent = ax.get_extent(ax.projection)
               _, _, Ys_trans, Xs_trans, Ms_trans  = cvt.vector_scalar_to_grid(
                  proj_data, projection, regrid,
                  Lons, Lats,
                  Ys[i] * Ms[i], Xs[i] * Ms[i],
                  (10**Ms[i]),
                  target_extent=target_extent
               )
            else:
               Ys_trans, Xs_trans = projection.transform_vectors(
                  proj_data,
                  Lons, Lats,
                  Ys[i] * Ms[i], Xs[i] * Ms[i]
               )
               Ms_trans = 10**Ms[i]
            Q.set_UVC(Ys_trans, Xs_trans, Ms_trans)

            # change scatter plot "offsets"
            avail = ObsFits[i].astype(bool)
            S_avail.set_offsets(np.array(list(zip(ObsLons, ObsLats)))[avail])
            S_miss.set_offsets(np.array(list(zip(ObsLons, ObsLats)))[~avail])

            # change observatory vector data, don't update other plot elements
            ObsYs_trans, ObsXs_trans = projection.transform_vectors(
               proj_data,
               ObsLons, ObsLats,
               ObsYs[i] * ObsMs[i], ObsXs[i] * ObsMs[i]
            )
            ObsMs_trans = 10**ObsMs[i]
            Q_obs.set_UVC(ObsYs_trans, ObsXs_trans, ObsMs_trans)


         # labels
         ax.set_title(ut.isoformat())

         # tweak layout
         plt.subplots_adjust(left=0.02, right=0.86, top=0.98, bottom=0.02)

         # save to file
         baseName = re.split('_.', fname)[0] # gets basename up to first '_', excluding suffixes
         plotFilename = (plot_dir+baseName+"_%04d%02d%02dT%02d%02d%02d.%s"%
                         (ut.year,ut.month,ut.day,ut.hour,ut.minute,ut.second,map_ext))
         
         # plt.show()

         try:
            plt.savefig(plotFilename, dpi=map_dpi)
         except IOError:
            # if plot_dir didn't exist, create and try again
            os.mkdir(plot_dir)
            plt.savefig(plotFilename, dpi=map_dpi)
