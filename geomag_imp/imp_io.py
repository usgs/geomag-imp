"""Input and Output for IMPs

This module provides methods for reading/writing interpolated magnetic-
field perturbations (IMPs) from/to the following file formats:

 - zipped collection of ASCII files formatted similarly to what as produced by
   Antti Pulkkinen's original Matlab-based program(s)
 - JSON (requires json_tricks from PiPy)
 - NASA's CDF (requires SpacePy from LANL)


NOTES:

- This module does not attempt to implement a Geomag-Algorithms IO factory. To
  do so requires designing a data model that is compatible with ObsPy Traces
  Streams, and it's not clear how appropriate these are for gridded data sets.
  This is not to say this is impossible, but it requires considerable thought.
  This module is just a starting point, and will likely evolve.

"""

# required imports
import numpy as np
import pkgutil
import io
#from StringIO import StringIO # not in Py3...io.BytesIO is preferred fix below
import gzip
import zipfile
import tempfile
import re
import os
import shutil
import warnings
import datetime as dt


# issue warnings if these are not available, but don't fail immediately
if pkgutil.find_loader('spacepy') is None or \
        pkgutil.find_loader('spacepy.pycdf') is None:
   print ("spacepy.pycdf package not available; " +
          "cannot import/export CDF files")
else:
   from spacepy import pycdf


if pkgutil.find_loader('json_tricks') is None:
   print ("JSON Tricks package not available; " +
          "cannot import/export JSON files")
else:
   import json_tricks.np as json_t


def read_imp_JSON(filename):
   """read in a imp JSON file
   Read gridded interpolated magnetic perturbations (IMPs) from a specially
   formatted JSON file.
   """
   with open(filename,'r') as fh:
      data = json_t.loads(fh.read())

   Epoch = data['Epoch']
   Latitude = data['Latitude']
   Longitude = data['Longitude']
   Radius = data['Radius']
   X = data['X']
   Y = data['Y']
   Z = data['Z']
   Label = data['Label']
   ObsLat = data['ObsLat']
   ObsLon = data['ObsLon']
   ObsRad = data['ObsRad']
   ObsX = data['ObsX']
   ObsY = data['ObsY']
   ObsZ = data['ObsZ']
   ObsFit = data['ObsFit']
   ObsName = data['ObsName']

   return (Epoch, (Latitude, Longitude, Radius), X, Y, Z, Label,
                  (ObsLat, ObsLon, ObsRad), ObsX, ObsY, ObsZ, ObsFit, ObsName)


def write_imp_CDF(Epoch, lat_lon_r, X, Y, Z, Label,
                  olat_olon_or, ObsX, ObsY, ObsZ, ObsFit, ObsName,
                  filename='impOut.json'):
# def write_imp_json(Epoch, (Latitude, Longitude), X, Y, Z, Label,
#                            (ObsLat, ObsLon), ObsFit, ObsName,
#                            filename='impOut.json'):
   """Write imp files
   Write gridded interpolated magnetic perturbations (IMPs) to a JSON file.

   TODO: make ObsName, ObsLat, ObsLon, and ObsFit optional
   TODO: figure out how to store metadata...really, need to figure out a
         imp metadata standard and use it for all inputs and outputs.
   """

   # unpack former tuple arguments (see PEP-3113)
   Latitude, Longitud, Radius = lat_lon_r
   ObsLat, ObsLon, ObsRad = olat_olon_or

   data = {}

   data['Epoch'] = (Epoch)
   data['Latitude'] = (Latitude)
   data['Longitude'] = (Longitude)
   data['Radius'] = (Radius)
   data['X'] = (X)
   data['Y'] = (Y)
   data['Z'] = (Z)
   data['Label'] = (Label)
   data['ObsLat'] = (ObsLat)
   data['ObsLon'] = (ObsLon)
   data['ObsRad'] = (ObsRad)
   data['ObsX'] = (ObsX)
   data['ObsY'] = (ObsY)
   data['ObsZ'] = (ObsZ)
   data['ObsFit'] = (ObsFit)
   data['ObsName'] = (ObsName)

   with open(filename, 'w') as fh:
      fh.write(json_t.dumps(data))



def read_imp_CDF(filename):
   """read in a imp CDF file
   read gridded interpolated magnetic perturbations (IMPs) from a specially
   formatted CDF file.

   TODO:
   """
   cdf = pycdf.CDF(filename)

   Epoch = cdf['Epoch'][:]
   Latitude = cdf['Latitude'][:]
   Longitude = cdf['Longitude'][:]
   Radius = cdf['Radius'][:]
   X = cdf['X'][:]
   Y = cdf['Y'][:]
   Z = cdf['Z'][:]
   Label = cdf['Label'][:]
   ObsLat = cdf['ObsLat'][:]
   ObsLon = cdf['ObsLon'][:]
   ObsRad = cdf['ObsRad'][:]
   ObsX = cdf['ObsX'][:]
   ObsY = cdf['ObsY'][:]
   ObsZ = cdf['ObsZ'][:]
   ObsFit = cdf['ObsFit'][:]
   ObsName = cdf['ObsName'][:]

   return (Epoch, (Latitude, Longitude, Radius), X, Y, Z, Label,
                  (ObsLat, ObsLon, ObsRad), ObsX, ObsY, ObsZ, ObsFit, ObsName)


def write_imp_CDF(Epoch, lat_lon_r, X, Y, Z, Label,
                  olat_olon_or, ObsX, ObsY, ObsZ, ObsFit, ObsName,
                  filename='impOut.cdf'):
   """Write imp files
   Write gridded interpolated magnetic perturbations (IMPs) to a nominally
   ISTP-compliant CDF file.

   TODO: make Obs* optional
   """
   # unpack former tuple arguments (see PEP-3113)
   Latitude, Longitud, Radius = lat_lon_r
   ObsLat, ObsLon, ObsRad = olat_olon_or

   # create a new CDF file object
   try:
      cdf = pycdf.CDF(filename, '')
   except pycdf.CDFError:
      warnings.warn(
         "File with same name exists, overwriting previous backup if it exists"
      )
      os.rename(filename, filename + ".BAK")
      cdf = pycdf.CDF(filename, '')

   cdf.compress(pycdf.const.GZIP_COMPRESSION)

   #
   # add global attributes (taken from an ISTP-compliant CDF; should re-
   #                        validate to see if actually compliant)
   #
   cdf.attrs['Project'] = "IMAG>Intermagnet"
   cdf.attrs['Source_name'] = "USGS>U.S. Geological Survey Geomagnetism Program"
   cdf.attrs['Discipline'] = ""
   cdf.attrs['Data_type'] = "K0>Key Parameter"
   cdf.attrs['Descriptor'] = "IMP>Interpolated Magnetic Perturbations"
   cdf.attrs['File_naming_convention'] = "source_datatype_descriptor_yyyyMMdd"
   cdf.attrs['Data_version'] = "01"
   cdf.attrs['PI_name'] = "E. Joshua Rigler"
   cdf.attrs['PI_affiliation'] = "U.S. Geological Survey"
   cdf.attrs['TEXT'] = ""
   cdf.attrs['Instrument_type'] = "Ground-Based Magnetometers, Riometers Sounders"
   cdf.attrs['Mission_group'] = "Ground-Based Investigations"
   cdf.attrs['Logical_source'] = filename.rsplit('.', 1)[0] # trim file suffix
   cdf.attrs['Logical_file_id'] = filename.rsplit('.', 1)[0] # trim file suffix
   cdf.attrs['Logical_source_description'] = ""
   cdf.attrs['Time_resolution'] = ""
   cdf.attrs['Rules_of_use'] = ""
   cdf.attrs['Generated_by'] = ""
   cdf.attrs['Generation_date'] = dt.date.today().isoformat()
   cdf.attrs['Acknowledgement'] = ""
   cdf.attrs['MODS'] = ""
   cdf.attrs['ADID_ref'] = ""
   cdf.attrs['LINK_TEXT'] = ""
   cdf.attrs['LINK_TITLE'] = ""
   cdf.attrs['HTTP_LINK'] = ""
   cdf.attrs['TITLE'] = "Interpolated Geomagnetic Data"
   cdf.attrs['FormatDescription'] = ("Geographic distribution of geomagnetic " +
                                     "vectors interpolated from observations, " +
                                     "plus original observations")
   cdf.attrs['FormatVersion'] = "1.1"

   cdf.attrs['TermsOfUse'] = ("INTERMAGNET terms of use are probably overly-" +
                              "restrictive for this, figure something else out")

   cdf.attrs['Institution'] = "U.S. Geological Survey Geomagnetism Program"
   cdf.attrs['Source'] = "U.S. Geological Survey Geomagnetism Program"


   #
   # add variables and attributes
   #

   # time stamps
   cdf.new('Epoch', data=Epoch, type=pycdf.const.CDF_EPOCH,
           recVary=True, dimVarys=[])
   cdf['Epoch'].attrs['CATDESC'] = "Default time"
   cdf['Epoch'].attrs['FIELDNAM'] = "EPOCH"
   cdf['Epoch'].attrs['FILLVAL'] = dt.datetime(9999,12,31,23,59,59,999999)
   cdf['Epoch'].attrs['LABLAXIS'] = "Epoch"
   cdf['Epoch'].attrs['MONOTON'] = "INCREASE"
   cdf['Epoch'].attrs['SCALETYP'] = "linear"
   cdf['Epoch'].attrs['TIME_BASE'] = "0 AD"
   cdf['Epoch'].attrs['UNITS'] = "ms"
   cdf['Epoch'].attrs['VALIDMAX'] = dt.datetime(202,12,31,23,59,59,999999)
   cdf['Epoch'].attrs['VALIDMIN'] = dt.datetime(1990,1,1,0,0,0,0)
   cdf['Epoch'].attrs['VAR_TYPE'] = "support_data"


   # real observations used for interpolation
   cdf.new('ObsName', data=ObsName, type=pycdf.const.CDF_CHAR,
           recVary=False)
   cdf['ObsName'].attrs['CATDESC'] = "ObsName"
   cdf['ObsName'].attrs['FIELDNAM'] = "ObsName"
   cdf['ObsName'].attrs['FORMAT'] = "A11"
   cdf['ObsName'].attrs['LABLAXIS'] = "ObsName"
   cdf['ObsName'].attrs['VAR_NOTES'] = "Name of magnetic observatory"
   cdf['ObsName'].attrs['VAR_TYPE'] = "metadata"

   cdf.new('ObsLat', data=ObsLat, type=pycdf.const.CDF_REAL4,
           recVary=False)
   cdf['ObsLat'].attrs['CATDESC'] = "ObsLat"
   cdf['ObsLat'].attrs['FIELDNAM'] = "ObsLat"
   cdf['ObsLat'].attrs['FILLVAL'] = -1e+31
   cdf['ObsLat'].attrs['FORMAT'] = "E12.2"
   cdf['ObsLat'].attrs['LABLAXIS'] = "ObsLat"
   cdf['ObsLat'].attrs['SCALETYP'] = "linear"
   cdf['ObsLat'].attrs['VALIDMAX'] = 90.0
   cdf['ObsLat'].attrs['VALIDMIN'] = -90.0
   cdf['ObsLat'].attrs['VAR_NOTES'] = "Latitude of ObsName observatory"
   cdf['ObsLat'].attrs['VAR_TYPE'] = "support_data"

   cdf.new('ObsLon', data=ObsLon, type=pycdf.const.CDF_REAL4,
           recVary=False)
   cdf['ObsLon'].attrs['CATDESC'] = "ObsLon"
   cdf['ObsLon'].attrs['FIELDNAM'] = "ObsLon"
   cdf['ObsLon'].attrs['FILLVAL'] = -1e+31
   cdf['ObsLon'].attrs['FORMAT'] = "E12.2"
   cdf['ObsLon'].attrs['LABLAXIS'] = "ObsLon"
   cdf['ObsLon'].attrs['SCALETYP'] = "linear"
   cdf['ObsLon'].attrs['VALIDMAX'] = 360.0
   cdf['ObsLon'].attrs['VALIDMIN'] = -180.0
   cdf['ObsLon'].attrs['VAR_NOTES'] = "Longitude of ObsName observatory (degrees)"
   cdf['ObsLon'].attrs['VAR_TYPE'] = "support_data"

   cdf.new('ObsRad', data=ObsRad, type=pycdf.const.CDF_REAL4,
           recVary=False)
   cdf['ObsRad'].attrs['CATDESC'] = "ObsRad"
   cdf['ObsRad'].attrs['FIELDNAM'] = "ObsRad"
   cdf['ObsRad'].attrs['FILLVAL'] = -1e+31
   cdf['ObsRad'].attrs['FORMAT'] = "E12.2"
   cdf['ObsRad'].attrs['LABLAXIS'] = "ObsLon"
   cdf['ObsRad'].attrs['SCALETYP'] = "linear"
   cdf['ObsRad'].attrs['VALIDMAX'] = 1e+31
   cdf['ObsRad'].attrs['VALIDMIN'] = 0
   cdf['ObsRad'].attrs['VAR_NOTES'] = "Radius of ObsName observatory (meters)"
   cdf['ObsRad'].attrs['VAR_TYPE'] = "support_data"

   cdf.new('ObsX', data=ObsX, type=pycdf.const.CDF_REAL8,
           recVary=True)
   cdf['ObsX'].attrs['CATDESC'] = "X component of observed magnetic field"
   cdf['ObsX'].attrs['DEPEND_0'] = "Epoch"
   cdf['ObsX'].attrs['DEPEND_1'] = "Latitude"
   cdf['ObsX'].attrs['DEPEND_2'] = "Longitude"
   cdf['ObsX'].attrs['DISPLAY_TYPE'] = 'no_plot'
   cdf['ObsX'].attrs['ElemRec'] = 'X'
   cdf['ObsX'].attrs['FIELDNAM'] = "Geomagnetic Field Element 1"
   cdf['ObsX'].attrs['FILLVAL'] = -1e+31
   cdf['ObsX'].attrs['FORMAT'] = "F12.4"
   cdf['ObsX'].attrs['OrigFreq'] = 99999.0
   cdf['ObsX'].attrs['SCALETYP'] = "linear"
   cdf['ObsX'].attrs['SampPer'] = 60.0
   cdf['ObsX'].attrs['UNITS'] = 'nT'
   cdf['ObsX'].attrs['VALIDMAX'] = 88000.0
   cdf['ObsX'].attrs['VALIDMIN'] = -88000.0
   cdf['ObsX'].attrs['VAR_NOTES'] = "X component points toward geographic north"
   cdf['ObsX'].attrs['VAR_TYPE'] = "data"

   cdf.new('ObsY', data=ObsY, type=pycdf.const.CDF_REAL8,
           recVary=True)
   cdf['ObsY'].attrs['CATDESC'] = "Y component of observed magnetic field"
   cdf['ObsY'].attrs['DEPEND_0'] = "Epoch"
   cdf['ObsY'].attrs['DEPEND_1'] = "Latitude"
   cdf['ObsY'].attrs['DEPEND_2'] = "Longitude"
   cdf['ObsY'].attrs['DISPLAY_TYPE'] = 'no_plot'
   cdf['ObsY'].attrs['ElemRec'] = 'Y'
   cdf['ObsY'].attrs['FIELDNAM'] = "Geomagnetic Field Element 2"
   cdf['ObsY'].attrs['FILLVAL'] = -1e+31
   cdf['ObsY'].attrs['FORMAT'] = "F12.4"
   cdf['ObsY'].attrs['OrigFreq'] = 99999.0
   cdf['ObsY'].attrs['SCALETYP'] = "linear"
   cdf['ObsY'].attrs['SampPer'] = 60.0
   cdf['ObsY'].attrs['UNITS'] = 'nT'
   cdf['ObsY'].attrs['VALIDMAX'] = 88000.0
   cdf['ObsY'].attrs['VALIDMIN'] = -88000.0
   cdf['ObsY'].attrs['VAR_NOTES'] = "Y component points toward geographic east"
   cdf['ObsY'].attrs['VAR_TYPE'] = "data"

   cdf.new('ObsZ', data=ObsZ, type=pycdf.const.CDF_REAL8,
           recVary=True)
   cdf['ObsZ'].attrs['CATDESC'] = "Z component of observed magnetic field"
   cdf['ObsZ'].attrs['DEPEND_0'] = "Epoch"
   cdf['ObsZ'].attrs['DEPEND_1'] = "ObsLat"
   cdf['ObsZ'].attrs['DEPEND_2'] = "ObsLon"
   cdf['ObsZ'].attrs['DISPLAY_TYPE'] = 'no_plot'
   cdf['ObsZ'].attrs['ElemRec'] = 'Z'
   cdf['ObsZ'].attrs['FIELDNAM'] = "Geomagnetic Field Element 3"
   cdf['ObsZ'].attrs['FILLVAL'] = -1e+31
   cdf['ObsZ'].attrs['FORMAT'] = "F12.4"
   cdf['ObsZ'].attrs['OrigFreq'] = 99999.0
   cdf['ObsZ'].attrs['SCALETYP'] = "linear"
   cdf['ObsZ'].attrs['SampPer'] = 60.0
   cdf['ObsZ'].attrs['UNITS'] = 'nT'
   cdf['ObsZ'].attrs['VALIDMAX'] = 88000.0
   cdf['ObsZ'].attrs['VALIDMIN'] = -88000.0
   cdf['ObsZ'].attrs['VAR_NOTES'] = "Z component points toward center of Earth"
   cdf['ObsZ'].attrs['VAR_TYPE'] = "data"

   cdf.new('ObsFit', data=ObsFit, type=pycdf.const.CDF_INT1,
           recVary=True)
   cdf['ObsFit'].attrs['CATDESC'] = "ObsFit"
   cdf['ObsFit'].attrs['DEPEND_0'] = "Epoch"
   cdf['ObsFit'].attrs['DEPEND_1'] = "ObsName"
   cdf['ObsFit'].attrs['FIELDNAM'] = "ObsFit"
   cdf['ObsFit'].attrs['FILLVAL'] = -128
   cdf['ObsFit'].attrs['FORMAT'] = "I3"
   cdf['ObsFit'].attrs['LABLAXIS'] = "ObsFit"
   cdf['ObsFit'].attrs['SCALETYP'] = "linear"
   cdf['ObsFit'].attrs['VALIDMAX'] = 9
   cdf['ObsFit'].attrs['VALIDMIN'] = 0
   cdf['ObsFit'].attrs['VAR_NOTES'] = ("Time-varying flag specifying number " +
                                       "(0..9; to accommodate ascii formats) " +
                                       "of ObsName's components used for " +
                                       "inversion")
   cdf['ObsFit'].attrs['VAR_TYPE'] = "support_data"


   # "virtual observatories", or the interpolated data
   # real observations used for interpolation
   cdf.new('Label', data=Label, type=pycdf.const.CDF_CHAR,
           recVary=False)
   cdf['Label'].attrs['CATDESC'] = "Label"
   cdf['Label'].attrs['FIELDNAM'] = "Label"
   cdf['Label'].attrs['FORMAT'] = "A11"
   cdf['Label'].attrs['LABLAXIS'] = "Label"
   cdf['Label'].attrs['VAR_NOTES'] = "Label for virtual magnetic observatory"
   cdf['Label'].attrs['VAR_TYPE'] = "metadata"

   cdf.new('Latitude', data=Latitude, type=pycdf.const.CDF_REAL4,
           recVary=False)
   cdf['Latitude'].attrs['CATDESC'] = "Latitude"
   cdf['Latitude'].attrs['FIELDNAM'] = "Latitude"
   cdf['Latitude'].attrs['FILLVAL'] = -1e+31
   cdf['Latitude'].attrs['FORMAT'] = "E12.2"
   cdf['Latitude'].attrs['LABLAXIS'] = "Lat_N"
   cdf['Latitude'].attrs['SCALETYP'] = "linear"
   cdf['Latitude'].attrs['VALIDMAX'] = 90.0
   cdf['Latitude'].attrs['VALIDMIN'] = -90.0
   cdf['Latitude'].attrs['VAR_NOTES'] = "Latitude of virtual magnetic observatories"
   cdf['Latitude'].attrs['VAR_TYPE'] = "support_data"

   cdf.new('Longitude', data=Longitude, type=pycdf.const.CDF_REAL4,
           recVary=False)
   cdf['Longitude'].attrs['CATDESC'] = "Longitude"
   cdf['Longitude'].attrs['FIELDNAM'] = "Longitude"
   cdf['Longitude'].attrs['FILLVAL'] = -1e+31
   cdf['Longitude'].attrs['FORMAT'] = "E12.2"
   cdf['Longitude'].attrs['LABLAXIS'] = "Lon_E"
   cdf['Longitude'].attrs['SCALETYP'] = "linear"
   cdf['Longitude'].attrs['VALIDMAX'] = 360.0
   cdf['Longitude'].attrs['VALIDMIN'] = -180.0
   cdf['Longitude'].attrs['VAR_NOTES'] = "Longitude of virtual magnetic observatories"
   cdf['Longitude'].attrs['VAR_TYPE'] = "support_data"

   cdf.new('Radius', data=Radius, type=pycdf.const.CDF_REAL4,
           recVary=False)
   cdf['Radius'].attrs['CATDESC'] = "Radius"
   cdf['Radius'].attrs['FIELDNAM'] = "Radius"
   cdf['Radius'].attrs['FILLVAL'] = -1e+31
   cdf['Radius'].attrs['FORMAT'] = "E12.2"
   cdf['Radius'].attrs['LABLAXIS'] = "Lon_E"
   cdf['Radius'].attrs['SCALETYP'] = "linear"
   cdf['Radius'].attrs['VALIDMAX'] = 1e+31
   cdf['Radius'].attrs['VALIDMIN'] = 0
   cdf['Radius'].attrs['VAR_NOTES'] = "Radius of interpolated sites"
   cdf['Radius'].attrs['VAR_TYPE'] = "support_data"

   cdf.new('X', data=X, type=pycdf.const.CDF_REAL8,
           recVary=True)
   cdf['X'].attrs['CATDESC'] = "X component of estimated magnetic field"
   cdf['X'].attrs['DEPEND_0'] = "Epoch"
   cdf['X'].attrs['DEPEND_1'] = "Latitude"
   cdf['X'].attrs['DEPEND_2'] = "Longitude"
   cdf['X'].attrs['DISPLAY_TYPE'] = 'no_plot'
   cdf['X'].attrs['ElemRec'] = 'X'
   cdf['X'].attrs['FIELDNAM'] = "Geomagnetic Field Element 1"
   cdf['X'].attrs['FILLVAL'] = -1e+31
   cdf['X'].attrs['FORMAT'] = "F12.4"
   cdf['X'].attrs['OrigFreq'] = 99999.0
   cdf['X'].attrs['SCALETYP'] = "linear"
   cdf['X'].attrs['SampPer'] = 60.0
   cdf['X'].attrs['UNITS'] = 'nT'
   cdf['X'].attrs['VALIDMAX'] = 88000.0
   cdf['X'].attrs['VALIDMIN'] = -88000.0
   cdf['X'].attrs['VAR_NOTES'] = "X component points toward geographic north"
   cdf['X'].attrs['VAR_TYPE'] = "data"

   cdf.new('Y', data=Y, type=pycdf.const.CDF_REAL8,
           recVary=True)
   cdf['Y'].attrs['CATDESC'] = "Y component of estimated magnetic field"
   cdf['Y'].attrs['DEPEND_0'] = "Epoch"
   cdf['Y'].attrs['DEPEND_1'] = "Latitude"
   cdf['Y'].attrs['DEPEND_2'] = "Longitude"
   cdf['Y'].attrs['DISPLAY_TYPE'] = 'no_plot'
   cdf['Y'].attrs['ElemRec'] = 'Y'
   cdf['Y'].attrs['FIELDNAM'] = "Geomagnetic Field Element 2"
   cdf['Y'].attrs['FILLVAL'] = -1e+31
   cdf['Y'].attrs['FORMAT'] = "F12.4"
   cdf['Y'].attrs['OrigFreq'] = 99999.0
   cdf['Y'].attrs['SCALETYP'] = "linear"
   cdf['Y'].attrs['SampPer'] = 60.0
   cdf['Y'].attrs['UNITS'] = 'nT'
   cdf['Y'].attrs['VALIDMAX'] = 88000.0
   cdf['Y'].attrs['VALIDMIN'] = -88000.0
   cdf['Y'].attrs['VAR_NOTES'] = "Y component points toward geographic east"
   cdf['Y'].attrs['VAR_TYPE'] = "data"

   cdf.new('Z', data=Z, type=pycdf.const.CDF_REAL8,
           recVary=True)
   cdf['Z'].attrs['CATDESC'] = "Z component of estimated magnetic field"
   cdf['Z'].attrs['DEPEND_0'] = "Epoch"
   cdf['Z'].attrs['DEPEND_1'] = "Latitude"
   cdf['Z'].attrs['DEPEND_2'] = "Longitude"
   cdf['Z'].attrs['DISPLAY_TYPE'] = 'no_plot'
   cdf['Z'].attrs['ElemRec'] = 'Z'
   cdf['Z'].attrs['FIELDNAM'] = "Geomagnetic Field Element 3"
   cdf['Z'].attrs['FILLVAL'] = -1e+31
   cdf['Z'].attrs['FORMAT'] = "F12.4"
   cdf['Z'].attrs['OrigFreq'] = 99999.0
   cdf['Z'].attrs['SCALETYP'] = "linear"
   cdf['Z'].attrs['SampPer'] = 60.0
   cdf['Z'].attrs['UNITS'] = 'nT'
   cdf['Z'].attrs['VALIDMAX'] = 88000.0
   cdf['Z'].attrs['VALIDMIN'] = -88000.0
   cdf['Z'].attrs['VAR_NOTES'] = "Z component points toward center of Earth"
   cdf['Z'].attrs['VAR_TYPE'] = "data"

   # close the CDF file
   cdf.close()


def read_imp_ASCII(filename):
   """Read Antti Pulkinnen's multi-file (ASCII) data.
   """

   # create a temporary directory
   tmpDir = tempfile.mkdtemp()

   # unzip filename to tmpDir
   with zipfile.ZipFile(filename, 'r') as inZip:
      inZip.extractall(tmpDir)

   # set filenames
   dt_file = os.path.join(tmpDir, 'DateTime.txt')
   location_file = os.path.join(tmpDir, 'LatLon.txt')
   bx_file = os.path.join(tmpDir, 'BX.txt')
   by_file = os.path.join(tmpDir, 'BY.txt')
   bz_file = os.path.join(tmpDir, 'BZ.txt')
   obx_file = os.path.join(tmpDir, 'obsBX.txt')
   oby_file = os.path.join(tmpDir, 'obsBY.txt')
   obz_file = os.path.join(tmpDir, 'obsBZ.txt')
   station_file = os.path.join(tmpDir, 'Stations.txt')

   DT = _read_antti_datetime(dt_file)

   Lat, Lon, Rad, Label = _read_antti_location(location_file)

   BX = _read_antti_component(bx_file)
   BY = _read_antti_component(by_file)
   BZ = _read_antti_component(bz_file)

   obsX = _read_antti_component(obx_file)
   obsY = _read_antti_component(oby_file)
   obsZ = _read_antti_component(obz_file)

   obsLat, obsLon, obsRad, obsInc, obsID = _read_antti_stations(station_file)

   shutil.rmtree(tmpDir)

   return (DT, (Lat, Lon, Rad), BX, BY, BZ, Label,
               (obsLat, obsLon, obsRad), obsX, obsY, obsZ, obsInc, obsID)


def write_imp_ASCII(DT, lat_lon_r, BX, BY, BZ, Label,
                    olat_olon_or, obsX, obsY, obsZ, obsInc, obsID,
                    filename='impOut.zip'):

# def write_antti(DT, Lat, Lon, BX, BY, BZ, Label,
#                 obsLat, obsLon, obsInc, obsID,
#                 dt_file = 'DateTime.txt.gz',
#                 location_file = 'LatLon.txt.gz',
#                 bx_file = 'BX.txt.gz',
#                 by_file = 'BY.txt.gz',
#                 bz_file = 'BZ.txt.gz',
#                 station_file = 'Stations.txt.gz'):
   """
   Write Antti Pulkinnen's multi-file (ASCII) data to a zipfile.
   """

   # unpack former tuple arguments (see PEP-3113)
   Lat, Lon, Rad = lat_lon_r
   obsLat, obsLon, obsRad = olat_olon_or

   # create a temporary directory
   tmpDir = tempfile.mkdtemp()

   # set filenames
   dt_file = os.path.join(tmpDir, 'DateTime.txt')
   location_file = os.path.join(tmpDir, 'LatLon.txt')
   bx_file = os.path.join(tmpDir, 'BX.txt')
   by_file = os.path.join(tmpDir, 'BY.txt')
   bz_file = os.path.join(tmpDir, 'BZ.txt')
   obx_file = os.path.join(tmpDir, 'obsBX.txt')
   oby_file = os.path.join(tmpDir, 'obsBY.txt')
   obz_file = os.path.join(tmpDir, 'obsBZ.txt')
   station_file = os.path.join(tmpDir, 'Stations.txt')

   # write out ASCII files
   _write_antti_datetime(DT, dt_file)
   _write_antti_location(Lat, Lon, Rad, Label, location_file)
   _write_antti_component(BX, 'X (northward) component', bx_file)
   _write_antti_component(BY, 'Y (eastward) component', by_file)
   _write_antti_component(BZ, 'Z (downward) component', bz_file)
   _write_antti_stations(obsLat, obsLon, obsRad, obsInc, obsID, station_file)

   # not a part of original ASCII format, but included for completeness
   _write_antti_component(obsX, 'observed X (northward) component', obx_file)
   _write_antti_component(obsY, 'observed Y (eastward) component', oby_file)
   _write_antti_component(obsZ, 'observed Z (downward) component', obz_file)

   # open up output zip file
   with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as outZip:
      outZip.write(dt_file, os.path.basename(dt_file))
      outZip.write(location_file, os.path.basename(location_file))
      outZip.write(bx_file, os.path.basename(bx_file))
      outZip.write(by_file, os.path.basename(by_file))
      outZip.write(bz_file, os.path.basename(bz_file))
      outZip.write(obx_file, os.path.basename(obx_file))
      outZip.write(oby_file, os.path.basename(oby_file))
      outZip.write(obz_file, os.path.basename(obz_file))
      outZip.write(station_file, os.path.basename(station_file))

   shutil.rmtree(tmpDir)

def _read_antti_datetime(dt_file):
   """
   Read datetimes from Antti Pulkinnen's DateTime.txt[.gz] file
   """
   # NOTE: genfromtxt() doesn't work with gzipped files as it should, so we
   #       unzip the file ourself, and use io.BytesIO to fake out genfromtext()
   if dt_file.split('.')[-1] == 'gz':
      ff = gzip.open(dt_file, 'r')
   else:
      ff = open(dt_file, 'r')

   sIO = io.BytesIO(ff.read())
   ff.close()

   ymdHMS = np.genfromtxt(sIO, comments="%")
   DT = np.array([dt.datetime(*elem) for elem in ymdHMS.astype('int')])
   sIO.close()

   return DT


def _write_antti_datetime(DT, dt_file):
   """
   Write datetimes into the ASCII format used by Antti Pulkinnen
   """
   if dt_file.split('.')[-1] == 'gz':
      ff = gzip.open(dt_file, 'w')
   else:
      ff = open(dt_file, 'w')

   ff.write("%% Date and time of the geoelectric field distribution. " +
            " Data produced on %s\n"%(dt.datetime.utcnow()))
   ff.write("%% \n")
   ff.write("%% This data comes together with files BX.txt, BY.txt, LatLon.txt" +
            " and Stations.txt. \n")
   ff.write("%% \n")
   ff.write("%% Contact:  \n")
   ff.write("%% \n")
   ff.write("%% The format of the data is as follows:\n")
   ff.write("%% \n")
   ff.write("%% year1 month1 day1 hour1 minute1 second1 \n")
   ff.write("%% year2 month2 day2 hour2 minute2 second2 \n")
   ff.write("%%  .     .      .     .      .      . \n")
   ff.write("%%  .     .      .     .      .      . \n")
   ff.write("%%  .     .      .     .      .      . \n")
   ff.write("%% \n")
   ff.write("\n")

   for d in DT:
      ff.write("%02.0f %02.0f %02.0f %02.0f %02.0f %02.0f\n"%
               (d.year, d.month, d.day, d.hour, d.minute, d.second))

   ff.close()



def _read_antti_component(component_file):
   """
   Read vector component from Antti Pulkinnen's [BX|BY|BZ].txt[.gz] file
   """
   # NOTE: genfromtxt() doesn't work with gzipped files as it should, so we
   #       unzip the file ourself, and use io.BytesIO to fake out genfromtext()
   if component_file.split('.')[-1] == 'gz':
      ff = gzip.open(component_file, 'r')
   else:
      ff = open(component_file, 'r')

   sIO = io.BytesIO(ff.read())
   ff.close()

   # read array
   component = np.genfromtxt(sIO, comments="%").T
   sIO.close()

   return component


def _write_antti_component(component, component_id, component_file):
   """
   Write vector components into the ASCII format used by Antti Pulkinnen.

   component - 2D matrix, rows for locations, columns for time steps
   component_ID - string describing component (e.g., 'X (northward) component')
   component_file - name of file to write out
   """
   if component_file.split('.')[-1] == 'gz':
      ff = gzip.open(component_file, 'w')
   else:
      ff = open(component_file, 'w')

   ff.write("%%%% %s of the magnetic field distribution."%component_id +
            " Data produced on %s\n"%dt.datetime.utcnow())
   ff.write("%% \n")
   ff.write("%% This data comes together with files DateTime.txt, LatLon.txt" +
            " and Stations.txt. \n")
   ff.write("%% \n")
   ff.write("%% Contact:  \n")
   ff.write("%% \n")
   ff.write("%% The format of the data is as follows:\n")
   ff.write("%% \n")
   ff.write("%% Comp(loc1,t1) Comp(loc1,t2) Comp(loc1,t3) ... \n")
   ff.write("%% Comp(loc2,t1) Comp(loc2,t2) Comp(loc2,t3) ... \n")
   ff.write("%%      .             .             . \n")
   ff.write("%%      .             .             . \n")
   ff.write("%%      .             .             . \n")
   ff.write("%% \n")
   ff.write("\n")

   fmt = ''.join(['%02.4f ' for row in component] + ['\n'])
   for loc in component.T:
      ff.write(fmt%tuple(loc))
   ff.close()


def _read_antti_location(location_file):
   """
   Read latitudes, longitudes, and (possibly blank) IDs from Antti Pulkinnen's
   latlon.txt[.gz] file
   """
   # NOTE: genfromtxt() doesn't work with gzipped files as it should, so we
   #       unzip the file ourself, and use io.BytesIO to fake out genfromtext()
   if location_file.split('.')[-1] == 'gz':
      ff = gzip.open(location_file, 'r')
   else:
      ff = open(location_file, 'r')

   sIO = io.BytesIO(ff.read())
   ff.close()

   # read LatLon array (with optional labels...
   #  either all have labels, or none, else genfromtxt() chokes)
   lll = zip(*np.atleast_1d(np.genfromtxt(
      sIO, comments="%", dtype=None,
      names=['latReal','lonReal','radReal','labelString']
   )))

   # handles older style(s) with no radius and/or labels
   if len(lll) > 3:
      lat, lon, rad = np.array(lll[0:3])
      label = np.array(lll[3])
   elif len(lll) > 2:
      lat, lon, rad = np.array(lll[0:3])
      if isinstance(rad[0], basestring):
         label = rad
         rad = np.ones(lat.shape)
      else:
         label = np.tile('', lat.shape)
   elif len(lll) == 2:
      lat, lon = np.array(lll[0:2])
      rad = np.ones(lat.shape)
      label = np.tile('', lat.shape)
   else:
      raise Exception('Requires (at least) latitude and longitude')

   return lat, lon, rad, label


def _write_antti_location(lat, lon, rad, label, location_file):
   """
   Write latitudes, longitudes, radius, and IDs of the locations of vector
   components into the ASCII format used by Antti Pulkinnen
   """
   if location_file.split('.')[-1] == 'gz':
      ff = gzip.open(location_file, 'w')
   else:
      ff = open(location_file, 'w')

   ff.write("%% Geographic coordinates of the geoelectric field distribution " +
            " Data produced on %s\n"%(dt.datetime.utcnow()))
   ff.write("%% \n")
   ff.write("%% This data comes together with files DateTime.txt, B?.txt," +
            " and Stations.txt. \n")
   ff.write("%% \n")
   ff.write("%% Contact:  \n")
   ff.write("%% \n")
   ff.write("%% The format of the data is as follows:\n")
   ff.write("%% \n")
   ff.write("%% lat1 lon1 rad1 label1 \n")
   ff.write("%% lat2 lon2 rad2 label2 \n")
   ff.write("%%  .     .      .  \n")
   ff.write("%%  .     .      .  \n")
   ff.write("%%  .     .      .  \n")
   ff.write("%% \n")
   ff.write("\n")

   for l in range(len(lat)):
      ff.write("%02.2f %02.2f %08e %s\n"%(lat[l], lon[l], rad[l], label[l]))

   ff.close()


def _read_antti_stations(station_file):
   """
   Function to parse contents of Antti Pulkinnen's Stations.txt[.gz] file.
   """
   if station_file.split('.')[-1] == 'gz':
      ff = gzip.open(station_file, 'r')
   else:
      ff = open(station_file, 'r')

   sIO = io.BytesIO(ff.read())
   ff.close()

   # extract and convert single line with observatory IDs
   obsList = []
   llList = []
   incList = []
   nObs = 0
   nLL = 0
   nInc = 0
   for line in sIO:
      if re.search("^%", line):
         # skip comments
         continue

      if re.search(r"^\s*$", line):
         # skip blank lines
         continue

      # first line of consequence should be a list of quoted strings holding
      # observatory IDs for observatories considered in this solution; convert
      # to a list of strings
      if len(obsList) == 0:
         obsList =  re.sub('\'', '', line).split()
         nObs = len(obsList)
         continue

      # assume next nobs lines read are observatory locations
      if nLL < nObs:
         llList.append([float(elem) for elem in line.split()])
         nLL = nLL+1
         continue

      # assume next nobs lines read are observatory inclusion (boolean) lists
      if nInc < nObs:
         #incList.append(line.strip())
         incList.append([int(elem) for elem in line.strip()])
         nInc = nInc+1
         continue

   # close sIO
   sIO.close()

   if len(llList) > 2:
      obsLat, obsLon, obsRad = zip(*llList)
   elif len(llList) == 2:
      obsLat, obsLon = zip(*llList)
      obsRad = np.ones(obsLat.shape)
   else:
      raise Exception('Requires (at least) latitude and longitude')

   obsInc = zip(*incList)

   return (np.array(obsLat), np.array(obsLon), np.array(obsRad),
           np.array(obsInc), np.array(obsList))


def _write_antti_stations(obs_lat, obs_lon, obs_rad, obs_inc, obs_id,
                          station_file):
   """
   Write latitudes, longitudes, radii, and IDs of the stations used to generate
   interpolated magnetic vector data into ASCII format used by Antti Pulkinnen,
   plus a table of flags indicating if the quality of the observatory was used
   for inversion at a particular time step.
   """
   if station_file.split('.')[-1] == 'gz':
      ff = gzip.open(station_file, 'w')
   else:
      ff = open(station_file, 'w')

   ff.write("%% Geographic coordinates and ID of stations used to generate" +
            " SECS-interpolated magnetic vector comonents. " +
            " Data produced on %s\n"%(dt.datetime.utcnow()))
   ff.write("%% \n")
   ff.write("%% This data comes together with files DateTime.txt, B?.txt," +
            " and LatLon.txt. \n")
   ff.write("%% \n")
   ff.write("%% Contact:  \n")
   ff.write("%% \n")
   ff.write("%% The format of the data is as follows:\n")
   ff.write("%% \n")
   ff.write("%% First row: the list of station codes used in SECS" +
            " calculations followed by the geographic coordinates of" +
            " the stations:\n")
   ff.write("%% \n")
   ff.write("%% lat1 lon1 rad1 \n")
   ff.write("%% lat2 lon2 rad2 \n")
   ff.write("%%  .     .   \n")
   ff.write("%%  .     .   \n")
   ff.write("%%  .     .   \n")
   ff.write("%% \n")
   ff.write("%% The rest of the data are an array of integers indicating the" +
            " quality [0-9] of station in the first row used in SECS inversion:\n")
   ff.write("%% \n")
   ff.write("%% bool(station1,t1) bool(station1,t2) bool(station1,t3) ...\n")
   ff.write("%% bool(station2,t1) bool(station2,t2) bool(station2,t3) ...\n")
   ff.write("\n")

   # write observatory ids as single line
   for obs in obs_id:
      ff.write("%s "%obs)

   ff.write("\n")
   ff.write("\n")

   # write observatory locations
   for l in range(len(obs_lat)):
      ff.write("%03.2f %03.2f %08e\n"%(obs_lat[l], obs_lon[l], obs_rad[l]))

   ff.write("\n")

   # write quality factor array
   fmt = ''.join(['%01.0f' for row in obs_inc] + ['\n'])
   for loc in obs_inc.T:
      ff.write(fmt%tuple(loc))

   ff.close()
