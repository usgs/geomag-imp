USGS Geomagnetism Program<br/>Interpolated Magnetic Perturbations (Geomag-IMP)
============================================================

**This project has been migrated to https://code.usgs.gov/ghsc/geomag/geomag-imp,<br>**
**no further updates will be applied on GitHub.**


# Background

Geomag-IMP is an open-source Python library for interpolating magnetic field
perturbations onto geographic coordinates.

Geomag-IMP provides two (for now) classes for interpolating magnetic fields:

- a thin wrapper to the Gaussian Process regressor found in Scikit-learn machine
  learning package, a kind of optimal interpolation with little to no physical
  basis.
- an adaptation of a technique first described by Olaf Amm and Ari Viljanen in
  their 1999 EPS article ["Ionospheric disturbance magnetic field continuation
  from the ground to ionosphere using spherical elementary current systems"][1].
  The algorithm implemented here drew heavily from subsequent work by NASA's
  Antti Pulkinnen and the Finnish Meteorological Institute's Ari Viljanen,
  perhaps most significantly, the 2003 JGR article ["Ionospheric equivalent
  current distributions determined with the method of spherical elementary
  current systems"][2].

Geomag-IMP is built on top of well-known Python scientific packages like:
[NumPy][3], [SciPy][4], and [Matplotlib][5]; as well as more focused scientific
packages like: [ObsPy][6], the USGS's [Geomag-Algorithms][7], and [SpacePy][8]
(originally published by Los Alamos National Laboratory)

# Installation

To install Geomag-IMP, first be sure to have a modern version of Python (>=3) 
properly installed on your computer. Any Python distribution should work, but
instructions are only provided to get this up-and-running using Miniconda from
Continuum Analytics (http://conda.pydata.org/miniconda.html).

1. install **Miniconda** distribution for your operating system, and set up a
   pristine Conda environment using (mostly) the conda-forge channel:
  - conda config --add channels conda-forge
  - conda create -n G-IMP_env python=3
  - conda activate G-IMP_env
2. install most Python packages (and their dependencies) using `conda` command:  
   some packages must be installed using `pip`
  - Minimum requirements:
    - `conda install obspy` (with dependencies, required by geomag-algorithms)
    - `conda install pycurl libcurl libssh2` (required for geomag-algorithms)
    - `conda install scikit-learn` (required for Gaussian Process Regression)
    - `conda install cartopy` (required for test and diagnostic plots)
    - `pip install git+https://github.com/usgs/geomag-algorithms.git` (required
       for working with USGS geomagnetic data and services)
  - Extras:
    - `conda install json_tricks` (robust data serialization for NumPy objects)
    - `pip install spacepy` (moslty for the included pycdf module)
       - requires [NASA's common data format (CDF) library][9] to be 
         properly installed and configured
       - spacepy can be tricky to install, but there are platform-specific
         tricks; best to refer to [online docs][8]
    - `conda install Jupyter` (needed re-run Notebook)
       - `conda install netcdf4` (read/write NetCDF4 test data)
       - `conda install xarray` (read/write NetCDF4 test data more easily
       - `conda install ffmpeg` (generate animations in Notebook)<br/>
         (the Notebook should be updated to use Cartopy; cells that generate
          maps will fail unless obsolete [Basemap][13] package is installed)
3. clone, then install geomag-imp using the setup.py file in the top folder of 
   this package:
  - `git clone git@github.com:usgs/geomag-imp.git`
  - `cd geomag-imp`
  - `python setup.py install`
  <br/>(can also just add the geomag-imp folder to $PYTHONPATH)



# Usage

## Application Program Interface (API)

The heart of Geomag-IMP is a collection of Python modules that take as input a
set of geographically located magnetic perturbations, and interpolate between
them to arbitrary locations. For now, relevant documentation is in the source:

- [*geomag_imp*](./geomag_imp/__init__.py) - a wrapper module that provides
  access to:  

  - [*geomag_imp.gpKernels*][11] - an alias to scikit-learn's Gaussian Process
    Kernel module
  - [*geomag_imp.gpRegressor*][12] - an alias to scikit-learn's Gaussian Process
    Regressor class
  - [*geomag_imp.secs*](./geomag_imp/secs.py) - object class designed to hold
    spherical elementary current system (SECS) used to interpolate magnetic
    perturbations. It is analagous (but not the same as) a GP kernel.
  - [*geomag_imp.secsRegressor*](./geomag_imp/secs.py) - object class that does
    the actual interpolation of magnetic perturbations. It optimizes the secs
    object to minimize the discrepancy between measured and secs-predicted
    perturbations. It is analagous to (but not the same as) a GP regressor.
  - [*geomag_imp.imp_io*](./geomag_imp/imp_io.py) - module with miscellaneous
    IO methods. Some have non-standard dependencies. Eventually, this will be
    refactored to work with a more formal data model.

## Command Line Interface (CLI)

Geomag-IMP does not currently provide a full-featured application, but it
provides a small set of useful "demonstration" scripts that may be adapted to
relatively simple operational requirements (by modifying parameters set at the
top of the scripts). These are:

- [*make_svsqdist.py*](./bin/make_svsqdist.py) - downloads XYZF geomagnetic
  data, then decomposes into  secular variation (sv), solar quiet (sq), and
  disturbance (dt) components. Also generates a time-dependent standard
  deviation (sd) of disturbance, which can be a useful diagnostic.

  Only a list of valid observatories can be specified on the command line, but
  additional configuration options are available and explained at the beginning
  of the script.

  An example call:
  ```
  > python make_svsqdist.py BOU
  ```
  ...generates
  [IAGA2002][10]
  output files:
  ```
  ./Disturbance/bouYYYYMMDDv_dt_min.min
  ./StandardDeviation/bouYYYYMMDDv_sd_min.min
  ./SolarQuiet/bouYYYYMMDDv_sq_min.min
  ./SecularVariation/bouYYYYMMDDv_sv_min.min
  ```
  ...and state files for each channel:
  ```
  ./PriorState/SqDistState_BOUX
  ./PriorState/SqDistState_BOUY
  ./PriorState/SqDistState_BOUZ
  ./PriorState/SqDistState_BOUF
  ```
  These allow the script to be run repeatedly, each time generating new sv, sq,
  dt, and sd data from the previous run up to the "present", assuming they are
  configured properly. Note that, eventually, the USGS will provide a real time
  data service for these data, so *make_svsqdist.py* will no longer be
  necessary.

- [*make_imp_secs.py*](./bin/make_imp_secs.py) - this reads sq and dt data
  produced by *make_svsqdist.py* for a collection of magnetic observatories,
  combines them, then generates gridded maps of synthetic magnetic perturbations
  using the secsRegressor.

  An example call:
  ```
  > python make_imp_secs.py BOU CMO FRN
  ```
  ...generates a single file:
  ```
  imp_YYYY-MM-DDThh:mm:ss--YYYY-MM-DDThh:mm:ss.ext
  ```
  This file holds the original observatory data for the input observatories
  specified on the command line, in addition to the gridded data. The interval
  in the filename reflects the interval of data processed. The extension will
  correspond to the output format configured at the beginning of the script.

- [*make_imp_gp.py*](./bin/make_imp_gp.py) - similar to *make_imp_secs.py*, but
  uses Gaussian Process regressor instead of SECS. This is included mostly for
  validation purposes.

- [*make_impmaps.py*](./bin/make_impmaps.py) - this generates diagnostic plots
  of the horizontal magnetic vector field over North America. It takes as input
  the filename generated by *make_imp&ast;.py*, and simply processes the data
  held in that file. It generates PNG images, and places them in a *Plots/*
  subdirectory.

### CLI Demonstration

A **very** simple demonstration of how to use these tools is provided in the
[data/](./data/) folder. The shell script [run_me.sh](./data/run_me.sh) calls
Python scrips in the recommended sequence. The directories into which data is
placed are configurable from the respective Python scripts, but leaving them
unchanged will facilitate future diagnoses should anything go wrong.

[1]: https://www.terrapub.co.jp/journals/EPS/pdf/5106/51060431.pdf
[2]: http://onlinelibrary.wiley.com/doi/10.1029/2001JA005085/full
[3]: http://www.numpy.org/
[4]: https://www.scipy.org/
[5]: http://matplotlib.org/
[6]: https://github.com/obspy/obspy/wiki
[7]: https://github.com/usgs/geomag-algorithms
[8]: https://spacepy.github.io/install.html
[9]: http://cdaweb.gsfc.nasa.gov/pub/software/cdf/
[10]: https://www.ngdc.noaa.gov/IAGA/vdat/IAGA2002/iaga2002format.html
[11]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/gaussian_process/kernels.py
[12]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/gaussian_process/gpr.py
[13]: https://matplotlib.org/basemap/
