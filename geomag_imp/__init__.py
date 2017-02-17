"""
The GIMP Module provides tools for gridded interpolated magnetic field maps.
"""
import imp_io # read/write data from/to gimp data files
from .secs import secs # spherical elementary current system class
from .secs import secsRegressor # SECS regressor class
from sklearn.gaussian_process import kernels as gpKernels
from sklearn.gaussian_process import GaussianProcessRegressor as gpRegressor
#import sc # spherical cap regressor...not yet implemented
#import sh # spherical harmonic regressor...not yet implemented

__all__ = [
    'imp_io',
    'secs',
    'secsRegressor',
    'gpKernels',
    'gpRegressor' # ,
    #'sc',
    #'sh'
]
