"""Spherical Elementary Current System regression

This module uses elementary current systems to estimate ground interpolated
magnetic-field perturbations (IMPs).
"""

# required imports
import numpy as np
from scipy import linalg as la
from sklearn.base import BaseEstimator, RegressorMixin, clone
from copy import copy,deepcopy

class secs(object):
   """Spherical elementary current system (SECS)

   This simple object class holds the pole locations, amplitudes, and standard
   deviations of a system of spherical elementary currents. For details, refer
   to the 1999 EPS paper by Olaf Amm and Ari Viljanen entitled "Ionospheric
   disturbance magnetic field continuation from the ground to ionosphere using
   spherical elementary current systems."

   Parameters
   ----------
   lat_lon_r : array-like (shape(n_secs, 3), required)
      Geodetic latitudes (degrees north), longitude (degrees east), and
      radii (meters) of the poles of a system of SECs
      NOTE: this cannot be easily modified after initialization;
            re-init if you want to change lat_lon_r

   amps : array-like (shape(n_secs), optional)
      Amplitude(s) of spherical elementary currents (Amps)
      (default is None)

   amps_var : array-like (shape(n_secs), optional)
      Variance of estimated secs.
      Note: This is currently obtained from the truncated SVD used to solve for
            amps (allows us to get by with fewer observations than parameters).
            Truncated SVD generates a smoothed, and therefore biased, estimate
            of amps. So, while the variance of amps can-and-is still estimated
            analytically, it only captures the random portion of the secs'
            uncertainty, and misses the bias error entirely.
      (default is None)

   Notes
   -----
   For now this is limited to divergence-free SECs, but may one day include
   curl-free SECs as well.
   """
   def __init__(self, lat_lon_r, amps=None, amps_var=None):

      # enforce 3 columns for latitude, longitude, and radius
      if not (np.ndim(lat_lon_r) == 2 and np.shape(lat_lon_r)[1] == 3):
         raise ValueError("lat_lon_r must have 3 columns!")

      # probably should check dims for amps* if not None
      self.amps = amps
      self.amps_var = amps_var

      # this discourages manual changes to _lat_lon_r after init;
      self._lat_lon_r = lat_lon_r
      self._lat_lon_r.flags.writeable = False

   @property
   def lat_lon_r(self):
      """return _lat_lon_r...discourages manual replacement of lat_lon_r
      """
      return self._lat_lon_r

   @property
   def n_secs(self):
      """return number of elementary currents
      """
      return np.shape(self.lat_lon_r)[0]

   @property
   def latitude(self):
      """return array of latitudes
      """
      return self.lat_lon_r[:,0]

   @property
   def longitude(self):
      """return array of longitudes
      """
      return self.lat_lon_r[:,1]

   @property
   def radius(self):
      """return array of radii
      """
      return self.lat_lon_r[:,2]

   @property
   def unit_amps(self):
      """return column array with 1 Amp for each elementary current
      """
      return  np.ones(self.n_secs)

   def div_free(self, lat_lon_r=None):
      """STUB to generate divergence-free current vectors

      NOTE: I need to understand better if non-uniform radii make sense here.
            They can certainly be used for fitting B-fields, but how does one
            superimpose sheet currents at locations that are not coincident?
            Maybe just check r to see if all are identical, and if not, fail.

      Parameters
      ----------
      lat_lon_r : array-like (shape(n_sites, 3), optional)
         Geodetic latitudes (degrees north), longitude (degrees east), and
         radii (meters) of a sites at which to sample divergence-free currents
         (default is self.lat_lon_r)

      Returns
      -------
      array-like (shape(n_sites, 2))
         A 2-column array holding Jx (northward) and Jy (eastward) for each site
      """
      if lat_lon_r is None:
         lat_lon_r = self.lat_lon_r


class secsRegressor(BaseEstimator, RegressorMixin):
   """Spherical elementary current system (SECS) regression.

   This algorithm is based largely on a 1999 EPS paper by Olaf Amm and Ari
   Viljanen entitled "Ionospheric disturbance magnetic field continuation
   from the ground to ionosphere using spherical elementary current systems".

   It is a translation of Antti Pulkinnen's Matlab package for ECS inversion
   and ground magnetic-field interpolation, but refactored with an object-
   oriented structure that resembles Python's scikit-learn Gaussian Process
   Regressor (GPR) class.

   Parameters
   ----------
   secs : secs object (required)
      Contains geodetic coordinates, amplitudes, and other attributes of the
      basis SECS (see secs class docstring for details).

   epsilon : float (optional)
      Multiplied by the largest singular value obtained from SVD of the data
      matrix in order to regularize/smooth the estimated SECS amplitudes.
      (default is 5e-2)

   Attributes
   ----------
   secs_ : secs object
      Actual secs used for prediction. The structure of this secs is the same
      as the one passed as a parameter, but (re)optimized whenever changes are
      made to obs_lat_lon_r_, obs_Btheta_Bphi_Br_, or obs_sigma_Btheta_Bphi_Br_.

   obs_lat_lon_r_ : array-like (shape(n_stations, 3))
      Geodetic latitude (degrees north), longitude (degrees east), and
      radii (meters) of magnetic field observations used to train secs_.

   obs_Btheta_Bphi_Br_ : array-like (shape(n_stations, 3))
      theta (nT southward), phi (nT eastward), and r (nT upward) components of
      magnetic field vectors used to train secs_

   obs_sigma_Btheta_Bphi_Br_ : array-like (shape(n_stations, 3)
      Standard error of magnetic vector components; these weight observations
      with lower sigma values more, although a zero value will lead to undefined
      behavior. Setting an element of sigma to infinity (e.g., float('inf'))
      effectively removes the corresponding observation from the SVD inversion
      by giving it a zero-weight.

   obs_T_ : array_like (shape(n_stations * 3, n_secs))
      "Transformation matrix" containing geometric factors that describe the
      contribution of each unit-valued elementary current in secs (columns) to
      the vector components of each observation (rows). This is re-computed
      if there is a change in obs_lat_lon_r_.

   U_ : array_like (shape())
      Left singular vectors from singular value decomposition. This is re-
      computed when obs_lat_lon_r_ and/or obs_sigma_Btheta_Bphi_Br_ changes.

   S_ : array_like (shape())
      Singular values from singular value decomposition. This is recomputed
      when obs_lat_lon_r_ and/or obs_sigma_Btheta_Bphi_Br_ changes.

   Vh_ : array_like (shape())
      Right singular vectors from singular value decomposition. This is re-
      computed when obs_lat_lon_r_ and/or obs_sigma_Btheta_Bphi_Br_ changes.

   pred_lat_lon_r_ : array-like (shape(n_sites, 3))
      Geodetic latitude (degrees north), longitude (degrees east), and
      radii (meters) of desired magnetic field predictions.

   pred_T_ : array_like (shape(n_sites * 3, n_secs))
      "Transformation matrix" containing geometric factors that describe the
      contribution of each unit-valued elementary current in ecs (columns) to
      the vector components of each prediction (rows). This is re-computed
      if there is a change in pred_lat_lon_r_.

   pred_sigma_ : array_like (shape(n_sites, 3))
      This is not implemented yet, and since it cannot be calculated
      analytically, it will require a cpu-intensive Monte-Carlo simulation
      based on random perturbations of SECS. Perhaps we will define it as
      a property once the sample method is complete. -EJR 12/2016


   NOTES
   -----

   """

   def __init__(self, secs, epsilon):

      self.secs = secs
      self.epsilon = epsilon

      # there is no need for a user to __init__ these, but they need to exist
      # for the fit and/or predict methods to work the first time
      self.obs_lat_lon_r_ = None
      self.obs_Btheta_Bphi_Br_ = None
      self.obs_sigma_Btheta_Bphi_Br_ = None
      self.pred_lat_lon_r_ = None


   def fit(self, lat_lon_r, Btheta_Bphi_Br,
                 sigma_Btheta_Bphi_Br = None):
      """Fit secs to training data

      Given a set of observed magnetic fields, optimize secs_.

      Parameters
      ----------
      lat_lon_r : array_like (shape(n_stations, 3), required)
         Geodetic latitude (degrees north), longitude (degrees east), and
         radii (meters) of magnetic field observations used to train ecs_

      Btheta_Bphi_Br : array-like (shape(n_stations, 3), required)
         theta (nT southward), phi (nT eastward), and r (nT upward) components of
         magnetic field vectors used to train secs_

      sigma_Btheta_Bphi_Br : array-like (shape(n_stations, 3), optional)
         Standard error of magnetic vector components; these weight observations
         with lower sigma values more, although a zero value leads to undefined
         behavior. Setting an element of sigma to infinity (e.g., float('inf'))
         effectively removes the corresponding observation from SVD inversion
         by giving it a zero-weight.


      Returns
      -------
      self : returns instance of self
         I'm not sure why this is done, but I'm following the example of gpr.py

      Notes
      -----
      The inputs are used only if they are NOT the same as corresponding class
      attributes. This allows us to bypass re-estimation of cpu-intensive, but
      often highly redundant, operations unless absolutely necessary. This does
      not compare array values, but rather the array objects themselves, which
      is much quicker.

      """
      # much below relies on inputs being NumPy arrays
      # (copy=False make sure only a reference is assigned, if possible;
      #  this is necessary for the bypass/optimization logic below)
      lat_lon_r = np.array(lat_lon_r, copy=False)
      Btheta_Bphi_Br = np.array(Btheta_Bphi_Br, copy=False)


      if sigma_Btheta_Bphi_Br is None:
         if self.obs_sigma_Btheta_Bphi_Br_ is None:
            # default sigmas equal to 1
            sigma_Btheta_Bphi_Br = np.ones(Btheta_Bphi_Br.shape)
         else:
            sigma_Btheta_Bphi_Br = self.obs_sigma_Btheta_Bphi_Br_
      sigma_Btheta_Bphi_Br = np.array(sigma_Btheta_Bphi_Br, copy=False)


      # enforce 3 columns for latitude, longitude, and radius
      if not (np.ndim(lat_lon_r) == 2 and np.shape(lat_lon_r)[1] == 3):
         raise ValueError("lat_lon_r must have 3 columns!")
      # do same for other inputs

      # not sure the value in doing this, but it emulates gpr.py
      if self.secs is None:
         raise Exception('Default secs not implemented yet, must define secs')
      else:
         self.secs_ = copy(self.secs)


      # "M" and "N" here are consistent with Antti Pulkkinen's original ML
      # code, but NOT with scipy.linalg.svd docs...just be careful
      M = self.secs_.n_secs
      N = np.prod(np.shape(lat_lon_r))


      if not lat_lon_r is self.obs_lat_lon_r_:
         # (re)estimate obs_T_
         self.obs_lat_lon_r_ = lat_lon_r
         self.obs_T_ = self.make_T(self.obs_lat_lon_r_)
         # force SVD calculation
         self.obs_sigma_Btheta_Bphi_Br_ = None

      if not (lat_lon_r is self.obs_lat_lon_r_ and
              sigma_Btheta_Bphi_Br is self.obs_sigma_Btheta_Bphi_Br_):
         # (re)stimate U_, S_, and Vh_
         self.obs_sigma_Btheta_Bphi_Br_ = sigma_Btheta_Bphi_Br
         (self.U_,
          self.S_,
          self.Vh_) = la.svd(self.obs_T_ /
                       np.tile(self.obs_sigma_Btheta_Bphi_Br_.reshape(N,1), M),
                       full_matrices=False)
         # force SECS re-estimation
         self.obs_Btheta_Bphi_Br_ = None

      # if not (lat_lon_r is self.obs_lat_lon_r_ and
      #         sigma_Btheta_Bphi_Br is self.obs_sigma_Btheta_Bphi_Br_ and
      #         Btheta_Bphi_Br is self.obs_Btheta_Bphi_Br_):
      #
      #    # NOTE: if none of the inputs were different, nothing gets updated...
      #    #       should we issue some kind of warning?...
      #    #       Never mind, we will ALWAYS reestimate secs_

      # (re)estimate secs_
      # NOTE: we preserve self.[U_|S_|Vh_] while re-calculatig VWU with each
      #       call; this is somewhat inefficient, but the SVD values have
      #       potential future uses other than just inverting for secs_,
      #       while VWU...not so much.
      self.obs_Btheta_Bphi_Br_ = Btheta_Bphi_Br
      
      if self.epsilon >= 1:
         # retain the first int(epsilon) singular values
         kk = np.arange(self.S_.size) > (self.epsilon - 1)
      elif self.epsilon < 1 and self.epsilon >= 0:
         # retain singular values greater than epsilon*S_.max()
         kk = self.S_ < self.epsilon * self.S_.max()
      elif self.epsilon < 0 and self.epsilon > -1:
         # retain singular values less than -epsilon*S_.max()
         kk = self.S_ > -self.epsilon * self.S_.max()
      else:
         # retain the last int(-epsilon) singular values
         kk = np.arange(self.S_.size) <= (-1 * self.epsilon - 1)
      
      self.S_[kk] = 0.
      S_ = self.S_.copy() # preserve self.S_ without Infs
      S_[kk] = np.inf # temporarily set singular values to Inf
      W = 1. / S_ # divide by Inf gives zero-weight
      VWU = np.dot(self.Vh_.T, np.dot(np.diag(W), self.U_.T))
      self.secs_.amps = np.dot(VWU,
                               self.obs_Btheta_Bphi_Br_.reshape(N) /
                               self.obs_sigma_Btheta_Bphi_Br_.reshape(N))
      self.secs_.amps_var = np.sum((self.Vh_.T / S_)**2, axis=1)

      # The preceding estimate for amps_var is taken directly from Pulkkinen's
      # original ML code, and it is consistent with Numerical Recipes, but a
      # classical reference that can be found here:
      # https://pdfs.semanticscholar.org/aef2/68c21be034bfd6228bf3946cb46e3c62cdb1.pdf
      # ...sugests that something is missing. Specifically, the variance of the
      # prediction errors. I need to revisit this when time allows -EJR 2/2017

      # Furthermore, if a full covariance is desired, the following (taken from:
      #  http://math.stackexchange.com/questions/1169744/) should work...
      #self.secs_.amps_var = np.dot(self.Vh_.T,
      #                               np.dot(np.diag(W**2), self.Vh_))
      # ...but 1) it requires a lot more memory, and 2) as explained at the
      # website, the resulting covariance is probably not valid because the
      # truncated SVD solution is inherently biased.

      return self


   def sample_secs(self, n_samples=1, random_state=0):
      """generate random perturbations of secs

      Parameters
      ----------
      n_samples : int (optional)
         Number of times to sample SECS distribution
         (default is 1)

      random_state : RandomState or int seed (optional)
         A numpy.random.RandomState or an integer that can be used to __init__
         a numpy.random.RandomState obect. An array of integers may also be
         passed, although it's not clear what good this does.

      Returns
      -------
      amps_samples : array_like (shape(n_samples, n_secs))
         Array of amps values that are drawn from a Gaussian distribution
         defined by means secs.amps, and variance secs.amps_var.

      """
      # convert secs_.amps_var into a diagonal covariance matrix if it has
      # the form originally specified by Antti Pulkinnen (i.e., a vector, with
      # no covariances on the off-diagonal)
      cov_matrix = np.diag(self.secs_.amps_var)
      #cov_matrix = self.secs_.amps_var # assumes full covariance

      try:
         # start by assuming random_state is a np.random.RandomState object
         amps_samples = random_state.multivariate_normal(self.secs_.amps,
                                                         cov_matrix,
                                                         n_samples)
      except AttributeError:
         # if a RandomState was not passed, generate one, then generate output
         random_state = np.random.RandomState(random_state)
         amps_samples = random_state.multivariate_normal(self.secs_.amps,
                                                         cov_matrix,
                                                         n_samples)

      return amps_samples


   def sample_Btheta_Bphi_Br(self, lat_lon_r, n_samples=1, random_state=0):
      """generate random perturbations of magnetic vector field

      This is mostly a wrapper for sample_secs() and predict()

      Parameters
      ----------
      lat_lon_r : array_like (shape(n_sites, 3), required)
         Geodetic latitude (degrees north), longitude (degrees east), and
         radii (meters) of desired magnetic field predictions

      n_samples : int (optional)
         Number of times to sample SECS distribution
         (default is 1)

      random_state : RandomState or int seed (optional)
         A numpy.random.RandomState or an integer that can be used to __init__
         a numpy.random.RandomState obect. An array of integers may also be
         passed, although it's not clear what good this does.

      Returns
      -------
      Btheta_Bphi_Br : array_like (shape(n_samples, n_sites, 3))

      """
      # call sample_secs() to get perturbed secs amplitudes
      amps_samples = self.sample_secs(n_samples = n_samples,
                                      random_state = random_state)

      # loop over rows in amps_samples
      Btheta_Bphi_Br = []
      for secs_amps in amps_samples:
         Btheta_Bphi_Br.append(self.predict(lat_lon_r,
                                            secs_amps = secs_amps))
         # bypass redundant re-estimation of transformation matrix
         lat_lon_r = self.pred_lat_lon_r_

      return np.array(Btheta_Bphi_Br)


   def predict(self, lat_lon_r, secs_amps=None):
      """predict magnetic fields

      Parameters
      ----------
      lat_lon_r : array_like (shape(n_sites, 3), required)
         Geodetic latitude (degrees north), longitude (degrees east), and
         radii (meters) of desired magnetic field predictions

      secs_amps : array_like (shape(n_secs), optional)
         If None, use self.secs_.amps, otherwise use secs_amps to generate a
         prediction of magnetic vectors located at lat_lon_r

      Returns
      -------
      Btheta_Bphi_Br : array_like (shape(n_sites, 3))

      Notes
      -----
      The lat_lon_r is used only if it is NOT the same as corresponding class
      attribute. This allows us to bypass re-estimation of cpu-intensive, but
      often highly redundant, estimation of pred_T_. This does not compare
      array values, but rather the array object itself, which is much quicker.
      """
      if secs_amps is None:
         secs_amps = self.secs_.amps

      # much below relies on lat_lon_r being a NumPy array
      # (copy=False make sure only a reference is assigned, if possible;
      #  this is necessary for the bypass/optimization logic below)
      lat_lon_r = np.array(lat_lon_r, copy=False)

      if not lat_lon_r is self.pred_lat_lon_r_:
         # (re)estimate pred_T_
         self.pred_lat_lon_r_ = lat_lon_r
         self.pred_T_ = self.make_T(self.pred_lat_lon_r_)

      # matrix multiplication generates a single long vector that is reshaped
      # to give 3 columns (note: NumPy row/column majority differs from Matlab)
      Btheta_Bphi_Br = np.dot(self.pred_T_, secs_amps).reshape(-1,3)

      return Btheta_Bphi_Br


   def make_T(self, lat_lon_r):
      """create secs_ transformation matrix

      This is the workhorse for the entire class. It is a vectorized version
      of the equations first presented by Amm & Viljanen (1999) that exploits
      the Python Numpy library's optimizations, and also attempts to minimize
      the number of trigonometric calls. This was done to overcome performance
      limitations encountered in the original Matlab algorithms. It has been
      carefully validated against the original algorithms, which were used in
      an earlier version of Geomag-IMP.

      Parameters
      ----------
      lat_lon_r : array_like (shape(n_stations, 3))
         Geodetic latitude (degrees north), longitude (degrees east), and
         radii (meters) of magnetic field observations.

      Returns
      -------
      T : array_like (shape(n_obs * 3, n_secs))
         "Transformation matrix" containing geometric factors that describe
         the contribution of unit-valued elementary currents at coordinates
         self.secs.lat_lon_r to each magnetic vector at coordinates lat_lon_r.
         The output is in standard spherical coordinates (theta, phi, r), not
         (North, East, Down).

      Notes
      -----
      - We do not implement the much simpler "Cartesian" case here because:
        1) it is unlikely we will ever use it; and
        2) if we do use it, we should create a new class
      - This function borrows heavily from Greg Lucas' elegant pysecs package
        (https://github.com/greglucas/pysecs), with the author's permission,
        of course. The only reason we did not simply "wrap" pysecs' secs class
        is the pysecs API was not easily adapted to resemble scikit-learn's
        Gaussian Process Regressor (GPR) class...in hindsight, it is not clear
        that this was a particularly useful API to emulate, but to drop it now
        would break backward compatibility for Geomag-IMP.
      """

      nobs = len(lat_lon_r)
      nsec = len(self.secs.lat_lon_r)

      obs_r = lat_lon_r[:, 2][:, np.newaxis]
      sec_r = self.secs.lat_lon_r[:, 2][np.newaxis, :]

      theta = calc_angular_distance(lat_lon_r[:, :2], self.secs.lat_lon_r[:, :2])
      alpha = calc_bearing(lat_lon_r[:, :2], self.secs.lat_lon_r[:, :2])

      # magnetic permeability
      mu0 = 4*np.pi*1e-7

      # simplify calculations by storing this ratio
      x = obs_r/sec_r

      sin_theta = np.sin(theta)
      cos_theta = np.cos(theta)
      factor = 1./np.sqrt(1 - 2*x*cos_theta + x**2)

      # Amm & Viljanen: Equation 9
      Br = mu0/(4*np.pi*obs_r) * (factor - 1)

      # Amm & Viljanen: Equation 10 (transformed to try and eliminate trig 
      #                              operations and divide by zeros)
      Btheta = -mu0/(4*np.pi*obs_r) * (factor*(x - cos_theta) + cos_theta)
      # If sin(theta) == 0: Btheta = 0
      # There is a possible 0/0 in the expansion when sec_loc == obs_loc
      Btheta = np.divide(Btheta, sin_theta, out=np.zeros_like(sin_theta),
                        where=sin_theta != 0)

      # When observation points radii are outside of the sec locations
      under_locs = sec_r < obs_r

      # NOTE: If any SECs are below observations the math will be done on all
      #       points. This could be updated to only work on the locations where
      #       this condition occurs, but would make the code messier, with 
      #       minimal performance gain except for very large matrices.
      if np.any(under_locs):
         # Flipped from previous case
         x = sec_r/obs_r

         # Amm & Viljanen: Equation A.7
         Br2 = mu0*x/(4*np.pi*obs_r) * (1./np.sqrt(1 - 2*x*cos_theta + x**2) - 1)

         # Amm & Viljanen: Equation A.8
         Btheta2 = - mu0 / (4*np.pi*obs_r) * ((obs_r-sec_r*cos_theta) /
                                                np.sqrt(obs_r**2 -
                                                      2*obs_r*sec_r*cos_theta +
                                                      sec_r**2) - 1)
         Btheta2 = np.divide(Btheta2, sin_theta, out=np.zeros_like(sin_theta),
                              where=sin_theta != 0)

         # Update only the locations where secs are under observations
         Btheta[under_locs] = Btheta2[under_locs]
         Br[under_locs] = Br2[under_locs]

      T = np.empty((nobs, 3, nsec))
      T[:, 0, :] = Btheta*np.sin(alpha)
      T[:, 1, :] = Btheta*np.cos(alpha)
      T[:, 2, :] = Br
      
      # output expected to be 2D, not 3D
      return T.reshape(nobs * 3, -1)


def calc_angular_distance(latlon1, latlon2):
   """Calculate the angular distance between a set of points.
   This function calculates the angular distance in radians
   between any number of latitude and longitude points.
   Parameters
   ----------
   latlon1 : ndarray (n x 2 [lat, lon])
      An array of n (latitude, longitude) points.
   latlon2 : ndarray (m x 2 [lat, lon])
      An array of m (latitude, longitude) points.
   Returns
   -------
   ndarray (n x m)
      The array of distances between the input arrays.
   """
   lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
   lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
   lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
   lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

   dlon = lon2 - lon1

   # theta == angular distance between two points
   theta = np.arccos(np.sin(lat1)*np.sin(lat2) +
                     np.cos(lat1)*np.cos(lat2)*np.cos(dlon))
   return theta


def calc_bearing(latlon1, latlon2):
   """Calculate the bearing (direction) between a set of points.
   This function calculates the bearing in radians
   between any number of latitude and longitude points.
   It is the direction from point 1 to point 2 going from the
   cartesian x-axis towards the cartesian y-axis.
   Parameters
   ----------
   latlon1 : ndarray (n x 2 [lat, lon])
      An array of n (latitude, longitude) points.
   latlon2 : ndarray (m x 2 [lat, lon])
      An array of m (latitude, longitude) points.
   Returns
   -------
   ndarray (n x m)
      The array of bearings between the input arrays.
   """
   lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
   lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
   lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
   lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

   dlon = lon2 - lon1

   # alpha == bearing, going from point1 to point2
   #          angle (from cartesian x-axis (By), going towards y-axis (Bx))
   # Used to rotate the SEC coordinate frame into the observation coordinate
   # frame.
   # SEC coordinates are: theta (colatitude (+ away from North Pole)),
   #                      phi (longitude, + east), r (+ out)
   # Obs coordinates are: X (+ north), Y (+ east), Z (+ down)
   alpha = np.pi/2 - np.arctan2(np.sin(dlon)*np.cos(lat2),
                              np.cos(lat1)*np.sin(lat2) -
                              np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
   return alpha