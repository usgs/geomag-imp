# import NumPy to work with arrays
import numpy as np

# use NumPy array-aware assert functions
from numpy.testing import assert_equal
from numpy.testing import assert_approx_equal

# import only the SECS classes from geomag_imp
from geomag_imp import secs, secsRegressor

# ignore divide-by-zero warnings (is this wise?)
np.seterr(divide='ignore', invalid='ignore')

# start by unit-testing basic class construction and object manipulation;
# progress to higher level tests that rely on the more basic functionality...
# if the basic stuff fails, the derived stuff will fail too.

def test_secs_construct():
   '''
   Test construction of spherical elementary current system
   '''
   # place SEC at north pole, 0 latitude, and 100 km altitude
   lats = np.array([90])
   lons = np.array([0])
   rads = np.array([6378000 + 100000])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec1 = secs(secs_llr, amps=None, amps_var=None)

   assert_equal(sec1.n_secs, 1)
   assert_equal(sec1.latitude, 90)
   assert_equal(sec1.longitude, 0)
   assert_equal(sec1.radius, 6378000 + 100000)



def test_regress_construct():
   '''
   Test construction of secs regressor
   '''
   # place SEC at north pole, 0 latitude, and 100 km altitude
   lats = np.array([90])
   lons = np.array([0])
   rads = np.array([6378000 + 100000])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec1 = secs(secs_llr, amps=None, amps_var=None)

   # initialize secsRegressor object
   epsilon = 0.1
   secsR = secsRegressor(sec1, epsilon)

   assert_equal(secsR.secs.n_secs, 1)
   assert_equal(secsR.secs.latitude, 90)
   assert_equal(secsR.secs.longitude, 0)
   assert_equal(secsR.secs.radius, 6378000 + 100000)
   assert_equal(secsR.epsilon, 0.1)


def test_predict_sec_above():
   '''
   Test SEC prediction at fixed locations relative to pole of SECS above ground
   '''
   # place SEC at north pole, 0 longitude, and 100 km altitude
   lats = np.array([90])
   lons = np.array([0])
   rads = np.array([6378000. + 100000.])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec_above = secs(secs_llr, amps=None, amps_var=None)

   # initialize secsRegressor object
   epsilon = 0.1
   secsR = secsRegressor(sec_above, epsilon)

   # predict magnetic disturbance at equator and at pole
   latp = np.array([0,90])
   lonp = np.array([0,0])
   radp = np.array([6378000., 6378000.])
   pred_llr = np.array(list(zip(latp, lonp, radp)))

   # set SEC amplitude to 10,000 Amps, and make prediction
   amps = 1e4
   pred = secsR.predict(pred_llr, amps)

   # calculate exact solution using relationships found in Pulkinnen et al.'s
   # 2003 Earth Planets Space article "Separation of the geomagnetic variation
   # field on the ground into external and internal parts using the elementary
   # current system method"
   mu0 = 4 * np.pi * 1e-7 # N / A^2
   Btheta90 = -(mu0 * amps) / (4. * np.pi * radp[0] *
                               np.sin((90-latp[0]) * np.pi/180.)) * \
      (((radp[0]/rads[0]) - np.cos((90-latp[0]) * np.pi/180.)) / \
       np.sqrt(1 - (2. * radp[0] * np.cos((90-latp[0]) * np.pi/180.)) / \
               rads[0] + (radp[0]/rads[0])**2) + \
       np.cos((90-latp[0]) * np.pi/180.) )
   Brad0 = (mu0 * amps) / (4. * np.pi * radp[1]) * \
       (1. / (np.sqrt(1. - 2. * radp[1] * np.cos((90-latp[1]) * np.pi/180.) / \
                      rads[0] + (radp[1] / rads[0])**2) ) - 1. )


   assert_approx_equal(pred[0,0], Btheta90, significant=9)
   assert_approx_equal(pred[1,2], Brad0, significant=9)


def test_predict_sec_above_Bphis():
   '''
   Test SEC prediction at fixed locations relative to pole of SECS above ground;
   this test actually checks the phi (longitudinal) vector component
   '''
   # place SEC at equator, 0 longitude, and 100 km altitude
   lats = np.array([0])
   lons = np.array([0])
   rads = np.array([6378000. + 100000.])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec_above = secs(secs_llr, amps=None, amps_var=None)

   # initialize secsRegressor object
   epsilon = 0.1
   secsR = secsRegressor(sec_above, epsilon)

   # predict magnetic disturbance at equator, but 90 degrees east and west
   # of pole position; these are where the theta component caculated in the
   # SEC frame will equal the negative of the phi component in geographic
   # coordinates
   latp = np.array([0,0])
   lonp = np.array([-90,90])
   radp = np.array([6378000., 6378000.])
   pred_llr = np.array(list(zip(latp, lonp, radp)))

   # set SEC amplitude to 10,000 Amps, and make prediction
   amps = 1e4
   pred = secsR.predict(pred_llr, amps)

   # calculate exact solution using relationships found in Pulkinnen et al.'s
   # 2003 Earth Planets Space article "Separation of the geomagnetic variation
   # field on the ground into external and internal parts using the elementary
   # current system method"
   mu0 = 4 * np.pi * 1e-7 # N / A^2
   Btheta90 = -(mu0 * amps) / (4. * np.pi * radp[0] *
                               np.sin((lonp[0] - lons[0]) * np.pi/180.)) * \
      (((radp[0]/rads[0]) - np.cos((lonp[0] - lons[0]) * np.pi/180.)) / \
       np.sqrt(1 - (2. * radp[0] * np.cos((lonp[0] - lons[0]) * np.pi/180.)) / \
               rads[0] + (radp[0]/rads[0])**2) + \
       np.cos((lonp[0] - lons[0]) * np.pi/180.) )
   Bphi90minus = -Btheta90
   Bphi90plus = Btheta90

   print(pred[0,1], Bphi90minus)
   print(pred[1,1], Bphi90plus)

   assert_approx_equal(pred[0,1], Bphi90minus, significant=9)
   assert_approx_equal(pred[1,1], Bphi90plus, significant=9)


def test_predict_sec_below():
   '''
   Test SEC prediction at fixed locations relative to pole of SECS below ground
   '''
   # place SEC at north pole, 0 longitude, and 100 km altitude
   lats = np.array([90])
   lons = np.array([0])
   rads = np.array([6378000. - 100000.])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec_below = secs(secs_llr, amps=None, amps_var=None)

   # initialize secsRegressor object
   epsilon = 0.1
   secsR = secsRegressor(sec_below, epsilon)

   # predict magnetic disturbance at equator and at pole
   latp = np.array([0,90])
   lonp = np.array([0,0])
   radp = np.array([6378000., 6378000.])
   pred_llr = np.array(list(zip(latp, lonp, radp)))

   # set SEC amplitude to 10,000 Amps, and make prediction
   amps = 1e4
   pred = secsR.predict(pred_llr, amps)

   # calculate exact solution using relationships found in Pulkinnen et al.'s
   # 2003 Earth Planets Space article "Separation of the geomagnetic variation
   # field on the ground into external and internal parts using the elementary
   # current system method"
   mu0 = 4 * np.pi * 1e-7 # N / A^2
   Btheta90 = -(mu0 * amps) / (4. * np.pi * radp[0] *
                               np.sin((90-latp[0]) * np.pi/180.)) * \
      ((radp[0] - rads[0] * np.cos((90-latp[0]) * np.pi/180.)) / \
       np.sqrt(radp[0]**2 -
               (2. * radp[0] * rads[0] * np.cos((90-latp[0]) * np.pi/180.)) + \
               rads[0]**2) - 1.)
   Brad0 = (mu0 * amps * rads[0]) / (4. * np.pi * radp[1]**2) * \
       (1. / (np.sqrt(1. - 2. * rads[0] * np.cos((90-latp[1]) * np.pi/180.) / \
                      radp[1] + (rads[0] / radp[1])**2) ) - 1. )

   assert_approx_equal(pred[0,0], Btheta90, significant=9)
   assert_approx_equal(pred[1,2], Brad0, significant=9)


def test_predict_sec_above_below():
   '''
   Test SEC prediction at fixed locations relative to pole of SECS below ground
   '''
   # place SEC at north pole, 0 longitude, and 100 km altitude
   lats = np.array([90, 90])
   lons = np.array([0, 0])
   rads = np.array([6378000. + 100000., 6378000. - 100000.])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec_above_below = secs(secs_llr, amps=None, amps_var=None)

   # initialize secsRegressor object
   epsilon = 0.1
   secsR = secsRegressor(sec_above_below, epsilon)

   # predict magnetic disturbance at equator and at pole
   latp = np.array([0,90])
   lonp = np.array([0,0])
   radp = np.array([6378000., 6378000.])
   pred_llr = np.array(list(zip(latp, lonp, radp)))

   # set SEC amplitude to 10,000 Amps, and make prediction
   amps = np.array([1e4,-1e4])
   pred = secsR.predict(pred_llr, amps)

   # calculate exact solution using relationships found in Pulkinnen et al.'s
   # 2003 Earth Planets Space article "Separation of the geomagnetic variation
   # field on the ground into external and internal parts using the elementary
   # current system method"
   mu0 = 4 * np.pi * 1e-7 # N / A^2
   Btheta90 = -(mu0 * amps[0]) / (4. * np.pi * radp[0] *
                               np.sin((90-latp[0]) * np.pi/180.)) * \
      (((radp[0]/rads[0]) - np.cos((90-latp[0]) * np.pi/180.)) / \
       np.sqrt(1 - (2. * radp[0] * np.cos((90-latp[0]) * np.pi/180.)) / \
               rads[0] + (radp[0]/rads[0])**2) + \
       np.cos((90-latp[0]) * np.pi/180.) )
   Btheta90 += -(mu0 * amps[1]) / (4. * np.pi * radp[0] *
                               np.sin((90-latp[0]) * np.pi/180.)) * \
      ((radp[0] - rads[1] * np.cos((90-latp[0]) * np.pi/180.)) / \
       np.sqrt(radp[0]**2 -
               (2. * radp[0] * rads[1] * np.cos((90-latp[0]) * np.pi/180.)) + \
               rads[1]**2) - 1.)

   Brad0 = (mu0 * amps[0]) / (4. * np.pi * radp[1]) * \
       (1. / (np.sqrt(1. - 2. * radp[1] * np.cos((90-latp[1]) * np.pi/180.) / \
                      rads[0] + (radp[1] / rads[0])**2) ) - 1. )
   Brad0 += (mu0 * amps[1] * rads[1]) / (4. * np.pi * radp[1]**2) * \
       (1. / (np.sqrt(1. - 2. * rads[1] * np.cos((90-latp[1]) * np.pi/180.) / \
                      radp[1] + (rads[1] / radp[1])**2) ) - 1. )

   assert_approx_equal(pred[0,0], Btheta90, significant=9)
   assert_approx_equal(pred[1,2], Brad0, significant=9)


def test_fit_sec_above_below():
   '''
   Fit uncorrupted B-field predictions generated by known SECs above and below
   the pole
   '''
   # place SEC at north pole, 0 latitude, and 100 km altitude
   lats = np.array([90, 90])
   lons = np.array([0, 0])
   rads = np.array([6378000. + 100000., 6378000. - 100000.])
   secs_llr = np.array(list(zip(lats, lons, rads)))

   # initialize secs object
   sec_above_below = secs(secs_llr, amps=None, amps_var=None)

   # initialize secsRegressor object
   epsilon = 0.1
   secsR = secsRegressor(sec_above_below, epsilon)

   # predict magnetic disturbance at equator and at pole
   latp = np.linspace(90,0,901)
   lonp = np.zeros(latp.shape)
   radp = np.zeros(latp.shape) + 6378000.
   pred_llr = np.array(list(zip(latp, lonp, radp)))

   # set SEC amplitude to 10,000 Amps, and make prediction
   amps = np.array([1e4,-1e4])
   pred = secsR.predict(pred_llr, amps)

   # fit the 2 SECs assuming 1nT uncertainty on observations
   secsR.fit(pred_llr, pred, np.ones(pred.shape) * 1e-9)

   assert_approx_equal(secsR.secs_.amps[0], 10000., significant=9)
   assert_approx_equal(secsR.secs_.amps[1], -10000., significant=9)

   assert_approx_equal(np.sqrt(secsR.secs_.amps_var[0]),
                               286.46658513, significant=9)
   assert_approx_equal(np.sqrt(secsR.secs_.amps_var[1]),
                               299.97580929, significant=9)
