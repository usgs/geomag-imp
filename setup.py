from distutils.core import setup

setup(
   name='geomag-imp',
   version='1.0.0',
   description='USGS Geomag Interpolated Magnetic Perturbations',
   url='https://github.com/usgs/geomag-imp',
   packages=[
      'geomag_imp'
   ],
   install_requires=[
      'numpy',
      'matplotlib',
	   'cartopy',
      'scipy',
      'scikit-learn',
      'obspy',
	   'geomag-algorithms',
      'spacepy',
      'json_tricks'
   ],
   scripts=[
      'bin/make_imp_secs.py',
      'bin/make_imp_gp.py',
      'bin/make_impmaps.py',
      'bin/make_svsqdist.py'
   ]
)
