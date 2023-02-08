import setuptools
import platform
from setuptools.extension import Extension


from committee_learning  import (
  __pkgname__ as PKG_NAME,
  __author__  as AUTHOR,
  __version__ as VERSION
)

# if platform.system() == 'Darwin':
#   import os
#   os.environ["CC"] = "g++-12"

def get_extensions():
  from Cython.Build import cythonize
  import numpy

  include_dirs = [
    numpy.get_include(),
    'external_libraries/cpp-boost-math/include'
  ]

  extra_compile_args = [
    '-O3',
    '-funroll-loops',
    '-std=c++20'
  ]

  cython_ode_erf = Extension(
      name='committee_learning.ode.cython_erf',
      sources=['committee_learning/ode/erf.pyx', 'committee_learning/ode/erf_integrals.cpp'],
      include_dirs=include_dirs,
      language = 'cpp',
      extra_compile_args=extra_compile_args
  )

  cython_risk = Extension(
      name='committee_learning._cython.risk',
      sources=['committee_learning/_cython/risk.pyx', 'committee_learning/ode/erf_integrals.cpp'],
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
  )

  return cythonize(
    [cython_risk, cython_ode_erf],
    compiler_directives={'language_level':3},
    annotate=True
  )

setuptools.setup(
  setup_requires= [
    'Cython',
    'numpy',
    'setuptools>=18.0' 
  ],
  ext_modules=get_extensions(),
  package_data={
    'committee_learning/_config':['*.pxd']
  },
  name = PKG_NAME,
  author  =  AUTHOR,
  version = VERSION,
  packages = setuptools.find_packages(),
  python_requires = '>=3.7', # Probably it works even with newer version of python, but still...
  install_requires = [
    'numpy',
    'torch',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'pyyaml',
    'tqdm',
  ],
  zip_safe=False
)