import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension


setup(
  name        = 'thesis_forecasting_modeling',
  packages    = ['algorithms', 'modelings', 'trainers'],
  package_dir = {'': '.'},

  ext_modules = cythonize([
    # algorithms
    Extension(
      'algorithms.initializers_cythonize',
      ['algorithms/initializers_cythonize.pyx'],
      include_dirs = [ numpy.get_include() ]
    ),

    Extension(
      'algorithms.optimizers_cythonize',
      ['algorithms/optimizers_cythonize.pyx'],
      include_dirs = [ numpy.get_include() ]
    ),

    Extension(
      'algorithms.bidirectional_cythonize',
      ['algorithms/bidirectional_cythonize.pyx'],
      include_dirs = [ numpy.get_include() ]
    ),

    Extension(
      'algorithms.gated_recurrent_unit_cythonize',
      ['algorithms/gated_recurrent_unit_cythonize.pyx'],
      include_dirs = [ numpy.get_include() ]
    ),

    # modelings
    Extension(
      'modelings.forecasting_bigru_model_cythonize',
      ['modelings/forecasting_bigru_model_cythonize.pyx'],
      include_dirs = [ numpy.get_include() ]
    ),

    # trainers
    Extension(
      'trainers.forecasting_bigru_trainer_cythonize',
      ['trainers/forecasting_bigru_trainer_cythonize.pyx'],
      include_dirs = [ numpy.get_include() ]
    )
  ]), zip_safe = False
)
