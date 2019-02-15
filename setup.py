from setuptools import find_packages, setup

TEST_REQUIRES = ["pytest"]

INSTALL_REQUIRES = ['bokeh',
                    'netcdf4',
                    'cartopy',
                    'dask',
                    'dask-jobqueue',
                    'distributed',
                    'jupyterlab',
                    'matplotlib',
                    'numpy',
                    'pandas',
                    'scipy',
                    'seaborn',
                    'tqdm',
                    'xarray']


AUTHOR = ['Sebastian Milinski', 'Aaron Spring']
AUTHOR_EMAIL = ['sebastian.milinski@mpimet.mpg.de',
                'aaron.spring@mpimet.mpg.de']
DESCRIPTION = [
    'Wrapper for parallel, effective and efficient computations via slurm and dask on our supercomputer']
setup(name='pymistral',
      version='0.1',
      description=DESCRIPTION,
      url='http://github.com/aaronspring/pymistral',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=INSTALL_REQUIRES,
      tests_require=TEST_REQUIRES,
      py_modules=["cdo"],
      requires_python=['>=3.6']
      )
