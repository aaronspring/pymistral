from setuptools import find_packages, setup

test_requirements = ["pytest"]

install_requires = ['bokeh',
                    'cartopy',
                    'cdo',
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

setup(name='pymistral',
      version='0.1',
      description=[
          'Wrapper for parallel, effective and efficient computations via slurm and dask on our supercomputer'],
      url='http://github.com/aaronspring/pymistral',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=install_requires,
      test_suite="tests",
      tests_require=test_requirements
      )
