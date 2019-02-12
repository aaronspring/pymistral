from setuptools import find_packages, setup

setup(name='pymistral',
      version='0.1',
      description=[
          'Wrapper for parallel, effective and efficient computations via slurm and dask on our supercomputer'],
      url='http://github.com/aaronspring/pymistral',
      author=['Sebastian Milinski', 'Aaron Spring'],
      author_email=['sebastian.milinski@mpimet.mpg.de',
                    'aaron.spring@mpimet.mpg.de'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_require=['xarray', 'numpy'],
      extras_require={
          'testing': ['pytest']
      }
      python_requires=['>=3.6']
      )
