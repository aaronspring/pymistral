from setuptools import setup

setup(name='pymistral',
      version='0.1',
      description='Wrapper for parallel, effective and efficient computations via slurm and dask on our supercomputer',
      url='http://github.com/aaronspring/pymistral',
      author=['Sebastian Milinski', 'Aaron Spring'],
      author_email=['sebastian.milinski@mpimet.mpg.de',
                    'aaron.spring@mpimet.mpg.de'],
      license='MIT',
      packages=['pymistral'],
      zip_safe=False,
      test_require=['pytest']
      )
