from setuptools import find_packages, setup

test_requirements = ["pytest"]

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
      test_suite="tests",
      tests_require=test_requirements,
      python_requires=['>=3.6']
      )
