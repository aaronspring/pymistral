# pymistral

Wrapper for parallel, effective and efficient computations via slurm and dask
on our supercomputer

# Contact

-   Sebastian Milinski (sebastian.milinski@mpimet.mpg.de)

-   Aaron Spring (aaron.spring@mpimet.mpg.de)

# System configuration

The following steps are necessary to use pymistral.

## Environment

Create a new conda environment using the pymistral.yml file. This has to be
done on the supercomputer when using pymistral on an HPC system or locally for
a standalone configuration.

`conda env create -f pymistral.yml`

If the environment already exists and needs to be updated, use:

    source activate pymistral
    conda env update -f=pymistral.yml

## Script to start jupyterlab on HPC

This part is only necessary when jupyterlab and pymistral is used on a remote
HPC system.

### pymistral_preload

Place the pymistral_preload file into your home directory on mistral and
change the conda environment name if necessary.

### start-pymistral.sh

Adjust the start-pymistral.sh script with your username and project number.

Change SJ_INCFILE if pymistral_preload has been renamed.
