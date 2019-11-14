# pymistral

Effective use of python with `xarray` and `dask` to analyse Earth-System-Modelling output on the `mistral` supercomputer at DKRZ.

# Examples

Check out our examples:

-   easy access via `intake-esm` on CMIP5, CMIP6 and MiKlip output on `mistral`
-   easy access via `intake` to ICDC observations on `mistral`
-   grid handling of `MPIOM` via `xgcm`
-   plotting the curvilinear `MPIOM` with `cartopy`: `xr.DataArray.plot_map()`

# Contact

-   Sebastian Milinski (sebastian.milinski@mpimet.mpg.de)

-   Aaron Spring (aaron.spring@mpimet.mpg.de)

# System configuration

The following steps are necessary to use `pymistral`.

## Environment

Create a new `conda environment` using the `pymistral.yml` file. This has to be
done on the supercomputer when using `pymistral` on an HPC system or locally for
a standalone configuration.

`conda env create -f pymistral.yml`

If the environment already exists and needs to be updated, use:

    source activate pymistral
    conda env update -f=pymistral.yml

## Script to start jupyterlab on HPC

This part is only necessary when `jupyterlab` and `pymistral` is used on a remote
HPC system. A detailed explanation is at the DKRZ website: <https://www.dkrz.de/up/systems/mistral/programming/jupyter-notebook>

### pymistral_preload

Place the `jupyter_preload` file into your home directory on `mistral` and
change the conda environment name if necessary. Alternatively, you can create your own `conda`: <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>
Here don't use your `$HOME` on `mistral`, specify path like `/work/yourgroup/m??????/miniconda3`.

### ./start-pymistral

Adjust the `start-jupyter` script with your username and project number.
