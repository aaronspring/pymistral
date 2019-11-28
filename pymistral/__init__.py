"""
pymistral
--------

A package for analyzing, processing, and mapping ESM output on mistral.

Available Modules
-----------------
- setup
- plot: Cartopy Mapping for MPIOM curvilinear grids
- hamocc: HAMOCC-specific helper functions
- slurm_post: write python code to file and send to SLURM (experimental)
- testing: FDR for xarray masked
- cdo_post: postprocessing with CDO into xarray
"""

from . import cdo_post, hamocc, plot, setup, slurm_post, testing
from .setup import cdo
