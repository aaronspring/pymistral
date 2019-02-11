from collections import OrderedDict

import cartopy as cp
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


def my_plot(data,
            projection=ccrs.PlateCarree(),
            coastline_color='gray',
            curv=False,
            **kwargs):
    """Wrap xr.plot with cartopy."""
    plt.figure(figsize=(10, 5))
    if curv:
        data = _rm_singul_lon(data)
    ax = plt.subplot(projection=projection)
    data.plot.pcolormesh(
        'lon', 'lat', ax=ax, transform=ccrs.PlateCarree(), **kwargs)
    # data.plot.contourf('lon', 'lat', ax=ax,
    #                     transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines(color=coastline_color, linewidth=1.5)
    if curv:
        ax.add_feature(cp.feature.LAND, zorder=100, edgecolor='k')
    if projection == ccrs.PlateCarree():
        _set_lon_lat_axis(ax, projection)


def _rm_singul_lon(ds):
    """Remove singularity from coordinates.

    http://nbviewer.jupyter.org/gist/pelson/79cf31ef324774c97ae7
    """
    lons = ds['lon'].values
    fixed_lons = lons.copy()
    for i, start in enumerate(np.argmax(np.abs(np.diff(lons)) > 180, axis=1)):
        fixed_lons[i, start + 1:] += 360
    lons_da = xr.DataArray(fixed_lons, ds.lat.coords)
    ds = ds.assign_coords(lon=lons_da)
    return ds


def my_facetgrid(da,
                 projection=ccrs.PlateCarree(),
                 coastline_color='gray',
                 curv=False,
                 col='year',
                 col_wrap=2,
                 **kwargs):
    """Wrap xr.facetgrid plot with cartopy"""
    transform = ccrs.PlateCarree()
    p = da.plot.pcolormesh(
        'lon',
        'lat',
        transform=transform,
        col=col,
        col_wrap=col_wrap,
        subplot_kws={'projection': projection},
        **kwargs)
    for ax in p.axes.flat:
        if curv:
            ax.add_feature(cp.feature.LAND, zorder=100, edgecolor='k')
        if projection == ccrs.PlateCarree():
            _set_lon_lat_axis(ax, projection)
            ax.set_xlabel('')
            ax.set_ylabel('')
        ax.coastlines()
        # ax.set_extent([-160, -30, 5, 75])
        # ax.set_aspect('equal', 'box-forced')


def _set_lon_lat_axis(ax, projection, talk=False):
    """Add longitude and latitude coordinates to cartopy plots."""
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=projection)
    ax.set_yticks([-60, -30, 0, 30, 60, 90], crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if talk:
        ax.outline_patch.set_edgecolor('black')
        ax.outline_patch.set_linewidth('1.5')
        ax.tick_params(labelsize=15)
        ax.tick_params(width=1.5)


def _set_integer_xaxis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
