import cartopy as cp
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.ticker import MaxNLocator


@xr.register_dataarray_accessor('plot_map')
class CartopyMap(object):
    """
    Plot the given 2D array on a cartopy axes with ('xc','lon','longitude') assumed as Longitude and ('yc','lat','latitude') assumed as Latitude.
    The default projection is PlateCarree, but can be:
        cartopy.crs.<ProjectionName>()
    If you would like to create a figure with multiple subplots
    you can pass an axes object to the function with keyword argument `ax,
    BUT then you need to specify the projection when you create the axes:
        plt.axes([x0, y0, w, h], projection=cartopy.crs.<ProjectionName>())
    Additional keywords can be given to the function as you would to
    the xr.DataArray.plot function. The only difference is that `robust`
    is set to True by default.
    The function returns a GeoAxes object to which features can be added with:
        ax.add_feature(feature.<FeatureName>, **kwargs)
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ax=None, proj=ccrs.PlateCarree(), plot_lon_lat_axis=True, feature='land', plot_type='pcolormesh', rm_cyclic=True, **kwargs):
        return self._cartopy(ax=ax, proj=proj, feature=feature, plot_lon_lat_axis=plot_lon_lat_axis, plot_type=plot_type, rm_cyclic=rm_cyclic, **kwargs)

    def _cartopy(self, ax=None, proj=ccrs.PlateCarree(), feature='land', plot_lon_lat_axis=True, plot_type='pcolormesh', rm_cyclic=True, **kwargs):

        # TODO: CESM2 and GFDL-ESM4 have lon issue

        xda = self._obj
        # da, convert to da or error
        if not isinstance(xda, xr.DataArray):
            if len(xda.data_vars) == 1:
                xda = xda[xda.data_vars[0]]
            else:
                raise ValueError(
                    f'Please provide xr.DataArray, found {type(xda)}')

        assert (xda.ndim == 2) or (xda.ndim == 3 and 'col' in kwargs or 'row' in kwargs) or (
            xda.ndim == 4 and 'col' in kwargs and 'row' in kwargs)
        single_plot = True if xda.ndim == 2 else False

        stereo_maps = (ccrs.Stereographic,
                       ccrs.NorthPolarStereo,
                       ccrs.SouthPolarStereo)
        if isinstance(proj, stereo_maps):
            raise ValueError(
                'Not implemented, see https://github.com/luke-gregor/xarray_tools/blob/master/accessors.py#L222')

        # find whether curv or not
        curv = False
        lon = None
        lat = None
        for c in xda.coords:
            if len(xda[c].dims) == 2:
                curv = True
                if c in ['xc', 'lon', 'longitude']:
                    lon = c
                if c in ['yc', 'lat', 'latitude']:
                    lat = c
        assert lon != None
        assert lat != None

        if proj in [ccrs.Robinson()]:
            plot_lon_lat_axis = False

        if plot_type is 'contourf':
            rm_cyclic = False
        if curv and rm_cyclic:
            xda = _rm_singul_lon(xda, lon=lon, lat=lat)

        if 'robust' not in kwargs:
            kwargs['robust'] = True
        if 'cbar_kwargs' not in kwargs:
            kwargs['cbar_kwargs'] = {'shrink': .6}

        if ax is None:
            if single_plot:
                axm = getattr(xda.plot, plot_type)(
                    lon, lat, ax=plt.axes(projection=proj), transform=ccrs.PlateCarree(), **kwargs)
            else:
                axm = getattr(xda.plot, plot_type)(
                    lon, lat, subplot_kws={'projection': proj}, transform=ccrs.PlateCarree(), **kwargs)
        else:
            axm = getattr(xda.plot, plot_type)(
                lon, lat, ax=ax, transform=ccrs.PlateCarree(), **kwargs)

        def work_on_axes(axes):
            if 'coastline_color' in kwargs:
                coastline_color = kwargs['coastline_color']
            else:
                coastline_color = 'gray'
            axes.coastlines(color=coastline_color, linewidth=1.5)
            if feature is not None:
                axes.add_feature(getattr(cp.feature, feature.upper()),
                                 zorder=100, edgecolor='k')

            if plot_lon_lat_axis:
                _set_lon_lat_axis(axes, proj)
        if single_plot:
            if ax is None:
                ax = plt.gca()
            work_on_axes(ax)
        else:
            for axes in axm.axes.flat:
                work_on_axes(axes)
        return axm


def _rm_singul_lon(ds, lon='lon', lat='lat'):
    """Remove singularity from coordinates.

    http://nbviewer.jupyter.org/gist/pelson/79cf31ef324774c97ae7
    """
    lons = ds[lon].values
    fixed_lons = lons.copy()
    for i, start in enumerate(np.argmax(np.abs(np.diff(lons)) > 180, axis=1)):
        fixed_lons[i, start + 1:] += 360
    lons_da = xr.DataArray(fixed_lons, ds[lat].coords)
    ds = ds.assign_coords({lon: lons_da})
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
