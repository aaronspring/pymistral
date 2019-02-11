import xarray as xr


def standardize(ds, time_dim='year', index=True):
    return (ds - ds.mean(time_dim)) / ds.std(time_dim)


mask_folder = '/work/mh0727/m300524/experiments/masks'


def calc_enso_index(ds, type='12', index=True, time_dim='time'):
    if type is '12':
        enso_weights = xr.open_dataset(
            mask_folder +
            '/GR15_lon_-90--80_lat_-10-0.weights.nc')['area'].squeeze()

    del enso_weights['depth']
    del enso_weights['time']
    sst = ds['tos']
    sst_clim = sst.groupby('time.month').mean(dim='time')
    sst_anom = sst.groupby('time.month') - sst_clim
    sst_anom_nino = (sst_anom * enso_weights).sum(['y', 'x'])
    if index:
        return standardize(sst_anom_nino, time_dim=time_dim)
    else:
        return sst_anom_nino
