import os

import cdo
import numpy as np
import xarray as xr

# builds on export WORK, GROUP to be set in your bashrc
user = os.environ['LOGNAME']
my_system = None
cdo_mistral = True
group = 'mh0727'
try:
    host = os.environ['HOSTNAME']
    assert group == os.environ['GROUP']
    for node in ['mlogin', 'mistralpp']:
        if node in host:
            my_system = 'mistral'
    if 'm' is host[0]:
        my_system = 'mistral'
    if my_system is None:
        my_system = 'local'
except:
    my_system = 'local'

# setup folders for working on mistral
if my_system is 'mistral':
    mistral_work = '/work/'
    work = f'{mistral_work}{group}/{user}/'
    tmp = work + 'tmp'
    if not os.path.exists(tmp):
        os.makedirs(tmp)
# setup folder for working via sshfs_mistral in ~/mistral_work
elif my_system is 'local':
    mistral_work = f'~/mistral_work/'
    work = f'{mistral_work}{group}/{user}/'
    cdo_mistral = True
    if cdo_mistral:
        tmp = os.path.expanduser('~/tmp')
    else:
        tmp = work + 'tmp'
    if not os.path.exists(tmp):
        os.makedirs(tmp)

# start
cdo = cdo.Cdo(tempdir=tmp)


cmip6_folder = mistral_work+'ik1017/CMIP6/data/CMIP6'
cmip5_folder = mistral_work+'kd0956/CMIP5/data/cmip5/output1'
GE_folder = mistral_work+'mh1007'


def remap_cdo(da):
    if not isinstance(da, xr.core.dataset.Dataset):
        da = da.to_dataset()
    remap = cdo.remapbil(
        'r360x180', input=da, returnXDataset=True, options='-P 8')
    return remap


def _decode_ym_cftime_to_int(ds):
    ds['time'] = ds.time.dt.year
    return ds


def _squeeze_dims(ds):
    """Get rid of extra dimensions."""
    ds = ds.squeeze()
    for dim in ['lon', 'lat', 'bnds', 'depth', 'depth_2', 'depth_3']:
        if dim in ds:
            if ds[dim].size <= 1:
                del ds[dim]
    drop = []
    for dim in [
            'hyai', 'hybi', 'hyam', 'hybm', 'time_bnds', 'lat_bnds', 'lon_bnds'
    ]:
        if dim in ds:
            drop.append(dim)
    ds = ds.drop(drop)
    return ds.squeeze()


def _set_LY(ds, first=1, dim='lead'):
    """Set integer lead index starting with first."""
    return ds.assign({dim: np.arange(first, first + ds[dim].size)})


def yearmean(ds, dim='time'):
    return ds.groupby('{dim}.year').mean(dim).rename({'year': dim})


def yearsum(ds, dim='time'):
    return ds.groupby('{dim}.year').sum(dim).rename({'year': dim})


def standardize(ds, dim='time'):
    return (ds-ds.mean(dim))/ds.std(dim)
