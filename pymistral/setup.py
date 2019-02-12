import os

import cdo
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm_notebook

# TODO: adapt for every user
try:
    my_system = None
    host = os.hostname()
    for node in ['mlogin', 'mistralpp']:
        if node in host:
            my_system = 'mistral'
    if my_system is None:
        my_system = 'local'
except:
    my_system = 'local'

if my_system is 'mistral':
    file_origin = '/work/mh0727/m300524/'
elif my_system is 'local':
    file_origin = '/Users/aaron.spring/mistral_work/'

cdo = cdo.Cdo(tempdir=file_origin + 'tmp')

# TODO: load all cmip cmorized varnames?
sample_file_dir = file_origin + 'experiments/sample_files/'
# hamocc_data_2d_varnamelist = cdo.showname(
#    input=sample_file_dir + 'hamocc_data_2d_*')[0].split()
# echam6_co2_varnamelist = cdo.showname(
#    input=sample_file_dir + 'echam6_co2*')[0].split()
# mpiom_data_2d_varnamelist = cdo.showname(
#    input=sample_file_dir + 'mpiom_data_2d_*')[0].split()

PM_path = file_origin + 'experiments/'
GE_path = file_origin + 'experiments/GE/'
center = 'MPI-M'
model = 'MPI-ESM-LR'
cmip_folder = '/work/ik0555/cmip5/archive/CMIP5/output/' + center + '/' + model
my_GE_path = file_origin + '160701_Grand_Ensemble/'
GE_post = my_GE_path + 'postprocessed/'
PM_post = PM_path + 'postprocessed/'


def _get_path_cmip(base_folder=cmip_folder,
                   exp='esmControl',
                   period='mon',
                   varname='co2',
                   comp='atmos',
                   run_id='r1i1p1',
                   ending='.nc',
                   timestr='*'):
    return base_folder + '/' + exp + '/' + period + '/' + comp + '/' + varname + '/' + run_id + '/' + varname + '_' + comp[
        0].upper(
    ) + period + '_' + model + '_' + exp + '_' + run_id + '_' + timestr + ending


# TODO: adapt for CMIP6, maybe with CMIP=5 arg
def load_cmip(exp='esmControl',
              period='mon',
              varname='co2',
              comp='atmos',
              run_id='r1i1p1',
              ending='.nc',
              timestr='*',
              operator='',
              levelstr=''):
    """Load a variable from CMIP5."""
    ncfiles_cmip = _get_path_cmip(
        exp=exp,
        period=period,
        varname=varname,
        comp=comp,
        run_id=run_id,
        ending=ending,
        timestr=timestr)
    return xr.open_dataset(
        cdo.addc(
            '0',
            input=operator + ' -select,name=' + varname + levelstr + ' ' +
            ncfiles_cmip,
            options='-r')).squeeze()[varname]


def read_table_file(table_file_str):
    """Read partab/.codes file."""
    table_file = pd.read_fwf(
        table_file_str,
        header=None,
        names=[
            'code', 'a', 'varname', 'b', 'c', 'long_name_and_unit', 'd', 'e'
        ])
    table_file.index = table_file['code']
    for a in 'abcde':
        del table_file[a]
    table_file['novarname'] = 'var' + table_file['code'].apply(str)
    table_file['unit'] = table_file['long_name_and_unit'].str.split(
        '[', expand=True).get(1)
    table_file['long_name'] = table_file['long_name_and_unit'].str.split(
        '[', expand=True).get(0)
    table_file['unit'] = table_file['unit'].str.replace(']', '')
    del table_file['long_name_and_unit']
    return table_file


def set_table(ds, table_file_str):
    """Replace variables in ds with table."""
    table = read_table_file(table_file_str)
    table_dict = {}
    for i in table.index:
        key, item = table[['novarname', 'varname']].loc[i]
        table_dict[key] = item
    return ds.rename(table_dict)


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


def _set_LY(ds, first=1):
    """Set integer time index starting with first."""
    ds = ds.assign(time=np.arange(first, first + ds.time.size))
    return ds


def _get_path(varname=None, exp='PM', prefix='ds', ta='ym', **kwargs):
    """Get postprocessed path."""
    if exp is 'PM':
        path = PM_path + 'postprocessed/'
    elif exp is 'GE':
        path = my_GE_path + 'postprocessed/'

    if varname is None:
        if isinstance(ds, xr.DataArray):
            varname = ds.name
        else:
            raise ValueError('specify varname')

    suffix = ''
    if prefix not in ['ds', 'control']:
        for key, value in kwargs.items():
            if prefix in ['skill']:
                if str(key) in ['sig', 'bootstrap']:
                    continue
            if isinstance(value, str):
                suffix += "_" + key + "_" + str(value)
            else:
                suffix += "_" + key + "_" + str(value)

    filename = prefix + '_' + varname + '_' + ta + suffix + '.nc'
    full_path = path + filename
    return full_path


def save(ds, varname=None, exp='PM', prefix='ds', ta='ym', **kwargs):
    """Save xr.object to _get_path location."""
    full_path = _get_path(
        varname=varname, exp=exp, prefix=prefix, ta=ta, **kwargs)
    print('save in:', full_path)
    ds.to_netcdf(full_path)


def _set_mm_span(ds):
    """Set monthly mean time axis.

    ### TODO: make possible for year 2300 or 1100.
    Starts in 1900 because of cftime limit."""
    span = pd.date_range(start='1/1/1900', periods=ds.time.size, freq='M')
    return ds.assign(time=span)


def yearmonmean(ds):
    return ds.groupby('time.year').mean('time').rename({'year': 'time'})


def yearsum(ds):
    return ds.groupby('time.year').sum('time').rename({'year': 'time'})


r_ppmw2ppmv = 28.8 / 44.0095
CO2_to_C = 44.0095 / 12.0111


def convert_C(ds):
    """Converts CO2 from ppmw to ppmv and co2_flux to C."""
    if 'CO2' in ds.data_vars:
        ds = ds * 1e6 * r_ppmw2ppmv
    if 'co2_fl' in ds.data_vars:
        ds = ds / CO2_to_C
    return ds


def _get_codes_str(file_type):
    return sample_file_dir + 'log/*' + file_type + '.codes'


def _get_GE_path(ext='hist',
                 m=1,
                 model='hamocc',
                 outdatatype='data_2d_mm',
                 timestr='*',
                 ending='.nc'):
    return GE_path + ext + '/' + ext + str(m).zfill(
        4) + '/outdata/' + model + '/' + ext + str(m).zfill(
            4) + '_' + model + '_' + outdatatype + '_' + timestr + ending


def _get_PM_path(init=3014,
                 m=0,
                 model='hamocc',
                 outdatatype='data_2d_mm',
                 timestr='*',
                 ending='.nc',
                 control=False):
    if control:
        run_id = 'vga0214'
    else:
        run_id = 'asp_esmControl_ens' + str(init) + '_m' + str(m).zfill(3)
    return PM_path + run_id + '/outdata/' + model + '/' + run_id + '_' + model + '_' + outdatatype + '_' + timestr + ending


def _get_GE_full_path(ext=['hist', 'rcp26'],
                      m=1,
                      model='hamocc',
                      outdatatype='data_2d_mm',
                      timestr='*',
                      ending='.nc'):
    path_list = []
    for ext in ext:
        path_list.append(
            _get_GE_path(
                ext=ext,
                m=m,
                model=model,
                outdatatype=outdatatype,
                timestr=timestr,
                ending=ending))
    return ' '.join(path_list)


def _agg_over_time(file_str,
                   varnamelist,
                   options='',
                   cdo_op=' -yearmonmean ',
                   levelstr=''):
    """Aggregate files along time dimension. Converts to netcdf. Optional cdo
    operator applicable."""
    varnstr = ','.join(varnamelist)
    return cdo.addc(
        0,
        input=cdo_op + ' -select,name=' + varnstr + levelstr + ' ' + file_str,
        options='-r ' + options)


def load(varnamelist=['tos'],
         exp='PM',
         cdo_op='-yearmonmean ',
         model='hamocc',
         outdatatype='data_2d_mm',
         ending='.nc',
         levelstr='',
         **kwargs):
    """Load variable. """
    if exp is 'PM':
        file_str = _get_PM_path(
            model=model, outdatatype=outdatatype, ending=ending, **kwargs)
    elif exp is 'GE':
        file_str = _get_GE_path(
            model=model, outdatatype=outdatatype, ending=ending, **kwargs)
    else:
        print('no fs')

    if ending == '.grb':
        if model == 'echam6':
            if outdatatype in ['co2_mm', 'co2']:
                codes = _get_codes_str('echam6_co2')
                options = '-f nc -t ' + codes
            elif outdatatype in ['tracer', 'tracer_mm']:
                codes = _get_codes_str('echam6_tracer')
                options = '-f nc -t ' + codes
            elif outdatatype == 'BOT_mm':
                options = '-f nc -t echam6'
            else:
                raise ValueError('outdatatype not specified yet!')
        else:
            raise ValueError('model not specified yet!')
    else:
        options = ''

    loaded = _agg_over_time(
        file_str,
        varnamelist,
        options=options,
        cdo_op=cdo_op,
        levelstr=levelstr)
    return loaded


def _load_PM(mmin=0,
             mmax=9,
             initlist=[3014],
             varnamelist=['tos'],
             curv=False,
             exp='PM',
             drop_none=False,
             cdo_op='-yearmonmean ',
             model='hamocc',
             outdatatype='data_2d_mm',
             ending='.nc',
             levelstr='',
             **kwargs):
    if curv:
        chunks = {'time': 21, 'x': 256, 'y': 220}
    else:
        chunks = {'time': 21, 'lat': 96, 'lon': 192}
    dslist = []
    for init in tqdm_notebook(
            initlist, desc='initialization loop', leave=False):
        many_member_ds = xr.concat([
            xr.open_mfdataset(
                load(
                    varnamelist=varnamelist,
                    exp=exp,
                    init=init,
                    m=m,
                    outdatatype=outdatatype,
                    model=model,
                    cdo_op=cdo_op,
                    levelstr=levelstr,
                    ending=ending,
                    **kwargs),
                decode_times=False,
                chunks=chunks,
                preprocess=_squeeze_dims) for m in np.arange(mmin, mmax + 1)
        ],
            dim='member')
        many_member_ds = many_member_ds.assign(
            member=np.arange(mmin, mmax + 1))
        many_member_ds = _set_LY(many_member_ds)
        dslist.append(many_member_ds)
    ds = xr.concat(dslist, dim='initialization')
    ds = ds.assign(initialization=initlist)
    print(ds.nbytes / 1e9, 'GB')
    print(ds.dims)
    return ds


def _load_GE(memberlist=['rcp26', 'rcp45', 'rcp85'],
             initlist=[1, 2, 3, 4, 5],
             varnamelist=['sst'],
             curv=False,
             exp='GE',
             drop_none=False,
             cdo_op='-yearmonmean ',
             model='mpiom',
             outdatatype='data_2d_mm',
             ending='.nc',
             levelstr='',
             **kwargs):
    if curv:
        chunks = {'time': 21, 'x': 256, 'y': 220}
    else:
        chunks = {'time': 21, 'lat': 96, 'lon': 192}
    dslist = []
    for m in tqdm_notebook(initlist, desc='initialization loop', leave=False):
        many_rcp_ds = xr.concat([
            xr.open_mfdataset(
                load(
                    varnamelist=varnamelist,
                    exp=exp,
                    ext=rcp,
                    m=m,
                    outdatatype=outdatatype,
                    model=model,
                    cdo_op=cdo_op,
                    levelstr=levelstr,
                    ending=ending,
                    **kwargs),
                decode_times=False,
                chunks=chunks,
                preprocess=_squeeze_dims) for rcp in memberlist
        ],
            dim='member')
        many_rcp_ds = many_rcp_ds.assign(member=memberlist)
        many_rcp_ds = _set_LY(many_rcp_ds)
        dslist.append(many_rcp_ds)
    ds = xr.concat(dslist, dim='initialization')
    ds = ds.assign(initialization=initlist)
    print(ds.nbytes / 1e9, 'GB')
    print(ds.dims)
    return ds


def postprocess_PM(varnames,
                   initlist=[3014, 3023],
                   model='mpiom',
                   outdatatype='data_2d_mm',
                   levelstr='',
                   timestr='*',
                   ending='.nc',
                   curv=True):
    """Create lead year timeseries for perfect-model experiment.

    Args:
        varnames (type): Description of parameter `varnames`.
        initlist (type): Description of parameter `initlist`. Defaults to [3014, 3023].
        model (type): Description of parameter `model`. Defaults to 'mpiom'.
        outdatatype (type): Description of parameter `outdatatype`. Defaults to 'data_2d_mm'.
        levelstr (type): Description of parameter `levelstr`. Defaults to ''.
        timestr (type): Description of parameter `timestr`. Defaults to '*'.
        ending (type): Description of parameter `ending`. Defaults to '.nc'.
        curv (type): Description of parameter `curv`. Defaults to True.

    Returns:
        type: Description of returned object.

    """
    """Create ym and mm output."""
    cdo_op = ' '  # ' -yearmonmean '
    for control in [False, True]:
        for varname in varnames:
            print(varname, 'control =', control)
            if control:
                ds = load(
                    varnamelist=[varname],
                    cdo_op=cdo_op,
                    model=model,
                    outdatatype=outdatatype,
                    levelstr=levelstr,
                    timestr=timestr,
                    ending=ending,
                    control=True)
                ds = _squeeze_dims(xr.open_dataset(ds))
                ds = convert_C(ds)
            else:
                ds = _load_PM(
                    varnamelist=[varname],
                    initlist=initlist,
                    cdo_op=cdo_op,
                    model=model,
                    outdatatype=outdatatype,
                    levelstr=levelstr,
                    timestr=timestr,
                    curv=curv,
                    ending=ending)
                ds = convert_C(ds)
                ds = _set_mm_span(ds)
            # save mm
            ta = 'mm'
            save(ds, exp='PM', name=varname, control=control, ta=ta)
            # save ym
            if control:
                ds = yearmonmean(ds)
                # ds = _set_LY(ds, first=3000)
                pass
            else:
                ds = _set_LY(yearmonmean(ds))
            ta = 'ym'
            save(ds, exp='PM', name=varname, control=control, ta=ta)


def postprocess_GE(varnames,
                   memberlist=['rcp26', 'rcp45', 'rcp85'],
                   initlist=[1, 2, 3, 4, 5],
                   model='mpiom',
                   outdatatype='data_2d_mm',
                   levelstr='',
                   timestr='*',
                   ending='.nc',
                   curv=True):
    """Create lead year timeseries for Grand Ensemble experiment of a list of varnames for list of extensions and members.

    Args:
        varnames (type): Description of parameter `varnames`.
        memberlist (type): Description of parameter `memberlist`. Defaults to ['rcp26', 'rcp45', 'rcp85'].
        initlist (type): Description of parameter `initlist`. Defaults to [1, 2, 3, 4, 5].
        model (type): Description of parameter `model`. Defaults to 'mpiom'.
        outdatatype (type): Description of parameter `outdatatype`. Defaults to 'data_2d_mm'.
        levelstr (type): Description of parameter `levelstr`. Defaults to ''.
        timestr (type): Description of parameter `timestr`. Defaults to '*'.
        ending (type): Description of parameter `ending`. Defaults to '.nc'.
        curv (type): Description of parameter `curv`. Defaults to True.

    Returns:
        nothing

    Saves:
        ds (xr.Dataset): lead year timeseries ('time','member','ensemble')

    """
    """Create ym."""
    cdo_op = ' -yearmonmean '
    control = False
    for varname in varnames:
        ds = _load_GE(
            varnamelist=[varname],
            memberlist=memberlist,
            initlist=initlist,
            cdo_op=cdo_op,
            model=model,
            outdatatype=outdatatype,
            levelstr=levelstr,
            timestr=timestr,
            curv=curv,
            ending=ending)
        ds = convert_C(ds)
        ds = _set_mm_span(ds)
        # save mm
        # ta = 'mm'
        # save(ds, exp='GE', name=varname, control=control, ta=ta)
        # save ym
        # ds = _set_LY(yearmonmean(ds))
        ta = 'ym'
        save(ds, exp='GE', name=varname, control=control, ta=ta)


def merge_monitoring(exp):
    """Merge all monitoring files of an experiment into one file."""
    pass
