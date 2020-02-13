import glob
import os
import warnings

import pandas as pd
import xarray as xr

import cdo

warnings.simplefilter('ignore')
xr.set_options(keep_attrs=True)

cdo = cdo.Cdo()
cdo.forceOutput = True


expid = 'asp_esmControl_PMassim_3014_TSDICALK_over_3170'
exppath = '/work/bm1124/m300524/experiments'
year = 3171  # year to get output labels from
outpath = os.path.expanduser('~/pymistral/')  # folder to save output_df to


def find_all_outdatatypes_in_exp(expid=expid, exppath=exppath, year=year):
    """Find all outdatatypes (tracer,atm,data_2d,...) from experiment `expid`
     from path `exppath` of a given `year`."""
    outdatatypes = []
    for model in ['echam6', 'jsbach', 'mpiom', 'hamocc']:
        path = f'{exppath}/{expid}/outdata/{model}/{expid}_{model}_*_{year}*'
        paths = glob.glob(path)
        for f in paths:
            file = os.path.basename(f)
            # remove expid
            for s in expid.split('_'):
                file = file.strip(s + '_')
            file = file.strip(str(year))
            # print(file)
            parts = file.split('_')
            model = parts[0]
            # print(parts)
            if len(parts[1:]) < 2:
                # print(parts[1:])
                outdatatype = parts[1].split('.')[0]
            else:
                outdatatype = '_'.join(
                    ('_'.join(parts[1:-1]), parts[-1].split('.')[0])
                )
            outdatatype = outdatatype.strip(str(year))
            outdatatype = outdatatype.strip(f'{year}0101_{year}1231')
            if outdatatype == 'co':
                outdatatype = 'co2'
            ending = parts[-1].split('.')[-1]
            print(
                f'Found file: model={model} outdatatype={outdatatype} '
                f'ending={ending}.'
            )
            outdatatypes.append(f'{model}_{outdatatype}')
    return outdatatypes


def read_all_outdatatype_files_to_ds(
    outdatatypes, expid=expid, exppath=exppath, year=year, outpath='~/.'
):
    """Read all outdatatypes from experiment `expid` from path `exppath` of a
     given `year` and return xr.Dataset."""
    ds_list = []
    for outdatatype_id in outdatatypes:
        try:
            print(f'Read {outdatatype_id} to xr.Dataset.')
            parts = outdatatype_id.split('_')
            model = parts[0]
            outdatatype = '_'.join(parts[1:])
            path = (
                f'{exppath}/{expid}/outdata/{model}/'
                + f'{expid}_{model}_{outdatatype}_{year}*'
            )
            if model in ['jsbach', 'echam6']:
                options = ' '
                if (
                    'BOT' in outdatatype
                    or 'ATM' in outdatatype
                    or 'LOG' in outdatatype
                ):
                    options += ' -t echam6'
                else:
                    table = (
                        f'{exppath}/{expid}/log/'
                        + f'{expid}_{model}_{outdatatype[:2]}*.codes'
                    )
                    options += ' -t ' + table
            else:
                options = ''
            ds = cdo.copy(input=path, options=options, returnXDataset=True)
            if 'time' not in ds.dims:
                ds = ds.expand_dims('time')
            ds.to_netcdf(f'{outpath}/sample_files/{model}_{outdatatype}.nc')
            # add outdatatype
            for v in ds.data_vars:
                ds[v].attrs['outdatatype'] = outdatatype
                ds[v].attrs['model'] = model
                ds[v].attrs['dims'] = list(ds[v].dims)
            ds_list.append(ds.isel(time=0).mean())
        except:
            print(f'{outdatatype_id} failed'
    return xr.merge(ds_list, compat='override')


def create_dataframe_of_output_info(ds, outpath=outpath):
    """Create pd.Dataframe about output from `ds` and save to `outpath`."""
    df = pd.DataFrame(
        index=ds.data_vars,
        columns=[
            'varname',
            'long_name',
            'code',
            'table',
            'units',
            'model',
            'outdatatype',
            'dims',
        ],
    )
    for v in list(ds.data_vars):
        for c in list(df.columns):
            df[c].loc[v] = ds[v].attrs[c] if c in ds[v].attrs else ''
    df['stream'] = df['model'] + '_' + df['outdatatype']
    df.to_csv(outpath + 'MPI-ESM-1-2-LR_output.csv')
    return df


def generate_output_df(
    expid=expid, exppath=exppath, year=year, outpath=outpath, recalc=False
):
    """Combine all functions above to generate output from `expid` or just load
     if not `recalc`."""
    path = outpath + 'MPI-ESM-1-2-LR_output.csv'
    if not recalc and os.path.exists(path):
        print(f'Read df from path: {path}')
        output_df = (
            pd.read_csv(path, index_col='varname')
            .rename(columns={'Unnamed: 0': 'varname'})
            .set_index('varname')
        )
    else:
        outdatatypes = find_all_outdatatypes_in_exp(
            expid=expid, exppath=exppath, year=year
        )
        ds = read_all_outdatatype_files_to_ds(
            outdatatypes, expid=expid, exppath=exppath, year=year, outpath=outpath
        )
        output_df = create_dataframe_of_output_info(ds, outpath=outpath)
    return output_df


output_df = generate_output_df(recalc=False)


def get_model_outdatatype_from_var(
    var, output_df=output_df, expid=expid, exppath=exppath
):
    if isinstance(var, list):  # if list check first
        var = var[0]
    if not isinstance(output_df, pd.DataFrame):
        raise ValueError(
            f'df needs to be pd.Dataframe, found {type(output_df)}'
        )
    model = output_df.T[var]['model']
    outdatatype = output_df.T[var]['outdatatype']
    if model in ['mpiom', 'hamocc']:
        ending = 'nc'
        options = ''
    else:
        ending = 'grb'
        options = ' -f nc '
        if (
            'BOT' in outdatatype
            or 'ATM' in outdatatype
            or 'LOG' in outdatatype
        ):
            options += ' -t echam6 '
        else:
            table = f'{exppath}/{expid}/log/{expid}_{model}_{outdatatype[:2]}*.codes'
            options += '-t ' + table + ' '
    return model, outdatatype, options, ending


def load_var_cdo(
    var, expid=expid, exppath=exppath, output_df=output_df, timestr='', sel=''
):
    """Load variable `var` from experiment_id `expid` with cdo into xarray."""
    model, outdatatype, options, ending = get_model_outdatatype_from_var(
        var, output_df=output_df
    )

    def load_cdo(var, path, options, sel=sel):
        print(f'cdo {options} -select,name={var}{sel} {path}')
        return cdo.select(
            f'name={var}{sel}',
            input=path,
            returnXDataset=True,
            options=options,
        )

    path = f'{exppath}/{expid}/outdata/{model}/{expid}_{model}_{outdatatype}_{timestr}*'
    ds = load_cdo(var, path, options, sel=sel).squeeze()

    return ds
