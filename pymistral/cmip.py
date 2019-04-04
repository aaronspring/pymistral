# workaround until this works:
# # The library auto-initializes upon import.
# import pyessv

import glob
import itertools
import os

import cdo
import pandas as pd
import xarray as xr

from .setup import _squeeze_dims, cmip5_folder, my_system, tmp

cdo = cdo.Cdo(tempdir=tmp)

if my_system is 'local':
    CV_basefolder = '/Users/aaron.spring/Coding/'
elif my_system is 'mistral':
    CV_basefolder = '/home/mpim/m300524/CMIP6_CVs/'

# CMIP6
# read in all institutions
name = 'institution_id'
cvpath = CV_basefolder+'CMIP6_CVs/CMIP6_'+name+'.json'
institution_ids = pd.read_json(
    cvpath).index[:-6].drop(['CV_collection_version', 'CV_collection_modified'])

# read in all models
name = 'source_id'
cvpath = CV_basefolder+'CMIP6_CVs/CMIP6_'+name+'.json'
model_ids = pd.read_json(
    cvpath).index[:-6].drop(['CV_collection_version', 'CV_collection_modified']).values

# read in all sources
name = 'source_id'
cvpath = CV_basefolder+'CMIP6_CVs/CMIP6_'+name+'.json'
source_ids = pd.read_json(cvpath)

# read in all activities/MIPs
name = 'activity_id'
cvpath = CV_basefolder+'CMIP6_CVs/CMIP6_'+name+'.json'
mip_ids = pd.read_json(
    cvpath).index[:-6].drop(['CV_collection_version', 'CV_collection_modified']).values
mip_table = pd.read_json(cvpath).drop(
    ['CV_collection_version', 'CV_collection_modified'])
mip_longnames = pd.read_json(cvpath)[
    'activity_id'][:-6].drop(['CV_collection_version', 'CV_collection_modified']).values

# read in experiments
name = 'experiment_id'
cvpath = CV_basefolder+'CMIP6_CVs/CMIP6_'+name+'.json'
experiment_ids = pd.read_json(cvpath).index.drop(
    ['CV_collection_modified', 'CV_collection_version', 'author']).values

# wrappers using the above


def CMIP6_CV_model_participations(model):
    """Returns MIPs a model participates in.
    Args:
        model (str): model from model_ids
    Returns:
        (list) of MIPs a model participates in.

    Example:
        CMIP6_CV_model_participations('MPI-ESM1-2-HR')
    """
    s = source_ids.loc[model].values[0]
    return s['activity_participation']


def participation_of_models(mip):
    """Return a list of all CMIP6 models participating in a MIP.

    Args:
        mip (str): MIP from mip_ids

    Example:
        participation_of_models('C4MIP')
        participation_of_models('DCPP')
    """
    mip_models = []
    for model in model_ids:
        if mip in CMIP6_CV_model_participations(model):
            mip_models.append(model)
    return mip_models


# CMIP5 on mistral
cmip5_centers_mistral = os.listdir(cmip5_folder)

cmip5_models_mistral = {}
for center in cmip5_centers_mistral:
    models = os.listdir('/'.join((cmip5_folder, center)))
    cmip5_models_mistral[center] = models

cmip5_all_models_mistral = list(
    itertools.chain.from_iterable(cmip5_models_mistral.values()))


def _get_path_cmip(base_folder=cmip5_folder,
                   model='MPI-ESM-LR',
                   center='MPI-M',
                   exp='historical',
                   period='mon',
                   varname='tos',
                   comp='ocean',
                   run_id='r1i1p1',
                   ending='.nc',
                   timestr='*',
                   **kwargs):
    try:
        path_v = sorted(glob.glob('/'.join([base_folder, center, model, exp,
                                            period, comp, comp[0].upper()+period, run_id, 'v????????'])))[-1]
        return path_v + '/' + varname + '/' + '_'.join([varname, comp[0].upper()+period, model, exp, run_id, timestr]) + ending
    except:
        return '/'.join([base_folder, center, model, exp,
                         period, comp, comp[0].upper()+period, run_id])


# wrapper to check which data is available
def find_cmip5_output(**kwargs):
    """Find available CMIP5 output on mistral. Returns model and center list."""
    output_models = []
    output_centers = []
    for center in cmip5_centers_mistral:
        for model in cmip5_models_mistral[center]:
            filestr = _get_path_cmip(model=model, center=center, **kwargs)
            if glob.glob(filestr) != []:
                # print(model,center,'exists')
                output_models.append(model)
                output_centers.append(center)
    print(len(output_models))
    return output_centers, output_models


# TODO: adapt for CMIP6, maybe with CMIP=5 arg
def load_cmip(base_folder=cmip5_folder,
              model='MPI-ESM-LR',
              center='MPI-M',
              exp='historical',
              period='mon',
              varname='tos',
              comp='ocean',
              run_id='r1i1p1',
              ending='.nc',
              timestr='*',
              operator='',
              select=''):
    """Load a variable from CMIP5."""
    ncfiles_cmip = _get_path_cmip(
        base_folder=cmip5_folder,
        model=model,
        center=center,
        exp=exp,
        period=period,
        varname=varname,
        comp=comp,
        run_id=run_id,
        ending=ending,
        timestr=timestr)
    nfiles = len(glob.glob(ncfiles_cmip))
    if nfiles is 0:
        raise ValueError('no files found in', ncfiles_cmip)
        # # TODO: check all args for reasonable inputs, check path exists explicitly
    print('Load', nfiles, 'files from:', ncfiles_cmip)
    if operator is not '':
        print('preprocessing: cdo', operator, ncfiles_cmip)
        return xr.open_dataset(
            cdo.addc(
                '0',
                input=operator + ' -select,name='+varname + select + ' ' +
                ncfiles_cmip,
                options='-r')).squeeze()[varname]
    else:
        print('xr.open_mfdataset('+ncfiles_cmip+')['+varname+']')
        return xr.open_mfdataset(ncfiles_cmip, concat_dim='time')[varname]


def load_cmip5_from_center_model_list(center_list=['MPI-M', 'NCAR'], model_list=['MPI-ESM-LR', 'CCSM4'], **cmip_kwargs):
    data = []
    for center, model in zip(center_list, model_list):
        print('Load', center, model)
        data.append(load_cmip(center=center, model=model, **cmip_kwargs))
    data = xr.concat(data, 'mode')
    data['model'] = model_list
    return data


def get_center_for_cmip5_model(model):
    """Get center name for a CMIP5 model based on CMIP5 centers and models found on mistral."""
    for center in cmip5_centers_mistral:
        if model in cmip5_models_mistral[center]:
            return center


def load_cmip5_from_model_list(model_list=['MPI-ESM-LR', 'CCSM4'], **cmip_kwargs):
    """Load CMIP5 output from mistral based on model_list.

    experiment_id, variables, ... to be specified in **cmip_kwargs."""
    data = []
    ml = model_list.copy()
    for model in model_list:
        center = get_center_for_cmip5_model(model)
        print(center, model)
        filestr = _get_path_cmip(model=model, center=center, **cmip_kwargs)
        if glob.glob(filestr) != []:
            new = load_cmip(center=center, model=model, **cmip_kwargs)
            new = _squeeze_dims(new)
            data.append(new)
        else:
            print('not found', filestr)
            ml.remove(model)
    try:
        data = xr.concat(data, 'model')
        data['model'] = ml
    except:
        print('some error: returns list')
    return data


def load_cmip5_many_varnames(varnamelist=['tos', 'sos'], **cmip_kwargs):
    """Load many variables from varnamelist from CMIP5 output from mistral.

    experiment_id, model_ids, ... to be specified in **cmip_kwargs."""
    data = []
    for varname in varnamelist:
        print('Load', varname)
        data.append(load_cmip(varname=varname, **cmip_kwargs))
    data = xr.merge(data)
    return data
