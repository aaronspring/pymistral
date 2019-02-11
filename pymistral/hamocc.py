import numpy as np


def nutrients_in_po4_units(ds):
    nutrient_varnames = ['no3os', 'po4os', 'dfeos']
    for nutrient in nutrient_varnames:
        if nutrient not in ds.data_vars:
            nutrient_varnames.remove(nutrient)
    if 'no3os' in ds.data_vars:
        ds['no3os'] = ds['no3os'] * 16
    if 'dfeos' in ds.data_vars:
        ds['dfeos'] = ds['dfeos'] * 2732.240437
    return ds[nutrient_varnames]


def get_nutlimf(ds):
    for _ in ['po4os', 'no3os', 'dfeos']:
        if _ not in ds:
            raise ValueError('missing', _)
    # convert all into phos units
    iron = ds['dfeos'] * 2732.240437
    nitrate = ds['no3os'] / 16
    phos = ds['po4os']
    nmask = nitrate < phos
    pmask = phos < nitrate
    pmaskphos = pmask * phos
    nmasknit = nmask * nitrate
    xa1 = pmaskphos + nmasknit
    imask = iron < xa1
    xmask = xa1 < iron
    tmp1 = iron * imask
    tmp2 = xa1 * xmask
    xa = tmp1 + tmp2
    xabkphyt = 1e-8 + xa
    nutlim = xa / xabkphyt
    nutlim.name = 'Nutrient availability growth factor'
    # iron lowest
    iron_lim = iron.where(iron < nitrate) < phos
    iron_lim.name = 'Iron limitation'
    nitrate_lim = nitrate.where(nitrate < iron) < phos
    nitrate_lim.name = 'Nitrate limitation'
    phos_lim = phos.where(phos < iron) < nitrate
    phos_lim.name = 'Phosphorous limitation'
    return nutlim, iron_lim, nitrate_lim, phos_lim


def temfa_phofa(ds):
    """Get primary productivity growth factor due to light and temperature at surface."""
    temfa = .6 * 1.066**(ds['tsw'] - 273.15)
    phofa = ds['soflwac'] * 0.02
    return temfa * phofa / (np.sqrt(phofa**2 + temfa**2))
