import os

import pytest
import xarray as xr

import intake

cat_str = '/home/mpim/m300524/pymistral/intake/obs.yml'


#@pytest.mark.skip(reason='Takes too long')
@pytest.mark.skipif(not os.path.exists(cat_str), reason="needs to find file on mistral for testing")
def test_every_item_in_intake_obs_catalog():
    cat = intake.open_catalog(cat_str)
    ignore = ['AVISO','MODIS']
    for i in list(cat):
        if i in ignore:
            continue
        try:
            cat[i].to_dask()
            print(i, 'works')
        except:
            print(i, 'fails')
