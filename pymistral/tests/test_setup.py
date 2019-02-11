import pytest
from pymistral.setup import my_system


def test_host_is_local_or_mistral():
    assert my_system in ['local', 'mistral']
