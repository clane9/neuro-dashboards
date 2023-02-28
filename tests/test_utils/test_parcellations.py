import numpy as np
import pytest

from neuro_dashboards.utils.parcellations import get_schaefer


def test_get_schaefer():
    parc = get_schaefer(parcels=200, networks=7)
    assert len(parc) == 200

    poly_list = parc.poly_list(values=np.random.randn(200))
    assert len(poly_list) == 272


if __name__ == "__main__":
    pytest.main(["-s", __file__])
