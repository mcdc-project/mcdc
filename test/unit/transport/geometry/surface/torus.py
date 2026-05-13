import pytest

import mcdc


def test_zero_axis_error():
    with pytest.raises(SystemExit):
        mcdc.Surface.Torus(axis=[0.0, 0.0, 0.0], R=1.0, r=0.5)
