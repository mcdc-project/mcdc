import mcdc
import mcdc.type_
import numpy as np

# ======================================================================================
# Nuclide
# ======================================================================================


def test_nuclide_basic():
    """Create a nuclide with complete definition."""
    mcdc.reset()

    n1 = mcdc.nuclide(
        capture=np.array([1.0, 2.0, 3.0, 4.0]),
        fission=np.array([1.0, 2.0, 3.0, 4.0]),
        scatter=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        nu_s=np.array([1.0, 2.0, 3.0, 4.0]),
        nu_p=np.array([1.0, 2.0, 3.0, 4.0]),
        nu_d=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        chi_p=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        chi_d=np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        speed=np.array([1.0, 2.0, 3.0, 4.0]),
        decay=np.array([1.0, 2.0, 3.0]),
    )

    assert n1.tag == "Nuclide"
    assert n1.ID == 0
    assert n1.G == 4
    assert n1.J == 3
    assert (n1.speed == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (n1.decay == np.array([1.0, 2.0, 3.0])).all()
    assert (n1.capture == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (n1.scatter == np.array([4.0, 8.0, 12.0, 16.0])).all()
    assert (n1.fission == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (n1.total == np.array([6.0, 12.0, 18.0, 24.0])).all()
    assert (n1.nu_s == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (n1.nu_p == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (
        n1.nu_d
        == np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
    ).all()
    assert (n1.nu_f == np.array([4.0, 8.0, 12.0, 16.0])).all()
    assert (n1.chi_s == np.ones([4, 4]) * 0.25).all()
    assert (n1.chi_p == np.ones([4, 4]) * 0.25).all()
    assert (n1.chi_d == np.ones([3, 4]) * 0.25).all()


def test_nuclide_default():
    """Create nuclides with incomplete definitions."""
    mcdc.reset()

    n1 = mcdc.nuclide(capture=np.ones(5))
    n2 = mcdc.nuclide(scatter=np.ones((5, 5)))
    n3 = mcdc.nuclide(fission=np.ones(5), nu_p=np.ones(5), chi_p=np.ones((5, 5)))

    # Checks
    assert n1.tag == "Nuclide"
    assert n1.ID == 0
    assert n1.G == 5
    assert n1.J == 0
    assert (n1.speed == np.ones(5)).all()
    assert (n1.decay == np.zeros(0)).all()
    assert (n1.capture == np.ones(5)).all()
    assert (n1.scatter == np.zeros(5)).all()
    assert (n1.fission == np.zeros(5)).all()
    assert (n1.nu_s == np.ones(5)).all()
    assert (n1.nu_p == np.zeros(5)).all()
    assert (n1.nu_f == np.zeros(5)).all()
    assert (n1.nu_d == np.zeros((5, 0))).all()
    assert (n1.chi_s == np.zeros((5, 5))).all()
    assert (n1.chi_p == np.zeros((5, 5))).all()
    assert (n1.chi_d == np.zeros((0, 5))).all()

    assert (n2.capture == np.zeros(5)).all()
    assert (n2.scatter == np.ones(5) * 5.0).all()
    assert (n2.fission == np.zeros(5)).all()
    assert (n1.nu_s == np.ones(5)).all()
    assert (n2.nu_p == np.zeros(5)).all()
    assert (n2.nu_f == np.zeros(5)).all()
    assert (n2.nu_d == np.zeros((5, 0))).all()
    assert (n2.chi_s == np.ones((5, 5)) * 0.2).all()
    assert (n2.chi_p == np.zeros((5, 5))).all()
    assert (n2.chi_d == np.zeros((0, 5))).all()

    assert (n3.capture == np.zeros(5)).all()
    assert (n3.scatter == np.zeros(5)).all()
    assert (n3.fission == np.ones(5)).all()
    assert (n1.nu_s == np.ones(5)).all()
    assert (n3.nu_p == np.ones(5)).all()
    assert (n3.nu_f == np.ones(5)).all()
    assert (n3.nu_d == np.zeros((5, 0))).all()
    assert (n3.chi_s == np.zeros((5, 5))).all()
    assert (n3.chi_p == np.ones((5, 5)) * 0.2).all()
    assert (n3.chi_d == np.zeros((0, 5))).all()


# ======================================================================================
# Material
# ======================================================================================


def test_material_single():
    # Start fresh
    mcdc.reset()

    # Create a single-nuclide material
    m1 = mcdc.material(
        capture=np.array([1.0, 2.0, 3.0, 4.0]),
        fission=np.array([1.0, 2.0, 3.0, 4.0]),
        scatter=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        nu_s=np.array([1.0, 2.0, 3.0, 4.0]),
        nu_p=np.array([1.0, 2.0, 3.0, 4.0]),
        nu_d=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        chi_p=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
        chi_d=np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        speed=np.array([1.0, 2.0, 3.0, 4.0]),
        decay=np.array([1.0, 2.0, 3.0]),
    )

    # Checks
    assert m1.tag == "Material"
    assert m1.ID == 0
    assert m1.N_nuclide == 1
    assert (m1.nuclide_IDs == np.array([0])).all()
    assert (m1.nuclide_densities == np.array([1.0])).all()
    assert m1.G == 4
    assert m1.J == 3
    assert (m1.speed == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (m1.capture == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (m1.scatter == np.array([4.0, 8.0, 12.0, 16.0])).all()
    assert (m1.fission == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (m1.total == np.array([6.0, 12.0, 18.0, 24.0])).all()
    assert (m1.nu_s == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (m1.nu_p == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (
        m1.nu_d
        == np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
    ).all()
    assert (m1.nu_f == np.array([4.0, 8.0, 12.0, 16.0])).all()
    assert (m1.chi_s == np.ones([4, 4]) * 0.25).all()
    assert (m1.chi_p == np.ones([4, 4]) * 0.25).all()

    # Check if the nuclide was registered
    n2 = mcdc.nuclide(capture=np.ones(5))
    assert n2.ID == 1


def test_material_multi():
    # Start fresh
    mcdc.reset()

    # Create a multi-nuclide material
    n1 = mcdc.nuclide(capture=np.ones(5), speed=np.ones(5) * 1)
    n2 = mcdc.nuclide(scatter=np.ones((5, 5)), speed=np.ones(5) * 2)
    n3 = mcdc.nuclide(
        fission=np.ones(5), nu_p=np.ones(5), chi_p=np.ones((5, 5)), speed=np.ones(5) * 3
    )
    m1 = mcdc.material(nuclides=[(n1, 1.0), (n2, 2.0), (n3, 3.0)])

    # Checks
    assert m1.tag == "Material"
    assert m1.ID == 0
    assert m1.N_nuclide == 3
    assert (m1.nuclide_IDs == np.array([0, 1, 2])).all()
    assert (m1.nuclide_densities == np.array([1.0, 2.0, 3.0])).all()
    assert m1.G == 5
    assert m1.J == 0
    assert (m1.total == np.ones(5) * 14).all()
    assert (m1.speed == np.ones(5) * 30 / 14).all()


# ======================================================================================
# Surface
# ======================================================================================


def test_surface_input_lower():
    type_ = "pLaNe x"
    result = mcdc.surface(type_, bc="RefLeCtiVe", x=0.0)
    assert result.boundary_type == "reflective"
    assert result.A == 0.0
    assert result.B == 0.0
    assert result.C == 0.0
    assert result.D == 0.0
    assert result.E == 0.0
    assert result.F == 0.0
    assert result.G == 1.0
    assert result.H == 0.0
    assert result.I == 0.0
    assert (result.J == np.array([[-0.0, 0.0]])).all()
    assert result.linear


# ======================================================================================
# Reset
# ======================================================================================


def test_reset():
    # Start fresh
    mcdc.reset()

    # ID reset
    n = mcdc.nuclide(capture=np.ones(5))
    assert n.ID == 0

    mcdc.reset()

    n1 = mcdc.nuclide(capture=np.ones(5))
    assert n1.ID == 0
