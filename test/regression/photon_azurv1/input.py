"""AZURV1 photon benchmark with energy-independent cross sections.

An isotropic photon pulse at the origin of an effectively infinite
medium with constant cross sections, scattering ratio c = 0.5. This is
the analytical transport problem the constant-cross-section treatment
exists for, so it exercises that path rather than the tabulated one.

The tally time grid is expressed in mean free times and converted to
seconds, because MC/DC's time unit is the second and its speed of light
is cm/s -- the pre-port deck used cm/ns.

Migrated from ``photon_transport_code/MCNP_Verification_Tests/Error-Convergence_testing/AZURV1_photon_v3.py``.
The physics is unchanged; the API is not. Energies are eV rather than MeV,
times are seconds rather than nanoseconds, materials are element compositions
rather than ``PhotonMaterial``, and settings hang off the ``Simulation``.
"""

import numpy as np
import os

import mcdc

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

simulation = mcdc.Simulation("AZURV1 photon")

from mcdc.constant import LIGHT_SPEED

# ======================================================================================
# Set model
# ======================================================================================

SIGMA_TOTAL = 1.0
SCATTERING_RATIO = 0.5
SIGMA_SCATTER = SCATTERING_RATIO * SIGMA_TOTAL
SIGMA_ABSORB = SIGMA_TOTAL - SIGMA_SCATTER

medium = mcdc.Material(
    photon_constant_xs=mcdc.PhotonConstantXSData(
        scatter=SIGMA_SCATTER, absorb=SIGMA_ABSORB
    ),
    name="AZURV1 medium",
)

# Reflecting walls far enough away to stand in for an infinite medium.
s1 = mcdc.Surface.PlaneX(x=-1.0e10, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneX(x=1.0e10, boundary_condition="reflective")
simulation.set_model([mcdc.Cell(region=+s1 & -s2, fill=medium)])

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(
    x=0.0,
    y=0.0,
    z=0.0,
    time=0.0,
    isotropic=True,
    energy=1.0e6,
    particle_type="photon",
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

# Mean free times, converted to seconds through the photon speed.
MEAN_FREE_TIMES = np.linspace(0.0, 20.0, 21)
times = MEAN_FREE_TIMES / (SIGMA_TOTAL * LIGHT_SPEED)

mesh = mcdc.MeshStructured(x=np.linspace(-20.5, 20.5, 202))
simulation.set_tallies(
    [mcdc.Tally(mesh=mesh, scores=["flux"], time=times, particle_type="photon")]
)

simulation.settings.N_particle = 1000
simulation.settings.N_batch = 2
simulation.settings.rng_seed = 42

simulation.run()
