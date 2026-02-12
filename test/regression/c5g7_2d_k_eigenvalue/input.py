import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("c5g7_xs.h5", "r")


# Materials
def set_mat(mat):
    return mcdc.MaterialMG(
        capture=mat["capture"][:],
        scatter=mat["scatter"][:],
        fission=mat["fission"][:],
        nu_p=mat["nu_p"][:],
        nu_d=mat["nu_d"][:],
        chi_p=mat["chi_p"][:],
        chi_d=mat["chi_d"][:],
        speed=mat["speed"][:],
        decay_rate=mat["decay"][:],
    )


# Set the material
mat_uo2 = set_mat(lib["uo2"])  # Fuel: UO2
mat_mox43 = set_mat(lib["mox43"])  # Fuel: MOX 4.3%
mat_mox7 = set_mat(lib["mox7"])  # Fuel: MOX 7.0%
mat_mox87 = set_mat(lib["mox87"])  # Fuel: MOX 8.7%
mat_gt = set_mat(lib["gt"])  # Guide tube
mat_fc = set_mat(lib["fc"])  # Fission chamber
mat_cr = set_mat(lib["cr"])  # Control rod
mat_mod = set_mat(lib["mod"])  # Moderator

# =============================================================================
# Pin cells
# =============================================================================

pitch = 1.26
radius = 0.54

# Surfaces
cy = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)

# Cells
uo2 = mcdc.Cell(region=-cy, fill=mat_uo2)
mox4 = mcdc.Cell(region=-cy, fill=mat_mox43)
mox7 = mcdc.Cell(region=-cy, fill=mat_mox7)
mox8 = mcdc.Cell(region=-cy, fill=mat_mox87)
gt = mcdc.Cell(region=-cy, fill=mat_gt)
fc = mcdc.Cell(region=-cy, fill=mat_fc)
cr = mcdc.Cell(region=-cy, fill=mat_cr)
mod = mcdc.Cell(region=+cy, fill=mat_mod)
modi = mcdc.Cell(region=-cy, fill=mat_mod)  # For all-water lattice

# Universes
u = mcdc.Universe(cells=[uo2, mod])
l = mcdc.Universe(cells=[mox4, mod])
m = mcdc.Universe(cells=[mox7, mod])
n = mcdc.Universe(cells=[mox8, mod])
g = mcdc.Universe(cells=[gt, mod])
f = mcdc.Universe(cells=[fc, mod])
c = mcdc.Universe(cells=[cr, mod])
w = mcdc.Universe(cells=[modi, mod])

# =============================================================================
# Assemblies
# =============================================================================

# Lattices
lattice_uo2 = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, g, u, u, g, u, u, g, u, u, u, u, u],
        [u, u, u, g, u, u, u, u, u, u, u, u, u, g, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, g, u, u, g, u, u, g, u, u, g, u, u, g, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, g, u, u, g, u, u, f, u, u, g, u, u, g, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, g, u, u, g, u, u, g, u, u, g, u, u, g, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, g, u, u, u, u, u, u, u, u, u, g, u, u, u],
        [u, u, u, u, u, g, u, u, g, u, u, g, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
    ],
)

lattice_mox = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, m, m, m, m, g, m, m, g, m, m, g, m, m, m, m, l],
        [l, m, m, g, m, n, n, n, n, n, n, n, m, g, m, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, g, n, n, g, n, n, g, n, n, g, n, n, g, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, g, n, n, g, n, n, f, n, n, g, n, n, g, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, g, n, n, g, n, n, g, n, n, g, n, n, g, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, m, g, m, n, n, n, n, n, n, n, m, g, m, m, l],
        [l, m, m, m, m, g, m, m, g, m, m, g, m, m, m, m, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
    ],
)

lattice_mod = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch * 17, 1],
    y=[-pitch * 17 / 2, pitch * 17, 1],
    universes=[[w]],
)

# Assembly cells
# Surfaces
x0 = mcdc.Surface.PlaneX(x=-pitch * 17 / 2)
x1 = mcdc.Surface.PlaneX(x=pitch * 17 / 2)
y0 = mcdc.Surface.PlaneY(y=-pitch * 17 / 2)
y1 = mcdc.Surface.PlaneY(y=pitch * 17 / 2)
# Cells
assembly_uo2 = mcdc.Cell(region=+x0 & -x1 & +y0 & -y1, fill=lattice_uo2)
assembly_mox = mcdc.Cell(region=+x0 & -x1 & +y0 & -y1, fill=lattice_mox)
assembly_mod = mcdc.Cell(region=+x0 & -x1 & +y0 & -y1, fill=lattice_mod)

# Set assemblies in their respective universes
u_ = mcdc.Universe(cells=[assembly_uo2])
m_ = mcdc.Universe(cells=[assembly_mox])
w_ = mcdc.Universe(cells=[assembly_mod])

# =============================================================================
# Root universe: core
# =============================================================================

# Lattice
lattice_core = mcdc.Lattice(
    x=[-pitch * 17 * 3 / 2, pitch * 17, 3],
    y=[-pitch * 17 * 3 / 2, pitch * 17, 3],
    universes=[[u_, m_, w_], [m_, u_, w_], [w_, w_, w_]],
)

# Core cell
# Surfaces
x0_ = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
x1_ = mcdc.Surface.PlaneX(x=pitch * 17 * 3, boundary_condition="vacuum")
y0_ = mcdc.Surface.PlaneY(y=-pitch * 17 * 3, boundary_condition="vacuum")
y1_ = mcdc.Surface.PlaneY(y=0.0, boundary_condition="reflective")
# Cell
core = mcdc.Cell(
    region=+x0_ & -x1_ & +y0_ & -y1_,
    fill=lattice_core,
    translation=[pitch * 17 * 3 / 2, -pitch * 17 * 3 / 2, 0.0],
)

# Root universe
mcdc.simulation.set_root_universe(cells=[core])

# =============================================================================
# Set source
# =============================================================================

mcdc.Source(
    x=[0.0, pitch * 17 * 2],
    y=[-pitch * 17 * 2, 0.0],
    isotropic=True,
    energy_group=6,
)

# =============================================================================
# Set tallies, settings, techniques, and run MC/DC
# =============================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.linspace(0.0, pitch * 17 * 3, 17 * 3 + 1),
    y=np.linspace(-pitch * 17 * 3, 0.0, 17 * 3 + 1),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

# Settings
mcdc.settings.N_particle = 20
mcdc.settings.census_bank_buffer_ratio = 4.0
mcdc.settings.source_bank_buffer_ratio = 3.0
mcdc.settings.set_eigenmode(N_inactive=1, N_active=2, gyration_radius="infinite-z")

# Techniques
mcdc.simulation.population_control()

# Run
mcdc.settings.set_eigenmode(N_inactive=1, N_active=2, gyration_radius="infinite-z")


mcdc.run()
