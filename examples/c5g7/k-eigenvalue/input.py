import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("../MGXS-C5G7-TD.h5", "r")


# Setter
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


# Materials
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
core_height = 128.52
refl_thick = 21.42

# Control rod banks fractions
#   All out: 0.0
#   All in : 1.0
cr1 = 0.0
cr2 = 0.0
cr3 = 0.0
cr4 = 0.0
# Control rod banks interfaces
cr1 = core_height * (0.5 - cr1)
cr2 = core_height * (0.5 - cr2)
cr3 = core_height * (0.5 - cr3)
cr4 = core_height * (0.5 - cr4)

# Surfaces
cy = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)
z1 = mcdc.Surface.PlaneZ(z=cr1)  # Control rod banks interfaces
z2 = mcdc.Surface.PlaneZ(z=cr2)
z3 = mcdc.Surface.PlaneZ(z=cr3)
z4 = mcdc.Surface.PlaneZ(z=cr4)
zf = mcdc.Surface.PlaneZ(z=core_height / 2)

# Fission chamber
fc = mcdc.Cell(-cy, mat_fc)
mod = mcdc.Cell(+cy, mat_mod)
fission_chamber = mcdc.Universe(cells=[fc, mod])

# Fuel rods
uo2 = mcdc.Cell(-cy & -zf, mat_uo2)
mox4 = mcdc.Cell(-cy & -zf, mat_mox43)
mox7 = mcdc.Cell(-cy & -zf, mat_mox7)
mox8 = mcdc.Cell(-cy & -zf, mat_mox87)
moda = mcdc.Cell(-cy & +zf, mat_mod)  # Water above pin
fuel_uo2 = mcdc.Universe(cells=[uo2, mod, moda])
fuel_mox43 = mcdc.Universe(cells=[mox4, mod, moda])
fuel_mox7 = mcdc.Universe(cells=[mox7, mod, moda])
fuel_mox87 = mcdc.Universe(cells=[mox8, mod, moda])

# Control rods and guide tubes
cr1 = mcdc.Cell(-cy & +z1, mat_cr)
cr2 = mcdc.Cell(-cy & +z2, mat_cr)
cr3 = mcdc.Cell(-cy & +z3, mat_cr)
cr4 = mcdc.Cell(-cy & +z4, mat_cr)
gt1 = mcdc.Cell(-cy & -z1, mat_gt)
gt2 = mcdc.Cell(-cy & -z2, mat_gt)
gt3 = mcdc.Cell(-cy & -z3, mat_gt)
gt4 = mcdc.Cell(-cy & -z4, mat_gt)
control_rod1 = mcdc.Universe(cells=[cr1, gt1, mod])
control_rod2 = mcdc.Universe(cells=[cr2, gt2, mod])
control_rod3 = mcdc.Universe(cells=[cr3, gt3, mod])
control_rod4 = mcdc.Universe(cells=[cr4, gt4, mod])

# =============================================================================
# Fuel lattices
# =============================================================================

# UO2 lattice 1
u = fuel_uo2
c = control_rod1
f = fission_chamber
lattice_1 = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, c, u, u, c, u, u, c, u, u, u, u, u],
        [u, u, u, c, u, u, u, u, u, u, u, u, u, c, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, c, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, f, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, c, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, c, u, u, u, u, u, u, u, u, u, c, u, u, u],
        [u, u, u, u, u, c, u, u, c, u, u, c, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
    ],
)

# MOX lattice 2
l = fuel_mox43
m = fuel_mox7
n = fuel_mox87
c = control_rod2
f = fission_chamber
lattice_2 = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, m, m, m, m, c, m, m, c, m, m, c, m, m, m, m, l],
        [l, m, m, c, m, n, n, n, n, n, n, n, m, c, m, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, c, n, n, c, n, n, c, n, n, c, n, n, c, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, c, n, n, c, n, n, f, n, n, c, n, n, c, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, c, n, n, c, n, n, c, n, n, c, n, n, c, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, m, c, m, n, n, n, n, n, n, n, m, c, m, m, l],
        [l, m, m, m, m, c, m, m, c, m, m, c, m, m, m, m, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
    ],
)

# MOX lattice 3
l = fuel_mox43
m = fuel_mox7
n = fuel_mox87
c = control_rod3
f = fission_chamber
lattice_3 = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, m, m, m, m, c, m, m, c, m, m, c, m, m, m, m, l],
        [l, m, m, c, m, n, n, n, n, n, n, n, m, c, m, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, c, n, n, c, n, n, c, n, n, c, n, n, c, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, c, n, n, c, n, n, f, n, n, c, n, n, c, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, c, n, n, c, n, n, c, n, n, c, n, n, c, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, m, c, m, n, n, n, n, n, n, n, m, c, m, m, l],
        [l, m, m, m, m, c, m, m, c, m, m, c, m, m, m, m, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
    ],
)

# UO2 lattice 4
u = fuel_uo2
c = control_rod4
f = fission_chamber
lattice_4 = mcdc.Lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, c, u, u, c, u, u, c, u, u, u, u, u],
        [u, u, u, c, u, u, u, u, u, u, u, u, u, c, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, c, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, f, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, c, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, c, u, u, u, u, u, u, u, u, u, c, u, u, u],
        [u, u, u, u, u, c, u, u, c, u, u, c, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
    ],
)

# =============================================================================
# Assemblies and core
# =============================================================================

# Surfaces
x0 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=pitch * 17)
x2 = mcdc.Surface.PlaneX(x=pitch * 17 * 2)
x3 = mcdc.Surface.PlaneX(x=pitch * 17 * 3, boundary_condition="vacuum")

y0 = mcdc.Surface.PlaneY(y=-pitch * 17 * 3, boundary_condition="vacuum")
y1 = mcdc.Surface.PlaneY(y=-pitch * 17 * 2)
y2 = mcdc.Surface.PlaneY(y=-pitch * 17)
y3 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="reflective")

z0 = mcdc.Surface.PlaneZ(z=-(core_height / 2 + refl_thick), boundary_condition="vacuum")
z1 = mcdc.Surface.PlaneZ(z=-(core_height / 2))
z2 = mcdc.Surface.PlaneZ(z=(core_height / 2 + refl_thick), boundary_condition="vacuum")

# Assembly cells
center = np.array([pitch * 17 / 2, -pitch * 17 / 2, 0.0])
assembly_1 = mcdc.Cell(+x0 & -x1 & +y2 & -y3 & +z1 & -z2, lattice_1, translation=center)

center += np.array([pitch * 17, 0.0, 0.0])
assembly_2 = mcdc.Cell(+x1 & -x2 & +y2 & -y3 & +z1 & -z2, lattice_2, translation=center)

center += np.array([-pitch * 17, -pitch * 17, 0.0])
assembly_3 = mcdc.Cell(+x0 & -x1 & +y1 & -y2 & +z1 & -z2, lattice_3, translation=center)

center += np.array([pitch * 17, 0.0, 0.0])
assembly_4 = mcdc.Cell(+x1 & -x2 & +y1 & -y2 & +z1 & -z2, lattice_4, translation=center)

# Bottom reflector cell
reflector_bottom = mcdc.Cell(+x0 & -x3 & +y0 & -y3 & +z0 & -z1, mat_mod)

# Side reflectors
reflector_south = mcdc.Cell(+x0 & -x3 & +y0 & -y1 & +z1 & -z2, mat_mod)
reflector_east = mcdc.Cell(+x2 & -x3 & +y1 & -y3 & +z1 & -z2, mat_mod)

# Root universe
mcdc.simulation.set_root_universe(
    cells=[
        assembly_1,
        assembly_2,
        assembly_3,
        assembly_4,
        reflector_bottom,
        reflector_south,
        reflector_east,
    ],
)

# =============================================================================
# Set source
# =============================================================================

mcdc.Source(
    x=[0.0, pitch * 17 * 2],
    y=[-pitch * 17 * 2, 0.0],
    z=[-core_height / 2, core_height / 2],
    isotropic=True,
    energy_group=0,  # Highest energy
)

# =============================================================================
# Set tallies, settings, techniques and run MC/DC
# =============================================================================

# Tally
x_grid = np.linspace(0.0, pitch * 17 * 3, 17 * 3 + 1)
y_grid = np.linspace(-pitch * 17 * 3, 0.0, 17 * 3 + 1)
z_grid = np.linspace(
    -(core_height / 2 + refl_thick), (core_height / 2 + refl_thick), 102 + 17 * 2 + 1
)
g_grid = np.array([-0.5, 3.5, 6.5])  # Collapsing to fast (1-4) and slow (5-7)
mesh = mcdc.MeshStructured(x=x_grid, y=y_grid, z=z_grid)
mcdc.TallyMesh(mesh=mesh, scores=["flux"], energy=g_grid)

# Settings
mcdc.settings.N_particle = 50
mcdc.settings.census_bank_buffer_ratio = 4.0
mcdc.settings.set_eigenmode(N_inactive=5, N_active=10, gyration_radius="all")

# Techniques
mcdc.simulation.population_control()

# Run
mcdc.run()
