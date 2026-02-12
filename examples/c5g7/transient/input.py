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
reflector_thickness = 21.42

# Control rod banks fractions
#   All out: 0.0
#   All in : 1.0
cr1 = np.array([1.0, 1.0, 0.89, 1.0])
cr1_t = np.array([0.0, 10.0, 15.0, 15.0 + 1.0 - cr1[-2]])

cr2 = np.array([1.0, 1.0, 0.0, 0.0, 0.8])
cr2_t = np.array([0.0, 5.0, 10.0, 15.0, 15.8])

cr3 = np.array([0.75, 0.75, 1.0])
cr3_t = np.array([0.0, 15.0, 15.25])

cr4 = np.array([1.0, 1.0, 0.5, 0.5, 1.0])
cr4_t = np.array(
    [0.0, 5.0, 5.0 + (cr4[1] - cr4[2]) / 2 * 10, 15.0, 15.0 + 1.0 - cr4[-2]]
)

# Tips of the control rod banks
cr1_bottom = core_height * (0.5 - cr1)
cr2_bottom = core_height * (0.5 - cr2)
cr3_bottom = core_height * (0.5 - cr3)
cr4_bottom = core_height * (0.5 - cr4)
cr1_top = cr1_bottom + core_height
cr2_top = cr2_bottom + core_height
cr3_top = cr3_bottom + core_height
cr4_top = cr4_bottom + core_height

# Durations of the moving tips
cr1_durations = cr1_t[1:] - cr1_t[:-1]
cr2_durations = cr2_t[1:] - cr2_t[:-1]
cr3_durations = cr3_t[1:] - cr3_t[:-1]
cr4_durations = cr4_t[1:] - cr4_t[:-1]

# Velocities of the moving tips
cr1_velocities = np.zeros((len(cr1) - 1, 3))
cr2_velocities = np.zeros((len(cr2) - 1, 3))
cr3_velocities = np.zeros((len(cr3) - 1, 3))
cr4_velocities = np.zeros((len(cr4) - 1, 3))
cr1_velocities[:, 2] = (cr1_top[1:] - cr1_top[:-1]) / cr1_durations
cr2_velocities[:, 2] = (cr2_top[1:] - cr2_top[:-1]) / cr2_durations
cr3_velocities[:, 2] = (cr3_top[1:] - cr3_top[:-1]) / cr3_durations
cr4_velocities[:, 2] = (cr4_top[1:] - cr4_top[:-1]) / cr4_durations

# Surfaces
cy = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)
# Control rod top and bottom tips
z1_top = mcdc.Surface.PlaneZ(z=cr1_top[0])
z1_bottom = mcdc.Surface.PlaneZ(z=cr1_bottom[0])
z2_top = mcdc.Surface.PlaneZ(z=cr2_top[0])
z2_bottom = mcdc.Surface.PlaneZ(z=cr2_bottom[0])
z3_top = mcdc.Surface.PlaneZ(z=cr3_top[0])
z3_bottom = mcdc.Surface.PlaneZ(z=cr3_bottom[0])
z4_top = mcdc.Surface.PlaneZ(z=cr4_top[0])
z4_bottom = mcdc.Surface.PlaneZ(z=cr4_bottom[0])
# Fuel top
#   (Bottom is bounded by the universe cell)
zf = mcdc.Surface.PlaneZ(z=0.5 * core_height)

# Move the control tips
z1_top.move(cr1_velocities, cr1_durations)
z1_bottom.move(cr1_velocities, cr1_durations)
z2_top.move(cr2_velocities, cr2_durations)
z2_bottom.move(cr2_velocities, cr2_durations)
z3_top.move(cr3_velocities, cr3_durations)
z3_bottom.move(cr3_velocities, cr3_durations)
z4_top.move(cr4_velocities, cr4_durations)
z4_bottom.move(cr4_velocities, cr4_durations)

# Fission chamber pin
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
cr1 = mcdc.Cell(-cy & +z1_bottom & -z1_top, mat_cr)
gt1_lower = mcdc.Cell(-cy & -z1_bottom, mat_gt)
gt1_upper = mcdc.Cell(-cy & +z1_top, mat_gt)
#
cr2 = mcdc.Cell(-cy & +z2_bottom & -z2_top, mat_cr)
gt2_lower = mcdc.Cell(-cy & -z2_bottom, mat_gt)
gt2_upper = mcdc.Cell(-cy & +z2_top, mat_gt)
#
cr3 = mcdc.Cell(-cy & +z3_bottom & -z3_top, mat_cr)
gt3_lower = mcdc.Cell(-cy & -z3_bottom, mat_gt)
gt3_upper = mcdc.Cell(-cy & +z3_top, mat_gt)
#
cr4 = mcdc.Cell(-cy & +z4_bottom & -z4_top, mat_cr)
gt4_lower = mcdc.Cell(-cy & -z4_bottom, mat_gt)
gt4_upper = mcdc.Cell(-cy & +z4_top, mat_gt)
#
control_rod1 = mcdc.Universe(cells=[cr1, gt1_lower, gt1_upper, mod])
control_rod2 = mcdc.Universe(cells=[cr2, gt2_lower, gt2_upper, mod])
control_rod3 = mcdc.Universe(cells=[cr3, gt3_lower, gt3_upper, mod])
control_rod4 = mcdc.Universe(cells=[cr4, gt4_lower, gt4_upper, mod])

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

z0 = mcdc.Surface.PlaneZ(
    z=-(core_height / 2 + reflector_thickness), boundary_condition="vacuum"
)
z1 = mcdc.Surface.PlaneZ(z=-(core_height / 2))
z2 = mcdc.Surface.PlaneZ(
    z=(core_height / 2 + reflector_thickness), boundary_condition="vacuum"
)

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
# Throughout the active center pin of Assembly four, at highest energy,
# for the first 15 seconds

source = mcdc.Source(
    x=np.array([pitch * 17 * 3 / 2] * 2) + np.array([-pitch / 2, +pitch / 2]),
    y=np.array([-pitch * 17 * 3 / 2] * 2) + np.array([-pitch / 2, +pitch / 2]),
    z=[-core_height / 2, core_height / 2],
    isotropic=True,
    energy_group=0,  # Highest energy
    time=[0.0, 15.0],
)

# =============================================================================
# Set tallies, settings, techniques and run MC/DC
# =============================================================================

# Tallies
Nt = 100
Nx = 17 * 2
Ny = 17 * 2
Nz = 17 * 6
t = np.linspace(0.0, 20.0, Nt + 1)
x = np.linspace(0.0, pitch * 17 * 2, Nx + 1)
y = np.linspace(-pitch * 17 * 2, 0.0, Ny + 1)
z = np.linspace(-core_height / 2, core_height / 2, Nz + 1)
mesh = mcdc.MeshStructured(x=x, y=y, z=z)
mcdc.TallyMesh(mesh=mesh, scores=["fission"], time=t)

# Settings
mcdc.settings.N_particle = 10000
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000

# Run
mcdc.run()
