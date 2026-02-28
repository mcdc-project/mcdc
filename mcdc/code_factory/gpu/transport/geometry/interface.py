from numba import njit

# ======================================================================================
# Geometry inspection
# ======================================================================================


@njit
def report_lost_particle(particle_container, mcdc):
    particle = particle_container[0]
    particle["alive"] = False
