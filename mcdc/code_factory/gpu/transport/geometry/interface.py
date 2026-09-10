from numba import njit
import mcdc.code_factory.gpu.substitution as sub
import mcdc.transport.geometry as geometry

# ======================================================================================
# Geometry inspection
# ======================================================================================


@sub.candidate(geometry.interface.report_lost_particle)
def report_lost_particle(particle_container, simulation):
    particle = particle_container[0]
    particle["alive"] = False
