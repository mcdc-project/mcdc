import math

from numba import njit

from mcdc.constant import SQRT_E_TO_SPEED

# ======================================================================================
# Particle properties
# ======================================================================================


@njit
def get_speed(particle_container, mcdc):
    """
    Get particle speed
    """
    particle = particle_container[0]

    # Multigroup
    if mcdc["setting"]["mode_MG"]:
        material_ID = particle["material_ID"]
        g = particle["g"]

        material = mcdc["materials"][material_ID]

        return material["speed"][g]

    # Continuoues energy
    else:
        return math.sqrt(particle["E"]) * SQRT_E_TO_SPEED
