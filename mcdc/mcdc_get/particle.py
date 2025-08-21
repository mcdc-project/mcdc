import mcdc.objects as objects


def speed(particle_container, mcdc):
    particle = particle_container[0]

    # Multigroup
    if mcdc["setting"]["MG_mode"]:
        objects.materials(particle["material_ID"])
        g = particle["g"]

        material = mcdc_new["materials"][material_ID]

        return material["speed"][g]

    # Continuoues energy
    else:
        return math.sqrt(particle["E"]) * SQRT_E_TO_SPEED
