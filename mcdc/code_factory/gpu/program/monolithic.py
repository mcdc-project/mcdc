import harmonize
import numba as nb

import mcdc.code_factory.gpu.program.builder as builder
import mcdc.code_factory.gpu.interface as interface
import mcdc.transport.util as util
import mcdc.numba_types as type_

from mcdc.transport.simulation import step_particle

data_shape = eval(f"{interface.data_shape}")


def step(program: nb.uintp, particle_input: interface.particle_gpu):
    simulation = interface.access_simulation(program)
    data_ptr = interface.access_data_ptr(program)
    data = harmonize.array_from_ptr(data_ptr, data_shape, nb.float64)

    particle_container = util.local_array(1, type_.particle)
    particle_container[0] = particle_input
    particle = particle_container[0]
    particle["alive"] = True
    particle["material_ID"] = -1
    particle["cell_ID"] = -1
    particle["surface_ID"] = -1
    particle["event"] = -1
    particle["fresh"] = False
    step_particle(particle_container, program, data)
    if particle["alive"]:
        interface.step_async(program, particle)


builder.async_functions.append(step)
