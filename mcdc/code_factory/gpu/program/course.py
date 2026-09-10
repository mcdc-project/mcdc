import numba as nb
import harmonize

import mcdc.code_factory.gpu.interface as interface


import mcdc.code_factory.gpu.substitution as sub
import mcdc.code_factory.gpu.program.builder as builder
import mcdc.transport.technique as technique
import mcdc.transport.physics.neutron.interface as neutron_interface
import mcdc.transport.physics.electron.interface as electron_interface
import mcdc.transport.simulation as sim
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.util as util
import mcdc.numba_types as type_
from mcdc.constant import *

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

    # Determine and move to event
    sim.move_to_event(particle_container, simulation, data)

    # Execute events
    if particle["event"] == EVENT_LOST:
        pass
    elif particle["event"] & EVENT_COLLISION:
        collision_data_container = util.local_array(1, type_.collision_data)
        if particle["particle_type"] == PARTICLE_NEUTRON:
            neutron_interface.collision(
                particle_container, collision_data_container, program, data
            )
            return
        elif particle["particle_type"] == PARTICLE_ELECTRON:
            electron_interface.collision(
                particle_container, collision_data_container, program, data
            )
            return
    elif particle["event"] & EVENT_SURFACE_CROSSING:
        sim.surface_crossing(particle_container, program, data)
        return
    elif particle["event"] & EVENT_TIME_CENSUS:
        particle_bank_module.bank_census_particle(particle_container, program)
        particle["alive"] = False
    elif particle["event"] & EVENT_TIME_BOUNDARY:
        particle["alive"] = False

    if particle["alive"]:
        interface.step_async(program, particle)


builder.async_functions.append(step)


real_surface_crossing = sim.surface_crossing


def surface_crossing(program: nb.uintp, particle_input: interface.particle_gpu):

    simulation = interface.access_simulation(program)
    data_ptr = interface.access_data_ptr(program)
    data = harmonize.array_from_ptr(data_ptr, data_shape, nb.float64)

    particle_container = util.local_array(1, type_.particle)
    particle_container[0] = particle_input
    particle = particle_container[0]

    real_surface_crossing(particle_container, program, data)

    sim.manage_weight(particle_container, program, data)


@sub.candidate(sim.surface_crossing)
def surface_crossing_intercept(particle_container, program, data):
    interface.surface_crossing_async(program, particle_container[0])


builder.async_functions.append(surface_crossing)


real_electron_collision = electron_interface.collision


def electron_collision(program: nb.uintp, particle_input: interface.particle_gpu):

    simulation = interface.access_simulation(program)
    data_ptr = interface.access_data_ptr(program)
    data = harmonize.array_from_ptr(data_ptr, data_shape, nb.float64)

    particle_container = util.local_array(1, type_.particle)
    particle_container[0] = particle_input
    particle = particle_container[0]

    collision_data_container = util.local_array(1, type_.collision_data)

    real_electron_collision(particle_container, collision_data_container, program, data)

    sim.close_out_collision(particle_container, collision_data_container, program, data)


@sub.candidate(electron_interface.collision)
def electron_collision_intercept(
    particle_container, collision_data_container, program, data
):
    interface.electron_collision_async(program, particle_container[0])


builder.async_functions.append(electron_collision)


real_neutron_collision = neutron_interface.collision


def neutron_collision(program: nb.uintp, particle_input: interface.particle_gpu):

    simulation = interface.access_simulation(program)
    data_ptr = interface.access_data_ptr(program)
    data = harmonize.array_from_ptr(data_ptr, data_shape, nb.float64)

    particle_container = util.local_array(1, type_.particle)
    particle_container[0] = particle_input
    particle = particle_container[0]

    collision_data_container = util.local_array(1, type_.collision_data)

    real_neutron_collision(particle_container, collision_data_container, program, data)

    sim.close_out_collision(particle_container, collision_data_container, program, data)


@sub.candidate(neutron_interface.collision)
def neutron_collision_intercept(
    particle_container, collision_data_container, program, data
):
    interface.neutron_collision_async(program, particle_container[0])


builder.async_functions.append(neutron_collision)


real_weight_windows = technique.weight_windows


def weight_windows(program: nb.uintp, particle_input: interface.particle_gpu):

    simulation = interface.access_simulation(program)
    data_ptr = interface.access_data_ptr(program)
    data = harmonize.array_from_ptr(data_ptr, data_shape, nb.float64)

    particle_container = util.local_array(1, type_.particle)
    particle_container[0] = particle_input
    particle = particle_container[0]

    real_weight_windows(particle_container, program, data)

    if particle["alive"]:
        interface.step_async(program, particle)


@sub.candidate(sim.manage_weight)
def manage_weight(particle_container, program, data):
    simulation = util.access_simulation(program)
    particle = particle_container[0]
    if simulation["technique"]["weight_windows"]["active"]:
        interface.weight_windows_async(program, particle_container[0])
        return
    elif simulation["technique"]["global_weight_roulette"]["active"]:
        technique.global_weight_roulette(particle_container, simulation)

    if particle["alive"]:
        interface.step_async(program, particle)


builder.async_functions.append(weight_windows)
