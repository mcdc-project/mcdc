import numba as nb
import numba.extending as nbxt
import numpy as np

from mpi4py import MPI

####

import mcdc.config as config

# ======================================================================================
# Transport function adapter
# ======================================================================================


def adapt_transport_functions():
    import mcdc.code_factory.gpu.transport as gpu_transport
    import mcdc.transport as transport

    # TODO: Make the following automatic
    transport.geometry.interface.report_lost_particle = (
        gpu_transport.geometry.interface.report_lost_particle
    )
    transport.particle_bank.bank_active_particle = (
        gpu_transport.particle_bank.bank_active_particle
    )
    transport.particle_bank.report_full_bank = (
        gpu_transport.particle_bank.report_full_bank
    )
    transport.particle_bank.report_empty_bank = (
        gpu_transport.particle_bank.report_empty_bank
    )
    # transport.simulation = gpu_transport.simulation
    transport.util.atomic_add = gpu_transport.util.atomic_add
    transport.util.local_array = gpu_transport.util.local_array


def adapt_transport_functions_post_declare():
    import mcdc.transport as transport

    transport.util.access_simulation = access_simulation


# ======================================================================================
# Forward declaration
# ======================================================================================

# Main types
none_type = None
simulation_type = None
data_type = None

# Access functions
state_spec = None
access_simulation = None
access_data_ptr = None
access_group = None
access_thread = None
particle_gpu = None
particle_record_gpu = None

# Asynchronous transport kernels
step_async = None
find_cell_async = None

# Memory allocations
alloc_managed_bytes = None
alloc_device_bytes = None


def forward_declare_gpu_program():
    import harmonize
    import mcdc.numba_types as type_

    # Get to set the globals
    global none_type, simulation_type, data_type
    global state_spec, access_simulation, access_data_ptr, access_group, access_thread, particle_gpu, particle_record_gpu
    global step_async, find_cell_async
    global alloc_managed_bytes, alloc_device_bytes

    # Compilation check
    if MPI.COMM_WORLD.Get_rank() == 0:
        if config.caching == False:
            harmonize.config.should_compile(harmonize.config.ShouldCompile.ALWAYS)
    else:
        harmonize.config.should_compile(harmonize.config.ShouldCompile.NEVER)

    # ROCm and CUDA paths
    if config.args.gpu_cuda_path != None:
        harmonize.config.set_cuda_path(config.args.gpu_cuda_path)
    if config.args.gpu_rocm_path != None:
        harmonize.config.set_rocm_path(config.args.gpu_rocm_path)

    # Main types: none, simulation structure, and simulation data
    none_type = nb.from_dtype(np.dtype([]))
    simulation_type = nb.types.Array(nb.from_dtype(type_.simulation), (1,), "C")
    data_type = nb.types.Array(nb.float64, 1, "C")

    # Set access functions
    state_spec = (
        {
            "simulation": simulation_type,
            "data": data_type,
        },
        none_type,
        none_type,
    )
    access_fns = harmonize.RuntimeSpec.access_fns(state_spec)
    access_simulation = access_fns["device"]["simulation"]["indirect"]
    access_data_ptr = access_fns["device"]["data"]["direct"]
    access_group = access_fns["group"]
    access_thread = access_fns["thread"]
    particle_gpu = nb.from_dtype(type_.particle)
    particle_record_gpu = nb.from_dtype(type_.particle_data)

    # Functions, and their signatures
    def step(program: nb.uintp, particle: particle_gpu):
        pass

    def find_cell(program: nb.uintp, particle: particle_gpu):
        pass

    # Asynchronous versions
    step_async, find_cell_async = harmonize.RuntimeSpec.async_dispatch(step, find_cell)

    # Program interfaces
    interface = harmonize.RuntimeSpec.program_interface()
    halt_early = interface["halt_early"]

    # Byte allocators
    alloc_managed_bytes = harmonize.alloc_managed_bytes
    alloc_device_bytes = harmonize.alloc_device_bytes


# ======================================================================================
# Program builder
# ======================================================================================

src_free_program = lambda pointer: None
free_state = lambda pointer: None


def build_gpu_program(data_size):
    import harmonize
    import mcdc.numba_types as type_
    import mcdc.transport.util as util

    from mcdc.transport.simulation import generate_source_particle, step_particle

    global src_free_program, free_state

    shape = eval(f"{(data_size,)}")

    # ==============
    # Base functions
    # ==============

    def make_work(program: nb.uintp) -> nb.boolean:
        simulation = access_simulation(program)
        data_ptr = access_data_ptr(program)
        data = harmonize.array_from_ptr(data_ptr, shape, nb.float64)

        util.atomic_add(simulation["mpi_work_iter"], 0, 1)
        idx_work = simulation["mpi_work_iter"][0]

        if idx_work >= simulation["mpi_work_size"]:
            return False

        work_start = simulation["mpi_work_start"]

        generate_source_particle(
            simulation["mpi_work_start"],
            nb.uint64(idx_work),
            simulation["source_seed"],
            program,
            data,
        )
        return True

    def initialize(program: nb.uintp):
        pass

    def finalize(program: nb.uintp):
        pass

    # ================
    # Async. functions
    # ================

    def step(program: nb.uintp, particle_input: particle_gpu):
        simulation = access_simulation(program)
        data_ptr = access_data_ptr(program)
        data = harmonize.array_from_ptr(data_ptr, shape, nb.float64)

        particle_container = util.local_array(1, type_.particle)
        particle_container[0] = particle_input
        particle = particle_container[0]
        particle["fresh"] = False
        step_particle(particle_container, data, program)
        if particle["alive"]:
            step_async(program, particle)

    # Bind them all
    base_fns = (initialize, finalize, make_work)
    async_fns = [step]
    async_fns = []
    src_spec = harmonize.RuntimeSpec("mcdc_source", state_spec, base_fns, async_fns)
    print("PASS")
    exit()
    harmonize.RuntimeSpec.bind_specs()

    # ==================================================================================
    #
    # ==================================================================================

    rank = MPI.COMM_WORLD.Get_rank()

    MPI.COMM_WORLD.Barrier()

    harmonize.RuntimeSpec.load_specs()

    if config.args.gpu_strategy == "async":
        config.args.gpu_arena_size = config.args.gpu_arena_size // 32
        src_fns = src_spec.async_functions()
    else:
        src_fns = src_spec.event_functions()

    arena_size = config.args.gpu_arena_size
    block_count = config.args.gpu_block_count

    alloc_state = src_fns["alloc_state"]
    free_state = src_fns["free_state"]

    src_alloc_program = src_fns["alloc_program"]
    src_free_program = src_fns["free_program"]
    src_load_global = src_fns["load_state_device_global"]
    src_store_global = src_fns["store_state_device_global"]
    src_store_pointer_global = src_fns["store_pointer_state_device_global"]
    src_load_data = src_fns["load_state_device_data"]
    src_store_data = src_fns["store_state_device_data"]
    src_store_pointer_data = src_fns["store_pointer_state_device_data"]
    src_init_program = src_fns["init_program"]
    src_exec_program = src_fns["exec_program"]
    src_complete = src_fns["complete"]
    src_clear_flags = src_fns["clear_flags"]
    src_set_device = src_fns["set_device"]

    # ==================================================================================
    #
    # ==================================================================================

    """
    global loop_source
    loop_source = gpu_loop_source
    #
    # Overwrite function
    for impl in target_rosters["cpu"].values():
        overwrite_func(impl, impl)
    """

    # ==================================================================================
    # Setup
    # ==================================================================================

    device_id = rank % config.args.gpu_share_stride

    mcdc = mcdc_container[0]
    src_set_device(device_id)
    mcdc["gpu_meta"]["state_pointer"] = cast_voidptr_to_uintp(alloc_state())
    if config.gpu_state_storage == "separate":
        src_store_pointer_global(
            mcdc["gpu_meta"]["state_pointer"], mcdc["gpu_meta"]["global_pointer"]
        )
        src_store_pointer_data(
            mcdc["gpu_meta"]["state_pointer"], mcdc["gpu_meta"]["tally_pointer"]
        )
    else:
        src_store_pointer_global(mcdc["gpu_meta"]["state_pointer"], mcdc_container)
        src_store_pointer_data(mcdc["gpu_meta"]["state_pointer"], data)

    mcdc["gpu_meta"]["source_program_pointer"] = cast_voidptr_to_uintp(
        src_alloc_program(mcdc["gpu_meta"]["state_pointer"], arena_size)
    )
    src_init_program(mcdc["gpu_meta"]["source_program_pointer"], block_count)
    return


def teardown_gpu_program(mcdc):
    src_free_program(cast_uintp_to_voidptr(mcdc["gpu_meta"]["source_program_pointer"]))
    free_state(cast_uintp_to_voidptr(mcdc["gpu_meta"]["state_pointer"]))


# ======================================================================================
# Simulation structure and data creators
# ======================================================================================


def create_data_array(size, dtype):
    if config.gpu_state_storage == "managed":
        data_tally_ptr = harmonize.alloc_managed_bytes(size)
    else:
        data_tally_ptr = harmonize.alloc_device_bytes(size)
    data_tally_uint = cast_voidptr_to_uintp(data_tally_ptr)
    data_tally = nb.carray(data_tally_ptr, (size,), dtype)
    return data_tally, data_tally_uint


def create_mcdc_container(dtype):
    if config.gpu_state_storage == "managed":
        mcdc_ptr = harmonize.alloc_managed_bytes(dtype.itemsize)
    else:
        mcdc_ptr = harmonize.alloc_device_bytes(dtype.itemsize)
    mcdc_uint = cast_voidptr_to_uintp(mcdc_ptr)
    mcdc_container = nb.carray(mcdc_ptr, (1,), dtype)
    return mcdc_container, mcdc_uint


# ======================================================================================
# Type casters
# ======================================================================================


@nbxt.intrinsic
def cast_uintp_to_voidptr(typingctx, src):
    # check for accepted types
    if isinstance(src, nb.types.Integer):
        # create the expected type signature
        result_type = nb.types.voidptr
        sig = result_type(nb.types.uintp)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)

        return sig, codegen


@nbxt.intrinsic
def cast_voidptr_to_uintp(typingctx, src):
    # check for accepted types
    if isinstance(src, nb.types.RawPointer):
        # create the expected type signature
        result_type = nb.types.uintp
        sig = result_type(nb.types.voidptr)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.ptrtoint(src, llrtype)

        return sig, codegen
