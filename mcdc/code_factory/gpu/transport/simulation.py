from mpi4py import MPI
import mcdc.code_factory.gpu.adapt as adapt

caching = config.caching

# =============================================================================
# Functions for GPU Interop
# =============================================================================

# The symbols declared below will be overwritten to reference external code that
# manages GPU execution (if GPU execution is supported and selected)
alloc_state, free_state = [None] * 2

src_alloc_program, src_free_program = [None] * 2
(
    src_load_global,
    src_load_constant,
    src_store_global,
    src_store_data,
    src_store_pointer_data,
) = [None] * 5
src_init_program, src_exec_program, src_complete, src_clear_flags = [None] * 4

pre_alloc_program, pre_free_program = [None] * 2
pre_load_global, pre_load_data, pre_store_global, pre_store_data = [None] * 4
pre_init_program, pre_exec_program, pre_complete, pre_clear_flags = [None] * 4


# If GPU execution is supported and selected, the functions shown below will
# be redefined to overwrite the above symbols and perform initialization/
# finalization of GPU state
@njit
def setup_gpu(mcdc, data_tally):
    pass


@njit
def teardown_gpu(mcdc):
    pass


def gpu_sources_spec():
    def make_work(prog: nb.uintp) -> nb.boolean:
        mcdc = adapt.mcdc_global(prog)

        atomic_add(mcdc["mpi_work_iter"], 0, 1)
        idx_work = mcdc["mpi_work_iter"][0]

        if idx_work >= mcdc["mpi_work_size"]:
            return False

        generate_source_particle(
            mcdc["mpi_work_start"], nb.uint64(idx_work), mcdc["source_seed"], prog
        )
        return True

    def initialize(prog: nb.uintp):
        pass

    def finalize(prog: nb.uintp):
        pass

    base_fns = (initialize, finalize, make_work)

    def step(prog: nb.uintp, P_input: adapt.particle_gpu):
        mcdc = adapt.mcdc_global(prog)
        data = adapt.mcdc_data(prog)
        particle_container = np.zeros(1, type_.particle)
        particle_container[0] = P_input
        particle = particle_container[0]
        if particle["fresh"]:
            prep_particle(particle_container, prog)
        particle["fresh"] = False
        step_particle(particle_container, data, prog)
        if particle["alive"]:
            adapt.step_async(prog, P)

    async_fns = [step]
    return adapt.harm.RuntimeSpec("mcdc_source", adapt.state_spec, base_fns, async_fns)


BLOCK_COUNT = config.args.gpu_block_count

ASYNC_EXECUTION = config.args.gpu_strategy == "async"


@njit(cache=caching)
def gpu_loop_source(seed, data, mcdc):

    # Progress bar indicator
    N_prog = 0

    if mcdc["technique"]["domain_decomposition"]:
        particle_bank_module.dd_check_in(mcdc)

    # =====================================================================
    # GPU Interop
    # =====================================================================

    # For async execution
    iter_count = 655360000
    # For event-based execution
    batch_size = 1

    full_work_size = mcdc["mpi_work_size"]
    if ASYNC_EXECUTION:
        phase_size = 1000000000
    else:
        phase_size = 1000000
    phase_count = (full_work_size + phase_size - 1) // phase_size

    for phase in range(phase_count):

        mcdc["mpi_work_iter"][0] = phase_size * phase
        mcdc["mpi_work_size"] = min(phase_size * (phase + 1), full_work_size)
        mcdc["source_seed"] = seed

        # Store the global state to the GPU
        src_store_constant(mcdc["gpu_state_pointer"], mcdc)
        src_store_data(mcdc["gpu_state_pointer"], data)

        # Execute the program, and continue to do so until it is done
        if ASYNC_EXECUTION:
            src_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, iter_count)
            while not src_complete(mcdc["source_program_pointer"]):
                particle_bank_module.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["source_program_pointer"], BLOCK_COUNT, iter_count
                )
        else:
            src_exec_program(mcdc["source_program_pointer"], BLOCK_COUNT, batch_size)
            while not src_complete(mcdc["source_program_pointer"]):
                particle_bank_module.dd_particle_send(mcdc)
                src_exec_program(
                    mcdc["source_program_pointer"], BLOCK_COUNT, batch_size
                )

        # Recover the original program state
        src_load_constant(mcdc, mcdc["gpu_state_pointer"])
        src_load_data(data, mcdc["gpu_state_pointer"])
        src_clear_flags(mcdc["source_program_pointer"])

    mcdc["mpi_work_size"] = full_work_size

    particle_bank_module.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


def build_gpu_progs(input_deck, args):

    STRAT = args.gpu_strategy

    src_spec = gpu_sources_spec()

    adapt.harm.RuntimeSpec.bind_specs()

    rank = MPI.COMM_WORLD.Get_rank()
    device_id = rank % args.gpu_share_stride

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()

    adapt.harm.RuntimeSpec.load_specs()

    if STRAT == "async":
        args.gpu_arena_size = args.gpu_arena_size // 32
        src_fns = src_spec.async_functions()
        pre_fns = pre_spec.async_functions()
    else:
        src_fns = src_spec.event_functions()
        pre_fns = pre_spec.event_functions()

    ARENA_SIZE = args.gpu_arena_size
    BLOCK_COUNT = args.gpu_block_count

    global alloc_state, free_state
    alloc_state = src_fns["alloc_state"]
    free_state = src_fns["free_state"]

    global src_alloc_program, src_free_program
    global src_load_global, src_store_global, src_load_data, src_store_data, src_store_pointer_data
    global src_init_program, src_exec_program, src_complete, src_clear_flags
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

    global pre_alloc_program, pre_free_program
    global pre_load_global, pre_store_global, pre_load_data, pre_store_data
    global pre_init_program, pre_exec_program, pre_complete, pre_clear_flags
    pre_alloc_state = pre_fns["alloc_state"]
    pre_free_state = pre_fns["free_state"]
    pre_alloc_program = pre_fns["alloc_program"]
    pre_free_program = pre_fns["free_program"]
    pre_load_global = pre_fns["load_state_device_global"]
    pre_store_global = pre_fns["store_state_device_global"]
    pre_load_data = pre_fns["load_state_device_data"]
    pre_store_data = pre_fns["store_state_device_data"]
    pre_init_program = pre_fns["init_program"]
    pre_exec_program = pre_fns["exec_program"]
    pre_complete = pre_fns["complete"]
    pre_clear_flags = pre_fns["clear_flags"]

    @njit
    def real_setup_gpu(mcdc_array, data_tally):
        mcdc = mcdc_array[0]
        src_set_device(device_id)
        arena_size = ARENA_SIZE
        mcdc["gpu_meta"]["state_pointer"] = adapt.cast_voidptr_to_uintp(alloc_state())
        # src_store_global(mcdc["gpu_meta"]["state_pointer"], mcdc_array[0])
        if config.gpu_state_storage == "separate":
            src_store_pointer_global(
                mcdc["gpu_meta"]["state_pointer"], mcdc["gpu_meta"]["global_pointer"]
            )
            src_store_pointer_data(
                mcdc["gpu_meta"]["state_pointer"], mcdc["gpu_meta"]["tally_pointer"]
            )
        else:
            src_store_pointer_global(mcdc["gpu_meta"]["state_pointer"], mcdc_array)
            src_store_pointer_data(mcdc["gpu_meta"]["state_pointer"], data_tally)

        mcdc["gpu_meta"]["source_program_pointer"] = adapt.cast_voidptr_to_uintp(
            src_alloc_program(mcdc["gpu_meta"]["state_pointer"], ARENA_SIZE)
        )
        src_init_program(mcdc["gpu_meta"]["source_program_pointer"], BLOCK_COUNT)
        return

    @njit
    def real_teardown_gpu(mcdc):
        src_free_program(
            adapt.cast_uintp_to_voidptr(mcdc["gpu_meta"]["source_program_pointer"])
        )
        free_state(adapt.cast_uintp_to_voidptr(mcdc["gpu_meta"]["state_pointer"]))

    global setup_gpu, teardown_gpu
    setup_gpu = real_setup_gpu
    teardown_gpu = real_teardown_gpu

    global loop_source
    loop_source = gpu_loop_source


# =============================================================================
# Functions for GPU Interop
# =============================================================================

# The symbols declared below will be overwritten to reference external code that
# manages GPU execution (if GPU execution is supported and selected)
alloc_state, free_state = [None] * 2

src_alloc_program, src_free_program = [None] * 2
src_load_constant, src_load_constant, src_store_constant, src_store_data = [None] * 4
src_init_program, src_exec_program, src_complete, src_clear_flags = [None] * 4

pre_alloc_program, pre_free_program = [None] * 2
pre_load_constant, pre_load_data, pre_store_constant, pre_store_data = [None] * 4
pre_init_program, pre_exec_program, pre_complete, pre_clear_flags = [None] * 4


# If GPU execution is supported and selected, the functions shown below will
# be redefined to overwrite the above symbols and perform initialization/
# finalization of GPU state
@njit
def setup_gpu(mcdc):
    pass


@njit
def teardown_gpu(mcdc):
    pass
