import numba as nb
import numba.extending as nbxt
import numpy as np
from mpi4py import MPI
from numba import njit

####
import mcdc.config as config
import mcdc.code_factory.gpu.substitution as sub

import mcdc.code_factory.gpu.interface as interface

state_spec = {}
async_functions = []

# ======================================================================================
# Transport function adapter
# ======================================================================================


# Overwrites global symbols in other modules with gpu-compatible counterparts
def adapt_transport_functions():
    import mcdc.code_factory.gpu.transport as gpu_transport
    import mcdc.transport as transport

    sub.SubstitutionRegistry.evaluate()


def adapt_transport_functions_post_setup():
    import mcdc.code_factory.gpu.transport as gpu_transport
    import mcdc.transport as transport

    transport.simulation.source_loop = gpu_transport.simulation.source_loop


# ======================================================================================
# Forward declaration
# ======================================================================================


def prepare_gpu_program(simulation_dtype, data_size):
    """Build shared GPU artifacts on rank zero before other ranks load them."""
    communicator = MPI.COMM_WORLD
    master = communicator.Get_rank() == 0

    if master:
        _prepare_gpu_program(simulation_dtype, data_size)

    if communicator.Get_size() > 1:
        communicator.Barrier()

    if not master:
        _prepare_gpu_program(simulation_dtype, data_size)

    if communicator.Get_size() > 1:
        communicator.Barrier()


def _prepare_gpu_program(simulation_dtype, data_size):
    forward_declare_gpu_program(simulation_dtype)
    adapt_transport_functions()
    build_gpu_program(data_size)


def forward_declare_gpu_program(simulation_dtype):
    import harmonize
    import mcdc.numba_types as type_

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

    bindings = {}

    bindings["ARENA_SIZE"] = config.args.gpu_arena_size
    bindings["BLOCK_COUNT"] = config.args.gpu_block_count

    # Main types: none, simulation structure, and simulation data
    bindings["none_type"] = nb.from_dtype(np.dtype([]))
    bindings["simulation_type"] = nb.types.Array(
        nb.from_dtype(simulation_dtype), (1,), "C"
    )
    bindings["data_type"] = nb.types.Array(nb.float64, 1, "C")

    # Set access functions
    global state_spec
    state_spec = (
        {
            "simulation": bindings["simulation_type"],
            "data": bindings["data_type"],
        },
        bindings["none_type"],
        bindings["none_type"],
    )
    access_fns = harmonize.RuntimeSpec.access_fns(state_spec)
    bindings["access_simulation"] = access_fns["device"]["simulation"]["indirect"]
    bindings["access_data_ptr"] = access_fns["device"]["data"]["direct"]
    bindings["access_group"] = access_fns["group"]
    bindings["access_thread"] = access_fns["thread"]
    bindings["particle_gpu"] = nb.from_dtype(type_.particle)
    bindings["particle_record_gpu"] = nb.from_dtype(type_.particle_data)
    interface.bind(bindings)


# ======================================================================================
# Program builder
# ======================================================================================


def build_gpu_program(data_size):
    import harmonize
    import mcdc.numba_types as type_
    import mcdc.transport.util as util
    from mcdc.transport.simulation import generate_source_particle, step_particle

    interface.bind({"data_shape": eval(f"{(data_size,)}")})

    # Bind them all
    import mcdc.code_factory.gpu.program.common as common

    base_fns = (common.initialize, common.finalize, common.make_work)

    if config.args.gpu_event_decomp == "monolithic":
        import mcdc.code_factory.gpu.program.monolithic as monolithic
    elif config.args.gpu_event_decomp == "course":
        import mcdc.code_factory.gpu.program.course as course
    else:
        raise RuntimeError(
            f"Unrecognized event decomposition scheme '{config.args.gpu_event_decomp}'"
        )

    sub.SubstitutionRegistry.evaluate(tag="async")

    bindings = {}

    global async_functions
    dispatch_fns = harmonize.RuntimeSpec.async_dispatch(*async_functions)

    for idx in range(len(async_functions)):
        py_fn = async_functions[idx]
        disp_fn = dispatch_fns[idx]
        bindings[py_fn.__name__ + "_async"] = disp_fn

    # Program interfaces
    prog_interface = harmonize.RuntimeSpec.program_interface()
    bindings["halt_early"] = prog_interface["halt_early"]

    # Byte allocators
    bindings["alloc_managed_bytes"] = harmonize.alloc_managed_bytes
    bindings["alloc_device_bytes"] = harmonize.alloc_device_bytes

    interface.bind(bindings)

    from mcdc.transport import util

    if config.ROCM_AVAILABLE:
        access_target = "hip"
    else:
        access_target = "gpu"

    @nb.extending.overload(util.access_simulation, target=access_target)
    def access_simulation_gpu_overload(program):
        def impl(program):
            return interface.access_simulation(program)

        return impl

    src_spec = harmonize.RuntimeSpec(
        "mcdc_source", state_spec, base_fns, async_functions
    )
    harmonize.RuntimeSpec.bind_specs()

    # Load the specs
    harmonize.RuntimeSpec.load_specs()

    if config.args.gpu_strategy == "async":
        config.args.gpu_arena_size = config.args.gpu_arena_size // 32
        bindings = src_spec.async_functions()
    else:
        bindings = src_spec.event_functions()

    interface.bind(bindings)


# ======================================================================================
# Setup GPU
# ======================================================================================


rank = MPI.COMM_WORLD.Get_rank()
device_id = rank % config.args.gpu_share_stride


@njit
def setup_gpu_program(simulation_container, data):
    simulation = simulation_container[0]

    interface.set_device(device_id)
    simulation["gpu_meta"]["state_pointer"] = cast_voidptr_to_uintp(
        interface.alloc_state()
    )
    if config.gpu_state_storage == "separate":
        interface.store_pointer_state_device_simulation(
            simulation["gpu_meta"]["state_pointer"],
            simulation["gpu_meta"]["simulation_pointer"],
        )
        interface.store_pointer_state_device_data(
            simulation["gpu_meta"]["state_pointer"],
            simulation["gpu_meta"]["data_pointer"],
        )
    else:
        interface.store_pointer_state_device_simulation(
            simulation["gpu_meta"]["state_pointer"], simulation_container
        )
        interface.store_pointer_state_device_data(
            simulation["gpu_meta"]["state_pointer"], data
        )

    simulation["gpu_meta"]["program_pointer"] = cast_voidptr_to_uintp(
        interface.alloc_program(
            simulation["gpu_meta"]["state_pointer"], interface.ARENA_SIZE
        )
    )
    interface.init_program(
        simulation["gpu_meta"]["program_pointer"], interface.BLOCK_COUNT
    )


@njit
def teardown_gpu_program(simulation):
    interface.free_program(
        cast_uintp_to_voidptr(simulation["gpu_meta"]["program_pointer"])
    )
    interface.free_state(cast_uintp_to_voidptr(simulation["gpu_meta"]["state_pointer"]))


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
