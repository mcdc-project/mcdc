import importlib
import numpy as np
from numba import njit, jit, objmode, literal_unroll, types
from numba.extending import intrinsic
import numba
import mcdc.type_ as type_
import mcdc.kernel as kernel
import mcdc.config as config


if importlib.util.find_spec("harmonize") is None:
    HAS_HARMONIZE = False
else:
    import harmonize as harm

    HAS_HARMONIZE = True


import math
import inspect

from mcdc.print_ import print_error

import mcdc.adapt as adapt


# =============================================================================
# Error Messangers
# =============================================================================


def unknown_target(target):
    print_error(f"ERROR: Unrecognized target '{target}'")


# =============================================================================
# uintp/voidptr casters
# =============================================================================


@intrinsic
def cast_uintp_to_voidptr(typingctx, src):
    # check for accepted types
    if isinstance(src, types.Integer):
        # create the expected type signature
        result_type = types.voidptr
        sig = result_type(types.uintp)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)

        return sig, codegen


@intrinsic
def cast_voidptr_to_uintp(typingctx, src):
    # check for accepted types
    if isinstance(src, types.RawPointer):
        # create the expected type signature
        result_type = types.uintp
        sig = result_type(types.voidptr)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.ptrtoint(src, llrtype)

        return sig, codegen


@njit()
def uintp_to_voidptr(value):
    val = numba.uintp(value)
    return cast_uintp_to_voidptr(val)


@njit()
def voidptr_to_uintp(value):
    return cast_voidptr_to_uintp(value)


def leak(arg):
    pass


@intrinsic
def leak_inner(typingctx, kind):
    def codegen(context, builder, signature, args):
        context.nrt.incref(builder, kind, args[0])

    return numba.void(kind), codegen


@numba.extending.overload(leak)
def leak_overload(arg):
    def impl(arg):
        leak_inner(arg)

    return impl


# =============================================================================
# Generic GPU/CPU Local Array Variable Constructors
# =============================================================================


def local_array(shape, dtype):
    return np.zeros(shape, dtype=dtype)


@numba.extending.type_callable(local_array)
def type_local_array(context):

    from numba.core.typing.npydecl import parse_dtype, parse_shape

    if isinstance(context, numba.core.typing.context.Context):

        # Function repurposed from Numba's ol_np_empty.
        def typer(shape, dtype):
            numba.np.arrayobj._check_const_str_dtype("empty", dtype)

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    raise numba.core.errors.UnsupportedError(
                        f"Integer shape type {shape} is not literal."
                    )
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral) for s in shape]):
                    raise numba.core.errors.UnsupportedError(
                        f"At least one element of shape tuple type{shape} is not an integer literal."
                    )
            else:
                raise numba.core.errors.UnsupportedError(
                    f"Shape is of unsupported type {shape}."
                )

            # No default arguments.
            nb_dtype = parse_dtype(dtype)
            nb_shape = parse_shape(shape)

            if nb_dtype is not None and nb_shape is not None:
                retty = types.Array(dtype=nb_dtype, ndim=nb_shape, layout="C")
                # Inlining the signature construction from numpy_empty_nd
                sig = retty(shape, dtype)
                return sig
            else:
                msg = f"Cannot parse input types to function np.empty({shape}, {dtype})"
                raise numba.errors.TypingError(msg)

        return typer

    elif isinstance(context, numba.cuda.target.CUDATypingContext):

        # Function repurposed from Numba's Cuda_array_decl.
        def typer(shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral) for s in shape]):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout="C")

        return typer

    elif isinstance(context, numba.hip.target.HIPTypingContext):

        def typer(shape, dtype):
            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral) for s in shape]):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                result = types.Array(dtype=nb_dtype, ndim=ndim, layout="C")
                return result

        return typer

    else:
        raise numba.core.errors.UnsupportedError(
            f"Unsupported target context {context}."
        )


@numba.extending.lower_builtin(local_array, types.IntegerLiteral, types.Any)
def builtin_local_array(context, builder, sig, args):

    shape, dtype = sig.args

    from numba.core.typing.npydecl import parse_dtype, parse_shape
    import numba.np.arrayobj as arrayobj

    if isinstance(context, numba.core.cpu.CPUContext):

        # No default arguments.
        nb_dtype = parse_dtype(dtype)
        nb_shape = parse_shape(shape)

        retty = types.Array(dtype=nb_dtype, ndim=nb_shape, layout="C")

        # In ol_np_empty, the reference type of the array is fed into the
        # signatrue as a third argument. This third argument is not used by
        # _parse_empty_args.
        sig = retty(shape, dtype)

        arrtype, shapes = arrayobj._parse_empty_args(context, builder, sig, args)
        ary = arrayobj._empty_nd_impl(context, builder, arrtype, shapes)

        return ary._getvalue()
    elif isinstance(context, numba.cuda.target.CUDATargetContext):
        length = sig.args[0].literal_value
        dtype = parse_dtype(sig.args[1])
        return numba.cuda.cudaimpl._generic_array(
            context,
            builder,
            shape=(length,),
            dtype=dtype,
            symbol_name="_cudapy_harm_lmem",
            addrspace=numba.cuda.cudadrv.nvvm.ADDRSPACE_LOCAL,
            can_dynsized=False,
        )
    elif isinstance(context, numba.hip.target.HIPTargetContext):
        length = sig.args[0].literal_value
        dtype = parse_dtype(sig.args[1])
        result = numba.hip.typing_lowering.hip.lowering._generic_array(
            context,
            builder,
            shape=(length,),
            dtype=dtype,
            symbol_name="_HIPpy_lmem",
            addrspace=numba.hip.amdgcn.ADDRSPACE_LOCAL,
            can_dynsized=False,
        )
        return result
    else:
        raise numba.core.errors.UnsupportedError(
            f"Unsupported target context {context}."
        )


# =============================================================================
# Decorators
# =============================================================================

toggle_rosters = {}

target_rosters = {}

late_jit_roster = set()

do_nothing_id = 0


def generate_do_nothing(arg_count, crash_on_call=None):
    global do_nothing_id
    name = f"do_nothing_{do_nothing_id}"
    args = ", ".join([f"arg_{i}" for i in range(arg_count)])
    source = f"def {name}({args}):\n"
    if crash_on_call != None:
        source += f"    assert False, '{crash_on_call}'\n"
    else:
        source += "    pass\n"
    exec(source)
    result = eval(name)
    do_nothing_id += 1
    return result


def overwrite_func(func, revised_func):
    mod_name = func.__module__
    fn_name = func.__name__
    new_fn_name = revised_func.__name__
    module = __import__(mod_name, fromlist=[fn_name])
    setattr(module, fn_name, revised_func)


def toggle(flag):
    def toggle_inner(func):
        global toggle_rosters
        if flag not in toggle_rosters:
            toggle_rosters[flag] = [False, []]
        toggle_rosters[flag][1].append(func)
        return func

    return toggle_inner


def set_toggle(flag, val):
    toggle_rosters[flag][0] = val


def eval_toggle():
    global toggle_rosters
    for _, pair in toggle_rosters.items():
        val = pair[0]
        roster = pair[1]
        for func in roster:
            if val:
                overwrite_func(func, numba.njit(func))
            else:
                global do_nothing_id
                name = func.__name__
                arg_count = len(inspect.signature(func).parameters)
                overwrite_func(func, numba.njit(generate_do_nothing(arg_count)))


blankout_roster = {}


def blankout_fn(func):
    global blankout_roster

    mod_name = func.__module__
    fn_name = func.__name__
    id = (mod_name, fn_name)

    if id not in blankout_roster:
        global do_nothing_id
        name = func.__name__
        arg_count = len(inspect.signature(func).parameters)
        blankout_roster[id] = generate_do_nothing(
            arg_count, crash_on_call=f"blankout fn for {name} should never be called"
        )

    blank = blankout_roster[id]

    return blank


def for_(target, on_target=[]):
    def for_inner(func):
        global target_rosters
        mod_name = func.__module__
        fn_name = func.__name__
        params = inspect.signature(func).parameters
        if target not in target_rosters:
            target_rosters[target] = {}
        target_rosters[target][(mod_name, fn_name)] = func
        # blank = blankout_fn(func)
        param_str = ", ".join(p for p in params)
        jit_str = f"def jit_func({param_str}):\n    global target_rosters\n    return target_rosters['{target}'][('{mod_name}','{fn_name}')]"
        exec(jit_str, globals(), locals())
        result = eval("jit_func")
        blank = blankout_fn(func)
        if target == "gpu":
            numba.core.extending.overload(blank, target=target)(result)
        else:
            numba.core.extending.overload(blank, target=target)(result)
        return blank

    return for_inner


def for_cpu(on_target=[]):
    return for_("cpu", on_target=on_target)


def for_gpu(on_target=[]):
    return for_("gpu", on_target=on_target)


def target_for(target):
    pass


def jit_on_target():
    def jit_on_target_inner(func):
        late_jit_roster.add(func)
        return func

    return jit_on_target_inner


def nopython_mode(is_on):
    if is_on:
        return
    if not isinstance(target_rosters["cpu"], dict):
        return

    for impl in target_rosters["cpu"].values():
        overwrite_func(impl, impl)


# =============================================================================
# GPU Type / Extern Functions Forward Declarations
# =============================================================================


@numba.njit()
def alloc_bytes_placeholder(size):
    return uintp_to_voidptr(0)


SIMPLE_ASYNC = True

none_type = None
mcdc_global_type = None
mcdc_data_type = None
mcdc_shared_type = None
state_spec = None
mcdc_global_gpu = None
mcdc_data_gpu = None
group_gpu = None
thread_gpu = None
particle_gpu = None
prep_gpu = None
step_async = None
halt_early = None
find_cell_async = None
tally_width = None
tally_length = None
tally_size = None
alloc_managed_bytes = alloc_bytes_placeholder
alloc_device_bytes = alloc_bytes_placeholder
tally_shape_literal = None


def gpu_forward_declare(args, tally_shape):

    if args.gpu_rocm_path != None:
        harm.config.set_rocm_path(args.gpu_rocm_path)

    if args.gpu_cuda_path != None:
        harm.config.set_cuda_path(args.gpu_cuda_path)

    global none_type, mcdc_global_type, mcdc_data_type, mcdc_shared_type
    global state_spec
    global mcdc_global_gpu, mcdc_data_gpu
    global group_gpu, thread_gpu
    global particle_gpu, particle_record_gpu
    global step_async, find_cell_async, halt_early
    global tally_width, tally_length, tally_size

    tally_size = tally_shape[0] * tally_shape[1] * 8

    global tally_shape_literal
    tally_shape_literal = tally_shape

    none_type = numba.from_dtype(np.dtype([]))
    mcdc_global_type = numba.types.Array(numba.from_dtype(type_.global_), (1,), "C")
    # mcdc_global_type = numba.from_dtype(type_.global_)

    tally_dims = len(tally_shape)
    mcdc_data_type = numba.types.Array(numba.float64, tally_dims, "C")
    state_spec = (
        {
            "global": mcdc_global_type,
            "data": mcdc_data_type,
        },
        none_type,
        none_type,
    )
    access_fns = harm.RuntimeSpec.access_fns(state_spec)
    mcdc_global_gpu = access_fns["device"]["global"]["indirect"]
    mcdc_data_gpu = access_fns["device"]["data"]["direct"]
    group_gpu = access_fns["group"]
    thread_gpu = access_fns["thread"]
    particle_gpu = numba.from_dtype(type_.particle)
    particle_record_gpu = numba.from_dtype(type_.particle_record)

    def step(prog: numba.uintp, P: particle_gpu):
        pass

    def find_cell(prog: numba.uintp, P: particle_gpu):
        pass

    step_async, find_cell_async = adapt.harm.RuntimeSpec.async_dispatch(step, find_cell)
    interface = adapt.harm.RuntimeSpec.program_interface()
    halt_early = interface["halt_early"]

    global alloc_managed_bytes
    global alloc_device_bytes
    alloc_managed_bytes = harm.alloc_managed_bytes
    alloc_device_bytes = harm.alloc_device_bytes


# =============================================================================
# Global GPU/CPU Arry Variable Constructors
# =============================================================================


@numba.njit()
def create_tally_array(width, length):
    if config.target == "gpu":
        if config.gpu_state_storage == "managed":
            data_tally_ptr = alloc_managed_bytes(tally_size)
        else:
            data_tally_ptr = alloc_device_bytes(tally_size)
        data_tally_uint = voidptr_to_uintp(data_tally_ptr)
        data_tally = numba.carray(data_tally_ptr, (width, length), type_.float64)
        return data_tally, data_tally_uint
    else:
        data_tally = np.zeros((width, length), dtype=type_.float64)
        return data_tally, 0


@numba.njit()
def create_mcdc_array():
    if config.target == "gpu":
        if config.gpu_state_storage == "managed":
            mcdc_ptr = alloc_managed_bytes(type_.global_size)
        else:
            mcdc_ptr = alloc_device_bytes(type_.global_size)
        mcdc_uint = voidptr_to_uintp(mcdc_ptr)
        mcdc_array = numba.carray(mcdc_ptr, (1,), type_.global_)
        return mcdc_array, mcdc_uint
    else:
        mcdc_array = np.zeros((1,), dtype=type_.global_)
        return mcdc_array, 0


# =============================================================================
# Seperate GPU/CPU Functions to Target Different Platforms
# =============================================================================


def mcdc_global(prog):
    return prog


@for_cpu()
def mcdc_global(prog):
    return prog


@for_gpu()
def mcdc_global(prog):
    return mcdc_global_gpu(prog)


@for_cpu()
def mcdc_data(prog):
    return None


@for_gpu()
def mcdc_data(prog):
    return mcdc_data_gpu(prog)


@for_cpu()
def group(prog):
    return prog


@for_gpu()
def group(prog):
    return group_gpu(prog)


@for_cpu()
def thread(prog):
    return prog


@for_gpu()
def thread(prog):
    return thread_gpu(prog)


@for_cpu()
def add_active(P_arr, prog):
    kernel.add_particle(P_arr, prog["bank_active"])


@for_gpu()
def add_active(P_rec_arr, prog):
    P_arr = local_array(1, type_.particle)
    kernel.recordlike_to_particle(P_arr, P_rec_arr)
    if SIMPLE_ASYNC:
        step_async(prog, P_arr[0])
    else:
        find_cell_async(prog, P_arr[0])


@for_cpu()
def add_source(P_arr, prog):
    kernel.add_particle(P_arr, prog["bank_source"])


@for_gpu()
def add_source(P_arr, prog):
    mcdc = mcdc_global(prog)
    kernel.add_particle(P_arr, mcdc["bank_source"])


@for_cpu()
def add_census(P_arr, prog):
    kernel.add_particle(P_arr, prog["bank_census"])


@for_gpu()
def add_census(P_arr, prog):
    mcdc = mcdc_global(prog)
    kernel.add_particle(P_arr, mcdc["bank_census"])


@for_cpu()
def add_future(P_arr, prog):
    kernel.add_particle(P_arr, prog["bank_future"])


@for_gpu()
def add_future(P_arr, prog):
    mcdc = mcdc_global(prog)
    kernel.add_particle(P_arr, mcdc["bank_future"])


@for_cpu()
def add_IC(P_arr, prog):
    kernel.add_particle(P_arr, prog["technique"]["IC_bank_neutron_local"])


@for_gpu()
def add_IC(P_arr, prog):
    mcdc = mcdc_global(prog)
    kernel.add_particle(P_arr, mcdc["technique"]["IC_bank_neutron_local"])


@for_cpu()
def global_add(ary, idx, val):
    result = ary[idx]
    ary[idx] += val
    return result


@for_gpu()
def global_add(ary, idx, val):
    return harm.array_atomic_add(ary, idx, val)


@for_cpu()
def global_max(ary, idx, val):
    result = ary[idx]
    if ary[idx] < val:
        ary[idx] = val
    return result


@for_gpu()
def global_max(ary, idx, val):
    return harm.array_atomic_max(ary, idx, val)


# =========================================================================
# Program Specifications
# =========================================================================

state_spec = None
one_event_fns = None
multi_event_fns = None


device_gpu, group_gpu, thread_gpu = None, None, None
iterate_async = None


def make_spec(target):
    global state_spec, one_event_fns, multi_event_fns
    global device_gpu, group_gpu, thread_gpu
    global iterate_async
    if target == "gpu":
        state_spec = (dev_state_type, grp_state_type, thd_state_type)
        one_event_fns = [iterate]
        # multi_event_fns = [source,move,scattering,fission,leakage,bcollision]
        device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
        (iterate_async,) = harm.RuntimeSpec.async_dispatch(iterate)
    elif target != "cpu":
        unknown_target(target)


@njit
def empty_base_func(prog):
    pass


def make_gpu_loop(
    state_spec,
    work_make_fn,
    step_fn,
    check_fn,
    arg_type,
    initial_fn=empty_base_func,
    final_fn=empty_base_func,
):
    async_fn_list = [step_fn]
    device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)

    def make_work(prog: numba.uintp) -> numba.boolean:
        return work_make_fn(prog)

    def initialize(prog: numba.uintp):
        initial_fn(prog)

    def finalize(prog: numba.uintp):
        final_fn(prog)

    def step(prog: numba.uintp, arg: arg_type):

        step_async()

    (step_async,) = harm.RuntimeSpec.async_dispatch(step)

    pass


# =========================================================================
# Compilation and Main Adapter
# =========================================================================


def compiler(func, target):
    if target == "cpu":
        return jit(func, nopython=True, nogil=True)  # , parallel=True)
    elif target == "cpus":
        return jit(func, nopython=True, nogil=True, parallel=True)
    elif target == "gpu_device":
        return cuda.jit(func, device=True)
    elif target == "gpu":
        return cuda.jit(func)
    else:
        unknown_target(target)
