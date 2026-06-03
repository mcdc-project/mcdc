from numba import njit, jit, objmode, literal_unroll, types
from numba.extending import intrinsic
import numba as nb
import numpy as np

import cffi

ffi = cffi.FFI()


# =============================================================================
# uintp/voidptr casters
# =============================================================================


@intrinsic
def cast_any_to_voidptr(typingctx, src):
    # create the expected type signature
    result_type = types.voidptr
    sig = result_type(src)

    # defines the custom code generation
    def codegen(context, builder, signature, args):
        # llvm IRBuilder code here
        [src] = args
        rtype = signature.return_type
        llrtype = context.get_value_type(rtype)
        return builder.bitcast(src, llrtype)

    return sig, codegen


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
    val = nb.uintp(value)
    return cast_uintp_to_voidptr(val)


@njit()
def voidptr_to_uintp(value):
    return cast_voidptr_to_uintp(value)


@njit()
def into_voidptr(value):
    return into_voidptr_python(value)


def into_voidptr_python(value):
    raise RuntimeError("`into_voidptr` is only supported in nopython mode.")


@nb.extending.overload(into_voidptr_python)
def into_voidptr_overload(value):

    if isinstance(value, nb.types.Array):

        def impl(value):
            ptr = ffi.from_buffer(value)
            vptr = cast_any_to_voidptr(ptr)
            return vptr

        return impl
    elif isinstance(value, nb.types.CPointer):

        def impl(value):
            return cast_any_to_voidptr(value)

        return impl
    elif isinstance(value, nb.types.Integer):

        def impl(value):
            return cast_uintp_to_voidptr(value)

        return impl
    else:
        raise RuntimeError(f"`into_voidptr` is not supported for type '{value}'")


###############################################################################
# New Code
###############################################################################


def array_result(array):
    return array


@nb.extending.overload(array_result)
def array_result_overload(array):

    if not isinstance(array, types.Array):
        raise nb.core.errors.TypingError(
            f"Expected array type argument for array_result, got {array}."
        )

    print("OVER LOADED")

    def impl(array):
        return (into_voidptr(array), len(array))

    return impl


def context_guard(context):
    if isinstance(context, nb.core.typing.context.Context):
        pass
    elif isinstance(context, nb.cuda.target.CUDATypingContext):
        pass
    elif isinstance(context, nb.hip.target.HIPTypingContext):
        pass
    else:
        raise nb.core.errors.UnsupportedError(f"Unsupported target context {context}.")


def array_return_typing(fn, elem_type):

    from inspect import signature

    arg_list = ",".join([param for param in signature(fn).parameters])
    template = "def typer({arg_list}):\n    return nb.types.Array(dtype=elem_type,ndim=1,layout='C')({arg_list})"

    gns = globals() | {"elem_type": elem_type}
    lns = {}
    exec(template.format(arg_list=arg_list), gns, lns)
    typer = lns["typer"]

    def typer_factory(context):
        from numba.np.numpy_support import as_dtype

        context_guard(context)

        return typer

    nb.extending.type_callable(fn)(typer_factory)


def array_return_lowering(fn, elem_type):

    from inspect import signature

    param_count = len(signature(fn).parameters)
    retty = nb.types.Array(dtype=elem_type, ndim=1, layout="C")
    sig = retty(*([nb.types.Any] * param_count))

    jit_fn = nb.njit(fn)

    def builtin(context, builder, sig, args):

        thing, data = args
        thing_type, data_type = sig.args

        import llvmlite.binding as ll
        from llvmlite import ir

        try:
            import numba.hip as hip

            ROCM_AVAILABLE = True
        except:
            ROCM_AVAILABLE = False

        if not ROCM_AVAILABLE:
            try:
                import numba.cuda as cuda

                CUDA_AVAILABLE = True
            except:
                CUDA_AVAILABLE = False
        else:
            CUDA_AVAILABLE = False

        lmod = builder.module
        retty = nb.types.Tuple([nb.types.voidptr, nb.types.uintp])
        ptr_sig = retty(*sig.args)

        res = context.compile_internal(builder, jit_fn.py_func, ptr_sig, args)
        ptr_res = builder.extract_value(res, 0)
        size_res = builder.extract_value(res, 1)
        shape = [size_res]
        dtype = data_type.dtype

        if ROCM_AVAILABLE and isinstance(context, nb.hip.target.HIPTargetContext):
            targetdata = ll.create_target_data(nb.hip.amdgcn.DATA_LAYOUT)
        elif CUDA_AVAILABLE and isinstance(context, nb.cuda.target.CUDATargetContext):
            targetdata = ll.create_target_data(nb.cuda.cudadrv.nvvm.NVVM().data_layout)
        lldtype = context.get_data_type(dtype)
        if isinstance(context, nb.core.cpu.CPUContext):
            itemsize = context.get_abi_sizeof(lldtype)
        elif ROCM_AVAILABLE and isinstance(context, nb.hip.target.HIPTargetContext):
            itemsize = lldtype.get_abi_size(targetdata)
        elif CUDA_AVAILABLE and isinstance(context, nb.cuda.target.CUDATargetContext):
            itemsize = lldtype.get_abi_size(targetdata)
        else:
            raise nb.core.errors.UnsupportedError(
                f"Unsupported target context {context}."
            )

        kstrides = [context.get_constant(types.intp, itemsize)]

        aryty = types.Array(dtype=elem_type, ndim=1, layout="C")
        ary = context.make_array(aryty)(context, builder)

        dataptr = builder.addrspacecast(
            ptr_res, ir.PointerType(ir.IntType(8)), "generic"
        )

        kshape = [size_res]
        context.populate_array(
            ary,
            data=builder.bitcast(dataptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=context.get_constant(types.intp, itemsize),
            meminfo=None,
        )
        print(ary._getvalue())
        return ary._getvalue()

    nb.extending.lower_builtin(fn, *sig.args)(builtin)


def array_return(sig):
    def array_return_true_decorator(fn):
        array_return_typing(fn, sig)
        array_return_lowering(fn, sig)
        return fn

    return array_return_true_decorator
