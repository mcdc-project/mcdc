import cffi
import numba as nb
import numpy as np
from numba import jit, literal_unroll, njit, objmode, types
from numba.extending import intrinsic
import mcdc.config as config

ffi = cffi.FFI()


# =============================================================================
# uintp/voidptr casting helper functions - for internal use only
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
def voidptr_to_uintp(value):
    return cast_voidptr_to_uintp(value)


@njit()
def into_voidptr(value):
    return into_voidptr_python(value)


# =============================================================================
# uintp/voidptr casting utility functions
# =============================================================================


# Converts a pointer-sized integer to a void*
@njit()
def uintp_to_voidptr(value):
    val = nb.uintp(value)
    return cast_uintp_to_voidptr(val)


# Placeholder function for casting to void*. There currently is no use case
# for void* values in python mode for mcdc.
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
# Helper decorators, functions, and builtins for returning arrays
###############################################################################


# Overload target
def array_result(array):
    return array


@nb.extending.overload(array_result)
def array_result_overload(array):

    if not isinstance(array, types.Array):
        raise nb.core.errors.TypingError(
            f"Expected array type argument for array_result, got {array}."
        )

    def impl(array):
        return (into_voidptr(array), array.shape)

    return impl


# Raises an error if the context is not recognized
def context_guard(context):
    if isinstance(context, nb.core.typing.context.Context):
        pass
    elif isinstance(context, nb.cuda.target.CUDATypingContext):
        pass
    elif isinstance(context, nb.hip.target.HIPTypingContext):
        pass
    else:
        raise nb.core.errors.UnsupportedError(f"Unsupported target context {context}.")


# Typing for the `array_return` builtin.
def array_return_typing(fn, elem_type, ndim):

    from inspect import signature

    arg_list = ",".join([param for param in signature(fn).parameters])
    template = "def typer({arg_list}):\n    return nb.types.Array(dtype=elem_type,ndim={ndim},layout='C')({arg_list})"

    gns = globals() | {"elem_type": elem_type}
    lns = {}
    exec(template.format(arg_list=arg_list, ndim=ndim), gns, lns)
    typer = lns["typer"]

    def typer_factory(context):
        from numba.np.numpy_support import as_dtype

        context_guard(context)

        return typer

    nb.extending.type_callable(fn)(typer_factory)


# The logic forthe `array_return` builtin
def array_return_lowering(fn, elem_type, ndim):

    from inspect import signature

    # The builtin returns an array with the given element
    # type and the given dimensionality (default 1)
    param_count = len(signature(fn).parameters)
    retty = nb.types.Array(dtype=elem_type, ndim=ndim, layout="C")
    sig = retty(*([nb.types.Any] * param_count))

    jit_fn = nb.njit(fn)

    # This builtin effectively replaces the original decorated function whenever it
    # is referenced in code. The original functions still exists, but it is called through
    # this builtin which converts the pointer/shape tuple that the function (should)
    # generate with `array_result` and return.
    def builtin(context, builder, sig, args):

        import llvmlite.binding as ll
        from llvmlite import ir

        lmod = builder.module
        retty = nb.types.Tuple(
            [nb.types.voidptr, nb.types.Tuple([nb.types.uintp] * ndim)]
        )
        ptr_sig = retty(*sig.args)

        res = context.compile_internal(builder, jit_fn.py_func, ptr_sig, args)
        ptr_res = builder.extract_value(res, 0)
        size_res = builder.extract_value(res, 1)
        shape = size_res
        dtype = elem_type

        # GPU platforms require a `targetdata` for array construction, which is created
        # slightly differently depending upon the platform.
        if config.ROCM_AVAILABLE and isinstance(
            context, nb.hip.target.HIPTargetContext
        ):
            targetdata = ll.create_target_data(nb.hip.amdgcn.DATA_LAYOUT)
        elif config.CUDA_AVAILABLE and isinstance(
            context, nb.cuda.target.CUDATargetContext
        ):
            targetdata = ll.create_target_data(nb.cuda.cudadrv.nvvm.NVVM().data_layout)
        lldtype = context.get_data_type(dtype)

        # The size of the item is derived either from the lldtype or `targetdata` depending
        # upon platform
        if isinstance(context, nb.core.cpu.CPUContext):
            itemsize = context.get_abi_sizeof(lldtype)
        elif config.ROCM_AVAILABLE and isinstance(
            context, nb.hip.target.HIPTargetContext
        ):
            itemsize = lldtype.get_abi_size(targetdata)
        elif config.CUDA_AVAILABLE and isinstance(
            context, nb.cuda.target.CUDATargetContext
        ):
            itemsize = lldtype.get_abi_size(targetdata)
        else:
            raise nb.core.errors.UnsupportedError(
                f"Unsupported target context {context}."
            )

        # The number of elements-worth of bytes that must be skipped to advance by 1 element
        # in a given dimension
        kstrides = [context.get_constant(types.intp, itemsize)]

        # Create array structure based on the supplied type information
        aryty = types.Array(dtype=elem_type, ndim=ndim, layout="C")
        ary = context.make_array(aryty)(context, builder)

        # Array populating logic expects pointers to the array buffer to be expressed as
        # a pointer to a byte in a generic address space.
        dataptr = builder.addrspacecast(
            ptr_res, ir.PointerType(ir.IntType(8)), "generic"
        )

        # Initialize the array structure with the data pointer, shape, and strides
        kshape = size_res
        context.populate_array(
            ary,
            data=builder.bitcast(dataptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=context.get_constant(types.intp, itemsize),
            meminfo=None,
        )
        return ary._getvalue()

    # To complete the illusion of the decorated function acting just like a normal
    # `njit` function, the decorated function is overloaded as the builtin that
    # was defined above.
    nb.extending.lower_builtin(fn, *sig.args)(builtin)


# A function decorated with `array_return` may return an array by passing
# it through the `array_result` function and returning the output
def array_return(sig, ndim=1):
    def array_return_true_decorator(fn):
        array_return_typing(fn, sig, ndim)
        array_return_lowering(fn, sig, ndim)
        return fn

    return array_return_true_decorator
