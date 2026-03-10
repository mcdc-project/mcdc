import harmonize

from numba import njit


@njit
def atomic_add(array, idx, value):
    harmonize.array_atomic_add(array, idx, value)


# =============================================================================
# Generic GPU/CPU local array variable constructors
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
