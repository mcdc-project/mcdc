import numba as nb
from inspect import getfullargspec

ARG_CHECK = True

njit_trace_template = """
def arg_check_{name}({args}):
    return fn({args})

@nb.extending.overload(arg_check_{name})
def arg_check_{name}_overload({args}):
    arg_list = [{args}]
    arg_name_list = [{arg_names}]
    for idx in range(len(arg_list)):
        arg = arg_list[idx]
        arg_name = arg_name_list[idx]
        if isinstance(arg,nb.types.Record):
            raise RuntimeError(f"Argument {{arg_name}} has a Record type. Records should be passed in an array.")
        elif isinstance(arg,nb.types.Optional):
            raise RuntimeError(f"Argument {{arg_name}} has a Record type. Records should be passed in an array.")
    return fn
"""


def wrap_with_check(fn, njit_fn):
    nb.extending.register_jitable(fn)
    arg_names = getfullargspec(fn).args
    args_str = ",".join(arg_names)
    arg_names_str = ",".join(f'"{n}"' for n in arg_names)
    name = fn.__name__
    print(f"wrapping {name}")
    gns = {"nb": nb, "fn": fn, "njit_fn": njit_fn}
    lns = {}
    code = njit_trace_template.format(args=args_str, arg_names=arg_names_str, name=name)
    exec(code, gns, lns)
    return lns[f"arg_check_{name}"]


def njit(*args, **kwargs):

    if (len(args) == 1) and (len(kwargs) == 0):
        if not ARG_CHECK:
            return njit(args[0])
        fn = args[0]
        njit_fn = nb.njit(args[0])
        return wrap_with_check(fn, njit_fn)

    else:
        if not ARG_CHECK:
            return nb.njit(*args, **kwargs)

        def wrapper(fn):
            njit_fn = nb.njit(*args, **kwargs)(fn)
            return wrap_with_check(fn, njit_fn)
