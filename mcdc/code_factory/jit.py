import numba as nb
from inspect import getfullargspec

ARG_CHECK = True

njit_trace_template = """
def arg_check_{name}({args}):
    return fn({args})

nb.extending.overload(arg_check_{name})
def arg_check_{name}_overload({args}):
    return fn
"""


def wrap_with_check(fn,njit_fn):
    args = ",".join(getfullargspec(fn).args)
    name = fn.__name__
    lns = {"nb":nb,"fn":fn,"njit_fn":njit_fn}
    exec(njit_trace_template.format(args=args,name=name),lns)
    return lns[f"arg_check_{name}"]


def njit(*args,**kwargs):

    if (len(args) == 1) and (len(kwargs) == 0):
        if not ARG_CHECK:
            return njit(args[0])
        fn = args[0]
        njit_fn = nb.njit(args[0])
        return wrap_with_check(fn,njit_fn)

    else:
        if not ARG_CHECK:
            return nb.njit(*args,**kwargs)

        def wrapper(fn):
            njit_fn = nb.njit(*args,**kwargs)(fn)
            return wrap_with_check(fn,njit_fn) 




