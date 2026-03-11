import numpy as np

from numba import njit


def _literalize(value):
    jit_str = f"@njit\ndef impl():\n    return {value}\n"
    exec(jit_str, globals(), locals())
    return eval("impl")


def make_literals(simulation):
    import mcdc.transport as transport

    # RPN evaluation buffer size
    if len(simulation.cells) == 0:
        rpn_evaluation_buffer_size = 1
    else:
        rpn_evaluation_buffer_size = int(
            max(
                [np.sum(np.array(x.region_RPN_tokens) >= 0.0) for x in simulation.cells]
            )
        )
    transport.literals.rpn_evaluation_buffer_size = _literalize(
        rpn_evaluation_buffer_size
    )
