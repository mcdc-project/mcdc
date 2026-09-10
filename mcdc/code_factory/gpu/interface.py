def bind(value_map):
    import sys

    module = sys.modules[__name__]
    for name, value in value_map.items():
        setattr(module, name, value)


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

# Memory allocations
alloc_managed_bytes = None
alloc_device_bytes = None
