import ast

targets = {
    'material': [
        ('mgxs_speed', 1),
        ('mgxs_decay_rate', 1),
        ('mgxs_capture', 1),
        ('mgxs_scatter', 1),
        ('mgxs_fission', 1),
        ('mgxs_total', 1),
        ('mgxs_nu_s', 1),
        ('mgxs_nu_p', 1),
        ('mgxs_nu_d', 2, ('G', 'J')),
        ('mgxs_nu_d_total', 1),
        ('mgxs_nu_f', 1),
        ('mgxs_chi_s', 2, ('G', 'G')),
        ('mgxs_chi_p', 2, ('G', 'G')),
        ('mgxs_chi_d', 2, ('J', 'G')),
    ],
    'nuclide': [
        ('from_material',),
    ],
    'settings': [
        ('census_time', 1),
    ]
}

def getter_1d_element(object_name, attribute_name):
    text = f"@njit\n"
    text += f"def {attribute_name}(index, {object_name}, data):\n"
    text += f"    offset = {object_name}[\"{attribute_name}_offset\"]\n"
    text += f"    return data[offset + index]\n\n\n"
    return text

def getter_1d_all(object_name, attribute_name):
    text = f"@njit\n"
    text += f"def {attribute_name}_all({object_name}, data):\n"
    text += f"    start = {object_name}[\"{attribute_name}_offset\"]\n"
    text += f"    end = start + {object_name}[\"{attribute_name}_length\"]\n"
    text += f"    return data[start:end]\n\n\n"
    return text

def getter_2d_element(object_name, attribute_name, stride):
    text = f"@njit\n"
    text += f"def {attribute_name}(index_1, index_2, {object_name}, data):\n"
    text += f"    offset = {object_name}[\"{attribute_name}_offset\"]\n"
    text += f"    stride = {object_name}[\"{stride}\"]\n"
    text += f"    return data[offset + index_1 * stride + index_2]\n\n\n"
    return text

def getter_2d_all(object_name, attribute_name, stride):
    text = f"@njit\n"
    text += f"def {attribute_name}_chunk(index_1, {object_name}, data):\n"
    text += f"    offset = {object_name}[\"{attribute_name}_offset\"]\n"
    text += f"    stride = {object_name}[\"{stride}\"]\n"
    text += f"    start = offset + index_1 * stride\n"
    text += f"    end = start + stride\n"
    text += f"    return data[start:end]\n\n\n"
    return text

def getter_from_other(object_name, other_name):
    text = f"@njit\n"
    text += f"def from_{other_name}(index, {other_name}, mcdc, data):\n"
    text += f"    offset = {other_name}[\"{object_name}_index_offset\"]\n"
    text += f"    {object_name}_ID = int(data[offset + index])\n"
    text += f"    return mcdc[\"{object_name}s\"][{object_name}_ID]\n\n\n"
    return text

for object_name in targets:
    with open(f"mcdc_get/{object_name}.py", "w") as f:
        text = "from numba import njit\n\n\n"
        for attribute in targets[object_name]:
            attribute_name = attribute[0]
            if attribute_name[:5] == "from_":
                other_name = attribute_name[5:]
                text += getter_from_other(object_name, other_name)
                continue
            attribute_dim = attribute[1]
            if attribute_dim == 1:
                text += getter_1d_all(object_name, attribute_name)
                text += getter_1d_element(object_name, attribute_name)
            if attribute_dim == 2:
                stride = attribute[2][1]
                text += getter_2d_all(object_name, attribute_name, stride)
                text += getter_2d_element(object_name, attribute_name, stride)
        f.write(text[:-2])


with open(f"mcdc_get/__init__.py", "w") as f:
    text = ""
    for object_name in targets:
        text += f"import mcdc.mcdc_get.{object_name} as {object_name}\n"
    f.write(text[:-1])
