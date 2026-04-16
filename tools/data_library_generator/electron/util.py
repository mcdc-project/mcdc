import ACEtk
import h5py
import numpy as np


def print_error(message):
    print(f"\n  [ERROR]: {message}\n")
    exit()


def print_note(message):
    print(f"\n  [NOTE]: {message}\n")


def decode_epr_zaid(name: str):
    """
    Decode an EPR ACE ZAID like '1000.14p' into atomic number Z.
    All EPR tables are elemental: ZAID = Z * 1000 (A = 000).
    Returns Z.
    """
    zaid_str, _ = name.split(".")
    Z = int(zaid_str) // 1000
    return Z


def load_elastic_angular_distribution(block, h5_group: h5py.Group):
    """
    Load elastic angular distribution block into HDF5 group.
    Stores incident energy grid, cosine grid, PDF, and CDF per energy.
    """
    NE = block.number_of_energies

    energies = np.array(block.energies)
    dataset = h5_group.create_dataset("energy", data=energies)
    dataset.attrs["unit"] = "MeV"

    offset = np.zeros(NE, dtype=int)
    cosines = []
    pdf = []
    cdf = []
    for i in range(NE):
        dist = block.distribution(i + 1)
        offset[i] = len(cosines)
        cosines.extend(dist.cosines)
        pdf.extend(dist.pdf)
        cdf.extend(dist.cdf)

    h5_group.create_dataset("offset", data=offset)
    h5_group.create_dataset("cosine", data=np.array(cosines))
    h5_group.create_dataset("pdf", data=np.array(pdf))
    h5_group.create_dataset("cdf", data=np.array(cdf))


def load_bremsstrahlung(block, h5_group: h5py.Group):
    """
    Load bremsstrahlung energy distribution block into HDF5 group.
    Stores outgoing photon energy PDF for each incident electron energy.
    """
    NE = block.number_of_energies

    energies = np.array(block.energies)
    dataset = h5_group.create_dataset("energy", data=energies)
    dataset.attrs["unit"] = "MeV"

    offset = np.zeros(NE, dtype=int)
    energy_out = []
    pdf = []
    for i in range(NE):
        dist = block.distribution(i + 1)
        offset[i] = len(energy_out)
        energy_out.extend(dist.outgoing_energies)
        pdf.extend(dist.pdf)

    h5_group.create_dataset("offset", data=offset)
    dataset = h5_group.create_dataset("energy_out", data=np.array(energy_out))
    dataset.attrs["unit"] = "MeV"
    h5_group.create_dataset("pdf", data=np.array(pdf))


def load_electroionization_subshell(block, h5_group: h5py.Group):
    """
    Load electroionization energy distribution for a single subshell.
    Stores outgoing (knock-on) electron energy PDF per incident energy.
    """
    NE = block.number_of_energies

    energies = np.array(block.energies)
    dataset = h5_group.create_dataset("energy", data=energies)
    dataset.attrs["unit"] = "MeV"

    offset = np.zeros(NE, dtype=int)
    energy_out = []
    pdf = []
    for i in range(NE):
        dist = block.distribution(i + 1)
        offset[i] = len(energy_out)
        energy_out.extend(dist.outgoing_energies)
        pdf.extend(dist.pdf)

    h5_group.create_dataset("offset", data=offset)
    dataset = h5_group.create_dataset("energy_out", data=np.array(energy_out))
    dataset.attrs["unit"] = "MeV"
    h5_group.create_dataset("pdf", data=np.array(pdf))


# =============================================================================
# Constants
# =============================================================================

SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
}

Z_TO_SYMBOL = {value: key for key, value in SYMBOL_TO_Z.items()}
