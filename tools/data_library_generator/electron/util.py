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
    Stores incident energy grid, scattering cosine grid, and CDF per energy.
    Note: ACEtk provides CDF only (no PDF) for electron elastic angular data.
    """
    energies = np.array(block.energies)
    dataset = h5_group.create_dataset("energy_grid", data=energies)
    dataset.attrs["unit"] = "MeV"

    NE = len(energies)
    offset = np.zeros(NE, dtype=int)
    value = []
    cdf = []
    for i, dist in enumerate(block.distributions):
        offset[i] = len(value)
        value.extend(dist.cosines)
        cdf.extend(dist.cdf)

    h5_group.create_dataset("energy_offset", data=offset)
    h5_group.create_dataset("value", data=np.array(value))
    h5_group.create_dataset("cdf", data=np.array(cdf))


def load_electroionization_subshell(block, h5_group: h5py.Group):
    """
    Load electroionization energy distribution for a single subshell.
    Stores incident energy grid, outgoing (knock-on) electron energy grid, and
    CDF per incident energy.
    Note: ACEtk provides CDF only (no PDF) for electroionization distributions.
    """
    energies = np.array(block.energies)
    dataset = h5_group.create_dataset("energy_grid", data=energies)
    dataset.attrs["unit"] = "MeV"

    NE = len(energies)
    offset = np.zeros(NE, dtype=int)
    value = []
    cdf = []
    for i, dist in enumerate(block.distributions):
        offset[i] = len(value)
        value.extend(dist.outgoing_energies)
        cdf.extend(dist.cdf)

    h5_group.create_dataset("energy_offset", data=offset)
    dataset = h5_group.create_dataset("value", data=np.array(value))
    dataset.attrs["unit"] = "MeV"
    h5_group.create_dataset("cdf", data=np.array(cdf))


# =============================================================================
# Constants
# =============================================================================

ELECTRON_MF_CROSS_SECTIONS = 23
ELECTRON_MF_DISTRIBUTIONS = 26

ELECTRON_MT = {
    "total": 501,
    "ionization": 522,
    "large_angle_elastic": 525,
    "elastic": 526,
    "bremsstrahlung": 527,
    "excitation": 528,
}

ELECTRON_SUBSHELLS = (
    (1, "K", "1S1/2"),
    (2, "L1", "2s1/2"),
    (3, "L2", "2p1/2"),
    (4, "L3", "2p3/2"),
    (5, "M1", "3s1/2"),
    (6, "M2", "3p1/2"),
    (7, "M3", "3p3/2"),
    (8, "M4", "3d3/2"),
    (9, "M5", "3d5/2"),
    (10, "N1", "4s1/2"),
    (11, "N2", "4p1/2"),
    (12, "N3", "4p3/2"),
    (13, "N4", "4d3/2"),
    (14, "N5", "4d5/2"),
    (15, "N6", "4f5/2"),
    (16, "N7", "4f7/2"),
    (17, "O1", "5s1/2"),
    (18, "O2", "5p1/2"),
    (19, "O3", "5p3/2"),
    (20, "O4", "5d3/2"),
    (21, "O5", "5d5/2"),
    (22, "O6", "5f5/2"),
    (23, "O7", "5f7/2"),
    (24, "O8", "5g7/2"),
    (25, "O9", "5g9/2"),
    (26, "P1", "6s1/2"),
    (27, "P2", "6p1/2"),
    (28, "P3", "6p3/2"),
    (29, "P4", "6d3/2"),
    (30, "P5", "6d5/2"),
    (31, "P6", "6f5/2"),
    (32, "P7", "6f7/2"),
    (33, "P8", "6g7/2"),
    (34, "P9", "6g9/2"),
    (35, "P10", "6h7/2"),
    (36, "P11", "6h9/2"),
    (37, "Q1", "7s1/2"),
    (38, "Q2", "7p1/2"),
    (39, "Q3", "7p3/2"),
)

ELECTRON_SUBSHELL_LABEL = {
    designator: label for designator, label, _ in ELECTRON_SUBSHELLS
}
ELECTRON_SUBSHELL_ORBITAL = {
    designator: orbital for designator, _, orbital in ELECTRON_SUBSHELLS
}
ELECTRON_SUBSHELL_MT = {
    designator: 533 + designator for designator, _, _ in ELECTRON_SUBSHELLS
}

ELECTRON_MF_MT = {
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["total"]): "Total Electron Cross Sections",
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["ionization"]): (
        "Ionization (sum of subshells)"
    ),
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["large_angle_elastic"]): (
        "Large Angle Elastic Scattering Cross Section"
    ),
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["elastic"]): (
        "Elastic Scatter (Total) Cross Sections"
    ),
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["bremsstrahlung"]): (
        "Bremsstrahlung Cross Sections"
    ),
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["excitation"]): (
        "Excitation Cross Sections"
    ),
    (ELECTRON_MF_DISTRIBUTIONS, ELECTRON_MT["large_angle_elastic"]): (
        "Large Angle Elastic Angular Distributions"
    ),
    (ELECTRON_MF_DISTRIBUTIONS, ELECTRON_MT["bremsstrahlung"]): (
        "Bremsstrahlung Photon Energy Spectra and Electron Average Energy Loss"
    ),
    (ELECTRON_MF_DISTRIBUTIONS, ELECTRON_MT["excitation"]): (
        "Excitation Electron Average Energy Loss"
    ),
}

for designator, label, orbital in ELECTRON_SUBSHELLS:
    mt = ELECTRON_SUBSHELL_MT[designator]
    name = f"{label} ({orbital}) Electroionization Subshell"
    ELECTRON_MF_MT[(ELECTRON_MF_CROSS_SECTIONS, mt)] = f"{name} Cross Sections"
    ELECTRON_MF_MT[(ELECTRON_MF_DISTRIBUTIONS, mt)] = f"{name} Energy Spectra"

ELECTRON_SECTION_ABBREVS = {
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["total"]): "xs_tot",
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["ionization"]): "xs_ion",
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["large_angle_elastic"]): "xs_lge",
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["elastic"]): "xs_el",
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["bremsstrahlung"]): "xs_brem",
    (ELECTRON_MF_CROSS_SECTIONS, ELECTRON_MT["excitation"]): "xs_exc",
    (ELECTRON_MF_DISTRIBUTIONS, ELECTRON_MT["large_angle_elastic"]): "dist_lge",
    (ELECTRON_MF_DISTRIBUTIONS, ELECTRON_MT["bremsstrahlung"]): "dist_brem",
    (ELECTRON_MF_DISTRIBUTIONS, ELECTRON_MT["excitation"]): "dist_exc",
}

for designator, label, _ in ELECTRON_SUBSHELLS:
    mt = ELECTRON_SUBSHELL_MT[designator]
    ELECTRON_SECTION_ABBREVS[(ELECTRON_MF_CROSS_SECTIONS, mt)] = f"xs_{label}"
    ELECTRON_SECTION_ABBREVS[(ELECTRON_MF_DISTRIBUTIONS, mt)] = f"dist_{label}"

MF_MT = ELECTRON_MF_MT
SECTIONS_ABBREVS = ELECTRON_SECTION_ABBREVS


def get_electron_mt(reaction: str):
    return ELECTRON_MT[reaction]


def get_electron_subshell_mt(designator: int):
    designator = int(designator)
    try:
        return ELECTRON_SUBSHELL_MT[designator]
    except KeyError as exc:
        raise ValueError(f"Unknown EPR subshell designator: {designator}") from exc


def get_electron_subshell_name(designator: int):
    designator = int(designator)
    try:
        label = ELECTRON_SUBSHELL_LABEL[designator]
        orbital = ELECTRON_SUBSHELL_ORBITAL[designator]
    except KeyError as exc:
        raise ValueError(f"Unknown EPR subshell designator: {designator}") from exc
    return f"{label} ({orbital})"


def get_electron_subshell_mts(designators):
    return [get_electron_subshell_mt(designator) for designator in designators]


def create_mt_group(h5_group: h5py.Group, mt: int):
    group = h5_group.create_group(f"MT-{mt:03}")
    group.attrs["MT"] = mt
    return group


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
