import ACEtk
import h5py
import numpy as np


def print_error(message):
    print(f"\n  [ERROR]: {message}\n")
    exit()


def print_note(message):
    print(f"\n  [NOTE]: {message}\n")


def decode_interpolation(code):
    if code not in INTERPOLATION_MAP.keys():
        print_error(f"Unsupported interpolation law: {code}")
    return INTERPOLATION_MAP[code]


def decode_ace_name(name: str):
    """
    Decode an ACE file name into atomic number Z, mass number A, excitation state S,
    following the rule:
        ZAID = 1000*Z + A,                      (ground state),
        ZAID = 1000*Z + A + 300 + 100*S,        (excited, S >= 1),
    and temperature T.
    Returns (Z, A, S, T)
    """
    zaid, extension = name.split(".")

    zaid = int(zaid)
    Z = zaid // 1000
    remainder = zaid % 1000

    if remainder < 300:
        # ground state
        A = remainder
        S = 0
    else:
        # excited state
        offset = remainder - 300
        S = offset // 100
        A = offset % 100

    T = ACE_TEMPERATURE_LIB81[extension]

    return Z, A, S, T


def get_zaid(nuclide_name):
    nuclide_name = nuclide_name.strip().capitalize()

    # Find where the letters end and digits begin
    symbol = ""
    mass = 0
    for i, ch in enumerate(nuclide_name):
        if ch.isdigit():
            symbol = nuclide_name[:i]
            mass = int(nuclide_name[i:])
            break
    else:
        raise ValueError(f"No mass number found in '{nuclide_name}'")

    if symbol not in Z_MAP.keys():
        raise ValueError(f"Unknown element symbol '{symbol}'")

    Z = Z_MAP[symbol]
    A = mass
    return Z, A


def get_ace_name(Z, A, T, S=None):
    ID = Z * 1000 + A
    if S is not None:
        ID += 300 + S * 100
    extension = ACE_EXTENSION_LIB81[T]
    return f"{ID}{extension}"


def load_fission_multiplicity(data, h5_group: h5py.Group):
    # Polynomial
    if data.type == 1:
        h5_group.attrs["type"] = "polynomial"

        C = np.array(data.coefficients)
        dataset = h5_group.create_dataset("coefficient", data=C)
        dataset.attrs["unit-base"] = "MeV"

    # Tabulated
    elif data.type == 2:
        h5_group.attrs["type"] = "tabulated"

        if not data.interpolation_data.is_linear_linear:
            print(f"[ERROR] Non linear-linear tabulated multiplicity is not supported")
            exit()

        energy = np.array(data.energies)

        h5_group.create_dataset("value", data=data.multiplicities)
        dataset = h5_group.create_dataset("energy", data=energy)
        dataset.attrs["unit"] = "MeV"

    ## Yield - Unsupported
    else:
        print(f"[ERROR] Unsupported multiplicity type: {data.type}")
        exit()


def load_cosine_distribution(data, h5_group: h5py.Group):
    if isinstance(data, ACEtk.continuous.FullyIsotropicDistribution):
        h5_group.attrs["type"] = "isotropic"

    elif isinstance(data, ACEtk.continuous.DistributionGivenElsewhere):
        h5_group.attrs["type"] = "energy-correlated"

    else:
        h5_group.attrs["type"] = "tabulated"

        # Check distribution support: all tabulated
        NE = data.number_incident_energies
        for i in range(NE):
            idx = i + 1
            if data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated:
                print_error("Angular distribution is not all-tabulated")

        # Incident energy
        energy = np.array(data.incident_energies)
        energy = h5_group.create_dataset("energy", data=energy)
        energy.attrs["unit"] = "MeV"

        # Tabulated disstributions
        interpolation = np.zeros(NE, dtype=int)
        offset = np.zeros(NE, dtype=int)
        cosine = []
        pdf = []
        for i, distribution in enumerate(data.distributions):
            interpolation[i] = distribution.interpolation
            offset[i] = len(cosine)
            cosine.extend(distribution.cosines)
            pdf.extend(distribution.pdf)
        cosine = np.array(cosine)
        pdf = np.array(pdf)
        h5_group.create_dataset("offset", data=offset)
        h5_group.create_dataset("value", data=cosine)
        h5_group.create_dataset("pdf", data=pdf)

        if not all(interpolation == 2):
            print_error("Angular distribution is not linearly-iterpolable")


def load_energy_distribution(data, h5_group: h5py.Group):
    if isinstance(data, ACEtk.continuous.LevelScatteringDistribution):
        h5_group.attrs["type"] = "level-scattering"

        C1 = np.array(data.C1)
        C1 = h5_group.create_dataset("C1", data=C1)
        C1.attrs["unit"] = "MeV"

        h5_group.create_dataset("C2", data=data.C2)

    elif isinstance(data, ACEtk.continuous.EvaporationSpectrum):
        h5_group.attrs["type"] = "evaporation"

        if not data.interpolation_data.is_linear_linear:
            print_error(
                "Evaporation distribution temperature is not linearly interpolable"
            )

        energy = np.array(data.energies)
        temperature = np.array(data.temperatures)
        restriction_energy = np.array(data.restriction_energy)

        dataset = h5_group.create_dataset("temperature_energy_grid", data=energy)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("temperature", data=temperature)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("restriction_energy", data=restriction_energy)
        dataset.attrs["unit"] = "MeV"

    elif isinstance(data, ACEtk.continuous.SimpleMaxwellianFissionSpectrum):
        h5_group.attrs["type"] = "maxwellian"

        if all(np.array(data.interpolation_data.interpolants) == 2):
            interpolation = "linear"
        elif all(np.array(data.interpolation_data.interpolants) == 5):
            interpolation = "log"
        else:
            print_error(
                "Unsupported temperature interpolation law in Maxwellian distribution"
            )

        energy = np.array(data.energies)
        temperature = np.array(data.temperatures)
        restriction_energy = np.array(data.restriction_energy)

        h5_group.create_dataset("temperature_interpolation", data=interpolation)
        dataset = h5_group.create_dataset("temperature_energy_grid", data=energy)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("temperature", data=temperature)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("restriction_energy", data=restriction_energy)
        dataset.attrs["unit"] = "MeV"

    elif isinstance(data, ACEtk.continuous.OutgoingEnergyDistributionData):
        h5_group.attrs["type"] = "tabulated"

        if not data.interpolation_data.is_linear_linear:
            print_error(
                "Non-linearly-interpolated energy distribution is not supported"
            )

        # Incident energy
        energy = np.array(data.incident_energies)
        energy = h5_group.create_dataset("energy", data=energy)
        energy.attrs["unit"] = "MeV"

        # Tabulated disstributions
        NE = data.number_incident_energies
        offset = np.zeros(NE, dtype=int)
        energy_out = []
        pdf = []
        for i in range(NE):
            distribution = data.distribution(i + 1)
            offset[i] = len(energy_out)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)

        energy_out = np.array(energy_out)
        pdf = np.array(pdf)

        h5_group.create_dataset("offset", data=offset)
        dataset = h5_group.create_dataset("value", data=energy_out)
        dataset.attrs["unit"] = ["MeV"]
        h5_group.create_dataset("pdf", data=pdf)

    elif isinstance(data, ACEtk.continuous.KalbachMannDistributionData):
        h5_group.attrs["type"] = "kalbach-mann"

        if not data.interpolation_data.is_linear_linear:
            print_error("Non-linearly-interpolated kalbach-mann is not supported")

        # Check distribution support: all kalbach-mann
        NE = data.number_incident_energies

        # Incident energy
        energy = np.array(data.incident_energies)
        energy = h5_group.create_dataset("energy", data=energy)
        energy.attrs["unit"] = "MeV"

        # Tabulated distributions
        offset = np.zeros(NE, dtype=int)
        energy_out = []
        pdf = []
        precompound_factor = []
        angular_slope = []
        for i, distribution in enumerate(data.distributions):
            offset[i] = len(pdf)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)
            precompound_factor.extend(distribution.precompound_fraction_values)
            angular_slope.extend(distribution.angular_distribution_slope_values)

        energy_out = np.array(energy_out)
        pdf = np.array(pdf)
        precompound_factor = np.array(precompound_factor)
        angular_slope = np.array(angular_slope)

        h5_group.create_dataset("offset", data=offset)
        dataset = h5_group.create_dataset("energy_out", data=energy_out)
        dataset.attrs["unit"] = "MeV"
        h5_group.create_dataset("pdf", data=pdf)
        h5_group.create_dataset("precompound_factor", data=precompound_factor)
        h5_group.create_dataset("angular_slope", data=angular_slope)

    elif isinstance(data, ACEtk.continuous.EnergyAngleDistributionData):
        h5_group.attrs["type"] = "energy-angle-tabulated"

        if not data.interpolation_data.is_linear_linear:
            print_error(
                "Non-linearly-interpolated correlated-energy-angle is not supported"
            )

        # Check distribution support: all kalbach-mann
        NE = data.number_incident_energies

        # Incident energy
        energy = np.array(data.incident_energies)
        dataset = h5_group.create_dataset("energy", data=energy)
        dataset.attrs["unit"] = "MeV"

        # Tabulated distributions
        offset = np.zeros(NE, dtype=int)
        energy_out = []
        pdf = []
        cosine_offset = []
        cosine = []
        cosine_pdf = []
        for i, distribution in enumerate(data.distributions):
            offset[i] = len(pdf)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)

            for inner_distribution in distribution.distributions:
                cosine_offset.append(len(cosine_pdf))
                cosine.extend(inner_distribution.cosines)
                cosine_pdf.extend(inner_distribution.pdf)

        energy_out = np.array(energy_out)
        pdf = np.array(pdf)
        cosine_offset = np.array(cosine_offset)
        cosine = np.array(cosine)
        cosine_pdf = np.array(cosine_pdf)

        h5_group.create_dataset("offset", data=offset)
        dataset = h5_group.create_dataset("energy_out", data=energy_out)
        dataset.attrs["unit"] = "MeV"
        h5_group.create_dataset("pdf", data=pdf)
        h5_group.create_dataset("cosine_offset", data=cosine_offset)
        h5_group.create_dataset("cosine", data=cosine)
        h5_group.create_dataset("cosine_pdf", data=cosine_pdf)

    elif isinstance(data, ACEtk.continuous.NBodyPhaseSpaceDistribution):
        h5_group.attrs["type"] = "N-body"

        if data.interpolation != 2:
            print_error("Non-linearly-interpolable N-body energy distribution")

        dataset = h5_group.create_dataset("value", data=data.values)
        dataset.attrs["unit"] = "MeV"
        h5_group.create_dataset("pdf", data=data.pdf)

    else:
        print_error(f"Unsupported energy distribution: {data}")


# ======================================================================================
# Constants
# ======================================================================================

INTERPOLATION_MAP = {2: "linear-linear"}

ACE_TEMPERATURE_LIB81 = {
    "10c": 293.6,
    "11c": 600.0,
    "12c": 900.0,
    "13c": 1200.0,
    "14c": 2500.0,
    "15c": 0.1,
    "16c": 233.15,
    "17c": 273.15,
}

TEMPERATURE_TO_ACELIB81 = {value: key for key, value in ACE_TEMPERATURE_LIB81.items()}

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
