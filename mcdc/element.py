import h5py
import numpy as np
import os

####

from mcdc.reaction import (
    ReactionElectronBremsstrahlung,
    ReactionElectronExcitation,
    ReactionElectronElasticScattering,
    ReactionElectronIonization,
    decode_type,
)
from mcdc.objects import ObjectNonSingleton
from mcdc.prints import print_1d_array

# ======================================================================================
# Element class
# ======================================================================================


class Element(ObjectNonSingleton):
    def __init__(self, element_name):
        label = "element"
        super().__init__(label)

        # Set attributes from the hdf5 file
        dir_name = os.getenv("MCDC_ELECTRON_XSLIB")
        file_name = f"{element_name}.h5"
        with h5py.File(f"{dir_name}/{file_name}", "r") as f:
            self.name = f["element_name"][()].decode()
            self.atomic_number = f["atomic_number"][()]
            self.atomic_weight_ratio = f["atomic_weight_ratio"][()]
            self.xs_energy_grid = f["electron_reactions/xs_energy_grid"][()]

            self.reactions = []
            self.total_xs = np.zeros_like(self.xs_energy_grid)

            for reaction_type in f["electron_reactions"]:
                if reaction_type == "xs_energy_grid":
                    continue

                if reaction_type == "bremsstrahlung":
                    ReactionClass = ReactionElectronBremsstrahlung

                elif reaction_type == "excitation":
                    ReactionClass = ReactionElectronExcitation

                elif reaction_type == "elastic_scattering":
                    ReactionClass = ReactionElectronElasticScattering

                elif reaction_type == "ionization":
                    ReactionClass = ReactionElectronIonization

                reaction = ReactionClass.from_h5_group(f[f"electron_reactions/{reaction_type}"])
                self.reactions.append(reaction)

                # Accumulate total XS
                self.total_xs += reaction.xs

    def __repr__(self):
        text = "\n"
        text += "Element\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic number: {self.atomic_number}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        text += f"  - XS energy grid {print_1d_array(self.xs_energy_grid)} eV\n"
        text += "  - Reactions\n"
        for reaction in self.reactions:
            text += f"    - {decode_type(reaction.type)}\n"
        return text


# For future use
ISOTOPIC_ABUNDANCE = {
    "H": {
        "H1": (99.972 + 99.999) / 2,
        "H2": (0.001 + 0.028) / 2,
    },
    "He": {
        "He3": 0.0002,
        "He4": 99.9998,
    },
    "Li": {
        "Li6": (1.9 + 7.8) / 2,
        "Li7": (92.2 + 98.1) / 2,
    },
    "Be": {
        "Be9": 100.0,
    },
    "B": {
        "B10": (18.9 + 20.4) / 2,
        "B11": (79.6 + 81.1) / 2,
    },
    "C": {
        "C12": (98.84 + 99.04) / 2,
        "C13": (0.96 + 1.16) / 2,
    },
    "N": {
        "N14": (99.578 + 99.663) / 2,
        "N15": (0.337 + 0.422) / 2,
    },
    "O": {
        "O16": (99.738 + 99.776) / 2,
        "O17": (0.0367 + 0.04) / 2,
        "O18": (0.187 + 0.222) / 2,
    },
    "F": {
        "F19": 100.0,
    },
    "Ne": {
        "Ne20": 90.48,
        "Ne21": 0.27,
        "Ne22": 9.25,
    },
    "Na": {
        "Na23": 100.0,
    },
    "Mg": {
        "Mg24": (78.88 + 79.05) / 2,
        "Mg25": (9.988 + 10.034) / 2,
        "Mg26": (10.96 + 11.09) / 2,
    },
    "Al": {
        "Al27": 100.0,
    },
    "Si": {
        "Si28": (92.191 + 92.318) / 2,
        "Si29": (4.645 + 4.699) / 2,
        "Si30": (3.037 + 3.11) / 2,
    },
    "P": {
        "P31": 100.0,
    },
    "S": {
        "S32": (94.41 + 95.29) / 2,
        "S33": (0.729 + 0.797) / 2,
        "S34": (3.96 + 4.77) / 2,
        "S36": (0.0129 + 0.0187) / 2,
    },
    "Cl": {
        "Cl35": (75.5 + 76.1) / 2,
        "Cl37": (23.9 + 24.5) / 2,
    },
    "Ar": {
        "Ar36": 0.3336,
        "Ar38": 0.0629,
        "Ar40": 99.6035,
    },
    "K": {
        "K39": 93.2581,
        "K40": 0.0117,
        "K41": 6.7302,
    },
    "Ca": {
        "Ca40": 96.941,
        "Ca42": 0.647,
        "Ca43": 0.135,
        "Ca44": 2.086,
        "Ca46": 0.004,
        "Ca48": 0.187,
    },
    "Sc": {
        "Sc45": 100.0,
    },
    "Ti": {
        "Ti46": 8.25,
        "Ti47": 7.44,
        "Ti48": 73.72,
        "Ti49": 5.41,
        "Ti50": 5.18,
    },
    "V": {
        "V50": 0.25,
        "V51": 99.75,
    },
    "Cr": {
        "Cr50": 4.345,
        "Cr52": 83.789,
        "Cr53": 9.501,
        "Cr54": 2.365,
    },
    "Mn": {
        "Mn55": 100.0,
    },
    "Fe": {
        "Fe54": 5.845,
        "Fe56": 91.754,
        "Fe57": 2.119,
        "Fe58": 0.282,
    },
    "Co": {
        "Co59": 100.0,
    },
    "Ni": {
        "Ni58": 68.0769,
        "Ni60": 26.2231,
        "Ni61": 1.1399,
        "Ni62": 3.6345,
        "Ni64": 0.9256,
    },
    "Cu": {
        "Cu63": 69.15,
        "Cu65": 30.85,
    },
    "Zn": {
        "Zn64": 49.17,
        "Zn66": 27.73,
        "Zn67": 4.04,
        "Zn68": 18.45,
        "Zn70": 0.61,
    },
    "Ga": {
        "Ga69": 60.108,
        "Ga71": 39.892,
    },
    "Ge": {
        "Ge70": 20.52,
        "Ge72": 27.45,
        "Ge73": 7.76,
        "Ge74": 36.52,
        "Ge76": 7.75,
    },
    "As": {
        "As75": 100.0,
    },
    "Se": {
        "Se74": 0.86,
        "Se76": 9.23,
        "Se77": 7.6,
        "Se78": 23.69,
        "Se80": 49.8,
        "Se82": 8.82,
    },
    "Br": {
        "Br79": (50.5 + 50.8) / 2,
        "Br81": (49.2 + 49.5) / 2,
    },
    "Kr": {
        "Kr78": 0.355,
        "Kr80": 2.286,
        "Kr82": 11.593,
        "Kr83": 11.5,
        "Kr84": 56.987,
        "Kr86": 17.279,
    },
    "Rb": {
        "Rb85": 72.17,
        "Rb87": 27.83,
    },
    "Sr": {
        "Sr84": 0.56,
        "Sr86": 9.86,
        "Sr87": 7.0,
        "Sr88": 82.58,
    },
    "Y": {
        "Y89": 100.0,
    },
    "Zr": {
        "Zr90": 51.45,
        "Zr91": 11.22,
        "Zr92": 17.15,
        "Zr94": 17.38,
        "Zr96": 2.8,
    },
    "Nb": {
        "Nb93": 100.0,
    },
    "Mo": {
        "Mo92": 14.649,
        "Mo94": 9.187,
        "Mo95": 15.873,
        "Mo96": 16.673,
        "Mo97": 9.582,
        "Mo98": 24.292,
        "Mo100": 9.744,
    },
    "Ru": {
        "Ru96": 5.54,
        "Ru98": 1.87,
        "Ru99": 12.76,
        "Ru100": 12.6,
        "Ru101": 17.06,
        "Ru102": 31.55,
        "Ru104": 18.62,
    },
    "Rh": {
        "Rh103": 100.0,
    },
    "Pd": {
        "Pd102": 1.02,
        "Pd104": 11.14,
        "Pd105": 22.33,
        "Pd106": 27.33,
        "Pd108": 26.46,
        "Pd110": 11.72,
    },
    "Ag": {
        "Ag107": 51.839,
        "Ag109": 48.161,
    },
    "Cd": {
        "Cd106": 1.245,
        "Cd108": 0.888,
        "Cd110": 12.47,
        "Cd111": 12.795,
        "Cd112": 24.109,
        "Cd113": 12.227,
        "Cd114": 28.754,
        "Cd116": 7.512,
    },
    "In": {
        "In113": 4.281,
        "In115": 95.719,
    },
    "Sn": {
        "Sn112": 0.97,
        "Sn114": 0.66,
        "Sn115": 0.34,
        "Sn116": 14.54,
        "Sn117": 7.68,
        "Sn118": 24.22,
        "Sn119": 8.59,
        "Sn120": 32.58,
        "Sn122": 4.63,
        "Sn124": 5.79,
    },
    "Sb": {
        "Sb121": 57.21,
        "Sb123": 42.79,
    },
    "Te": {
        "Te120": 0.09,
        "Te122": 2.55,
        "Te123": 0.89,
        "Te124": 4.74,
        "Te125": 7.07,
        "Te126": 8.84,
        "Te128": 31.74,
        "Te130": 34.08,
    },
    "I": {
        "I127": 100.0,
    },
    "Xe": {
        "Xe124": 0.095,
        "Xe126": 0.089,
        "Xe128": 1.91,
        "Xe129": 26.4,
        "Xe130": 4.071,
        "Xe131": 21.232,
        "Xe132": 26.909,
        "Xe134": 10.436,
        "Xe136": 8.857,
    },
    "Cs": {
        "Cs133": 100.0,
    },
    "Ba": {
        "Ba130": 0.11,
        "Ba132": 0.1,
        "Ba134": 2.42,
        "Ba135": 6.59,
        "Ba136": 7.85,
        "Ba137": 11.23,
        "Ba138": 71.7,
    },
    "La": {
        "La138": 0.08881,
        "La139": 99.91119,
    },
    "Ce": {
        "Ce136": 0.186,
        "Ce138": 0.251,
        "Ce140": 88.449,
        "Ce142": 11.114,
    },
    "Pr": {
        "Pr141": 100.0,
    },
    "Nd": {
        "Nd142": 27.153,
        "Nd143": 12.173,
        "Nd144": 23.798,
        "Nd145": 8.293,
        "Nd146": 17.189,
        "Nd148": 5.756,
        "Nd150": 5.638,
    },
    "Sm": {
        "Sm144": 3.08,
        "Sm147": 15.0,
        "Sm149": 13.82,
        "Sm150": 7.37,
        "Sm152": 26.74,
        "Sm154": 22.74,
    },
    "Eu": {
        "Eu151": 47.81,
        "Eu153": 52.19,
    },
    "Gd": {
        "Gd152": 0.2,
        "Gd154": 2.18,
        "Gd155": 14.8,
        "Gd156": 20.47,
        "Gd157": 15.65,
        "Gd158": 24.84,
        "Gd160": 21.86,
    },
    "Tb": {
        "Tb159": 100.0,
    },
    "Dy": {
        "Dy156": 0.056,
        "Dy158": 0.095,
        "Dy160": 2.329,
        "Dy161": 18.889,
        "Dy162": 25.475,
        "Dy163": 24.896,
        "Dy164": 28.26,
    },
    "Ho": {
        "Ho165": 100.0,
    },
    "Er": {
        "Er162": 0.139,
        "Er164": 1.601,
        "Er166": 33.503,
        "Er167": 22.869,
        "Er168": 26.978,
        "Er170": 14.91,
    },
    "Tm": {
        "Tm169": 100.0,
    },
    "Yb": {
        "Yb168": 0.123,
        "Yb170": 2.982,
        "Yb171": 14.086,
        "Yb172": 21.686,
        "Yb173": 16.103,
        "Yb174": 32.025,
        "Yb176": 12.995,
    },
    "Lu": {
        "Lu175": 97.401,
        "Lu176": 2.599,
    },
    "Hf": {
        "Hf174": 0.16,
        "Hf176": 5.26,
        "Hf177": 18.6,
        "Hf178": 27.28,
        "Hf179": 13.62,
        "Hf180": 35.08,
    },
    "Ta": {
        "Ta180": 0.01201,
        "Ta181": 99.98799,
    },
    "W": {
        "W180": 0.12,
        "W182": 26.5,
        "W183": 14.31,
        "W184": 30.64,
        "W186": 28.43,
    },
    "Re": {
        "Re185": 37.4,
        "Re187": 62.6,
    },
    "Os": {
        "Os184": 0.02,
        "Os186": 1.59,
        "Os187": 1.97,
        "Os188": 13.24,
        "Os189": 16.15,
        "Os190": 26.26,
        "Os192": 40.78,
    },
    "Ir": {
        "Ir191": 37.3,
        "Ir193": 62.7,
    },
    "Pt": {
        "Pt190": 0.012,
        "Pt192": 0.782,
        "Pt194": 32.864,
        "Pt195": 33.77,
        "Pt196": 25.21,
        "Pt198": 7.356,
    },
    "Au": {
        "Au197": 100.0,
    },
    "Hg": {
        "Hg196": 0.15,
        "Hg198": 10.04,
        "Hg199": 16.94,
        "Hg200": 23.14,
        "Hg201": 13.17,
        "Hg202": 29.74,
        "Hg204": 6.82,
    },
    "Tl": {
        "Tl203": (29.44 + 29.59) / 2,
        "Tl205": (70.41 + 70.56) / 2,
    },
    "Pb": {
        "Pb204": 1.4,
        "Pb206": 24.1,
        "Pb207": 22.1,
        "Pb208": 52.4,
    },
    "Bi": {
        "Bi209": 100.0,
    },
    "Th": {
        "Th230": 0.02,
        "Th232": 99.98,
    },
    "Pa": {
        "Pa231": 100.0,
    },
    "U": {
        "U234": 0.0054,
        "U235": 0.7204,
        "U238": 99.2742,
    },
}
