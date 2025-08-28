import numpy as np

####

from mcdc import data_container
from mcdc.constant import (
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_EXCITATION,
)
from mcdc.data_container import DataMaxwellian, DataMultiPDF, DataPolynomial, DataTable
from mcdc.objects import ObjectPolymorphic, register_object
from mcdc.prints import print_1d_array
from mcdc.util import cdf_from_pdf


class ReactionBase(ObjectPolymorphic):
    def __init__(self, label, type_, h5_group):
        super().__init__(label, type_)
        self.xs = h5_group["xs"][()]

        # Register the instance
        register_object(self)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - XS {print_1d_array(self.xs)} /cm\n"
        return text


def decode_type(type_):
    if type_ == REACTION_NEUTRON_CAPTURE:
        return "Neutron capture"
    elif type_ == REACTION_NEUTRON_ELASTIC_SCATTERING:
        return "Neutron elastic scattering"
    elif type_ == REACTION_NEUTRON_FISSION:
        return "Neutron fission"
    elif type_ == REACTION_ELECTRON_BREMSSTRAHLUNG:
        return "Electron bremsstrahlung"
    elif type_ == REACTION_ELECTRON_EXCITATION:
        return "Electron excitation"


# ======================================================================================
# Neutron reactions
# ======================================================================================


class ReactionNeutronCapture(ReactionBase):
    def __init__(self, h5_group):
        label = "neutron_capture_reaction"
        type_ = REACTION_NEUTRON_CAPTURE
        super().__init__(label, type_, h5_group)


class ReactionNeutronElasticScattering(ReactionBase):
    def __init__(self, h5_group):
        label = "neutron_elastic_scattering_reaction"
        type_ = REACTION_NEUTRON_ELASTIC_SCATTERING
        super().__init__(label, type_, h5_group)

        # Scattering cosine
        base = "scattering_cosine"
        grid = h5_group[f"{base}/energy_grid"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        value = h5_group[f"{base}/value"][()]
        pdf = h5_group[f"{base}/PDF"][()]
        self.mu = DataMultiPDF(grid, offset, value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Scattering cosine: {data_container.decode_type(self.mu.type)} [ID: {self.mu.ID}]\n"
        return text


class ReactionNeutronFission(ReactionBase):
    def __init__(self, h5_group):
        label = "neutron_fission_reaction"
        type_ = REACTION_NEUTRON_FISSION
        super().__init__(label, type_, h5_group)

        products = h5_group["products"]
        prompt_product = products["prompt_neutron"]

        # Delayed neutron groups
        delayed_products = []
        self.N_delayed = 0
        if "delayed_neutrons" in products.keys():
            delayed_products = products["delayed_neutrons"]
            self.N_delayed = len(products["delayed_neutrons"])

        # Prompt neutron: yield
        self.prompt_yield = None
        if prompt_product["yield"].attrs["type"] == "table":
            x = prompt_product["yield"]["energy_grid"][()]
            y = prompt_product["yield"]["value"][()]
            self.prompt_yield = DataTable(x, y)
        elif prompt_product["yield"].attrs["type"] == "polynomial":
            coeffs = prompt_product["yield"]["polynomial_coefficient"][()]
            self.prompt_yield = DataPolynomial(coeffs)

        # Prompt neutron: energy spectrum
        self.prompt_spectrum = None
        if prompt_product["energy_out"].attrs["type"] == "multi_pdf":
            grid = prompt_product["energy_out"]["energy_grid"][()]
            offset = prompt_product["energy_out"]["energy_offset"][()]
            value = prompt_product["energy_out"]["value"][()]
            pdf = prompt_product["energy_out"]["PDF"][()]
            self.prompt_spectrum = DataMultiPDF(grid, offset, value, pdf)
        elif prompt_product["energy_out"].attrs["type"] == "maxwellian":
            print(prompt_product["energy_out"].keys())
            restriction_energy = prompt_product["energy_out"][
                "maxwell_restriction_energy"
            ][()]
            nuclear_temperature = prompt_product["energy_out"][
                "maxwell_nuclear_temperature"
            ]
            nuclear_temperature_energy_grid = nuclear_temperature["energy_grid"][()]
            nuclear_temperature_value = nuclear_temperature["value"][()]
            self.prompt_spectrum = DataMaxwellian(
                restriction_energy,
                nuclear_temperature_energy_grid,
                nuclear_temperature_value,
            )

        # Delayed products
        self.delayed_yields = [None] * self.N_delayed
        self.delayed_spectrums = [None] * self.N_delayed
        self.delayed_decay_rates = np.zeros(self.N_delayed)
        for i, name in enumerate(delayed_products):
            delayed_product = delayed_products[name]
            # Yield
            if delayed_product["yield"].attrs["type"] == "table":
                x = delayed_product["yield"]["energy_grid"][()]
                y = delayed_product["yield"]["value"][()]
                self.delayed_yields[i] = DataTable(x, y)
            elif delayed_product["yield"].attrs["type"] == "polynomial":
                coeffs = delayed_product["yield"]["polynomial_coefficient"][()]
                self.delayed_yields[i] = DataPolynomial(coeffs)

            # Energy spectrum
            if delayed_product["energy_out"].attrs["type"] == "multi_pdf":
                grid = delayed_product["energy_out"]["energy_grid"][()]
                offset = delayed_product["energy_out"]["energy_offset"][()]
                value = delayed_product["energy_out"]["value"][()]
                pdf = delayed_product["energy_out"]["PDF"][()]
                self.delayed_spectrums[i] = DataMultiPDF(grid, offset, value, pdf)
            elif delayed_product["energy_out"].attrs["type"] == "maxwellian":
                restriction_energy = delayed_product["energy_out"][
                    "reastriction_energy"
                ][()]
                nuclear_temperature = delayed_product["energy_out"][
                    "nuclear_temperature"
                ][()]
                nuclear_temperature_energy_grid = nuclear_temperature["energy_grid"][()]
                nuclear_temperature_value = nuclear_temperature["value"][()]
                self.delayed_spectrums[i] = DataMaxwellian(
                    restriction_energy,
                    nuclear_temperature_energy_grid,
                    nuclear_temperature_value,
                )

            # Decay rate
            self.delayed_decay_rates[i] = (
                1.0 / delayed_product["mean_emission_time"][()]
            )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Number of delayed groups: {self.N_delayed}\n"
        text += f"  - Prompt neutron\n"
        text += f"    - Yield: {data_container.decode_type(self.prompt_yield.type)} [ID: {self.prompt_yield.ID}]\n"
        text += f"    - Spectrum: {data_container.decode_type(self.prompt_spectrum.type)} [ID: {self.prompt_spectrum.ID}]\n"
        if self.N_delayed > 0:
            text += f"  - Delayed neutron groups\n"
        for i in range(self.N_delayed):
            text += f"    - Group {i+1}\n"
            text += f"      - Yield: {data_container.decode_type(self.delayed_yields[i].type)} [ID: {self.delayed_yields[i].ID}]\n"
            text += f"      - Spectrum: {data_container.decode_type(self.delayed_spectrums[i].type)} [ID: {self.delayed_spectrums[i].ID}]\n"
            text += f"      - Mean emission time: {1.0 / self.delayed_decay_rates[i]:.5g} s\n"

        return text


# ======================================================================================
# Electron reactions
# ======================================================================================


class ReactionElectronBremsstrahlung(ReactionBase):
    def __init__(self, h5_group):
        label = "electron_bremsstrahlung_reaction"
        type_ = REACTION_ELECTRON_BREMSSTRAHLUNG
        super().__init__(label, type_, h5_group)

        # Energy loss
        base = "energy_loss"
        self.eloss_energy_grid = h5_group[f"{base}/energy_grid"][()]
        self.eloss_energy_offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss = h5_group[f"{base}/value"][()]
        self.eloss_PDF = h5_group[f"{base}/PDF"][()]

        # CDF
        eloss = h5_group[f"{base}/value"][()]
        PDF = h5_group[f"{base}/PDF"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss_CDF = np.zeros_like(PDF)
        for i in range(len(offset)):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(PDF)
            for idx in range(start, end - 2):
                self.eloss_CDF[idx + 1] = (
                    self.eloss_CDF[idx]
                    + (PDF[idx] + PDF[idx + 1]) * (eloss[idx + 1] - eloss[idx]) * 0.5
                )
            # Ensure it ends at one
            self.eloss_CDF[end - 1] = 1.0

    def __repr__(self):
        text = super().__repr__()
        text += " - Energy loss\n"
        text += f"   - eloss_energy_grid {print_1d_array(self.eloss_energy_grid)}\n"
        text += f"   - eloss_energy_offset {print_1d_array(self.eloss_energy_offset)}\n"
        text += f"   - eloss {print_1d_array(self.eloss)}\n"
        text += f"   - eloss_PDF {print_1d_array(self.eloss_PDF)}\n"
        return text


class ReactionElectronExcitation(ReactionBase):
    def __init__(self, h5_group):
        label = "electron_excitation_reaction"
        type_ = REACTION_ELECTRON_EXCITATION
        super().__init__(label, type_, h5_group)

        # Energy loss
        base = "energy_loss"
        self.eloss_energy_grid = h5_group[f"{base}/energy_grid"][()]
        self.eloss_energy_offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss = h5_group[f"{base}/value"][()]
        self.eloss_PDF = h5_group[f"{base}/PDF"][()]

        # CDF
        eloss = h5_group[f"{base}/value"][()]
        PDF = h5_group[f"{base}/PDF"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        self.eloss_CDF = np.zeros_like(PDF)
        for i in range(len(offset)):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(PDF)
            for idx in range(start, end - 2):
                self.eloss_CDF[idx + 1] = (
                    self.eloss_CDF[idx]
                    + (PDF[idx] + PDF[idx + 1]) * (eloss[idx + 1] - eloss[idx]) * 0.5
                )
            # Ensure it ends at one
            self.eloss_CDF[end - 1] = 1.0

    def __repr__(self):
        text = super().__repr__()
        text += " - Energy loss\n"
        text += f"   - eloss_energy_grid {print_1d_array(self.eloss_energy_grid)}\n"
        text += f"   - eloss_energy_offset {print_1d_array(self.eloss_energy_offset)}\n"
        text += f"   - eloss {print_1d_array(self.eloss)}\n"
        text += f"   - eloss_PDF {print_1d_array(self.eloss_PDF)}\n"
        return text
