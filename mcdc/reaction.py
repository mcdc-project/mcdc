import numpy as np

####

from mcdc import data_container
from mcdc.constant import (
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    REACTION_ELECTRON_BREMSSTRAHLUNG,
    REACTION_ELECTRON_EXCITATION,
    REACTION_ELECTRON_ELASTIC_SCATTERING,
    REACTION_ELECTRON_IONIZATION,
    ELECTRON_REST_MASS_ENERGY,
)
from mcdc.data_container import DataMaxwellian, DataMultiPDF, DataPolynomial, DataTable
from mcdc.objects import ObjectPolymorphic
from mcdc.prints import print_1d_array


class ReactionBase(ObjectPolymorphic):
    def __init__(self, label, type_, xs):
        super().__init__(label, type_)
        self.xs = xs

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
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
    elif type_ == REACTION_ELECTRON_ELASTIC_SCATTERING:
        return "Electron elastic scattering"
    elif type_ == REACTION_ELECTRON_IONIZATION:
        return "Electron ionization"


# ======================================================================================
# Neutron reactions
# ======================================================================================


class ReactionNeutronCapture(ReactionBase):
    def __init__(self, xs):
        label = "neutron_capture_reaction"
        type_ = REACTION_NEUTRON_CAPTURE
        super().__init__(label, type_, xs)
    
    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]
        return cls(xs)


class ReactionNeutronElasticScattering(ReactionBase):
    def __init__(self, xs, mu):
        label = "neutron_elastic_scattering_reaction"
        type_ = REACTION_NEUTRON_ELASTIC_SCATTERING
        super().__init__(label, type_, xs)

        self.mu = mu

    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]
        
        # Scattering cosine
        base = "scattering_cosine"
        grid = h5_group[f"{base}/energy_grid"][()]
        offset = h5_group[f"{base}/energy_offset"][()]
        value = h5_group[f"{base}/value"][()]
        pdf = h5_group[f"{base}/PDF"][()]
        mu = DataMultiPDF(grid, offset, value, pdf)

        return cls(xs, mu)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Scattering cosine: {data_container.decode_type(self.mu.type)} [ID: {self.mu.ID}]\n"
        return text


class ReactionNeutronFission(ReactionBase):
    def __init__(self, xs, prompt_yield, prompt_spectrum, delayed_yields, delayed_spectrums, delayed_decay_rates):
        label = "neutron_fission_reaction"
        type_ = REACTION_NEUTRON_FISSION
        super().__init__(label, type_, xs)

        self.prompt_yield = prompt_yield
        self.prompt_spectrum = prompt_spectrum
        self.N_delayed = len(delayed_yields)
        self.delayed_yields = delayed_yields
        self.delayed_spectrums = delayed_spectrums
        self.delayed_decay_rates = delayed_decay_rates

    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]
       
        # The products
        products = h5_group["products"]
        prompt_product = products["prompt_neutron"]

        # Delayed neutron groups
        delayed_products = []
        N_delayed = 0
        if "delayed_neutrons" in products.keys():
            delayed_products = products["delayed_neutrons"]
            N_delayed = len(products["delayed_neutrons"])

        # Prompt neutron: yield
        prompt_yield = None
        if prompt_product["yield"].attrs["type"] == "table":
            x = prompt_product["yield"]["energy_grid"][()]
            y = prompt_product["yield"]["value"][()]
            prompt_yield = DataTable(x, y)
        elif prompt_product["yield"].attrs["type"] == "polynomial":
            coeffs = prompt_product["yield"]["polynomial_coefficient"][()]
            prompt_yield = DataPolynomial(coeffs)

        # Prompt neutron: energy spectrum
        prompt_spectrum = None
        if prompt_product["energy_out"].attrs["type"] == "multi_pdf":
            grid = prompt_product["energy_out"]["energy_grid"][()]
            offset = prompt_product["energy_out"]["energy_offset"][()]
            value = prompt_product["energy_out"]["value"][()]
            pdf = prompt_product["energy_out"]["PDF"][()]
            prompt_spectrum = DataMultiPDF(grid, offset, value, pdf)
        elif prompt_product["energy_out"].attrs["type"] == "maxwellian":
            restriction_energy = prompt_product["energy_out"][
                "maxwell_restriction_energy"
            ][()]
            nuclear_temperature = prompt_product["energy_out"][
                "maxwell_nuclear_temperature"
            ]
            nuclear_temperature_energy_grid = nuclear_temperature["energy_grid"][()]
            nuclear_temperature_value = nuclear_temperature["value"][()]
            prompt_spectrum = DataMaxwellian(
                restriction_energy,
                nuclear_temperature_energy_grid,
                nuclear_temperature_value,
            )

        # Delayed products
        delayed_yields = [None] * N_delayed
        delayed_spectrums = [None] * N_delayed
        delayed_decay_rates = np.zeros(N_delayed)
        for i, name in enumerate(delayed_products):
            delayed_product = delayed_products[name]
            # Yield
            if delayed_product["yield"].attrs["type"] == "table":
                x = delayed_product["yield"]["energy_grid"][()]
                y = delayed_product["yield"]["value"][()]
                delayed_yields[i] = DataTable(x, y)
            elif delayed_product["yield"].attrs["type"] == "polynomial":
                coeffs = delayed_product["yield"]["polynomial_coefficient"][()]
                delayed_yields[i] = DataPolynomial(coeffs)

            # Energy spectrum
            if delayed_product["energy_out"].attrs["type"] == "multi_pdf":
                grid = delayed_product["energy_out"]["energy_grid"][()]
                offset = delayed_product["energy_out"]["energy_offset"][()]
                value = delayed_product["energy_out"]["value"][()]
                pdf = delayed_product["energy_out"]["PDF"][()]
                delayed_spectrums[i] = DataMultiPDF(grid, offset, value, pdf)
            elif delayed_product["energy_out"].attrs["type"] == "maxwellian":
                restriction_energy = delayed_product["energy_out"][
                    "reastriction_energy"
                ][()]
                nuclear_temperature = delayed_product["energy_out"][
                    "nuclear_temperature"
                ][()]
                nuclear_temperature_energy_grid = nuclear_temperature["energy_grid"][()]
                nuclear_temperature_value = nuclear_temperature["value"][()]
                delayed_spectrums[i] = DataMaxwellian(
                    restriction_energy,
                    nuclear_temperature_energy_grid,
                    nuclear_temperature_value,
                )

            # Decay rate
            delayed_decay_rates[i] = (
                1.0 / delayed_product["mean_emission_time"][()]
            )
    
        return cls(xs, prompt_yield, prompt_spectrum, delayed_yields, delayed_spectrums, delayed_decay_rates)


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
    def __init__(self, xs, eloss):
        label = "electron_bremsstrahlung_reaction"
        type_ = REACTION_ELECTRON_BREMSSTRAHLUNG
        super().__init__(label, type_, xs)

        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]

        # Energy loss
        base = "energy_loss"
        grid = h5_group[f"{base}/energy"][()]
        value = h5_group[f"{base}/value"][()]
        eloss = DataTable(grid, value)

        return cls(xs, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: {data_container.decode_type(self.eloss.type)} [ID: {self.eloss.ID}]\n"
        return text


class ReactionElectronExcitation(ReactionBase):
    def __init__(self, xs, eloss):
        label = "electron_excitation_reaction"
        type_ = REACTION_ELECTRON_EXCITATION
        super().__init__(label, type_, xs)

        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]

        # Energy loss
        base = "energy_loss"
        grid = h5_group[f"{base}/energy"][()]
        value = h5_group[f"{base}/value"][()]
        eloss = DataTable(grid, value)

        return cls(xs, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: {data_container.decode_type(self.eloss.type)} [ID: {self.eloss.ID}]\n"
        return text
    

class ReactionElectronElasticScattering(ReactionBase):
    def __init__(self, xs, large_angle_xs, mu_large_angle, small_angle_xs, mu_small_angle):
        label = "electron_elastic_scattering_reaction"
        type_ = REACTION_ELECTRON_ELASTIC_SCATTERING
        super().__init__(label, type_, xs)

        self.xs_large_angle = large_angle_xs
        self.mu_large_angle = mu_large_angle
        self.xs_small_angle = small_angle_xs
        self.mu_small_angle = mu_small_angle

    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]

        # Large angle
        large_angle = h5_group["large_angle"]
        large_angle_xs = large_angle["xs"][()]
        large_angle_grid = large_angle["scattering_cosine"]["energy_grid"][()]
        large_angle_offset = large_angle["scattering_cosine"]["energy_offset"][()]
        large_angle_value = large_angle["scattering_cosine"]["value"][()]
        large_angle_pdf = large_angle["scattering_cosine"]["PDF"][()]
        mu_large_angle = DataMultiPDF(large_angle_grid, large_angle_offset, large_angle_value, large_angle_pdf)

        # Small Angle
        small_angle = h5_group["small_angle"]
        small_angle_xs = small_angle["xs"][()]
        small_angle_grid = small_angle["scattering_cosine"]["energy_grid"][()]
        small_angle_offset = small_angle["scattering_cosine"]["energy_offset"][()]
        small_angle_value = small_angle["scattering_cosine"]["value"][()]
        small_angle_pdf = small_angle["scattering_cosine"]["PDF"][()]
        mu_small_angle = DataMultiPDF(small_angle_grid, small_angle_offset, small_angle_value, small_angle_pdf)

        return cls(xs, large_angle_xs, mu_large_angle, small_angle_xs, mu_small_angle)

    def __repr__(self):
        text = super().__repr__()
        text += " - Large angle scattering:\n"
        text += f"   - XS {print_1d_array(self.xs_large_angle)} barn\n"
        text += f"   - Scattering cosine: {data_container.decode_type(self.mu_large_angle.type)} [ID: {self.mu_large_angle.ID}]\n"
        text += " - Small angle scattering:\n"
        text += f"   - XS {print_1d_array(self.xs_small_angle)} barn\n"
        text += f"   - Scattering cosine: {data_container.decode_type(self.mu_small_angle.type)} [ID: {self.mu_small_angle.ID}]\n"
        return text
    

class ReactionElectronIonization(ReactionBase):
    def __init__(self, xs, N_subshells, subshell_name, subshell_xs, subshell_binding_energy, subshell_product):
        label = "electron_ionization_reaction"
        type_ = REACTION_ELECTRON_IONIZATION
        super().__init__(label, type_, xs)

        self.N_subshells = N_subshells
        self.subshell_name = subshell_name
        self.subshell_xs = subshell_xs
        self.subshell_binding_energy = subshell_binding_energy
        self.subshell_product = subshell_product

    @classmethod
    def from_h5_group(cls, h5_group):
        xs = h5_group["xs"][()]

        subshells = h5_group["subshells"]
        N_subshells = len(subshells)
        subshell_name = [None] * N_subshells
        subshell_xs = [None] * N_subshells
        subshell_binding_energy = [None] * N_subshells
        subshell_product = [None] * N_subshells

        # Subshell data
        for i, name in enumerate(subshells):
            subshell = subshells[name]
            subshell_name[i] = name
            subshell_xs[i] = subshell["xs"][()]
            subshell_binding_energy[i] = subshell["binding_energy"][()]
            product_grid = subshell["product"]["energy_grid"][()]
            product_offset = subshell["product"]["energy_offset"][()]
            product_value = subshell["product"]["value"][()]
            product_pdf = subshell["product"]["PDF"][()]
            subshell_product[i] = DataMultiPDF(product_grid, product_offset, product_value, product_pdf)

        return cls(xs, N_subshells, subshell_name, subshell_xs, subshell_binding_energy, subshell_product)

    def compute_mu_delta(self, T_delta, T_prim):
        me = ELECTRON_REST_MASS_ENERGY
        pd = (T_delta * (T_delta + 2 * me)) ** 0.5
        pp = (T_prim * (T_prim + 2 * me)) ** 0.5
        mu = (T_delta * (T_prim + 2.0 * me)) / (pd * pp)

        if mu < -1.0:
            mu = -1.0
        if mu >  1.0:
            mu =  1.0

        return mu

    def sample_delta_direction(self, T_delta, T_prim, rng):
        mu = self.compute_mu_delta(T_delta, T_prim)
        phi = 2 * np.pi * rng()
        sin = (1 - mu*mu) ** 0.5
        ux = sin * np.cos(phi)
        uy = sin * np.sin(phi)
        uz = mu
        return ux, uy, uz

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Number of subshells: {self.N_subshells}\n"
        for i in range(self.N_subshells):
            text += f"    - Name: {self.subshell_name[i]}\n"
            text += f"    - XS: {print_1d_array(self.subshell_xs[i])} barn\n"
            text += f"    - Binding energy: {self.subshell_binding_energy[i]} eV\n"
            prod = self.subshell_product[i]
            text += f"    - Product: {data_container.decode_type(prod.type)} [ID: {prod.ID}]\n"
        return text
