"""Synthetic photon libraries and compiled simulations for the photon unit tests.

The libraries written here are deliberately small but structurally real: they
go through the same HDF5 reader, the same ``Element.set_photon_data`` and the
same Numba layer packing as a production library, so a schema mistake surfaces
here rather than in a deck.
"""

import h5py
import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_

from mcdc.constant import PARTICLE_PHOTON

# A coarse log-spaced grid spanning the interesting decades. Two energies are
# repeated, which is how ENDF writes an absorption edge, so the tests exercise
# the zero-width-bin path that real libraries always contain.
ENERGY_GRID = [
    1.0e2,
    1.0e3,
    8.8e4,  # Pb K edge, first copy: below
    8.8e4,  # Pb K edge, second copy: above
    1.0e6,
    1.022e6,  # pair-production threshold, first copy
    1.022e6,  # second copy
    1.0e7,
    1.0e8,
]

#: Subshell binding energies, chosen so the K edge sits inside the grid.
K_BINDING_ENERGY = 8.8e4
L_BINDING_ENERGY = 1.5e4

#: A third binding energy, so the L shell has somewhere to be filled from and
#: the de-excitation cascade has a second step to take.
M_BINDING_ENERGY = 2.0e3

#: The K-alpha line energy: the difference of two binding energies, which is
#: always strictly below the binding energy of the shell that emitted it.
K_ALPHA_ENERGY = K_BINDING_ENERGY - L_BINDING_ENERGY

#: The L line emitted when the K-alpha vacancy in the L shell is itself filled.
#: This is the cascade photon.
L_LINE_ENERGY = L_BINDING_ENERGY - M_BINDING_ENERGY


def _grid_length():
    return len(ENERGY_GRID)


def write_photon_library(
    path,
    *,
    atomic_number=82,
    coherent=1.0,
    incoherent=2.0,
    photoelectric=4.0,
    pair=8.0,
    fluorescence_yield=1.0,
    include_relaxation=True,
    cascade=False,
):
    """Write a minimal but schema-complete photon element library.

    Each channel is constant above its threshold, so the expected macroscopic
    cross sections are exactly computable and the tests do not depend on
    interpolation detail. Photoelectric opens at the K edge and pair production
    at its physical threshold, both expressed through shared-grid xs offsets.
    """
    N = _grid_length()

    with h5py.File(path, "w") as file:
        file.attrs["format"] = "mcdc-data"
        file.attrs["format-version"] = "0.1.0"
        file["atomic-number"] = np.int64(atomic_number)

        photon = file
        reactions = photon.create_group("reactions")
        grid = photon.create_dataset("xs-energy-grid", data=np.array(ENERGY_GRID))
        grid.attrs["unit"] = "eV"

        def interpolation(group, size):
            rule = group.create_group("interpolation")
            rule["laws"] = np.array(["linear"], dtype=h5py.string_dtype())
            rule["boundaries"] = [size]

        def table(group, x, y, x_unit, y_unit=None):
            group.attrs["type"] = "table"
            group.create_dataset("grid", data=x).attrs["unit"] = x_unit
            value = group.create_dataset("value", data=y)
            if y_unit:
                value.attrs["unit"] = y_unit
            interpolation(group, len(x))

        def channel(name, MT, value, threshold_index, category=None):
            group = reactions.create_group(name)
            group.attrs["ENDF-MT"] = MT
            group.attrs["category"] = category or name
            xs = group.create_group("xs")
            xs["offset"] = threshold_index
            xs.create_dataset("value", data=np.full(N - threshold_index, value)).attrs[
                "unit"
            ] = "barn"
            interpolation(xs, N - threshold_index)
            group["reference-frame"] = "LAB"
            group.create_group("cross-products")
            return group

        # Coherent, with a form factor whose forward value is Z, as F(0, Z) = Z.
        coherent_group = channel("coherent-scattering", 502, coherent, 0)
        table(
            coherent_group.create_group("form-factor"),
            [0.0, 1.0, 10.0, 1.0e4],
            [float(atomic_number), 0.5 * atomic_number, 0.01 * atomic_number, 0.0],
            "1/angstrom",
        )
        channel("incoherent-scattering", 504, incoherent, 0)

        # Photoelectric opens at the K edge, with two subshells.
        photoelectric_group = channel("photoelectric", 522, photoelectric, 2)
        subshells = photoelectric_group.create_group("subshells")
        for designator, name, binding_energy, fraction in (
            (1, "K", K_BINDING_ENERGY, 0.75),
            (3, "L2", L_BINDING_ENERGY, 0.25),
        ):
            subshell = subshells.create_group(name)
            subshell["designator"] = np.int64(designator)
            values = np.where(
                np.array(ENERGY_GRID) >= binding_energy,
                photoelectric * fraction,
                0.0,
            )
            table(
                subshell.create_group("xs"), np.array(ENERGY_GRID), values, "eV", "barn"
            )
            subshell.create_group("cross-products")
            binding = subshell.create_dataset(
                "binding-energy", data=np.float64(binding_energy)
            )
            binding.attrs["unit"] = "eV"

        # Pair production opens at its physical threshold.
        channel("pair-production-nuclear-field", 517, pair, 5, "pair-production")[
            "field"
        ] = "nuclear"

        if not include_relaxation:
            return

        # Atomic relaxation. The K shell has one radiative transition with the
        # given yield and, when the yield is below one, the remaining
        # probability is a non-radiative channel that deposits locally.
        #
        # origin_designator is the shell the filling electron comes from, so it
        # is where the vacancy moves next: the K line is filled from L, which
        # leaves an L vacancy. Whether that second vacancy can itself relax
        # radiatively is what ``cascade`` controls.
        relaxation = file.create_group("atomic-relaxation")
        subshell_group = relaxation.create_group("subshells")

        k_shell = subshell_group.create_group("vacancy-K")
        k_shell["designator"] = np.int64(1)
        k_shell["electron-count"] = np.float64(2.0)
        binding = k_shell.create_dataset(
            "binding-energy", data=np.float64(K_BINDING_ENERGY)
        )
        binding.attrs["unit"] = "eV"
        transitions = k_shell.create_group("transitions")
        if fluorescence_yield >= 1.0:
            energies = [K_ALPHA_ENERGY]
            probabilities = [1.0]
            secondary = [0]
        else:
            energies = [K_ALPHA_ENERGY, K_ALPHA_ENERGY]
            probabilities = [fluorescence_yield, 1.0 - fluorescence_yield]
            secondary = [0, 3]  # the second is non-radiative (Auger)
        energy = transitions.create_dataset("energy", data=np.array(energies))
        energy.attrs["unit"] = "eV"
        transitions.create_dataset("probability", data=np.array(probabilities))
        transitions.create_dataset(
            "origin-designator", data=np.array([3] * len(energies), dtype=np.int64)
        )
        transitions.create_dataset(
            "secondary-designator", data=np.array(secondary, dtype=np.int64)
        )

        # By default the L shell has nothing above it to fill the vacancy, so
        # no transitions at all -- which is how real libraries write outer
        # shells, and which terminates the cascade after one photon. With
        # ``cascade=True`` it gets one radiative transition filled from M, so a
        # K event emits the primary line plus one cascade line.
        l_shell = subshell_group.create_group("vacancy-L")
        l_shell["designator"] = np.int64(3)
        l_shell["electron-count"] = np.float64(8.0)
        binding = l_shell.create_dataset(
            "binding-energy", data=np.float64(L_BINDING_ENERGY)
        )
        binding.attrs["unit"] = "eV"

        if cascade:
            l_transitions = l_shell.create_group("transitions")
            energy = l_transitions.create_dataset(
                "energy", data=np.array([L_LINE_ENERGY])
            )
            energy.attrs["unit"] = "eV"
            l_transitions.create_dataset("probability", data=np.array([1.0]))
            l_transitions.create_dataset(
                "origin-designator", data=np.array([5], dtype=np.int64)
            )
            l_transitions.create_dataset(
                "secondary-designator", data=np.array([0], dtype=np.int64)
            )

        else:
            terminal = l_shell.create_group("transitions")
            for name in (
                "energy",
                "probability",
                "origin-designator",
                "secondary-designator",
            ):
                terminal[name] = np.zeros(0)

        m_shell = subshell_group.create_group("relaxation-only-M")
        m_shell["designator"] = 5
        m_shell["binding-energy"] = M_BINDING_ENERGY
        m_shell["electron-count"] = 2.0
        terminal = m_shell.create_group("transitions")
        for name in (
            "energy",
            "probability",
            "origin-designator",
            "secondary-designator",
        ):
            terminal[name] = np.zeros(0)

        # The shared atomic collection is independent of the projectile file.
        with h5py.File(path.parent / "atoms.h5", "a") as atoms:
            atoms.attrs["format"], atoms.attrs["format-version"] = "mcdc-data", "0.1.0"
            symbol = path.stem.removeprefix("photon-")
            if symbol in atoms:
                del atoms[symbol]
            record = atoms.create_group(symbol)
            record["atomic-number"] = atomic_number
            file.copy(file["atomic-relaxation"], record, name="atomic-relaxation")
        del file["atomic-relaxation"]


@pytest.fixture
def photon_model(tmp_path, monkeypatch, prepare_simulation):
    """A one-element photon material at unit atom density."""

    def build(density=1.0, edit=None, **library):
        write_photon_library(tmp_path / "photon-X.h5", **library)
        if edit is not None:
            with h5py.File(tmp_path / "photon-X.h5", "r+") as file:
                edit(file)
        monkeypatch.setenv("MCDC_LIB", str(tmp_path))

        material = mcdc.Material(number_density={"X": density})
        cell = mcdc.Cell(fill=material)

        def configure(simulation):
            simulation.settings.photon_transport.active = True

        simulation_container, data = prepare_simulation(
            cells=[cell], configure=configure
        )
        return simulation_container[0], data

    return build


def make_photon(E, seed=1, weight=1.0):
    """A live photon container at energy ``E`` travelling along +z."""
    container = np.zeros(1, dtype=type_.particle)
    particle = container[0]
    particle["particle_type"] = PARTICLE_PHOTON
    particle["E"] = E
    particle["w"] = weight
    particle["ux"] = 0.0
    particle["uy"] = 0.0
    particle["uz"] = 1.0
    particle["alive"] = True
    particle["material_ID"] = 0
    particle["cell_ID"] = 0
    particle["rng_seed"] = np.uint64(seed)
    return container


def make_interaction_data():
    """An empty interaction-data container, as the transport loop supplies."""
    return np.zeros(1, dtype=type_.interaction_data)
