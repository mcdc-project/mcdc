"""Shared CrossProduct parsing and projectile transport-support boundaries."""

import h5py
import numpy as np
import pytest

from mcdc.constant import (
    INF,
    PARTICLE_ANY,
    PARTICLE_ELECTRON,
    PARTICLE_NEUTRON,
    PARTICLE_PHOTON,
    PARTICLE_PROTON,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
)
from mcdc.object_.distribution import DistributionNBody, DistributionNone
from mcdc.object_.hdf5_reader import read_cross_products
from mcdc.object_.neutron_reaction import NeutronReactionCapture
from mcdc.object_.photon_reaction import PhotonReactionIncoherentScattering


def write_product(parent, index=0, particle="neutron"):
    """Write a prompt product with explicit independent angle-energy data."""
    product = parent.require_group("cross-products").create_group(str(index))
    product["particle"] = particle
    product["reference-frame"] = "LAB"
    product["decay-rate"] = np.inf
    product.create_group("yield").attrs["type"] = "polynomial"
    product["yield/coefficients"] = [index + 0.5]
    distribution = product.create_group("angle-energy-distribution")
    distribution.attrs["type"] = "independent-angle-energy"
    distribution.create_group("angular-cosine-distribution").attrs["type"] = "isotropic"
    energy = distribution.create_group("energy-distribution")
    energy.attrs["type"] = "level-scattering"
    energy["C1"], energy["C2"] = 0.0, 0.5
    return product


@pytest.mark.parametrize(
    "particle,particle_type,identity",
    [
        ("neutron", PARTICLE_NEUTRON, (0, 1)),
        ("proton", PARTICLE_PROTON, (1, 1)),
        ("electron", PARTICLE_ELECTRON, (-1, -1)),
        ("photon", PARTICLE_PHOTON, (0, 0)),
        ("deuteron", PARTICLE_ANY, (1, 2)),
        ("triton", PARTICLE_ANY, (1, 3)),
        ("helion", PARTICLE_ANY, (2, 3)),
        ("alpha", PARTICLE_ANY, (2, 4)),
    ],
)
def test_cross_product_identity_and_independent_frame(
    tmp_path, particle, particle_type, identity
):
    with h5py.File(tmp_path / "products.h5", "w") as group:
        group["reference-frame"] = "COM"
        write_product(group, particle=particle)
        (product,) = read_cross_products(group)

    assert product.particle_type == particle_type
    assert (product.atomic_number, product.mass_number) == identity
    assert product.reference_frame == REFERENCE_FRAME_LAB
    assert not product.delayed
    assert product.decay_constant == INF


def test_cross_products_keep_numeric_order(tmp_path):
    with h5py.File(tmp_path / "products.h5", "w") as group:
        assert read_cross_products(group) == []
        group.create_group("cross-products")
        assert read_cross_products(group) == []
        for index in reversed(range(12)):
            write_product(group, index)
        products = read_cross_products(group)

    assert [product.production_yield.coefficients[0] for product in products] == [
        index + 0.5 for index in range(12)
    ]


def test_delayed_joint_cross_product(tmp_path):
    with h5py.File(tmp_path / "products.h5", "w") as group:
        record = write_product(group)
        record["reference-frame"][()] = "COM"
        record["decay-rate"][()] = 0.25
        del record["angle-energy-distribution"]
        distribution = record.create_group("angle-energy-distribution")
        distribution.attrs["type"] = "n-body"
        distribution["number-particles"], distribution["total-mass-ratio"] = 3, 3.0
        distribution["value"], distribution["pdf"] = [0.0, 1.0], [1.0, 1.0]
        interpolation = distribution.create_group("interpolation")
        interpolation["laws"] = np.array(["linear"], dtype=h5py.string_dtype())
        interpolation["boundaries"] = [2]
        (product,) = read_cross_products(group)

    assert product.reference_frame == REFERENCE_FRAME_COM
    assert product.delayed
    assert product.decay_constant == 0.25
    assert isinstance(product.mu, DistributionNone)
    assert isinstance(product.energy_spectra[0], DistributionNBody)


@pytest.mark.parametrize("projectile", ["neutron", "photon"])
def test_unsupported_projectiles_reject_cross_products(tmp_path, projectile):
    with h5py.File(tmp_path / "products.h5", "w") as group:
        write_product(group)
        with pytest.raises(ValueError, match="cross-product transport is unsupported"):
            if projectile == "neutron":
                NeutronReactionCapture.from_h5_group(group, None)
            else:
                PhotonReactionIncoherentScattering.from_h5_group(group)
