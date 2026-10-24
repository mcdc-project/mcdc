"""Explicit isotropy shares runtime storage and follows generic angular sampling."""

import h5py
import numpy as np
import pytest

import mcdc
from mcdc.constant import (
    DISTRIBUTION_ISOTROPIC,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
)
from mcdc.object_.data import DataPolynomial
from mcdc.object_.distribution import (
    DistributionBase,
    DistributionIsotropic,
    DistributionNone,
    DistributionNBody,
    DistributionLevelScattering,
)
from mcdc.object_.hdf5_reader import read_distribution
from mcdc.object_.neutron_reaction import (
    NeutronReactionElasticScattering,
    NeutronReactionInelasticScattering,
    NeutronReactionFission,
)
from mcdc.object_.proton_reaction import ProtonReactionInelasticScattering
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.transport.distribution import (
    sample_distribution,
    sample_distribution_with_scale,
)


@pytest.mark.parametrize(
    "sampler", [sample_distribution, sample_distribution_with_scale]
)
@pytest.mark.parametrize(
    "xi, expected", [(0, -1), (0.25, -0.5), (0.5, 0), (0.9, 0.8), (1, 1)]
)
def test_isotropic_generic_sampling(
    tmp_path, sampler, xi, expected, mock_rng_sequence, prepare_simulation
):
    with h5py.File(tmp_path / "isotropic.h5", "w") as group:
        group.attrs["type"] = "isotropic"
        angular = read_distribution(group)
    assert isinstance(angular, DistributionIsotropic)
    container, data = prepare_simulation(objects=[angular])
    simulation = container[0]
    rng = mock_rng_sequence(xi)
    assert sampler(
        10.0, simulation["distributions"][angular.ID], rng, simulation, data
    ) == pytest.approx(expected)
    assert rng[0]["idx"] == 1


def test_stateless_distributions_share_packed_records(prepare_simulation):
    """Independent Python instances and reaction references use one record per type."""
    angular = [DistributionIsotropic(), DistributionIsotropic()]
    absent = [DistributionNone(), DistributionNone()]
    reactions = [
        NeutronReactionElasticScattering(2, np.ones(2), 0, REFERENCE_FRAME_COM, item)
        for item in angular
    ]
    container, _ = prepare_simulation(objects=[*reactions, *absent])
    simulation = container[0]
    assert simulation["N_isotropic_distribution"] == 1
    assert simulation["N_none_distribution"] == 1
    assert simulation["N_table_distribution"] == 0
    assert angular[0].ID == angular[1].ID
    assert angular[0].sub_ID == angular[1].sub_ID == 0
    assert absent[0].ID == absent[1].ID == 0
    assert angular[0].ID != absent[0].ID
    np.testing.assert_array_equal(
        simulation["neutron_elastic_scattering_reactions"]["mu_table_ID"],
        [angular[0].ID] * 2,
    )
    for name in (
        "neutron_elastic_scattering_reactions",
        "neutron_inelastic_scattering_reactions",
        "neutron_fission_reactions",
        "proton_inelastic_scattering_reactions",
        "secondary_products",
    ):
        assert "angle_type" not in simulation[name].dtype.names


def test_stateless_sharing_is_scoped_to_each_simulation_and_recompilation():
    models = []
    for _ in range(2):
        simulation = mcdc.Simulation()
        simulation.set_model([mcdc.Cell()])
        isotropic = [DistributionIsotropic(), DistributionIsotropic()]
        absent = [DistributionNone(), DistributionNone()]
        # Object discovery also traverses references owned by the model graph.
        simulation.root_universe.angular_distributions = [*isotropic, *absent]
        for _ in range(2):
            simulation.compile()
            assert (
                sum(
                    isinstance(item, DistributionIsotropic)
                    for item in simulation.distributions
                )
                == 1
            )
            assert (
                sum(
                    isinstance(item, DistributionNone)
                    for item in simulation.distributions
                )
                == 1
            )
            for item in [*isotropic, *absent]:
                assert item.compile_ID == simulation.compile_ID
                assert isinstance(simulation.distributions[item.ID], type(item))
            assert isotropic[0].ID == isotropic[1].ID
            assert absent[0].ID == absent[1].ID == 0
        models.append(simulation)
    first = next(
        item
        for item in models[0].distributions
        if isinstance(item, DistributionIsotropic)
    )
    second = next(
        item
        for item in models[1].distributions
        if isinstance(item, DistributionIsotropic)
    )
    assert first is not second


def test_absent_distribution_cannot_be_sampled(mock_rng_sequence, prepare_simulation):
    absent = DistributionNone()
    container, data = prepare_simulation(objects=[absent])
    simulation = container[0]
    rng = mock_rng_sequence()
    with pytest.raises(ValueError, match="absent Distribution"):
        sample_distribution(
            10.0, simulation["distributions"][absent.ID], rng, simulation, data
        )
    assert rng[0]["idx"] == 0


@pytest.mark.parametrize("kind", ["neutron", "proton", "fission", "secondary-product"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("missing_angle", [False, True])
def test_reactions_require_consistent_angle_energy_sources(kind, joint, missing_angle):
    """The absence marker must agree with the energy sampler's output quantities."""
    angular = DistributionNone() if missing_angle else DistributionIsotropic()
    spectrum = (
        DistributionNBody(3, 12.0, [0, 1], [1, 1])
        if joint
        else DistributionLevelScattering(0.0, 0.5)
    )
    frame = REFERENCE_FRAME_COM if joint else REFERENCE_FRAME_LAB

    def build():
        if kind == "secondary-product":
            return SecondaryProduct(
                0,
                DataPolynomial([1.0]),
                frame,
                angular,
                np.array([1.0, 10.0]),
                np.ones((1, 1)),
                [spectrum],
            )
        basic = (
            18 if kind == "fission" else 16,
            np.ones(2),
            0,
            frame,
            0.0,
        )
        if kind == "fission":
            return NeutronReactionFission(*basic, angular, spectrum)
        cls = (
            NeutronReactionInelasticScattering
            if kind == "neutron"
            else ProtonReactionInelasticScattering
        )
        return cls(
            *basic, 1, angular, np.array([1.0, 10.0]), np.ones((1, 1)), [spectrum]
        )

    if joint == missing_angle:
        reaction = build()
        assert isinstance(reaction.mu, DistributionBase)
        assert not hasattr(reaction, "angle_type")
    else:
        with pytest.raises(ValueError, match="correlation"):
            build()


def test_elastic_rejects_an_absent_angular_distribution():
    with pytest.raises(ValueError, match="explicit angular distribution"):
        NeutronReactionElasticScattering(
            2, np.ones(2), 0, REFERENCE_FRAME_COM, DistributionNone()
        )
