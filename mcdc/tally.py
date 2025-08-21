import mcdc.objects as objects

# Get input_deck
import mcdc.config as config
import mcdc.global_ as global_

import numpy as np
from numba import njit

from mcdc.card import (
    MeshTallyCard,
    SurfaceTallyCard,
    CellTallyCard,
    CSTallyCard,
)
from mcdc.constant import (
    INF,
    PI,
)
from mcdc.input_ import check_support
import mcdc.type_ as type_


def mesh_tally(
    x=np.array([-INF, INF]),
    y=np.array([-INF, INF]),
    z=np.array([-INF, INF]),
    t=np.array([-INF, INF]),
    mu=np.array([-1.0, 1.0]),
    azi=np.array([-PI, PI]),
    g=np.array([-INF, INF]),
    E=np.array([0.0, INF]),
    scores=["flux"],
):
    """
    Create a tally card to collect MC solutions.

    Parameters
    ----------
    x : array_like[float], optional
        x-coordinates that demarcate tally bins.
    y : array_like[float], optional
        y-coordinates that demarcate tally bins.
    z : array_like[float], optional
        z-coordinates that demarcate tally bins.
    t : array_like[float], optional
        Times that demarcate tally bins.
    mu : array_like[float], optional
        Angles that demarcate axial angular tally bins.
    azi : array_like[float], optional
        Angles that demarcate azimuthal angular tally bins.
    g : array_like[float] or str, optional
        Energy group halves that demarcate energy group tally bins.
        String value "all" can be used to tally each individual group.
    E : array_like[float], optional
        Energies that demarcate energy tally bins. This overrides `g` in
        continuous-energy mode.
    scores : list of str {"flux", "total", "fission", "density"}
        List of physical quantities to be scored.

    Returns
    -------
    MeshTallyCard
        The tally card.
    """

    # Make tally card
    card = MeshTallyCard()

    # Set ID
    card.ID = len(global_.input_deck.mesh_tallies)

    # Set mesh
    card.x = x
    card.y = y
    card.z = z

    # Set other filters
    card.t = t
    card.mu = mu
    card.azi = azi

    # Set energy group grid
    if type(g) == type("string") and g == "all":
        G = objects.materials[0].G
        card.g = np.linspace(0, G, G + 1) - 0.5
    else:
        card.g = g
    if not objects.settings.multigroup_mode:
        card.g = E

    # Calculate total number bins
    Nx = len(card.x) - 1
    Ny = len(card.y) - 1
    Nz = len(card.z) - 1
    Nt = len(card.t) - 1
    Nmu = len(card.mu) - 1
    N_azi = len(card.azi) - 1
    Ng = len(card.g) - 1
    card.N_bin = Nx * Ny * Nz * Nt * Nmu * N_azi * Ng

    # Scores
    for s in scores:
        score_checked = check_support(
            "score type",
            s,
            [
                "flux",
                "total",
                "fission",
                "density",
                "net-current",
                "mu-sq",
                "time-moment-flux",
                "space-moment-flux",
                "time-moment-current",
                "space-moment-current",
                "time-moment-mu-sq",
                "space-moment-mu-sq",
            ],
        )
        card.scores.append(score_checked)

    # Add to deck
    global_.input_deck.mesh_tallies.append(card)

    return card


def surface_tally(
    surface,
    scores=["net-current"],
):
    """
    Create a tally card to collect MC solutions.

    Parameters
    ----------
    surface : SurfaceCard
        Surface to which the tally is attached to
    scores : list of str {"flux", "net-current"}
        List of physical quantities to be scored.

    Returns
    -------
    SurfaceTallyCard
        The tally card.
    """

    # Make tally card
    card = SurfaceTallyCard(surface.ID)

    # Set ID
    card.ID = len(global_.input_deck.surface_tallies)

    # Set surface
    card.surface_ID = surface.ID
    surface.tally_IDs.append(card.ID)
    surface.N_tally += 1

    # Calculate total number bins
    card.N_bin = 1

    # Scores
    for s in scores:
        score_checked = check_support(
            "score type",
            s,
            ["flux", "net-current"],
        )
        card.scores.append(score_checked)

    # Add to deck
    global_.input_deck.surface_tallies.append(card)

    return card


def cell_tally(
    cell,
    t=np.array([-INF, INF]),
    g=np.array([-INF, INF]),
    E=np.array([0.0, INF]),
    scores=["flux"],
):
    """
    Create a tally card to collect MC solutions.

    Parameters
    ----------
    cell : CellCard
        Cell to which the tally is attached to
    t : array_like[float], optional
        Times that demarcate tally bins.
    g : array_like[float] or str, optional
        Energy group halves that demarcate energy group tally bins.
        String value "all" can be used to tally each individual group.
    E : array_like[float], optional
        Energies that demarcate energy tally bins. This overrides `g` in
        continuous-energy mode.
    scores : list of str {"flux", "net-current"}
        List of physical quantities to be scored.

    Returns
    -------
    CellTallyCard
        The tally card.
    """

    # Make tally card
    card = CellTallyCard(cell.ID)

    # Set ID
    card.ID = len(global_.input_deck.cell_tallies)

    card.t = t
    # Set energy group grid
    if type(g) == type("string") and g == "all":
        G = objects.materials[0].G
        card.g = np.linspace(0, G, G + 1) - 0.5
    else:
        card.g = g
    if not objects.settings.multigroup_mode:
        card.g = E

    # Set cell
    card.cell_ID = cell.ID
    cell.tally_IDs.append(card.ID)
    cell.N_tally += 1

    # Calculate total number bins
    Nt = len(card.t) - 1
    Ng = len(card.g) - 1
    card.N_bin = Nt * Ng

    # Scores
    for s in scores:
        score_checked = check_support(
            "score type",
            s,
            ["flux", "net-current", "fission"],
        )
        card.scores.append(score_checked)

    # Add to deck
    global_.input_deck.cell_tallies.append(card)

    return card


def cs_tally(
    N_cs_bins=10,
    cs_bin_size=([1.0, 1.0]),
    x=np.array([-INF, INF]),
    y=np.array([-INF, INF]),
    z=np.array([-INF, INF]),
    t=np.array([-INF, INF]),
    mu=np.array([-1.0, 1.0]),
    azi=np.array([-PI, PI]),
    g=np.array([-INF, INF]),
    E=np.array([0.0, INF]),
    scores=["flux"],
):
    # Make tally card
    card = CSTallyCard()

    # Set ID
    card.ID = len(global_.input_deck.cs_tallies)

    # Set mesh
    card.x = x
    card.y = y
    card.z = z

    # Set bin properties, convert bin size to problem units
    card.N_cs_bins = N_cs_bins
    card.cs_bin_size[0] = cs_bin_size[0] / (len(x) - 1) * (x[-1] - x[0])
    card.cs_bin_size[1] = cs_bin_size[1] / (len(y) - 1) * (y[-1] - y[0])

    # Set other filters
    card.t = t
    card.mu = mu
    card.azi = azi

    # Set energy group grid
    if type(g) == type("string") and g == "all":
        G = global_.input_deck.materials[0].G
        card.g = np.linspace(0, G, G + 1) - 0.5
    else:
        card.g = g
    if global_.input_deck.setting["mode_CE"]:
        card.g = E

    # Calculate total number bins
    Nx = len(card.x) - 1
    Ny = len(card.y) - 1
    Nz = len(card.z) - 1
    Nt = len(card.t) - 1
    Nmu = len(card.mu) - 1
    N_azi = len(card.azi) - 1
    Ng = len(card.g) - 1
    card.N_bin = Nx * Ny * Nz * Nt * Nmu * N_azi * Ng

    # Scores
    for s in scores:
        score_checked = check_support(
            "score type",
            s,
            ["flux", "total", "fission", "density"],
        )
        card.scores.append(score_checked)

    # Add to deck
    global_.input_deck.cs_tallies.append(card)

    return card
