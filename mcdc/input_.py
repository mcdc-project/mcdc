"""
This module contains functions for setting MC/DC input deck.
Docstrings use NumPy formatting.
"""

# Instantiate and get the global variable container
import mcdc.global_ as global_

import h5py, math, mpi4py, os
import numpy as np
import scipy as sp

from pathlib import Path

from mcdc.card import (
    NuclideCard,
    MaterialCard,
    RegionCard,
    SurfaceCard,
    CellCard,
    UniverseCard,
    LatticeCard,
    SourceCard,
    MeshTallyCard,
)
from mcdc.constant import (
    GYRATION_RADIUS_ALL,
    GYRATION_RADIUS_INFINITE_X,
    GYRATION_RADIUS_INFINITE_Y,
    GYRATION_RADIUS_INFINITE_Z,
    GYRATION_RADIUS_ONLY_X,
    GYRATION_RADIUS_ONLY_Y,
    GYRATION_RADIUS_ONLY_Z,
    INF,
    PCT_NONE,
    PCT_COMBING,
    PCT_COMBING_WEIGHT,
    PCT_SPLITTING_ROULETTE,
    PCT_SPLITTING_ROULETTE_WEIGHT,
    PI,
    REGION_ALL,
    TINY,
    WW_MIN,
    WW_PREVIOUS,
    WW_USER,
    WW_WOLLABER,
)
from mcdc.print_ import print_error
import mcdc.type_ as type_


def nuclide(
    capture=None,
    scatter=None,
    fission=None,
    nu_s=None,
    nu_p=None,
    nu_d=None,
    chi_p=None,
    chi_d=None,
    speed=None,
    decay=None,
):
    """
    Create a nuclide

    Parameters
    ----------
    capture : numpy.ndarray (1D), optional
        Capture microscopic cross-section [barn].
    scatter : numpy.ndarray (2D), optional
        Differential scattering microscopic cross-section [gout, gin] [barn].
    fission : numpy.ndarray (1D), optional
        Fission microscopic cross-section [barn].
    nu_s : numpy.ndarray (1D), optional
        Scattering multiplication.
    nu_p : numpy.ndarray (1D), optional
        Prompt fission neutron yield.
    nu_d : numpy.ndarray (2D), optional
        Delayed neutron precursor yield [dg, gin].
    chi_p : numpy.ndarray (2D), optional
        Prompt fission spectrum [gout, gin].
    chi_d : numpy.ndarray (2D), optional
        Delayed neutron spectrum [gout, dg].
    speed : numpy.ndarray (1D), optional
        Energy group speed [cm/s].
    decay : numpy.ndarray (1D), optional
        Precursor group decay constant [/s].

    Returns
    -------
    NuclideCard
        The nuclide

    Notes
    -----
    Parameters are set to zeros by default. Energy group size G is determined by the
    size of `capture`, `scatter`, or `fission`. Thus, at least `capture`, `scatter`,
    or `fission` needs to be provided. `nu_p` or `nu_d` is needed if `fission` is
    provided. `chi_p` and `chi_d` are needed if `nu_p` and `nu_d` are provided,
    respectively, and G > 1. Delayed neutron precursor group size J is determined by
    the size of `nu_d`; if `nu_d` is not given, J = 0.

    See also
    --------
    mcdc.material : A material can be defined as a collection of nuclides.
    """
    # Energy group size
    if capture is not None:
        G = len(capture)
    elif scatter is not None:
        G = len(scatter)
    elif fission is not None:
        G = len(fission)
    else:
        print_error("Need to supply capture, scatter, or fission to mcdc.nuclide")

    # Delayed group size
    J = 0
    if nu_d is not None:
        J = len(nu_d)

    # Make nuclide card
    card = NuclideCard(G, J)

    # Set ID
    card.ID = len(global_.input_deck.nuclides)

    # Speed (vector of size G)
    if speed is not None:
        card.speed[:] = speed[:]

    # Decay constant (vector of size J)
    if decay is not None:
        card.decay[:] = decay[:]

    # Cross-sections (vector of size G)
    if capture is not None:
        card.capture[:] = capture[:]
    if scatter is not None:
        card.scatter[:] = np.sum(scatter, 0)[:]
    if fission is not None:
        card.fission[:] = fission[:]
        card.fissionable = True
    card.total[:] = card.capture + card.scatter + card.fission

    # Scattering multiplication (vector of size G)
    if nu_s is not None:
        card.nu_s[:] = nu_s[:]

    # Check if nu_p or nu_d is not provided, give fission
    if fission is not None:
        if nu_p is None and nu_d is None:
            print_error("Need to supply nu_p or nu_d for fissionable mcdc.nuclide")

    # Prompt fission production (vector of size G)
    if nu_p is not None:
        card.nu_p[:] = nu_p[:]

    # Delayed fission production (matrix of size GxJ)
    if nu_d is not None:
        # Transpose: [dg, gin] -> [gin, dg]
        card.nu_d[:, :] = np.swapaxes(nu_d, 0, 1)[:, :]

    # Total fission production (vector of size G)
    card.nu_f += card.nu_p
    for j in range(J):
        card.nu_f += card.nu_d[:, j]

    # Scattering spectrum (matrix of size GxG)
    if scatter is not None:
        # Transpose: [gout, gin] -> [gin, gout]
        card.chi_s[:, :] = np.swapaxes(scatter, 0, 1)[:, :]
        for g in range(G):
            if card.scatter[g] > 0.0:
                card.chi_s[g, :] /= card.scatter[g]

    # Prompt fission spectrum (matrix of size GxG)
    if nu_p is not None:
        if G == 1:
            card.chi_p[:, :] = np.array([[1.0]])
        elif chi_p is None:
            print_error("Need to supply chi_p if nu_p is provided and G > 1")
        else:
            # Convert 1D spectrum to 2D
            if chi_p.ndim == 1:
                tmp = np.zeros((G, G))
                for g in range(G):
                    tmp[:, g] = chi_p
                chi_p = tmp
            # Transpose: [gout, gin] -> [gin, gout]
            card.chi_p[:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
            # Normalize
            for g in range(G):
                if np.sum(card.chi_p[g, :]) > 0.0:
                    card.chi_p[g, :] /= np.sum(card.chi_p[g, :])

    # Delayed fission spectrum (matrix of size JxG)
    if nu_d is not None:
        if G == 1:
            card.chi_d[:, :] = np.ones([J, G])
        else:
            if chi_d is None:
                print_error("Need to supply chi_d if nu_d is provided  and G > 1")
            # Transpose: [gout, dg] -> [dg, gout]
            card.chi_d[:, :] = np.swapaxes(chi_d, 0, 1)[:, :]
        # Normalize
        for dg in range(J):
            if np.sum(card.chi_d[dg, :]) > 0.0:
                card.chi_d[dg, :] /= np.sum(card.chi_d[dg, :])

    # Add to deck
    global_.input_deck.nuclides.append(card)

    return card


def material(
    nuclides=None,
    capture=None,
    scatter=None,
    fission=None,
    nu_s=None,
    nu_p=None,
    nu_d=None,
    chi_p=None,
    chi_d=None,
    speed=None,
    decay=None,
    name=None,
):
    """
    Create a material

    A material is defined either as a collection of nuclides or directly by its
    macroscopic constants.

    Parameters
    ----------
    nuclides : list of tuple of (dictionary, float), optional
        List of pairs of nuclide card and its density [/barn-cm].
    capture : numpy.ndarray (1D), optional
        Capture macroscopic cross-section [/cm].
    scatter : numpy.ndarray (2D), optional
        Differential scattering macroscopic cross-section [gout, gin] [/cm].
    fission : numpy.ndarray (1D), optional
        Fission macroscopic cross-section [/cm].
    nu_s : numpy.ndarray (1D), optional
        Scattering multiplication.
    nu_p : numpy.ndarray (1D), optional
        Prompt fission neutron yield.
    nu_d : numpy.ndarray (2D), optional
        Delayed neutron precursor yield [dg, gin].
    chi_p : numpy.ndarray (2D), optional
        Prompt fission spectrum [gout, gin].
    chi_d : numpy.ndarray (2D), optional
        Delayed neutron spectrum [gout, dg].
    speed : numpy.ndarray (1D), optional
        Energy group speed [cm/s].
    decay : numpy.ndarray (1D), optional
        Precursor group decay constant [/s].
    name : str, optional
        Name of the material (e.g. moderator)

    Returns
    -------
    MaterialCard
        The material

    See also
    --------
    mcdc.nuclide : A material can be defined as a collection of nuclides.
    """
    # If nuclides are not given, and macroscopic constants are given instead,
    # create a nuclide card and set a single-nuclide material
    if nuclides is None:
        card_nuclide = nuclide(
            capture,
            scatter,
            fission,
            nu_s,
            nu_p,
            nu_d,
            chi_p,
            chi_d,
            speed,
            decay,
        )
        nuclides = [[card_nuclide, 1.0]]

    # Number of nuclides
    N_nuclide = len(nuclides)

    # Continuous energy mode?
    if isinstance(nuclides[0][0], str):
        global_.input_deck.setting["mode_CE"] = True
        global_.input_deck.setting["mode_MG"] = False

        # Make material card
        card = MaterialCard(N_nuclide)

        # Set ID
        card.ID = len(global_.input_deck.materials)

        # Default values
        card.J = 6

        # Set the nuclides
        for i in range(N_nuclide):
            nuc_name = nuclides[i][0]
            density = nuclides[i][1]

            # Create nuclide card if not defined yet
            if not nuclide_registered(nuc_name):
                nuc_card = NuclideCard()
                nuc_card.name = nuc_name

                # Set ID
                nuc_card.ID = len(global_.input_deck.nuclides)

                # Default values
                nuc_card.J = 6

                # Check if the nuclide is available in the nuclear data library
                dir_name = os.getenv("MCDC_XSLIB")
                if dir_name == None:
                    print_error(
                        "Continuous energy data directory not configured \n       "
                        "see https://cement-psaapgithubio.readthedocs.io/en/latest"
                        "./install.html#configuring-continuous-energy-library \n"
                    )

                # Fissionable flag
                lib_file_name = dir_name + "/" + nuc_name + ".h5"
                if not Path(lib_file_name).is_file():
                    print_error(f"Nuclide data not found: {nuc_name}")
                with h5py.File(lib_file_name, "r") as f:
                    if max(f["fission"][:]) > 0.0:
                        nuc_card.fissionable = True
                        card.fissionable = True

                # Add to deck
                global_.input_deck.nuclides.append(nuc_card)
            else:
                nuc_card = get_nuclide(nuc_name)

            card.nuclide_IDs[i] = nuc_card.ID
            card.nuclide_densities[i] = density

        # Check if there is a material with identical composition already
        if global_.input_deck.setting["mode_CE"]:
            for card_registered in global_.input_deck.materials:
                identical = True
                for i in range(len(card_registered.nuclide_IDs)):
                    if not card_registered.nuclide_IDs[i] == card.nuclide_IDs[i]:
                        identical = False
                        break
                    if (
                        not card_registered.nuclide_densities[i]
                        == card.nuclide_densities[i]
                    ):
                        identical = False
                        break

                if identical:
                    return card_registered

        # Add to deck
        global_.input_deck.materials.append(card)
        return card

    # Nuclide and group sizes
    G = nuclides[0][0].G
    J = nuclides[0][0].J

    # Make material card
    card = MaterialCard(N_nuclide, G, J)

    # Set ID
    card.ID = len(global_.input_deck.materials)

    # Calculate basic XS
    for i in range(N_nuclide):
        nuc = nuclides[i][0]
        density = nuclides[i][1]
        card.nuclide_IDs[i] = nuc.ID
        card.nuclide_densities[i] = density

        card.capture += nuc.capture * density
        card.scatter += nuc.scatter * density
        card.fission += nuc.fission * density
        card.total += nuc.total * density

    # Calculate effective speed
    # Current approach: weighted by nuclide macroscopic total cross section
    # TODO: other more appropriate way?
    for i in range(N_nuclide):
        nuc = nuclides[i][0]
        density = nuclides[i][1]
        card.speed += nuc.speed * nuc.total * density
    # If vacuum material, just pick the last nuclide
    if max(card.total) == 0.0:
        card.speed[:] = nuc.speed
    else:
        card.speed /= card.total

    # Calculate effective spectra and multiplicities of scattering and prompt fission
    if max(card.scatter) > 0.0:
        nuSigmaS = np.zeros((G, G), dtype=float)
        for i in range(N_nuclide):
            nuc = nuclides[i][0]
            density = nuclides[i][1]
            SigmaS = np.diag(nuc.scatter) * density
            nu_s = np.diag(nuc.nu_s)
            chi_s = np.transpose(nuc.chi_s)
            nuSigmaS += chi_s.dot(nu_s.dot(SigmaS))
        chi_nu_s = nuSigmaS.dot(np.diag(1.0 / card.scatter))
        card.nu_s = np.sum(chi_nu_s, axis=0)
        card.chi_s = np.transpose(chi_nu_s.dot(np.diag(1.0 / card.nu_s)))
    if max(card.fission) > 0.0:
        nuSigmaF = np.zeros((G, G), dtype=float)
        for i in range(N_nuclide):
            nuc = nuclides[i][0]
            density = nuclides[i][1]
            SigmaF = np.diag(nuc.fission) * density
            nu_p = np.diag(nuc.nu_p)
            chi_p = np.transpose(nuc.chi_p)
            nuSigmaF += chi_p.dot(nu_p.dot(SigmaF))
        chi_nu_p = nuSigmaF.dot(np.diag(1.0 / card.fission))
        card.nu_p = np.sum(chi_nu_p, axis=0)
        card.chi_p = np.transpose(chi_nu_p.dot(np.diag(1.0 / card.nu_p)))

    # Calculate delayed and total fission multiplicities
    if max(card.fission) > 0.0:
        card.nu_f[:] = card.nu_p[:]
        for j in range(J):
            total = np.zeros(G)
            for i in range(N_nuclide):
                nuc = nuclides[i][0]
                density = nuclides[i][1]
                total += nuc.nu_d[:, j] * nuc.fission * density
            card.nu_d[:, j] = total / card.fission
            card.nu_f += card.nu_d[:, j]

    # Add to deck
    global_.input_deck.materials.append(card)

    return card


def surface(type_, bc="interface", **kw):
    """Create a surface to define the region of a cell.

    Parameters
    ----------
    type\_ : {"plane-x", "plane-y", "plane-z", "plane", "cylinder-x", "cylinder-y",
              "cylinder-z", "sphere", "quadric"}
        Surface type.
    bc : {"interface", "vacuum", "reflective"}
        Surface boundary condition.

    Other Parameters
    ----------------
    x : {float, array_like[float]}
        x-position [cm] for `"plane-x"`.
    y : {float, array_like[float]}
        y-position [cm] for `"plane-y"`.
    z : {float, array_like[float]}
        z-position [cm] for `"plane-z"`.
    center : array_like[float]
        Center point [cm] for `"cylinder-x"` (y,z), `"cylinder-y"` (x,z),
        `"cylinder-z"` (x,y), or `"sphere"` (x,y,z).
    radius : float
        Radius [cm] for `"cylinder-x"`, `"cylinder-y"`, `"cylinder-z"`, and `"sphere"`.
    A, B, C, D : float
        Coefficients [cm] for `"plane"`.
    A, B, C, D, E, F, G, H, I, J : float
        Coefficients [cm] for `"quadric"`.

    Returns
    -------
    SurfaceCard
        The surface card

    See also
    --------
    mcdc.cell : Create a cell whose region is defined by surfaces.
    """
    # Make surface card
    card = SurfaceCard()

    # Set ID
    card.ID = len(global_.input_deck.surfaces)

    # Check if the selected type is supported
    type_ = check_support(
        "surface type",
        type_,
        [
            "plane-x",
            "plane-y",
            "plane-z",
            "plane",
            "cylinder-x",
            "cylinder-y",
            "cylinder-z",
            "sphere",
            "quadric",
        ],
    )
    card.type = type_

    # Boundary condition
    bc = check_support(
        "surface boundary condition",
        bc,
        [
            "interface",
            "vacuum",
            "reflective",
        ],
    )
    card.boundary_type = bc

    # ==========================================================================
    # Surface attributes
    # ==========================================================================
    # Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0

    card.type = type_

    # Set up surface attributes
    if type_ == "plane-x":
        check_requirement("surface plane-x", kw, ["x"])
        card.G = 1.0
        card.J = -kw.get("x")
        card.linear = True
    elif type_ == "plane-y":
        check_requirement("surface plane-y", kw, ["y"])
        card.H = 1.0
        card.J = -kw.get("y")
        card.linear = True
    elif type_ == "plane-z":
        check_requirement("surface plane-z", kw, ["z"])
        card.I = 1.0
        card.J = -kw.get("z")
        card.linear = True
    elif type_ == "plane":
        check_requirement("surface plane", kw, ["A", "B", "C", "D"])
        card.G = kw.get("A")
        card.H = kw.get("B")
        card.I = kw.get("C")
        card.J = kw.get("D")
        card.linear = True
    elif type_ == "cylinder-x":
        check_requirement("surface cylinder-x", kw, ["center", "radius"])
        y, z = kw.get("center")[:]
        r = kw.get("radius")
        card.B = 1.0
        card.C = 1.0
        card.H = -2.0 * y
        card.I = -2.0 * z
        card.J = y**2 + z**2 - r**2
    elif type_ == "cylinder-y":
        check_requirement("surface cylinder-y", kw, ["center", "radius"])
        x, z = kw.get("center")[:]
        r = kw.get("radius")
        card.A = 1.0
        card.C = 1.0
        card.G = -2.0 * x
        card.I = -2.0 * z
        card.J = x**2 + z**2 - r**2
    elif type_ == "cylinder-z":
        check_requirement("surface cylinder-z", kw, ["center", "radius"])
        x, y = kw.get("center")[:]
        r = kw.get("radius")
        card.A = 1.0
        card.B = 1.0
        card.G = -2.0 * x
        card.H = -2.0 * y
        card.J = x**2 + y**2 - r**2
    elif type_ == "sphere":
        check_requirement("surface sphere", kw, ["center", "radius"])
        x, y, z = kw.get("center")[:]
        r = kw.get("radius")
        card.A = 1.0
        card.B = 1.0
        card.C = 1.0
        card.G = -2.0 * x
        card.H = -2.0 * y
        card.I = -2.0 * z
        card.J = x**2 + y**2 + z**2 - r**2
    elif type_ == "quadric":
        check_requirement(
            "surface quadric", kw, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        )
        card.A = kw.get("A")
        card.B = kw.get("B")
        card.C = kw.get("C")
        card.D = kw.get("D")
        card.E = kw.get("E")
        card.F = kw.get("F")
        card.G = kw.get("G")
        card.H = kw.get("H")
        card.I = kw.get("I")
        card.J = kw.get("J")

    # Normalize linear surfaces
    if card.linear:
        G = card.G
        H = card.H
        I = card.I
        norm = (G**2 + H**2 + I**2) ** 0.5
        card.G /= norm
        card.H /= norm
        card.I /= norm
        card.J /= norm
        card.nx = card.G
        card.ny = card.H
        card.nz = card.I

    # Add to deck
    global_.input_deck.surfaces.append(card)

    return card


def cell(region=None, fill=None, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    """
    Create a cell as model building block.

    Parameters
    ----------
    region : RegionCard
        Region that defines the cell geometry.
    fill : MaterialCard or UniverseCard or LatticeCard
        Material/universe/lattice that fills the cell.
    translation : array_like[float], optional
        To translate the origin of the fill (if universe or lattice).
    rotation : array_like[float], optional
        To rotate the the fill (if universe or lattice).

    Returns
    -------
    CellCard
        The cell card.

    See also
    --------
    mcdc.surface : Create a surface to define the region of a cell.
    mcdc.material : Create a material to fill a cell.
    mcdc.universe : Create a universe to fill a cell.
    mcdc.lattice : Create a lattice to fill a cell.
    """

    # Make cell card
    card = CellCard()

    # Set ID
    card.ID = len(global_.input_deck.cells)

    # If region is not assigned, create a region that encompass all
    if region is None:
        region = RegionCard("all")
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)

    # Assign region
    card.region_ID = region.ID

    # Set region Reverse Polish Notation and region description
    if region.type != "all":
        card.set_region_RPN()
        card.set_region()

    # Assign fill type and ID
    if fill.tag == "Material":
        card.fill_type = "material"
    elif fill.tag == "Universe":
        card.fill_type = "universe"
    elif fill.tag == "Lattice":
        card.fill_type = "lattice"
    card.fill_ID = fill.ID

    # Translation
    card.translation[:] = translation

    # Rotation
    card.rotation[:] = rotation

    # Get all surface IDs
    card.set_surface_IDs()

    # Add to deck
    global_.input_deck.cells.append(card)

    return card


def universe(cells, root=False):
    """
    Define a list of cells as a universe.

    Parameters
    ----------
    cells : list of CellCard
        List of cells that comprise the universe.
    root : bool
        Flag to edit the root universe

    Returns
    -------
    UniverseCard
        The universe card

    See also
    --------
    mcdc.cell : Creates a cell that can be used to define a universe.
    """

    # Edit root universe
    if root:
        # Create and replace placeholder if root is not yet created
        if global_.input_deck.universes[0] == None:
            card = UniverseCard()
            card.ID = 0
            global_.input_deck.universes[0] = card
        else:
            card = global_.input_deck.universes[0]

    # Create new universe
    else:
        card = UniverseCard()
        card.ID = len(global_.input_deck.universes)

    # Cells
    N_cell = len(cells)
    card.cell_IDs = np.zeros(N_cell, dtype=int)
    for i in range(N_cell):
        card.cell_IDs[i] = cells[i].ID

    # Push card
    if not root:
        global_.input_deck.universes.append(card)

    return card


def lattice(x=None, y=None, z=None, universes=None):
    """
    Create a lattice card.

    Parameters
    ----------
    x : array_like[float], optional
        x-coordinates that define the lattice grid (default None).
    y : array_like[float], optional
        y-coordinates that define the lattice grid (default None).
    z : array_like[float], optional
        z-coordinates that define the lattice grid (default None).
    universes : list of (list of dictionary), optional
        List of lists of universe cards that fill the lattice (default None).

    Returns
    -------
    dictionary
        Lattice card.
    """
    # Make lattice card
    card = LatticeCard()
    card.ID = len(global_.input_deck.lattices)

    # Set grid
    if x is not None:
        card.x0 = x[0]
        card.dx = x[1]
        card.Nx = x[2]
    if y is not None:
        card.y0 = y[0]
        card.dy = y[1]
        card.Ny = y[2]
    if z is not None:
        card.z0 = z[0]
        card.dz = z[1]
        card.Nz = z[2]

    # Set universe IDs
    get_ID = np.vectorize(lambda obj: obj.ID)
    universe_IDs = get_ID(universes)
    ax_expand = []
    if x is None:
        ax_expand.append(2)
    if y is None:
        ax_expand.append(1)
    if z is None:
        ax_expand.append(0)
    for ax in ax_expand:
        universe_IDs = np.expand_dims(universe_IDs, axis=ax)

    # Change indexing structure: [z(flip), y(flip), x] --> [x, y, z]
    tmp = np.transpose(universe_IDs)
    tmp = np.flip(tmp, axis=1)
    card.universe_IDs = np.flip(tmp, axis=2)

    # Push card
    global_.input_deck.lattices.append(card)
    return card


def source(**kw):
    """
    Create a source card.

    Other Parameters
    ----------------
    point : array_like
        [x, y, z] point position for point source.
    x : array_like
        [x_min and x_max] for uniform source.
    y : array_like
        [y_min and y_max] for uniform source.
    z : array_like
        [z_min and z_max] for uniform source.
    isotropic : bool
        Flag for whether source is isotropic.
    direction : array_like
        [ux, uy, uz] unit vector for parallel beam source.
    white_direction : array_like
        [nx, ny, nz] unit vector of the normal outward direction of the surface
        at which isotropic surface source is emitted. Note that it is similar to the
        mechanics of the typical white boundary condition in reactor physics.
    energy : array_like
        [MG] Probability mass function of the energy group for multigroup source.
        [CE] 2D array of piecewise linear pdf [eV, value].
    time : array_like
        [t_min and t_max] in/at which source is emitted.
    prob : float
        Relative probability (or strength) of the source.

    Returns
    -------
    dictionary
        A source card.
    """
    # Check the supplied keyword arguments
    for key in kw.keys():
        check_support(
            "source parameter",
            key,
            [
                "point",
                "x",
                "y",
                "z",
                "isotropic",
                "direction",
                "white_direction",
                "energy",
                "time",
                "prob",
            ],
            False,
        )

    # Get keyword arguments
    point = kw.get("point")
    x = kw.get("x")
    y = kw.get("y")
    z = kw.get("z")
    isotropic = kw.get("isotropic")
    direction = kw.get("direction")
    white = kw.get("white_direction")
    energy = kw.get("energy")
    time = kw.get("time")
    prob = kw.get("prob")

    # Make source card
    card = SourceCard()

    # Set ID
    card.ID = len(global_.input_deck.sources)

    # Set position
    if point is not None:
        card.x = point[0]
        card.y = point[1]
        card.z = point[2]
    else:
        card.box = True
        if x is not None:
            card.box_x = np.array(x)
        if y is not None:
            card.box_y = np.array(y)
        if z is not None:
            card.box_z = np.array(z)

    # Set direction
    if white is not None:
        card.isotropic = False
        card.white = True
        ux = white[0]
        uy = white[1]
        uz = white[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2) ** 0.5
        card.white_x = ux / norm
        card.white_y = uy / norm
        card.white_z = uz / norm
    elif direction is not None:
        card.isotropic = False
        ux = direction[0]
        uy = direction[1]
        uz = direction[2]
        # Normalize
        norm = (ux**2 + uy**2 + uz**2) ** 0.5
        card.ux = ux / norm
        card.uy = uy / norm
        card.uz = uz / norm

    # Set energy
    if energy is not None:
        if global_.input_deck.setting["mode_MG"]:
            group = np.array(energy)
            # Normalize
            card.group = group / np.sum(group)
        if global_.input_deck.setting["mode_CE"]:
            energy = np.array(energy)
            # Resize
            card.energy = np.zeros(energy.shape)
            # Set energy
            card.energy[0, :] = energy[0, :]
            # Normalize pdf
            card.energy[1, :] = energy[1, :] / np.trapz(energy[1, :], x=energy[0, :])
            # Make cdf
            card.energy[1, :] = sp.integrate.cumulative_trapezoid(
                card.energy[1], x=card.energy[0], initial=0.0
            )
    else:
        # Default for MG
        if global_.input_deck.setting["mode_MG"]:
            G = global_.input_deck.materials[0].G
            group = np.ones(G)
            card.group = group / np.sum(group)
        # Default for CE
        if global_.input_deck.setting["mode_CE"]:
            # Normalize pdf
            card.energy[1, :] = card.energy[1, :] / np.trapz(
                card.energy[1, :], x=card.energy[0, :]
            )
            # Make cdf
            card.energy[1, :] = sp.integrate.cumulative_trapezoid(
                card.energy[1], x=card.energy[0], initial=0.0
            )

    # Set time
    if time is not None:
        card.time = np.array(time)

    # Set probability
    if prob is not None:
        card.prob = prob

    # Push card
    global_.input_deck.sources.append(card)

    return card


# ==============================================================================
# Setting
# ==============================================================================


def setting(**kw):
    """
    Create a setting card.

    Other Parameters
    ----------------
    N_particle : int
        Number of MC particle histories to run (for k-eigen and iQMC its /iteration).
    N_batch : int
        Number of batches to run.
    rng_seed : int
        Random number seed.
    time_boundary : float
        The time edge of the problem, after which all particles will be killed.
    progress_bar : bool
        Whether to display the progress bar (default True; disable when running MC/DC in a loop).
    output_name : str
        Name of the output file MC/DC should save data in (default "output.h5").
    save_input_deck : bool
        Whether to save the input deck information to the output file (default False).
    k_eff : str
        Whether to run a k-eigenvalue problem.
    source_file : str
        Source file path and name.
    IC_file : str
        Path to a file containing a description of an initial condition.
    active_bank_buff : int
        Size of the particle active bank buffer.
    census_bank_buff : int
        Size of the particle census bank buffer (in multiples of N_particle).
    source_bank_buff : int
        Size of the particle source bank buffer (in multiples of N_particle).
    future_bank_buff : int
        Size of the particle future bank buffer (in multiples of N_particle).

    Returns
    -------
    dictionary
        A setting card.
    """

    # Check the supplied keyword arguments
    for key in kw.keys():
        check_support(
            "setting parameter",
            key,
            [
                "N_particle",
                "N_batch",
                "rng_seed",
                "time_boundary",
                "progress_bar",
                "output_name",
                "save_input_deck",
                "k_eff",
                "source_file",
                "IC_file",
                "active_bank_buff",
                "census_bank_buff",
                "source_bank_buff",
                "future_bank_buff",
            ],
            False,
        )

    # Get keyword arguments
    N_particle = kw.get("N_particle")
    N_batch = kw.get("N_batch")
    rng_seed = kw.get("rng_seed")
    time_boundary = kw.get("time_boundary")
    progress_bar = kw.get("progress_bar")
    output = kw.get("output_name")
    save_input_deck = kw.get("save_input_deck")
    k_eff = kw.get("k_eff")
    source_file = kw.get("source_file")
    IC_file = kw.get("IC_file")
    bank_active_buff = kw.get("active_bank_buff")
    bank_census_buff = kw.get("census_bank_buff")
    bank_source_buff = kw.get("source_bank_buff")
    bank_future_buff = kw.get("future_bank_buff")

    # Check if setting card has been initialized
    card = global_.input_deck.setting

    # Number of particles
    if N_particle is not None:
        card["N_particle"] = int(N_particle)

    # Number of batches
    if N_batch is not None:
        card["N_batch"] = int(N_batch)

    # Time boundary
    if time_boundary is not None:
        card["time_boundary"] = time_boundary

    # RNG seed and stride
    if rng_seed is not None:
        card["rng_seed"] = rng_seed

    # Output .h5 file name
    if output is not None:
        card["output_name"] = output

    # Progress bar
    if progress_bar is not None:
        card["progress_bar"] = progress_bar

    # k effective
    if k_eff is not None:
        card["k_init"] = k_eff

    # Maximum active bank size
    if bank_active_buff is not None:
        card["bank_active_buff"] = int(bank_active_buff)

    # Census bank size multiplier
    if bank_census_buff is not None:
        card["bank_census_buff"] = int(bank_census_buff)

    # Source bank size multiplier
    if bank_source_buff is not None:
        card["bank_source_buff"] = int(bank_source_buff)

    # Future bank size multiplier
    if bank_future_buff is not None:
        card["bank_future_buff"] = int(bank_future_buff)

    # Save input deck?
    if save_input_deck is not None:
        card["save_input_deck"] = save_input_deck

    # Source file
    if source_file is not None:
        card["source_file"] = True
        card["source_file_name"] = source_file

        # Set number of particles
        card_setting = global_.input_deck.setting
        with h5py.File(source_file, "r") as f:
            card_setting["N_particle"] = f["particles_size"][()]

    # IC file
    if IC_file is not None:
        card["IC_file"] = True
        card["IC_file_name"] = IC_file

        # Set number of particles
        card_setting = global_.input_deck.setting
        with h5py.File(IC_file, "r") as f:
            card_setting["N_particle"] = f["IC/neutrons_size"][()]
            card_setting["N_precursor"] = f["IC/precursors_size"][()]

    # TODO: Allow both source and IC files
    if IC_file and source_file:
        print_error("Using both source and IC files is not supported yet.")


def eigenmode(
    N_inactive=0, N_active=0, k_init=1.0, gyration_radius=None, save_particle=False
):
    """
    Create an eigenmode card.

    Parameters
    ----------
    N_inactive : int
        Number of cycles not included when averaging the k-eigenvalue (default 0).
    N_active : int
        Number of cycles to include for statistics of the k-eigenvalue (default 0).
    k_init : float
        Initial k value to iterate on (default 1.0).
    gyration_radius : float, optional
        Specify a gyration radius (default None).
    save_particle : bool
        Whether final particle bank outputs (default False).

    Returns
    -------
    dictionary
        A eigenmode card.
    """

    # Update setting card
    card = global_.input_deck.setting
    card["N_inactive"] = N_inactive
    card["N_active"] = N_active
    card["N_cycle"] = N_inactive + N_active
    card["mode_eigenvalue"] = True
    card["k_init"] = k_init
    card["save_particle"] = save_particle

    # Gyration radius setup
    if gyration_radius is not None:
        card["gyration_radius"] = True
        if gyration_radius == "all":
            card["gyration_radius_type"] = GYRATION_RADIUS_ALL
        elif gyration_radius == "infinite-x":
            card["gyration_radius_type"] = GYRATION_RADIUS_INFINITE_X
        elif gyration_radius == "infinite-y":
            card["gyration_radius_type"] = GYRATION_RADIUS_INFINITE_Y
        elif gyration_radius == "infinite-z":
            card["gyration_radius_type"] = GYRATION_RADIUS_INFINITE_Z
        elif gyration_radius == "only-x":
            card["gyration_radius_type"] = GYRATION_RADIUS_ONLY_X
        elif gyration_radius == "only-y":
            card["gyration_radius_type"] = GYRATION_RADIUS_ONLY_Y
        elif gyration_radius == "only-z":
            card["gyration_radius_type"] = GYRATION_RADIUS_ONLY_Z
        else:
            print_error("Unknown gyration radius type")


# ==============================================================================
# Technique
# ==============================================================================


def implicit_capture():
    """
    Activate implicit capture (implies no weighted emission).
    """
    card = global_.input_deck.technique
    card["implicit_capture"] = True
    card["weighted_emission"] = False


def weighted_emission(flag):
    """
    Activate weighted emission variance reduction technique.

    Parameters
    ----------
    flag : bool
        True to activate weighted emission.
    """

    card = global_.input_deck.technique
    card["weighted_emission"] = flag


def population_control(pct="splitting-roulette"):
    """
    Set population control techniques.

    Parameters
    ----------
    pct : str, optional
        Population control method (default "spliting-roulette").
    """
    # Check if the selected technique is supported
    pct = check_support(
        "population control technique",
        pct,
        [
            "combing",
            "combing-weight",
            "splitting-roulette",
            "splitting-roulette-weight",
        ],
    )
    card = global_.input_deck.technique
    card["pct"] = pct
    card["population_control"] = True
    card["weighted_emission"] = False


def branchless_collision():
    """
    Activate branchless collision variance reduction technique (implies no weighted emission).
    """
    card = global_.input_deck.technique
    card["branchless_collision"] = True
    card["weighted_emission"] = False


def delta_tracking(
    collision_estimator=False,
    cutoff_energy=0,
    material_exclusion=INF,
    plot_majorant=True,
):
    """
    Activate delta tracking and set hybrid parameters

    Parameters
    ----------
    collision_estimator : bool
        Use a collision estimator for flux tallies
    cutoff_energy : float
        A cut off energy, underwich do surface tracking (**continuous energy only**) (look at the majorant plot and make an informed decision!)
    material_exclusion : int
        Material ID of a single region not to do delta tracking in (TODO: Make multiple regions) (**continuous energy only**)
    plot_majorant : bool
        Saves a .pdf plot of the majorant and nuclides when produced

    Notes
    -----
    TODO: turn off other incompatible tracking techniques if any (k-eigenvalue, fission flux tallies, )
    Doesn't work with k-eigenvalue simulations

    """
    card = global_.input_deck.technique
    card["delta_tracking"] = True
    card["collision_estimator"] = collision_estimator
    card["dt_cutoff_energy"] = cutoff_energy
    card["dt_material_exclusion"] = material_exclusion
    card["plot_majorant"] = plot_majorant


def time_census(t, tally_frequency=None):
    """
    Set time-census boundaries.

    Parameters
    ----------
    t : array_like[float]
        The time-census boundaries.
    tally_frequency : integer, optional
        Number of uniform tally time mesh bins in census-based tallying.
        This overrides manual tally time mesh definitions.

    Returns
    -------
        None (in-place card alterations).
    """

    # Make sure that the time grid points are sorted
    if not is_sorted(t):
        print_error("Time census: Time grid points have to be sorted.")

    # Make sure that the starting point is larger than zero
    if t[0] <= 0.0:
        print_error("Time census: First census time should be larger than zero.")

    # Add the default, final census-at-infinity
    t = np.append(t, INF)

    # Set the time census parameters
    card = global_.input_deck.setting
    card["census_time"] = t
    card["N_census"] = len(t)

    # Set the census-based tallying
    if tally_frequency is not None and tally_frequency > 0:
        # Reset all tallies' time grids:
        card["census_based_tally"] = True
        card["census_tally_frequency"] = tally_frequency


def weight_window(
    x=np.array([-INF, INF]),
    y=np.array([-INF, INF]),
    z=np.array([-INF, INF]),
    mu=np.array([-1.0, 1.0]),
    azi=np.array([-PI, PI]),
    g=np.array([-INF, INF]),
    E=np.array([0.0, INF]),
    window=None,
    width=2.5,
    method={"user"},
    modifications={},
    save_ww_data=True,
):
    """
    Activate weight window variance reduction technique.

    Parameters
    ----------
    x : array_like[float], optional
        Location of the weight window in x (default None).
    y : array_like[float], optional
        Location of the weight window in y (default None).
    z : array_like[float], optional
        Location of the weight window in z (default None).
    t : array_like[float], optional
        Location of the weight window in t (default None).
    window : array_like[float], optional
        Center of the weight windows (default None).
    width : float, optional
        Width of the window (default 2.5).
    epsilon : float, optional
        Small values used for techniques (default empty list).
    techniques : list of str, optional
        List of techniques to use for ww
        {'user','previous','alpha','min_center','wollaber'} (default {'user'}).
    Returns
    -------
        A weight window card.

    """

    t = global_.input_deck.setting["census_time"]
    card = global_.input_deck.technique
    card["weight_window"] = True
    N_update = 0

    card["ww"]["save"] = save_ww_data
    # Set width
    if width is not None:
        card["ww"]["width"] = width

    # Checking WW method
    method_checked = check_support(
        "Weight window method",
        method,
        ["user", "previous"],
    )
    if method_checked == "user":
        card["ww"]["auto"] = WW_USER
    elif method_checked == "previous":
        card["ww"]["auto"] = WW_PREVIOUS

        scores = (["flux"],)
        # Make tally card
        tcard = MeshTallyCard()

        # Set ID
        tcard.ID = len(global_.input_deck.mesh_tallies)
        card["ww"]["tally_idx"] = tcard.ID

        # Set mesh
        tcard.x = x
        tcard.y = y
        tcard.z = z

        # Set other filters
        tcard.t = t
        tcard.mu = mu
        tcard.azi = azi

        # Set energy group grid
        if type(g) == type("string") and g == "all":
            G = global_.input_deck.materials[0].G
            tcard.g = np.linspace(0, G, G + 1) - 0.5
        else:
            tcard.g = g
        if global_.input_deck.setting["mode_CE"]:
            tcard.g = E

        # Calculate total number bins
        Nx = len(tcard.x) - 1
        Ny = len(tcard.y) - 1
        Nz = len(tcard.z) - 1
        Nt = len(tcard.t) - 1
        Nmu = len(tcard.mu) - 1
        N_azi = len(tcard.azi) - 1
        Ng = len(tcard.g) - 1
        tcard.N_bin = Nx * Ny * Nz * Nt * Nmu * N_azi * Ng
        tcard.scores.append("flux")
        # Add to deck
        global_.input_deck.mesh_tallies.append(tcard)

    # Checking techniques
    for mod in modifications:
        mod_checked = check_support(
            "Weight window modification",
            mod[0],
            ["min-center", "wollaber"],
        )
        if mod_checked == "min-center":
            card["ww"]["epsilon"][WW_MIN] = mod[1]
        if mod_checked == "wollaber":
            card["ww"]["epsilon"][WW_WOLLABER] = mod[1]
            card["ww"]["epsilon"][WW_WOLLABER + 1] = mod[2]

    # Set mesh
    card["ww"]["mesh"]["x"] = x
    card["ww"]["mesh"]["y"] = y
    card["ww"]["mesh"]["z"] = z
    card["ww"]["mesh"]["t"] = t
    card["ww"]["mesh"]["mu"] = mu
    card["ww"]["mesh"]["azi"] = azi

    # Set energy group grid
    if type(g) == type("string") and g == "all":
        G = global_.input_deck.materials[0].G
        card["ww"]["mesh"]["g"] = np.linspace(0, G, G + 1) - 0.5
    else:
        tcard.g = g
    if global_.input_deck.setting["mode_CE"]:
        card["ww"]["mesh"]["g"] = E

    if window is None:
        window = np.ones((Nt, Nx, Ny, Nz))
    """
    # Set window
    ax_expand = []
    if t is None:
        ax_expand.append(0)
    if x is None:
        ax_expand.append(1)
    if y is None:
        ax_expand.append(2)
    if z is None:
        ax_expand.append(3)
    for ax in ax_expand:
        window = np.expand_dims(window, axis=ax)
    """
    card["ww"]["center"] = window
    return card, tcard


def domain_decomposition(
    x=None,
    y=None,
    z=None,
    exchange_rate=100000,
    exchange_rate_padding=None,
    work_ratio=None,
):
    """
    Activate domain decomposition.

    Parameters
    ----------
    x : array_like[float], optional
        Location of subdomain boundaries in x (default None).
    y : array_like[float], optional
        Location of subdomain boundaries in y (default None).
    z : array_like[float], optional
        Location of subdomain boundaries in z (default None).
    exchange_rate : float, optional
        Number of particles to acumulate in the domain banks before sending.
    work_ratio : array_like[integer], optional
        Number of processors in each domain

    Returns
    -------
        A domain decomposition card.

    """
    card = global_.input_deck.technique
    card["domain_decomposition"] = True
    card["dd_exchange_rate"] = int(exchange_rate)
    card["dd_exchange_rate_padding"] = exchange_rate_padding
    dom_num = 1
    # Set mesh
    if x is not None:
        card["dd_mesh"]["x"] = x
        dom_num *= len(x)
    if y is not None:
        card["dd_mesh"]["y"] = y
        dom_num *= len(y)
    if z is not None:
        card["dd_mesh"]["z"] = z
        dom_num += len(z)

    card["dd_work_ratio"] = work_ratio
    card["dd_idx"] = 0
    card["dd_xp_neigh"] = []
    card["dd_xn_neigh"] = []
    card["dd_yp_neigh"] = []
    card["dd_yn_neigh"] = []
    card["dd_zp_neigh"] = []
    card["dd_zn_neigh"] = []
    return card


def iQMC(
    phi0=None,
    g=None,
    t=None,
    x=None,
    y=None,
    z=None,
    source0=None,
    source_x0=None,
    source_y0=None,
    source_z0=None,
    krylov_restart=None,
    fixed_source=None,
    maxit=25,
    tol=1e-6,
    fixed_source_solver="source iteration",
    sample_method="halton",
    mode="fixed",
    scores=[],
):
    """
    Activate the iterative Quasi-Monte Carlo (iQMC) neutron transport method.

    Parameters
    ----------
    phi0 : array_like[float], optional
        Initial scalar flux approximation (default None).
    g : array_like[float], optional
        Energy values that define energy mesh (default None).
    t : array_like[float], optional
        Time values that define time mesh (default None).
    x : array_like[float], optional
        x-coordinates that define spacial mesh (default None).
    y : array_like[float], optional
        y-coordinates that define spacial mesh (default None).
    z : array_like[float], optional
        z-coordinates that define spacial mesh (default None).

    Other Parameters
    ----------
    source0 : array_like[float], optional
        Initial particle source (default None).
    source_x0 : array_like[float], optional
        Initial source for source-x (default None).
    source_y0 : array_like[float], optional
        Initial source for source-y (default None).
    source_z0 : array_like[float], optional
        Initial source for source-z (default None).
    krylov_restart : int, optional
        Max number of iterations for Krylov iteration (default same as maxit).
    fixed_source : array_like[float], optional
        Fixed source (default same as phi0).
    iterations_max : int, optional
        Maximum number of iterations allowed before termination (default 25).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    fixed_source_solver : {'source iteration', 'gmres'}
        Deterministic solver for fixed-source problem (default "source iteration").
        Solver for k-eigenvalue problem (default "power_iteration").
    sample_method: {'halton', 'random'}
        Method for generating particle samples.
    mode: {'fixed', batched}
        Set iQMC to run with a fixed-seed or batched iteration scheme.
    scores : list of str, optional
        List of tallies to score in addition to the mandatory flux and
        source strength. Additional scores include
        {'source-x', 'source-y', 'source-z', 'fission-power'} (default empty list).

    Returns
    -------
        None (in-place card alterations).

    Notes
    -----
        phi0 is used to estimate the initial source strength. If source0 is
        provided, source0 will be used instead of phi0. Either phi0 or
        source0 must be provided as they are used to initialize particle
        weights.
    """

    card = global_.input_deck.technique
    card["iQMC"] = True
    card["iqmc"]["tol"] = tol
    card["iqmc"]["iterations_max"] = maxit
    card["iqmc"]["sample_method"] = sample_method
    card["iqmc"]["mode"] = mode

    # Set mesh
    if g is not None:
        card["iqmc"]["mesh"]["g"] = g
    if t is not None:
        card["iqmc"]["mesh"]["t"] = t
    if x is not None:
        card["iqmc"]["mesh"]["x"] = x
    if y is not None:
        card["iqmc"]["mesh"]["y"] = y
    if z is not None:
        card["iqmc"]["mesh"]["z"] = z

    ax_expand = []
    if g is None:
        ax_expand.append(0)
    if t is None:
        ax_expand.append(1)
    if x is None:
        ax_expand.append(2)
    if y is None:
        ax_expand.append(3)
    if z is None:
        ax_expand.append(4)
    for ax in ax_expand:
        phi0 = np.expand_dims(phi0, axis=ax)
        if fixed_source is not None:
            fixed_source = np.expand_dims(fixed_source, axis=ax)
        else:
            fixed_source = np.zeros_like(phi0)

    if krylov_restart is None:
        krylov_restart = maxit

    if source0 is None:
        source0 = np.zeros_like(phi0)

    score_list = card["iqmc"]["score_list"]
    for name in scores:
        score_list[name] = True

    if score_list["source-x"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_x0 is None:
            source_x0 = np.zeros_like(phi0)

    if score_list["source-y"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_y0 is None:
            source_y0 = np.zeros_like(phi0)

    if score_list["source-z"]:
        card["iqmc"]["krylov_vector_size"] += 1
        if source_z0 is None:
            source_z0 = np.zeros_like(phi0)

    card["iqmc"]["score"]["flux"] = phi0
    card["iqmc"]["score"]["source-x"] = source_x0
    card["iqmc"]["score"]["source-y"] = source_y0
    card["iqmc"]["score"]["source-z"] = source_z0
    card["iqmc"]["source"] = source0
    card["iqmc"]["fixed_source"] = fixed_source
    card["iqmc"]["fixed_source_solver"] = fixed_source_solver
    card["iqmc"]["krylov_restart"] = krylov_restart


def weight_roulette(w_threshold=0.2, w_survive=1.0):
    """
    Activate weight roulette technique.

    If neutron weight is below `w_threshold`, then enter weight roulette
    technique with survival weight `w_survive`.

    Parameters
    ----------
    w_threshold : float
        Weight_roulette() is called on a particle if P['w'] <= wr_threshold.
    w_survive : float
        Weight of surviving particle.

    Returns
    -------
        None (in-place card alterations).
    """
    card = global_.input_deck.technique
    card["weight_roulette"] = True
    card["wr_threshold"] = w_threshold
    card["wr_survive"] = w_survive


def IC_generator(
    N_neutron=0,
    N_precursor=0,
    cycle_stretch=1.0,
    neutron_density=None,
    max_neutron_density=None,
    precursor_density=None,
    max_precursor_density=None,
):
    """
    Activate initial condition generator.

    The initial condition generator samples initial neutrons and precursors
    during an eigenvalue simulation.

    Parameters
    ----------
    N_neutron : int
        Neutron target size.
    N_precursor : int
        Delayed neutron precursor target size.
    cycle_stretch : float
        Factor to stretch number of cycles. Higher cycle stretch reduces inter-cycle
        correlation.
    neutron_density, max_neutron_density : float
        Total and maximum neutron density, required if `N_neutron` > 0.
    precursor_density, max_precursor_density : float
        Total and maximum precursor density, required if `N_precursor` > 0.

    Returns
    -------
        None (in-place card alterations).
    """

    # Turn on eigenmode and population control
    eigenmode()
    population_control()

    # Set parameters
    card = global_.input_deck.technique
    card["IC_generator"] = True
    card["IC_N_neutron"] = N_neutron
    card["IC_N_precursor"] = N_precursor

    # Setting parameters
    card_setting = global_.input_deck.setting
    N_particle = card_setting["N_particle"]

    # Check optional parameters
    if N_neutron > 0.0:
        if neutron_density is None or max_neutron_density is None:
            print_error("IC generator requires neutron_density and max_neutron_density")
        card["IC_neutron_density"] = N_particle * neutron_density
        card["IC_neutron_density_max"] = max_neutron_density
    if N_precursor > 0.0:
        if precursor_density is None:
            print_error(
                "IC generator requires precursor_density and max_precursor_density"
            )
        card["IC_precursor_density"] = N_particle * precursor_density
        card["IC_precursor_density_max"] = max_precursor_density

    # Set number of active cycles
    n = card["IC_neutron_density"]
    n_max = card["IC_neutron_density_max"]
    C = card["IC_precursor_density"]
    C_max = card["IC_precursor_density_max"]
    N_cycle1 = 0.0
    N_cycle2 = 0.0
    if N_neutron > 0:
        N_cycle1 = math.ceil(cycle_stretch * math.ceil(n_max / n * N_neutron))
    if N_precursor > 0:
        N_cycle2 = math.ceil(cycle_stretch * math.ceil(C_max / C * N_precursor))
    N_cycle = max(N_cycle1, N_cycle2)
    card_setting["N_cycle"] = N_cycle
    card_setting["N_active"] = N_cycle


def uq(**kw):
    """
    Activate uncertainty quantification.

    Other Parameters
    ----------------
    material : dictionary, optional
        Material card of material with uncertain parameters.
    nuclide : dictionary, optional
        Nuclear card of nuclide with uncertain parameters.
    distribution : {"uniform"}
        Probability distribution of uncertain parameters.

    Returns
    -------
        None (in-place card alterations).
    """

    def append_card(delta_card, global_tag):
        delta_card.distribution = dist
        delta_card.flags = []
        for key in kw.keys():
            check_support(parameter.tag + " parameter", key, parameter_list, False)
            delta_card.flags.append(key)
            setattr(delta_card, key, kw[key])
        global_.input_deck.uq_deltas[global_tag].append(delta_card)

    global_.input_deck.technique["uq"] = True
    # Make sure N_batch > 1
    if global_.input_deck.setting["N_batch"] <= 1:
        print_error(
            "Must set N_batch>1 with global_.setting() prior to global_.uq() call."
        )

    # Check uq parameter
    parameter_ = check_support(
        "uq parameter",
        list(kw)[0],
        ["nuclide", "material", "surface", "source"],
        False,
    )
    parameter = kw[parameter_]
    del kw[parameter_]
    parameter.uq = True

    # Confirm supplied distribution
    check_requirement("uq", kw, ["distribution"])
    dist = check_support("distribution", kw["distribution"], ["uniform"], False)
    del kw["distribution"]

    # Only remaining keywords should be the parameter delta(s)

    if parameter.tag == "Material":
        parameter_list = [
            "capture",
            "scatter",
            "fission",
            "nu_s",
            "nu_p",
            "nu_d",
            "chi_p",
            "chi_d",
            "speed",
            "decay",
        ]
        global_tag = "materials"
        if parameter.N_nuclide == 1:
            nuc_card = NuclideCard(parameter.G, parameter.J)
            nuc_card.ID = parameter.nuclide_IDs[0]
            append_card(nuc_card, "nuclides")
        delta_card = MaterialCard(parameter.N_nuclide, parameter.G, parameter.J)
        for name in ["ID", "nuclide_IDs", "nuclide_densities"]:
            setattr(delta_card, name, getattr(parameter, name))
    elif parameter.tag == "Nuclide":
        parameter_list = [
            "capture",
            "scatter",
            "fission",
            "nu_s",
            "nu_p",
            "nu_d",
            "chi_p",
            "chi_d",
            "speed",
            "decay",
        ]
        global_tag = "nuclides"
        delta_card = make_card_nuclide(parameter.G, parameter.J)
        delta_card["ID"] = parameter.ID
    append_card(delta_card, global_tag)


# ==============================================================================
# Util
# ==============================================================================


def nuclide_registered(name):
    for card in global_.input_deck.nuclides:
        if name == card.name:
            return True
    return False


def get_nuclide(name):
    for card in global_.input_deck.nuclides:
        if name == card.name:
            return card


def print_card(card):
    if isinstance(card, SurfaceHandle):
        card = card.card
    for key in card:
        if key == "tag":
            print(card[key] + " card")
        else:
            print("  " + key + " : " + str(card[key]))


def check_support(label, value, supported, replace=True):
    if replace:
        value = value.replace("_", "-").replace(" ", "-").lower()
    supported_str = "{"
    for str_ in supported:
        supported_str += str_ + ", "
    supported_str = supported_str[:-2] + "}"
    if value not in supported:
        print_error("Unsupported " + label + ": " + value + "\n" + supported_str)
    return value


def check_requirement(label, kw, required):
    missing = "{"
    error = False
    for req in required:
        if req not in kw.keys():
            error = True
            missing += req + ", "
    missing = missing[:-2] + "}"
    if error:
        print_error("Parameters " + missing + " are required for" + label)


def make_particle_bank(size):
    struct = [
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("t", np.float64),
        ("ux", np.float64),
        ("uy", np.float64),
        ("uz", np.float64),
        ("g", np.uint64),
        ("E", np.float64),
        ("w", np.float64),
        ("rng_seed", np.uint64),
    ]
    iqmc_struct = [("w", np.float64, (1,))]
    struct += [("iqmc", iqmc_struct)]

    bank = np.zeros(size, dtype=np.dtype(struct))

    # Set default values
    for i in range(size):
        bank[i]["ux"] = 1.0
        bank[i]["w"] = 1.0
        bank[i]["rng_seed"] = 1

    return bank


def save_particle_bank(bank, name):
    with h5py.File(name + ".h5", "w") as f:
        f.create_dataset("particles", data=bank[:])
        f.create_dataset("particles_size", data=len(bank[:]))


# ==============================================================================
# Reset
# ==============================================================================


def reset():
    global_.input_deck.reset()


# ==============================================================================
# Misc
# ==============================================================================


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])
