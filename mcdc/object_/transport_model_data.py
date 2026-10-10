from types import NoneType
from typing import Annotated

import numpy as np
from numpy import float64
from numpy.typing import ArrayLike, NDArray

from mcdc.constant import (
    NEUTRON_MULTIGROUP_ENERGY_MIDPOINT,
    NEUTRON_MULTIGROUP_ENERGY_MIDPOINT_LOG,
    NEUTRON_MULTIGROUP_ENERGY_UNIFORM,
    NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG,
)
from mcdc.object_.base import MCDCObject
from mcdc.print_ import print_1d_array, print_error

_NEUTRON_MULTIGROUP_ENERGY_REPRESENTATIONS = {
    "midpoint": NEUTRON_MULTIGROUP_ENERGY_MIDPOINT,
    "log_midpoint": NEUTRON_MULTIGROUP_ENERGY_MIDPOINT_LOG,
    "uniform": NEUTRON_MULTIGROUP_ENERGY_UNIFORM,
    "log_uniform": NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG,
}
_NEUTRON_MULTIGROUP_LOG_ENERGY_REPRESENTATIONS = {
    NEUTRON_MULTIGROUP_ENERGY_MIDPOINT_LOG,
    NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG,
}


class NeutronMultigroupData(MCDCObject):
    """Groupwise macroscopic interaction data for neutron multigroup transport.

    Neutron multigroup transport represents neutron energy with discrete groups
    and describes interactions using groupwise cross sections, production
    spectra, speeds, and delayed-precursor data. For a multigroup-only material,
    :meth:`mcdc.Material.multigroup` provides the convenient entry point.
    Construct ``NeutronMultigroupData`` directly when a material needs an
    explicit physical energy grid, including when attaching it alongside a
    :ref:`native composition <user_native_transport>`.

    Parameters
    ----------
    capture : array_like of float, optional
        Macroscopic capture cross section in ``cm^-1`` for each incoming
        energy group, with shape ``(G,)``.
    scatter : array_like of float, optional
        Macroscopic scattering-production matrix in ``cm^-1`` with shape
        ``(G, G)``, indexed as ``scatter[g_out, g_in]``. Column sums define the
        scattering cross section for each incoming group. Columns are
        normalized internally to form the scattering spectrum.
    fission : array_like of float, optional
        Macroscopic fission cross section in ``cm^-1`` for each incoming
        energy group, with shape ``(G,)``. Supplying fission requires at least
        one of ``nu_p`` or ``nu_d``.
    nu_s : array_like of float, optional
        Mean number of neutrons produced per scattering event in each
        incoming group, with shape ``(G,)``. Defaults to one.
    nu_p : array_like of float, optional
        Mean prompt-fission yield for each incoming group, with shape
        ``(G,)``.
    nu_d : array_like of float, optional
        Mean delayed-fission yield with shape ``(J, G)``, indexed as
        ``nu_d[j, g_in]``. Its first dimension determines the number ``J`` of
        delayed precursor groups.
    chi_p : array_like of float, optional
        Prompt-fission spectrum. Shape ``(G,)`` applies one outgoing spectrum
        to every incoming group. Shape ``(G, G)`` is indexed as
        ``chi_p[g_out, g_in]``. Required when ``nu_p`` is supplied and
        ``G > 1``.
    chi_d : array_like of float, optional
        Delayed-fission spectrum with shape ``(G, J)``, indexed as
        ``chi_d[g_out, j]``. Required when ``nu_d`` is supplied and ``G > 1``.
    speed : array_like of float, optional
        Neutron speed in cm/s for each energy group, with shape ``(G,)``.
        Defaults to one.
    decay_rate : array_like of float, optional
        Decay constant in ``s^-1`` for each delayed precursor group, with
        shape ``(J,)``. Defaults to infinity.
    energy_grid : array_like of float, optional
        Physical energy-group boundaries in eV with shape ``(G + 1,)``.
        User-supplied boundaries must be strictly increasing, and group ``g``
        spans ``energy_grid[g] <= E < energy_grid[g + 1]``. The default is a
        zero-valued placeholder used only when physical energy mapping is not
        required.
    energy_representation : str or int, optional
        Policy used to reconstruct continuous energy from a group when a
        physical energy grid participates in transport.
        ``"midpoint"`` uses the arithmetic midpoint, ``"log_midpoint"`` uses
        the geometric midpoint, ``"uniform"`` samples uniformly in energy,
        and ``"log_uniform"`` samples uniformly in log-energy. The default is
        ``"midpoint"``. Only the default ``"midpoint"`` placeholder is
        accepted when ``energy_grid`` is omitted. Logarithmic policies require
        positive boundaries.
        The corresponding ``NEUTRON_MULTIGROUP_ENERGY_*`` integer constants
        are also accepted.

    Notes
    -----
    ``G`` is inferred from ``capture``, ``scatter``, or ``fission``. An
    explicit energy grid is required whenever simulation transport needs to
    map continuous neutron energy to material-local groups.

    Examples
    --------
    Construct one-group data with capture, scattering, and prompt fission:

    >>> import mcdc
    >>> import numpy as np
    >>> neutron_multigroup = mcdc.NeutronMultigroupData(
    ...     capture=np.array([1.0 / 3.0]),
    ...     scatter=np.array([[1.0 / 3.0]]),
    ...     fission=np.array([1.0 / 3.0]),
    ...     nu_p=np.array([2.3]),
    ...     energy_grid=np.array([1.0e-5, 20.0e6]),
    ... )

    Construct data with two energy groups. Scattering is indexed by outgoing
    then incoming group:

    >>> two_group = mcdc.NeutronMultigroupData(
    ...     capture=np.array([0.1, 0.2]),
    ...     scatter=np.array([
    ...         [1.0, 2.0],
    ...         [3.0, 0.0],
    ...     ]),
    ...     nu_s=np.array([1.1, 1.2]),
    ...     energy_grid=np.array([1.0e-5, 1.0, 20.0e6]),
    ... )

    Construct two-group fission data with two delayed precursor groups. The
    delayed yield is indexed by precursor then incoming energy group, while
    the delayed spectrum is indexed by outgoing energy then precursor group:

    >>> multiple_precursors = mcdc.NeutronMultigroupData(
    ...     fission=np.array([0.2, 0.3]),
    ...     nu_d=np.array([
    ...         [0.1, 0.2],
    ...         [0.3, 0.4],
    ...     ]),
    ...     chi_d=np.array([
    ...         [1.0, 3.0],
    ...         [3.0, 1.0],
    ...     ]),
    ...     decay_rate=np.array([0.01, 0.02]),
    ... )
    """

    # MC/DC framework metadata
    label = "neutron_multigroup_data"

    G: int
    J: int

    energy_grid: Annotated[NDArray[float64], ("G+1",)]
    energy_representation: int

    speed: Annotated[NDArray[float64], ("G",)]
    decay_rate: Annotated[NDArray[float64], ("J",)]

    capture: Annotated[NDArray[float64], ("G",)]
    scatter: Annotated[NDArray[float64], ("G",)]
    fission: Annotated[NDArray[float64], ("G",)]
    total: Annotated[NDArray[float64], ("G",)]

    nu_s: Annotated[NDArray[float64], ("G",)]
    nu_p: Annotated[NDArray[float64], ("G",)]
    nu_d: Annotated[NDArray[float64], ("G", "J")]
    nu_d_total: Annotated[NDArray[float64], ("G",)]
    nu_f: Annotated[NDArray[float64], ("G",)]

    chi_s: Annotated[NDArray[float64], ("G", "G")]
    chi_p: Annotated[NDArray[float64], ("G", "G")]
    chi_d: Annotated[NDArray[float64], ("J", "G")]

    fissionable: bool

    def __init__(
        self,
        capture: ArrayLike | NoneType = None,
        scatter: ArrayLike | NoneType = None,
        fission: ArrayLike | NoneType = None,
        nu_s: ArrayLike | NoneType = None,
        nu_p: ArrayLike | NoneType = None,
        nu_d: ArrayLike | NoneType = None,
        chi_p: ArrayLike | NoneType = None,
        chi_d: ArrayLike | NoneType = None,
        speed: ArrayLike | NoneType = None,
        decay_rate: ArrayLike | NoneType = None,
        energy_grid: ArrayLike | NoneType = None,
        energy_representation: str | int = "midpoint",
    ) -> None:
        super().__init__()

        # Convert user inputs to the runtime array representation
        capture = _as_array("capture", capture)
        scatter = _as_array("scatter", scatter)
        fission = _as_array("fission", fission)
        nu_s = _as_array("nu_s", nu_s)
        nu_p = _as_array("nu_p", nu_p)
        nu_d = _as_array("nu_d", nu_d)
        chi_p = _as_array("chi_p", chi_p)
        chi_d = _as_array("chi_d", chi_d)
        speed = _as_array("speed", speed)
        decay_rate = _as_array("decay_rate", decay_rate)
        energy_grid = _as_array("energy_grid", energy_grid)

        # Infer dimensions from the defining cross sections and delayed yields
        self.G = _infer_group_count(capture, scatter, fission)
        self.J = _infer_delayed_group_count(nu_d, self.G)

        # Validate all user-facing array shapes before deriving stored data
        _validate_shape("capture", capture, (self.G,))
        _validate_shape("scatter", scatter, (self.G, self.G))
        _validate_shape("fission", fission, (self.G,))
        _validate_shape("nu_s", nu_s, (self.G,))
        _validate_shape("nu_p", nu_p, (self.G,))
        _validate_shape("nu_d", nu_d, (self.J, self.G))
        _validate_prompt_spectrum_shape(chi_p, self.G)
        _validate_shape("chi_d", chi_d, (self.G, self.J))
        _validate_shape("speed", speed, (self.G,))
        _validate_shape("decay_rate", decay_rate, (self.J,))
        _validate_shape("energy_grid", energy_grid, (self.G + 1,))

        # Reject values that cannot represent physical groupwise data
        for name, array in (
            ("capture", capture),
            ("scatter", scatter),
            ("fission", fission),
            ("nu_s", nu_s),
            ("nu_p", nu_p),
            ("nu_d", nu_d),
            ("chi_p", chi_p),
            ("chi_d", chi_d),
        ):
            _validate_nonnegative(name, array)
        _validate_positive("speed", speed)
        _validate_decay_rate(decay_rate)

        # Validate relationships between fission yields and spectra
        if fission is not None and nu_p is None and nu_d is None:
            print_error("NeutronMultigroupData fission data requires nu_p or nu_d.")
        if fission is None and (nu_p is not None or nu_d is not None):
            print_error(
                "NeutronMultigroupData fission yields require fission cross sections."
            )
        if chi_p is not None and nu_p is None:
            print_error("NeutronMultigroupData chi_p requires nu_p.")
        if chi_d is not None and nu_d is None:
            print_error("NeutronMultigroupData chi_d requires nu_d.")
        if decay_rate is not None and nu_d is None:
            print_error("NeutronMultigroupData decay_rate requires nu_d.")

        # Resolve the energy grid and continuous-energy reconstruction policy
        self.energy_representation = _resolve_energy_representation(
            energy_representation
        )
        if energy_grid is None:
            if self.energy_representation != NEUTRON_MULTIGROUP_ENERGY_MIDPOINT:
                print_error(
                    "NeutronMultigroupData requires an explicit energy_grid when "
                    "energy_representation is not 'midpoint'."
                )
            self.energy_grid = np.zeros(self.G + 1, dtype=float64)
        else:
            if not np.all(np.isfinite(energy_grid)):
                print_error("NeutronMultigroupData energy grid entries must be finite.")
            if np.any(np.diff(energy_grid) <= 0.0):
                print_error(
                    "NeutronMultigroupData energy grid must be strictly increasing."
                )
            if (
                self.energy_representation
                in _NEUTRON_MULTIGROUP_LOG_ENERGY_REPRESENTATIONS
                and energy_grid[0] <= 0.0
            ):
                print_error(
                    "NeutronMultigroupData logarithmic energy representation requires positive "
                    "energy boundaries."
                )
            self.energy_grid = energy_grid

        # Apply transport defaults for group speed and precursor decay
        self.speed = np.ones(self.G, dtype=float64) if speed is None else speed
        self.decay_rate = (
            np.full(self.J, np.inf, dtype=float64) if decay_rate is None else decay_rate
        )

        # Build cross sections and the normalized scattering spectrum
        self.capture = np.zeros(self.G, dtype=float64) if capture is None else capture
        self.chi_s = np.zeros((self.G, self.G), dtype=float64)
        if scatter is None:
            self.scatter = np.zeros(self.G, dtype=float64)
        else:
            self.scatter = np.sum(scatter, axis=0)
            self.chi_s = np.swapaxes(scatter, 0, 1).copy()
            _normalize_rows(self.chi_s, self.scatter > 0.0, "scatter")

        self.fissionable = fission is not None
        self.fission = np.zeros(self.G, dtype=float64) if fission is None else fission
        self.total = self.capture + self.scatter + self.fission

        # Build scattering and fission yields
        self.nu_s = np.ones(self.G, dtype=float64) if nu_s is None else nu_s
        self.nu_p = np.zeros(self.G, dtype=float64) if nu_p is None else nu_p
        if nu_d is None:
            self.nu_d = np.zeros((self.G, self.J), dtype=float64)
        else:
            self.nu_d = np.swapaxes(nu_d, 0, 1).copy()
        self.nu_d_total = np.sum(self.nu_d, axis=1)
        self.nu_f = self.nu_p + self.nu_d_total

        # Build normalized prompt- and delayed-fission spectra
        self.chi_p = np.zeros((self.G, self.G), dtype=float64)
        if nu_p is not None:
            if self.G == 1:
                self.chi_p[:] = 1.0
            elif chi_p is None:
                print_error("NeutronMultigroupData with nu_p and G > 1 requires chi_p.")
            else:
                if chi_p.ndim == 1:
                    chi_p = np.tile(chi_p[:, np.newaxis], (1, self.G))
                self.chi_p = np.swapaxes(chi_p, 0, 1).copy()
                _normalize_rows(self.chi_p, self.nu_p > 0.0, "chi_p")

        self.chi_d = np.zeros((self.J, self.G), dtype=float64)
        if nu_d is not None:
            if self.G == 1:
                self.chi_d[:] = 1.0
            elif chi_d is None:
                print_error("NeutronMultigroupData with nu_d and G > 1 requires chi_d.")
            else:
                self.chi_d = np.swapaxes(chi_d, 0, 1).copy()
                active_delayed_groups = np.any(nu_d > 0.0, axis=1)
                _normalize_rows(self.chi_d, active_delayed_groups, "chi_d")

    def __repr__(self) -> str:
        text = super().__repr__()
        text += f"  - G: {self.G}\n"
        text += f"  - J: {self.J}\n"
        text += f"  - Energy grid {print_1d_array(self.energy_grid)}\n"
        text += f"  - Sigma_c {print_1d_array(self.capture)}\n"
        text += f"  - Sigma_s {print_1d_array(self.scatter)}\n"
        text += f"  - Sigma_f {print_1d_array(self.fission)}\n"
        text += f"  - nu_s {print_1d_array(self.nu_s)}\n"
        text += f"  - nu_p {print_1d_array(self.nu_p)}\n"
        text += f"  - nu_d {print_1d_array(self.nu_d.flatten())}\n"
        text += f"  - chi_s {print_1d_array(self.chi_s.flatten())}\n"
        text += f"  - chi_p {print_1d_array(self.chi_p.flatten())}\n"
        text += f"  - chi_d {print_1d_array(self.chi_d.flatten())}\n"
        text += f"  - speed {print_1d_array(self.speed)}\n"
        text += f"  - decay rate {print_1d_array(self.decay_rate)}\n"
        return text


# ======================================================================================
# Photon constant cross sections
# ======================================================================================


class PhotonConstantXSData(MCDCObject):
    """Energy-independent macroscopic cross sections for photon transport.

    A photon analogue of :class:`NeutronMultigroupData` for problems whose cross
    sections are constant in energy, which is what the analytical transport
    benchmarks (Case, de Hoffmann-Placzek) are posed with. Scattering is treated
    as isotropic and elastic and absorption as analog capture, so the whole of
    the physics is the two numbers below plus their sum.

    Parameters
    ----------
    scatter : float, optional
        Macroscopic scattering cross section in ``cm^-1``.
    absorb : float, optional
        Macroscopic absorption cross section in ``cm^-1``.

    Examples
    --------
    A purely scattering medium::

        >>> import mcdc
        >>> material = mcdc.Material(
        ...     name="scatterer",
        ...     photon_constant_xs=mcdc.PhotonConstantXSData(scatter=1.0),
        ... )
    """

    # MC/DC framework metadata
    label = "photon_constant_xs_data"

    scatter: float
    absorb: float
    total: float

    def __init__(self, scatter: float = 0.0, absorb: float = 0.0) -> None:
        super().__init__()

        scatter = float(scatter)
        absorb = float(absorb)
        if scatter < 0.0:
            print_error("PhotonConstantXSData scatter must be non-negative.")
        if absorb < 0.0:
            print_error("PhotonConstantXSData absorb must be non-negative.")

        self.scatter = scatter
        self.absorb = absorb
        self.total = scatter + absorb

    def __repr__(self) -> str:
        text = super().__repr__()
        text += f"  - Sigma_s: {self.scatter}\n"
        text += f"  - Sigma_a: {self.absorb}\n"
        text += f"  - Sigma_t: {self.total}\n"
        return text


def _as_array(name, value):
    """Convert an optional user value to a float64 array."""
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float64)
    except (TypeError, ValueError):
        print_error(f"NeutronMultigroupData {name} must be numeric array-like data.")


def _infer_group_count(capture, scatter, fission) -> int:
    """Infer the energy-group count from the first supplied cross section."""
    defining = capture if capture is not None else scatter
    defining = fission if defining is None else defining
    if defining is None:
        return 0
    if defining.ndim == 0:
        print_error(
            "NeutronMultigroupData cross sections must be arrays with an energy-group axis."
        )
    if defining.shape[0] == 0:
        print_error(
            "NeutronMultigroupData cross sections must define at least one energy group."
        )
    return defining.shape[0]


def _infer_delayed_group_count(nu_d, G: int) -> int:
    """Infer the delayed-group count from delayed-fission yields."""
    if nu_d is None:
        return 0
    if nu_d.ndim != 2:
        print_error(
            f"NeutronMultigroupData nu_d must have shape (J, G); got {nu_d.shape}."
        )
    if nu_d.shape[1] != G:
        print_error(
            f"NeutronMultigroupData nu_d must have shape (J, G) with G = {G}; got {nu_d.shape}."
        )
    return nu_d.shape[0]


def _validate_shape(name, array, expected) -> None:
    """Require an optional array to have its declared NeutronMultigroupData shape."""
    if array is not None and array.shape != expected:
        print_error(
            f"NeutronMultigroupData {name} must have shape {expected}; got {array.shape}."
        )


def _validate_prompt_spectrum_shape(chi_p, G: int) -> None:
    """Accept either a shared vector or an incoming-group spectrum matrix."""
    if chi_p is None:
        return
    if chi_p.shape not in ((G,), (G, G)):
        print_error(
            f"NeutronMultigroupData chi_p must have shape ({G},) or ({G}, {G}); got {chi_p.shape}."
        )


def _validate_nonnegative(name, array) -> None:
    """Reject negative or non-finite physical data."""
    if array is None:
        return
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        print_error(
            f"NeutronMultigroupData {name} entries must be finite and nonnegative."
        )


def _validate_positive(name, array) -> None:
    """Reject nonpositive or non-finite physical data."""
    if array is None:
        return
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        print_error(
            f"NeutronMultigroupData {name} entries must be finite and positive."
        )


def _validate_decay_rate(array) -> None:
    """Allow the infinite default sentinel while rejecting invalid rates."""
    if array is None:
        return
    if np.any(np.isnan(array)) or np.any(array < 0.0):
        print_error("NeutronMultigroupData decay_rate entries must be nonnegative.")


def _resolve_energy_representation(policy) -> int:
    """Resolve a public reconstruction policy name or integer code."""
    if isinstance(policy, str):
        if policy not in _NEUTRON_MULTIGROUP_ENERGY_REPRESENTATIONS:
            expected = ", ".join(_NEUTRON_MULTIGROUP_ENERGY_REPRESENTATIONS)
            print_error(
                f"Unknown NeutronMultigroupData energy representation {policy!r}. "
                f"Expected one of: {expected}."
            )
        return _NEUTRON_MULTIGROUP_ENERGY_REPRESENTATIONS[policy]

    if (
        not isinstance(policy, (bool, np.bool_))
        and isinstance(policy, (int, np.integer))
        and policy in _NEUTRON_MULTIGROUP_ENERGY_REPRESENTATIONS.values()
    ):
        return int(policy)

    print_error(f"Unknown NeutronMultigroupData energy representation {policy!r}.")


def _normalize_rows(array, required, name) -> None:
    """Normalize spectra over outgoing groups and validate active rows."""
    for index in range(len(array)):
        norm = np.sum(array[index])
        if required[index] and norm <= 0.0:
            print_error(
                f"NeutronMultigroupData {name} spectrum {index} must have positive mass."
            )
        if norm > 0.0:
            array[index] /= norm
