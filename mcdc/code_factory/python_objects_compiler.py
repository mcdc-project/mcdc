from mcdc.object_.base import MCDCObject, MCDCPolymorphic
from mcdc.object_.cell import Region, Cell
from mcdc.object_.data import DataBase, DataNone
from mcdc.object_.distribution import DistributionBase, DistributionNone
from mcdc.object_.electron_reaction import ElectronReactionBase
from mcdc.object_.element import Element
from mcdc.object_.material import Material
from mcdc.object_.transport_model_data import (
    NeutronMultigroupData,
    PhotonConstantXSData,
)
from mcdc.object_.mesh import MeshBase
from mcdc.object_.neutron_reaction import NeutronReactionBase
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.photon_reaction import PhotonReactionBase
from mcdc.object_.proton_reaction import ProtonReactionBase
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.object_.universe import Universe, Lattice
from mcdc.object_.simulation import Simulation
from mcdc.object_.source import Source
from mcdc.object_.surface import Surface
from mcdc.object_.tally import Tally
from mcdc.print_ import print_error

NONE_OBJECT_CLASSES = (DataNone, DistributionNone)  # Has customized compilation


def compile_simulation(simulation: Simulation):
    """Discover, register, and finalize one simulation's Python model.

    Recursive object hooks perform object-local compilation. Once discovery is
    complete, the owning :class:`~mcdc.object_.simulation.Simulation` resolves
    model-wide state before runtime packing begins.
    """
    # Require geometry rooted in at least one cell
    if len(simulation.root_universe.cells) == 0:
        print_error("Simulation model has not been set (root universe is empty).")

    # Preserve explicitly configured roots before resetting their registered
    # object lists. Geometry members may reference these objects and compile
    # them while the model graph is traversed.
    sources = simulation.sources
    tallies = simulation.tallies

    # Reset model
    simulation._reset_model()
    simulation.sources = []
    simulation.tallies = []

    # Reserved objects
    none_data = DataNone()
    none_distribution = DistributionNone()
    none_neutron_multigroup = NeutronMultigroupData()
    none_photon_constant_xs = PhotonConstantXSData()

    # Compile reserved objects
    none_data._compile_into_simulation(simulation)
    none_distribution._compile_into_simulation(simulation)
    none_neutron_multigroup._compile_into_simulation(simulation)
    none_photon_constant_xs._compile_into_simulation(simulation)

    # Compile model
    root_universe = simulation.root_universe
    root_universe._compile_into_simulation(simulation)

    # Compile source
    for source in sources:
        source._compile_into_simulation(simulation)

    # Compile tally
    for tally in tallies:
        tally._compile_into_simulation(simulation)

    # Compile remaining object members, including those owned by embedded
    # simulation configuration objects such as transport techniques.
    simulation._compile_members_into_simulation(simulation)

    # Resolve state that depends on the complete discovered model
    simulation._finalize_compilation()


def register_object(object_: MCDCObject, simulation: Simulation) -> bool:
    # Skip if already compiled
    if object_.compile_ID == simulation.compile_ID:
        return False

    # Get the object list
    if isinstance(object_, Cell):
        object_list = simulation.cells
    elif isinstance(object_, DataBase):
        object_list = simulation.data
    elif isinstance(object_, DistributionBase):
        object_list = simulation.distributions
    elif isinstance(object_, Lattice):
        object_list = simulation.lattices
    elif isinstance(object_, Material):
        object_list = simulation.materials
    elif isinstance(object_, NeutronMultigroupData):
        object_list = simulation.neutron_multigroup_data
    elif isinstance(object_, PhotonConstantXSData):
        object_list = simulation.photon_constant_xs_data
    elif isinstance(object_, MeshBase):
        object_list = simulation.meshes
    elif isinstance(object_, Element):
        object_list = simulation.elements
    elif isinstance(object_, ElectronReactionBase):
        object_list = simulation.electron_reactions
    elif isinstance(object_, Nuclide):
        object_list = simulation.nuclides
    elif isinstance(object_, NeutronReactionBase):
        object_list = simulation.neutron_reactions
    elif isinstance(object_, ProtonReactionBase):
        object_list = simulation.proton_reactions
    elif isinstance(object_, PhotonReactionBase):
        object_list = simulation.photon_reactions
    elif isinstance(object_, SecondaryProduct):
        object_list = simulation.secondary_products
    elif isinstance(object_, Region):
        object_list = simulation.regions
    elif isinstance(object_, Source):
        object_list = simulation.sources
    elif isinstance(object_, Surface):
        object_list = simulation.surfaces
    elif isinstance(object_, Tally):
        object_list = simulation.tallies
    elif isinstance(object_, Universe):
        object_list = simulation.universes
    else:
        object_list = []
        print_error(f"Unidentified object list for object {object_}")

    # Assign IDs
    object_.ID = len(object_list)
    if isinstance(object_, MCDCPolymorphic):
        counter_key = (id(object_list), object_.sub_type)
        object_.sub_ID = simulation._next_sub_ID.get(counter_key, 0)
        simulation._next_sub_ID[counter_key] = object_.sub_ID + 1
    object_.compile_ID = simulation.compile_ID

    # Assign name if needed (TODO: Resolve IDE error message)
    if hasattr(object_, "name") and object_.name.startswith("(Unnamed"):
        if isinstance(object_, MCDCPolymorphic):
            ID = object_.sub_ID
        else:
            ID = object_.ID
        object_.name = object_.label + f"_{ID}"

    # Register to simulation (TODO: Resolve IDE error message)
    object_list.append(object_)

    return True
