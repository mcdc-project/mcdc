import numpy as np
import sympy

from mcdc.constant import (
    BOOL_AND,
    BOOL_OR,
    BOOL_NOT,
    INF,
    PARTICLE_NEUTRON,
    PI,
)

# Get the global variable container
import mcdc.global_ as global_


class InputCard:
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        text = "%s card\n" % self.tag

        for name in [
            a
            for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a))
            and a != "tag"
            and not a.startswith("_")
        ]:
            text += "  %s : %s\n" % (name, str(getattr(self, name)))
        return text


class NuclideCard(InputCard):
    def __init__(self, G=1, J=0, name=None):
        InputCard.__init__(self, "Nuclide")

        # Continuous energy?
        if name is not None:
            G = 0
            J = 0

        # Set card data
        self.ID = None
        self.name = name
        self.G = G
        self.J = J
        self.fissionable = False
        self.speed = np.ones(G)
        self.decay = np.ones(J) * INF
        self.capture = np.zeros(G)
        self.scatter = np.zeros(G)
        self.fission = np.zeros(G)
        self.total = np.zeros(G)
        self.nu_s = np.ones(G)
        self.nu_p = np.zeros(G)
        self.nu_d = np.zeros([G, J])
        self.nu_f = np.zeros(G)
        self.chi_s = np.zeros([G, G])
        self.chi_p = np.zeros([G, G])
        self.chi_d = np.zeros([J, G])
        self.uq = False
        self.flags = []
        self.distribution = ""
        self.name = ""


class MaterialCard(InputCard):
    def __init__(self, N_nuclide, G=1, J=0):
        InputCard.__init__(self, "Material")

        # Set card data
        self.ID = None
        self.N_nuclide = N_nuclide
        self.nuclide_IDs = np.zeros(N_nuclide, dtype=int)
        self.nuclide_densities = np.zeros(N_nuclide, dtype=float)
        self.G = G
        self.J = J
        self.speed = np.zeros(G)
        self.capture = np.zeros(G)
        self.scatter = np.zeros(G)
        self.fission = np.zeros(G)
        self.total = np.zeros(G)
        self.nu_s = np.ones(G)
        self.nu_p = np.zeros(G)
        self.nu_d = np.zeros([G, J])
        self.nu_f = np.zeros(G)
        self.chi_s = np.zeros([G, G])
        self.chi_p = np.zeros([G, G])
        self.uq = False
        self.flags = []
        self.distribution = ""


class RegionCard(InputCard):
    def __init__(self, type_):
        InputCard.__init__(self, "Region")

        # Set card data
        self.ID = None
        self.type = type_
        self.A = None
        self.B = None

    def __and__(self, other):
        region = RegionCard("intersection")
        region.A = self.ID
        region.B = other.ID
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __or__(self, other):
        region = RegionCard("union")
        region.A = self.ID
        region.B = other.ID
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __invert__(self):
        region = RegionCard("complement")
        region.A = self.ID
        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __str__(self):
        if self.type == "halfspace":
            if self.B > 0:
                return "+s%i" % self.A
            else:
                return "-s%i" % self.A
        elif self.type == "intersection":
            return "r%i & r%i" % (self.A, self.B)
        elif self.type == "union":
            return "r%i | r%i" % (self.A, self.B)
        elif self.type == "complement":
            return "~r%i" % (self.A)
        elif self.type == "all":
            return "all"


class SurfaceCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Surface")

        # Set card data
        self.ID = None
        self.type = ""
        self.boundary_type = "interface"
        self.linear = False
        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.E = 0.0
        self.F = 0.0
        self.G = 0.0
        self.H = 0.0
        self.I = 0.0
        self.J = 0.0
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0
        self.N_tally = 0
        self.tally_IDs = []
        self.moving = False
        self.N_move = 1
        self.move_velocities = [(0.0, 0.0, 0.0)]
        self.move_durations = [INF]

    def _create_halfspace(self, positive):
        region = RegionCard("halfspace")
        region.A = self.ID
        if positive:
            region.B = 1
        else:
            region.B = -1

        # Check if an identical halfspace region already existed
        for idx, existing_region in enumerate(global_.input_deck.regions):
            if (
                existing_region.type == "halfspace"
                and region.A == existing_region.A
                and region.B == existing_region.B
            ):
                return global_.input_deck.regions[idx]

        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __pos__(self):
        return self._create_halfspace(True)

    def __neg__(self):
        return self._create_halfspace(False)

    def move(self, velocities, durations):
        self.moving = True
        self.N_move = len(durations) + 1

        if isinstance(velocities, np.ndarray):
            velocities = velocities.tolist()
            durations = durations.tolist()

        self.move_velocities = velocities
        self.move_velocities.append([0.0, 0.0, 0.0])
        self.move_velocities = np.array(self.move_velocities)

        self.move_durations = durations
        self.move_durations.append(INF)
        self.move_durations = np.array(self.move_durations)


class CellCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Cell")

        # Set card data
        self.ID = None
        self.region_ID = None
        self.region = "all"
        self.fill_type = "material"
        self.fill_ID = None
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.surface_IDs = np.zeros(0, dtype=int)
        self.N_tally = 0
        self.tally_IDs = []
        self._region_RPN = []  # Reverse Polish Notation

    def set_region_RPN(self):
        # Make alias and reset
        rpn = self._region_RPN
        rpn.clear()

        # Build RPN based on the assigned region
        region = global_.input_deck.regions[self.region_ID]
        stack = [region]
        while len(stack) > 0:
            token = stack.pop()
            if isinstance(token, RegionCard):
                if token.type == "halfspace":
                    rpn.append(token.ID)
                elif token.type == "intersection":
                    region_A = global_.input_deck.regions[token.A]
                    region_B = global_.input_deck.regions[token.B]
                    stack += ["&", region_A, region_B]
                elif token.type == "union":
                    region_A = global_.input_deck.regions[token.A]
                    region_B = global_.input_deck.regions[token.B]
                    stack += ["|", region_A, region_B]
                elif token.type == "complement":
                    region = global_.input_deck.regions[token.A]
                    stack += ["~", region]
            else:
                if token == "&":
                    rpn.append(BOOL_AND)
                elif token == "|":
                    rpn.append(BOOL_OR)
                elif token == "~":
                    rpn.append(BOOL_NOT)
                else:
                    print_error("Something is wrong with cell RPN creation.")

    def set_region(self):
        stack = []

        for token in self._region_RPN:
            if token >= 0:
                stack.append(token)
            else:
                if token == BOOL_AND or token == BOOL_OR:
                    item_1 = stack.pop()
                    if isinstance(item_1, int):
                        item_1 = sympy.symbols(str(global_.input_deck.regions[item_1]))

                    item_2 = stack.pop()
                    if isinstance(item_2, int):
                        item_2 = sympy.symbols(str(global_.input_deck.regions[item_2]))

                    if token == BOOL_AND:
                        stack.append(item_1 & item_2)
                    else:
                        stack.append(item_1 | item_2)

                elif token == BOOL_NOT:
                    item = stack.pop()
                    if isinstance(item, int):
                        item = sympy.symbols(str(global_.input_deck.regions[item]))
                    stack.append(~item)

        self.region = sympy.logic.boolalg.simplify_logic(stack[0])

    def set_surface_IDs(self):
        surface_IDs = []

        for token in self._region_RPN:
            if token >= 0:
                ID = global_.input_deck.regions[token].A
                if not ID in surface_IDs:
                    surface_IDs.append(ID)

        self.surface_IDs = np.sort(np.array(surface_IDs))


class UniverseCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Universe")

        # Set card data
        self.ID = None
        self.cell_IDs = np.zeros(0, dtype=int)


class LatticeCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Lattice")

        # Set card data
        self.ID = None
        self.universe_IDs = np.array([[[[0]]]])
        self.x0 = -INF
        self.x0 = -INF
        self.dx = 2 * INF
        self.Nx = 1
        self.y0 = -INF
        self.dy = 2 * INF
        self.Ny = 1
        self.z0 = -INF
        self.dz = 2 * INF
        self.Nz = 1


class SourceCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Source")

        # Set card data
        self.ID = None
        self.box = False
        self.isotropic = True
        self.white = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.box_x = np.array([0.0, 0.0])
        self.box_y = np.array([0.0, 0.0])
        self.box_z = np.array([0.0, 0.0])
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 0.0
        self.white_x = 0.0
        self.white_y = 0.0
        self.white_z = 0.0
        self.group = np.array([1.0])
        self.energy = np.array([[1e6 - 1.0, 1e6 + 1.0], [1.0, 1.0]])
        self.time = np.array([0.0, 0.0])
        self.particle_type = PARTICLE_NEUTRON
        self.prob = 1.0


# ======================================================================================
# Tally cards
# ======================================================================================


class TallyCard(InputCard):
    def __init__(self, type_):
        InputCard.__init__(self, type_)

        # Set card data
        self.ID = None
        self.scores = []
        self.N_bin = 0

        # Filters
        self.t = np.array([-INF, INF])
        self.mu = np.array([-1.0, 1.0])
        self.azi = np.array([-PI, PI])
        self.g = np.array([-INF, INF])


class MeshTallyCard(TallyCard):
    def __init__(self):
        TallyCard.__init__(self, "Mesh tally")

        # Set card data
        self.x = np.array([-INF, INF])
        self.y = np.array([-INF, INF])
        self.z = np.array([-INF, INF])
        self.N_bin = 1


class SurfaceTallyCard(TallyCard):
    def __init__(self, surface_ID):
        TallyCard.__init__(self, "Surface tally")

        # Set card data
        self.surface_ID = surface_ID
        self.N_bin = 1


class CellTallyCard(TallyCard):
    def __init__(self, cell_ID):
        TallyCard.__init__(self, "Cell tally")

        # Set card data
        self.cell_ID = cell_ID
        self.N_bin = 1


class CSTallyCard(TallyCard):
    def __init__(self):
        TallyCard.__init__(self, "CS tally")

        # Set card data
        self.x = np.array([-INF, INF])
        self.y = np.array([-INF, INF])
        self.z = np.array([-INF, INF])
        self.N_bin = 1
        self.N_cs_bins = 1
        self.cs_bin_size = np.array([1.0, 1.0])
