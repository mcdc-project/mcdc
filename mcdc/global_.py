import numpy as np

from mcdc.constant import INF, GYRATION_RADIUS_ALL, PI


# ======================================================================================
# Input Deck
# ======================================================================================


def make_card_mesh():
    return {
        "x": np.array([-INF, INF]),
        "y": np.array([-INF, INF]),
        "z": np.array([-INF, INF]),
        "t": np.array([-INF, INF]),
        "mu": np.array([-1.0, 1.0]),
        "azi": np.array([-PI, PI]),
        "g": np.array([-INF, INF]),
    }


class InputDeck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.surfaces = []
        self.regions = []
        self.cells = []
        self.universes = [None]  # Placeholder for the root universe
        self.lattices = []
        self.sources = []
        self.mesh_tallies = []
        self.surface_tallies = []
        self.cell_tallies = []
        self.cs_tallies = []

        self.technique = {
            "tag": "Technique",
            "weighted_emission": True,
            "implicit_capture": False,
            "population_control": False,
            "pct": "none",
            "pc_factor": 1.0,
            "weight_window": False,
            "ww": {
                "center": np.ones([1, 1, 1, 1]),
                "width": 2.5,
                "mesh": make_card_mesh(),
                "auto": 0,
                "epsilon": np.zeros(3),
                "save": False,
                "tally_idx": 0,
            },
            "domain_decomposition": False,
            "dd_idx": 0,
            "dd_local_rank": 0,
            "dd_mesh": make_card_mesh(),
            "dd_exchange_rate": 0,
            "dd_exchange_rate_padding": 0,
            "dd_work_ratio": np.array([1]),
            "weight_roulette": False,
            "wr_threshold": 0.0,
            "wr_survive": 1.0,
            "iQMC": False,
            "iqmc": {
                "sample_method": "halton",
                "mode": "fixed",
                "fixed_source_solver": "source iteration",
                "krylov_restart": 5,
                "krylov_vector_size": 1,
                "tol": 1e-6,
                "residual": 1.0,
                "iteration_count": 0,
                "iterations_max": 5,
                "fixed_source": np.ones([1, 1, 1, 1, 1]),
                "material_idx": np.ones([1, 1, 1, 1]),
                "source": np.ones([1, 1, 1, 1, 1]),
                "score": {
                    "flux": np.ones([1, 1, 1, 1]),
                    "source-x": np.zeros([1, 1, 1, 1]),
                    "source-y": np.zeros([1, 1, 1, 1]),
                    "source-z": np.zeros([1, 1, 1, 1]),
                    "fission-source": np.zeros([1, 1, 1, 1]),
                },
                "score_list": {
                    "flux": True,
                    "effective-scattering": True,
                    "effective-fission": True,
                    "source-x": False,
                    "source-y": False,
                    "source-z": False,
                    "fission-power": False,
                    "fission-source": False,
                },
                "mesh": {
                    "g": np.array([-INF, INF]),
                    "t": np.array([-INF, INF]),
                    "x": np.array([-INF, INF]),
                    "y": np.array([-INF, INF]),
                    "z": np.array([-INF, INF]),
                    "mu": np.array([-1.0, 1.0]),
                    "azi": np.array([-PI, PI]),
                },
            },
            "branchless_collision": False,
            "uq": False,
        }

        self.uq_deltas = {
            "tag": "Uq",
            "nuclides": [],
            "materials": [],
            "surfaces": [],
        }


input_deck = InputDeck()
