from mcdc.settings import Settings

class ObjectBase:
    def __init__(self, type_, derived_class=False):
        self.type = type_
        self.derived_class = derived_class
        self.numba_ID = -1


# The actual objects
materials = []
nuclides = []
reactions = []
settings = Settings
