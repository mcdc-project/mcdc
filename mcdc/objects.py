class ObjectBase:
    def __init__(self, label):
        self.label = label
        self.numbafied = False
        register_object(self)


class ObjectSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label)


class ObjectNonSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label)


class ObjectPolymorphic(ObjectNonSingleton):
    def __init__(self, label, type_):
        super().__init__(label)
        self.type = type_
        self.ID_numba = -1


class ObjectOverriding(ObjectPolymorphic):
    def __init__(self, label, type_):
        super().__init__(label, type_)


# The objects
settings = None  # Singleton
materials = []  # Overriding-polymorphic
nuclides = []  # Non-singleton
elements = []  # Non-singleton
reactions = []  # Polymorphic
data_containers = []  # Polymorphic


# Helper functions
def register_object(object_):
    from mcdc.data_container import DataContainer
    from mcdc.material import MaterialBase
    from mcdc.nuclide import Nuclide
    from mcdc.element import Element
    from mcdc.reaction import ReactionBase

    global materials, nuclides, elements, reactions, data_containers

    if isinstance(object_, MaterialBase):
        object_list = materials
    elif isinstance(object_, Nuclide):
        object_list = nuclides
    elif isinstance(object_, Element):
        object_list = elements
    elif isinstance(object_, ReactionBase):
        object_list = reactions
    elif isinstance(object_, DataContainer):
        object_list = data_containers

    if isinstance(object_, ObjectSingleton):
        object_list = object_
    if isinstance(object_, ObjectNonSingleton):
        object_.ID = len(object_list)
        object_list.append(object_)
