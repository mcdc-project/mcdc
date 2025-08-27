class ObjectBase:
    def __init__(self, label):
        self.label = label
        self.numbafied = False


class ObjectSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label)


class ObjectNonSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label)
        self.ID = -1


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
reactions = []  # Polymorphic
data_containers = []  # Polymorphic


# Helper functions
def register_polymorphic_object(object_):
    from mcdc.data_container import DataContainer
    from mcdc.reaction import ReactionBase

    global reactions, data_containers
    if isinstance(object_, DataContainer):
        object_list = data_containers
    elif isinstance(object_, ReactionBase):
        object_list = reactions

    object_.ID = len(object_list)
    object_.ID_numba = sum([x.type == object_.type for x in object_list])
    object_list.append(object_)
