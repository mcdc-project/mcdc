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


# The objects
settings = None  # Singleton
materials = []  # Polymorphic
nuclides = []  # Non-singleton
reactions = []  # Polymorphic
