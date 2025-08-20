from mcdc.settings import Settings


class ObjectBase:
    def __init__(self, label):
        self.label = label


class ObjectSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label)


class ObjectNonSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label)
        self.numba_ID = -1


class ObjectPolymorphic(ObjectNonSingleton):
    def __init__(self, label, type_):
        super().__init__(label)
        self.type = type_


# The actual objects
settings = Settings()  # Singleton
materials = []  # Polymorphic
nuclides = []  # Non-singleton
reactions = []  # Polymorphic
