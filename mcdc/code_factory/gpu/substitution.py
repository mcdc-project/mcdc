import inspect
import numba as nb


def always():
    return True


def identify_fn(fn):
    return f"'{fn.__name__}' at {inspect.getfile(fn)}:{inspect.getsourcelines(fn)[1]}"


def identity(x):
    return x


class SubstitutionRegistry:
    target_registry = {}
    candidate_registry = set()

    @classmethod
    def evaluate(cls, tag=None):

        remove_set = set()
        for item in cls.candidate_registry:
            if not item.condition():
                remove_set.add(item)

        for item in remove_set:
            cls.candidate_registry.remove(item)

        mapping = {}
        for item in cls.candidate_registry:
            if item.target_fn in mapping:
                target = item.target_fn
                existing_sub = mapping[item.target_fn].fn
                current_sub = item.fn
                raise RuntimeError(
                    f"\nMultiple subsitution candidates are active for the function {identify_fn(target)}\n"
                    + f"The first substitution candidate found was {identify_fn(existing_sub)}\n"
                    + f"The second substitution candidate found was {identify_fn(current_sub)}\n"
                )
            else:
                mapping[item.target_fn] = item

        for key, item in mapping.items():
            if not item.target_fn in cls.target_registry:
                raise RuntimeError(
                    f"\nA substitution is active for the function {identify_fn(item.target_fn)} - but it is not marked as a substitution target.\n"
                    + f"The active substitution is {identify_fn(item.fn)}\n"
                )
            if cls.target_registry[item.target_fn].tag != tag:
                continue
            item.evaluate()

        for item in cls.target_registry:
            if not item in mapping:
                cls.target_registry[item].evaluate()


class SubstitutionCandidate:

    def __init__(self, **kwargs):
        self.target_fn = kwargs["target_fn"]
        self.fn = kwargs["fn"]
        self.condition = kwargs["condition"]
        self.passthrough = kwargs["passthrough"]

    def evaluate(self):
        setattr(
            inspect.getmodule(self.target_fn),
            self.target_fn.__name__,
            self.passthrough(self.fn),
        )

    @staticmethod
    def register(**kwargs):
        SubstitutionRegistry.candidate_registry.add(SubstitutionCandidate(**kwargs))


class SubstitutionTarget:

    def __init__(self, **kwargs):
        self.fn = kwargs["fn"]
        self.passthrough = kwargs["passthrough"]
        self.tag = kwargs["tag"]

    def evaluate(self):
        setattr(
            inspect.getmodule(self.fn),
            self.fn.__name__,
            self.passthrough(self.fn),
        )

    @staticmethod
    def register(**kwargs):
        SubstitutionRegistry.target_registry[kwargs["fn"]] = SubstitutionTarget(
            **kwargs
        )


def candidate(target_fn, passthrough=nb.njit, condition=always, **kwargs):
    if isinstance(target_fn, nb.core.dispatcher.Dispatcher):
        target_fn = target_fn.py_func

    def deco(fn):
        SubstitutionCandidate.register(
            target_fn=target_fn,
            fn=fn,
            passthrough=passthrough,
            condition=condition,
            **kwargs,
        )
        return fn

    return deco


def target(passthrough=nb.njit, tag=None):
    def deco(fn):
        SubstitutionTarget.register(
            fn=fn,
            passthrough=passthrough,
            tag=tag,
        )
        return passthrough(fn)

    return deco
