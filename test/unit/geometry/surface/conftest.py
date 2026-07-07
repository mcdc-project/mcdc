import pytest


@pytest.fixture(autouse=True)
def reset_simulation():
    from mcdc.object_.simulation import simulation

    simulation.__init__()
    yield
    simulation.__init__()


@pytest.fixture
def compile_surfaces():
    def _compile(static_surface_obj, moving_surface_obj):
        from mcdc.main import preparation

        structure_container, data = preparation()
        structure = structure_container[0]
        static_surface = structure["surfaces"][static_surface_obj.ID]
        moving_surface = structure["surfaces"][moving_surface_obj.ID]
        return static_surface, moving_surface, data

    return _compile
