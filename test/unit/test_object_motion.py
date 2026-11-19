"""Motion schedules can be reused across surfaces, sources, and cell products."""

from copy import deepcopy

import numpy as np
import pytest

import mcdc


@pytest.mark.parametrize(
    "velocity_type,duration_type",
    [(list, list), (tuple, tuple), (np.asarray, np.asarray), (list, np.asarray)],
)
def test_reuse_motion_schedule(velocity_type, duration_type):
    velocities = velocity_type([[0.0, 0.0, -2.0], [0.0, 0.0, 3.0]])
    durations = duration_type([0.5, 1.0])
    original_velocities = deepcopy(velocities)
    original_durations = deepcopy(durations)

    surface = mcdc.Surface.PlaneZ(z=2.0)
    source = mcdc.Source(position=[0.0, 0.0, 2.0])
    cell = mcdc.Cell(region=+surface)

    for object_, move in (
        (surface, surface.move),
        (source, source.move),
        (cell, cell.move_product),
    ):
        assert not object_.moving
        move(velocities, durations)

        # Reuse must not append extra segments to the caller's schedule.
        np.testing.assert_array_equal(velocities, original_velocities)
        np.testing.assert_array_equal(durations, original_durations)
        assert object_.moving
        assert object_.N_move == 3
        np.testing.assert_allclose(object_.move_time_grid[:3], [0.0, 0.5, 1.5])
        np.testing.assert_allclose(
            object_.move_translations,
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
        )

        # Reconfiguration replaces the schedule rather than extending it.
        move(velocities, durations)
        assert object_.N_move == 3
