import numpy as np

from mcdc.constant import INF


def cmf_from_pmf(pmf):
    cmf = np.zeros(len(pmf) + 1)

    # Build CMF incrementally
    total = 0.0
    for idx in range(len(pmf)):
        total += pmf[idx]
        cmf[idx + 1] = total

    # Normalize this segment so CDF ends at 1
    norm = cmf[-1]
    pmf /= norm
    cmf /= norm

    return pmf, cmf


def cdf_from_pdf(value, pdf):
    cdf = np.zeros_like(pdf)

    # Build CDF incrementally with trapezoidal integration
    for idx in range(len(pdf) - 1):
        cdf[idx + 1] = (
            cdf[idx] + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
        )

    # Normalize this segment so CDF ends at 1
    norm = cdf[-1]
    pdf /= norm
    cdf /= norm

    return pdf, cdf


def multi_cdf_from_pdf(offset, value, pdf):
    cdf = np.zeros_like(pdf)

    for i in range(len(offset)):
        start = offset[i]
        end = offset[i + 1] if i < len(offset) - 1 else len(pdf)

        # Build CDF incrementally with trapezoidal integration
        for idx in range(start, end - 1):
            cdf[idx + 1] = (
                cdf[idx]
                + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
            )

        # Normalize this segment so CDF ends at 1
        norm = cdf[end - 1]
        pdf[start:end] /= norm
        cdf[start:end] /= norm

    return pdf, cdf


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])


def move_object(object_, velocities, durations):
    object_.moving = True
    object_.N_move = len(durations) + 1
    object_.N_move_grid = len(durations) + 2

    if isinstance(velocities, np.ndarray):
        velocities = velocities.tolist()
        durations = durations.tolist()

    # Add the statics for the rest of the simulation
    move_velocities = velocities
    move_velocities.append([0.0, 0.0, 0.0])
    object_.move_velocities = np.array(move_velocities)
    #
    move_durations = durations
    move_durations.append(INF)
    object_.move_durations = np.array(move_durations)

    # Set time grid and translations
    object_.move_time_grid = np.zeros(object_.N_move_grid)
    object_.move_translations = np.zeros((object_.N_move_grid, 3))
    for n in range(object_.N_move):
        t_start = object_.move_time_grid[n]
        object_.move_time_grid[n + 1] = t_start + object_.move_durations[n]

        trans_start = object_.move_translations[n]
        object_.move_translations[n + 1] = (
            trans_start + object_.move_velocities[n] * object_.move_durations[n]
        )
