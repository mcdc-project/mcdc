import numpy as np


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


# ======================================================================================
# Data
# ======================================================================================
