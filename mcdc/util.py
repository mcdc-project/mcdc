import numpy as np


def cdf_from_pdf(offset, value, pdf):
    cdf = np.zeros_like(pdf)
    for i in range(len(offset)):
        start = offset[i]
        end = offset[i + 1] if i < len(offset) - 1 else len(pdf)
        for idx in range(start, end - 2):
            cdf[idx + 1] = (
                cdf[idx]
                + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
            )
        # Ensure it ends at one
        cdf[end - 1] = 1.0

    return cdf
