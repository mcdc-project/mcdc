import numpy as np


def make_test_tabulated_energy_angle_data():
    Ei = [1.0, 3.0, 5.0]
    L_i = [0.0, 3.0, 6.0]

    E_i_k = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    p_i_k = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    c_i_k = [0.0, 0.5, 1.0, 0.0, 0.2, 1.0]

    L_i_k = [0.0, 3.0, 6.0]
    mu_i_j = [-1.0, 0.0, 1.0, -0.5, 0.5, 1.0]
    p_mu_i_j = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    c_mu_i_j = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0]

    data = np.array(
        Ei + L_i + E_i_k + p_i_k + c_i_k + L_i_k + mu_i_j + p_mu_i_j + c_mu_i_j,
        dtype=np.float64,
    )

    idx = 0
    energy_offset = idx
    idx += len(Ei)
    offset_offset = idx
    idx += len(L_i)
    energy_out_offset = idx
    idx += len(E_i_k)
    pdf_offset = idx
    idx += len(p_i_k)
    cdf_offset = idx
    idx += len(c_i_k)
    cosine_offset__offset = idx
    idx += len(L_i_k)
    cosine_offset = idx
    idx += len(mu_i_j)
    cosine_pdf_offset = idx
    idx += len(p_mu_i_j)
    cosine_cdf_offset = idx

    table = {
        "energy_offset": energy_offset,
        "energy_length": len(Ei),
        "offset_offset": offset_offset,
        "offset_length": len(L_i),
        "energy_out_offset": energy_out_offset,
        "energy_out_length": len(E_i_k),
        "pdf_offset": pdf_offset,
        "pdf_length": len(p_i_k),
        "cdf_offset": cdf_offset,
        "cdf_length": len(c_i_k),
        "cosine_offset__offset": cosine_offset__offset,
        "cosine_offset__length": len(L_i_k),
        "cosine_offset": cosine_offset,
        "cosine_length": len(mu_i_j),
        "cosine_pdf_offset": cosine_pdf_offset,
        "cosine_pdf_length": len(p_mu_i_j),
        "cosine_cdf_offset": cosine_cdf_offset,
        "cosine_cdf_length": len(c_mu_i_j),
    }
    return table, data
