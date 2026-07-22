import h5py
import numpy as np

def load_pstar_file(filepath):
    energies, sps = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                energies.append(float(parts[0]))
                sps.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(energies), np.array(sps)

file = h5py.File("../../../proton_generated_lib/p_in_H2O.h5", "w")
pstar_path = "../../../pstar_lib/H2O.txt"
E_s, S_s = load_pstar_file(pstar_path)
X0 = 36.08
sp = file.create_group("stopping_power")
sp.create_dataset("energy",               data=E_s).attrs["unit"] = "MeV"
sp.create_dataset("total_stopping_power", data=S_s).attrs["unit"] = "MeV cm2/g"
sp.create_dataset("radiation_length", data=X0).attrs["unit"] = "g/cm2"