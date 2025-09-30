from openalea.rhizosoil.mimics_cn_grid import MIMICS_CN
import numpy as np


if __name__ == "__main__":
    n = 100
    clay = np.full((n, n, n), 30)
    mm = MIMICS_CN(clay_percentage=clay)
    print("initialized")

    for k in range(10):
        print(k)
        mm(soil_temperature=np.full_like(clay, 10),
           labile_OC=np.full_like(clay, 5e-1),
           labile_ON=np.full_like(clay, 1e-2),
           labile_IN=np.full_like(clay, 1e-6),
           net_N_uptake=np.full_like(clay, 0.),
           litter_inputs=np.full_like(clay, 50),
           litter_CN=np.full_like(clay, 15.),
           rhizodeposits_inputs=np.full_like(clay, 50),
           rhizodeposits_CN=np.full_like(clay, 15.)
           )
    print("final state", mm.state[0])