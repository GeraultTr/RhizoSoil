from openalea.rhizosoil.mimics_cn_grid import MIMICS_CN
import numpy as np


if __name__ == "__main__":
    clay = np.full((3, 3, 3), 30)
    mm = MIMICS_CN(clay_percentage=clay)

    # for _ in range(1000):
    #     mm()
    print("final state", mm.state)