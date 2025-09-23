from openalea.rhizosoil.mimics_cn import MIMICS_CN


if __name__ == "__main__":
    mm = MIMICS_CN()

    for _ in range(1000):
        mm()
    print("final state", mm.state)