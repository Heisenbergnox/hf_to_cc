import numpy as np


def ao2mo(C, eri, h):
    mo = np.einsum("ma, mnls->anls", C, eri)
    mo = np.einsum("nb, mnls->mbls", C, mo)
    mo = np.einsum("lc, mnls->mncs", C, mo)
    mo = np.einsum("sd, mnls->mnld", C, mo)

    h_mo = C.T @ h @ C
    return mo, h_mo


def spin_orbitals(mo, h_mo):
    nso = 2 * len(h_mo)
    hso = np.zeros((nso, nso))
    g = np.zeros((nso, nso, nso, nso))
    for p in range(nso):
        for q in range(nso):
            if p % 2 == q % 2:
                hso[p, q] = h_mo[p // 2, q // 2]

    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    v = 0.0
                    if p % 2 == r % 2 and q % 2 == s % 2:
                        v += mo[p // 2, r // 2, q // 2, s // 2]
                    if p % 2 == s % 2 and q % 2 == r % 2:
                        v -= mo[p // 2, s // 2, q // 2, r // 2]
                    g[p, q, r, s] = v
    return hso, g
