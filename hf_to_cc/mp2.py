import numpy as np


def mp2(N, mo, eps):
    nocc = N // 2
    eo = eps[:nocc]
    ev = eps[nocc:]
    ovov = mo[:nocc, nocc:, :nocc, nocc:]

    D = eo[:, None, None, None] - ev[None, :, None, None] + eo[None, None, :, None] - ev[None, None, None, :]

    E_mp2 = np.sum(ovov * (2 * ovov - ovov.transpose(0, 3, 2, 1)) / D)
    return E_mp2
