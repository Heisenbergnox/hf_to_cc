import numpy as np


def ccd(g, e, N, quad, tol=1e-12):
    o = slice(0, N)
    v = slice(N, None)
    eo = e[o]
    ev = e[v]
    oooo = g[o, o, o, o]
    vvvv = g[v, v, v, v]
    ovvo = g[o, v, v, o]
    oovv = g[o, o, v, v]
    D = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]

    t_o = oovv / D
    E_o = 0.25 * np.einsum("ijab, ijab->", oovv, t_o)

    for n_iter in range(1, 201):
        R = oovv + quad(t_o)
        R += 0.5 * np.einsum("abcd, ijcd->ijab", vvvv, t_o)
        R += 0.5 * np.einsum("klij, klab->ijab", oooo, t_o)
        X = np.einsum("kbcj, ikac->ijab", ovvo, t_o)
        R += X - X.transpose(1, 0, 2, 3) - X.transpose(0, 1, 3, 2) + X.transpose(1, 0, 3, 2)
        t_n = R / D
        E_n = 0.25 * np.einsum("ijab, ijab->", oovv, t_n)

        if np.abs(E_n - E_o) < tol:
            return E_n, n_iter
        t_o = t_n
        E_o = E_n
    raise RuntimeError("ccd did not converge")
