import numpy as np
import scipy.linalg


def solve(F, S):
    e, C = scipy.linalg.eigh(F, S)
    return e, C


def rhf(S, h, eri, E_nuc, N, d_x=None):
    tol = 1e-10
    if d_x is None:
        F = h
        eps, C = solve(F, S)
        c_o = C[:, :N // 2]
        d_o = 2 * c_o @ c_o.T
    else:
        d_o = d_x

    for it in range(1, 101):
        J = np.einsum("mnls, ls->mn", eri, d_o)
        K = np.einsum("mlns, ls->mn", eri, d_o)
        F = h + J - 0.5 * K
        E_hf = 0.5 * np.einsum("mn, mn->", d_o, h+F) + E_nuc
        eps, C = solve(F, S)
        c_n = C[:, :N // 2]
        d_n = 2 * c_n @ c_n.T
        if np.abs(d_n - d_o).max() < tol:
            return E_hf, eps, C, it
        d_o = d_n
    raise RuntimeError("scf did not converge")


def scf_diis(S, h, eri, E_nuc, N, nkeep=8, d_x=None):
    tol = 1e-10
    if d_x is None:
        F = h
        eps, C = solve(F, S)
        c_o = C[:, :N // 2]
        d_o = 2 * c_o @ c_o.T
    else:
        d_o = d_x
    Gs = []
    Fs = []

    for n_iter in range(1, 101):
        J = np.einsum("mnls, ls->mn", eri, d_o)
        K = np.einsum("mlns, ls->mn", eri, d_o)
        F = h + J - (0.5 * K)
        E = 0.5 * np.einsum("mn, mn->", d_o, h+F) + E_nuc

        Fs.append(F)
        G = F @ d_o @ S - S @ d_o @ F
        Gs.append(G)
        Fs = Fs[-nkeep:]
        Gs = Gs[-nkeep:]

        m = len(Gs)
        A = np.zeros((m+1, m+1))
        for k in range(m):
            for l in range(m):
                A[k, l] = np.sum(Gs[k] * Gs[l])
        A[:m, m] = -1
        A[m, :m] = -1
        rhs = np.zeros(m+1)
        rhs[m] = -1
        c = np.linalg.solve(A, rhs)
        c = c[:m]

        F_mix = np.einsum("k, kmn->mn", c, Fs)

        eps, C = solve(F_mix, S)
        c_n = C[:, :N // 2]
        d_n = 2 * c_n @ c_n.T
        if np.abs(d_n - d_o).max() < tol:
            eps, C = solve(F, S)
            return E, eps, C, n_iter
        d_o = d_n
    raise RuntimeError("scf did not converge")


def uhf(S, h, eri, E_nuc, na, nb, Ca=None, Cb=None, maxit=5000):
    tol = 1e-10
    if Ca is None:
        eps, Ca = solve(h, S)
        Cb = Ca.copy()
    ca_o = Ca[:, :na]
    cb_o = Cb[:, :nb]
    da_o = ca_o @ ca_o.T
    db_o = cb_o @ cb_o.T

    for n_iter in range(1, maxit):
        J = np.einsum("mnls, ls->mn", eri, da_o + db_o)
        Ka = np.einsum("mlns, ls->mn", eri, da_o)
        Kb = np.einsum("mlns, ls->mn", eri, db_o)
        Fa = h + J - Ka
        Fb = h + J - Kb
        E = 0.5 * np.einsum("mn, mn->", da_o + db_o, h)
        E += 0.5 * np.einsum("mn, mn->", da_o, Fa)
        E += 0.5 * np.einsum("mn, mn->", db_o, Fb)
        E += E_nuc

        eps, Ca = solve(Fa, S)
        eps, Cb = solve(Fb, S)
        ca_n = Ca[:, :na]
        cb_n = Cb[:, :nb]
        da_n = ca_n @ ca_n.T
        db_n = cb_n @ cb_n.T

        if abs(db_n - db_o).max() < tol and abs(da_n - da_o).max() < tol:
            return E, Ca, Cb, n_iter
        da_o = da_n
        db_o = db_n
    raise RuntimeError("uhf did not converge")


def uhf_diis(S, h, eri, E_nuc, na, nb, Ca=None, Cb=None, nkeep=8, maxit=5000):
    tol = 1e-10
    if Ca is None:
        eps, Ca = solve(h, S)
        Cb = Ca.copy()
    ca_o = Ca[:, :na]
    cb_o = Cb[:, :nb]
    da_o = ca_o @ ca_o.T
    db_o = cb_o @ cb_o.T
    Gs = []
    Fas = []
    Fbs = []

    for n_iter in range(1, maxit):
        J = np.einsum("mnls, ls->mn", eri, da_o + db_o)
        Ka = np.einsum("mlns, ls->mn", eri, da_o)
        Kb = np.einsum("mlns, ls->mn", eri, db_o)
        Fa = h + J - Ka
        Fb = h + J - Kb
        E = 0.5 * np.einsum("mn, mn->", da_o + db_o, h)
        E += 0.5 * np.einsum("mn, mn->", da_o, Fa)
        E += 0.5 * np.einsum("mn, mn->", db_o, Fb)
        E += E_nuc

        Fas.append(Fa)
        Fbs.append(Fb)
        G1 = Fa @ da_o @ S - S @ da_o @ Fa
        G2 = Fb @ db_o @ S - S @ db_o @ Fb
        G = np.concatenate([G1.ravel(), G2.ravel()])
        Gs.append(G)
        Fas = Fas[-nkeep:]
        Fbs = Fbs[-nkeep:]
        Gs = Gs[-nkeep:]

        m = len(Gs)
        A = np.zeros((m+1, m+1))
        for k in range(m):
            for l in range(m):
                A[k, l] = np.sum(Gs[k] * Gs[l])
        A[:m, m] = -1
        A[m, :m] = -1
        rhs = np.zeros(m+1)
        rhs[m] = -1
        c = np.linalg.solve(A, rhs)
        c = c[:m]

        Fa_mix = np.einsum("k, kmn->mn", c, Fas)
        Fb_mix = np.einsum("k, kmn->mn", c, Fbs)

        eps, Ca = solve(Fa_mix, S)
        eps, Cb = solve(Fb_mix, S)
        ca_n = Ca[:, :na]
        cb_n = Cb[:, :nb]
        da_n = ca_n @ ca_n.T
        db_n = cb_n @ cb_n.T

        if abs(db_n - db_o).max() < tol and abs(da_n - da_o).max() < tol:
            eps, Ca = solve(Fa, S)
            eps, Cb = solve(Fb, S)
            return E, Ca, Cb, n_iter
        da_o = da_n
        db_o = db_n
    raise RuntimeError("uhf did not converge")
