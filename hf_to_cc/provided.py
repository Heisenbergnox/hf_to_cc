import numpy as np
import scipy.linalg
from pyscf import gto, scf

H2O = "O 0 0 0.119262; H 0 0.763239 -0.477047; H 0 -0.763239 -0.477047"


def integrals(atom, basis="sto-3g"):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    return mol, S, h, eri, mol.energy_nuc(), mol.nelectron


def ccd_quad(t, g, N):
    o, v = slice(0, N), slice(N, None)
    oovv = g[o, o, v, v]
    P_ij = lambda x: x - x.transpose(1, 0, 2, 3)
    P_ab = lambda x: x - x.transpose(0, 1, 3, 2)
    es = lambda *a: np.einsum(*a, optimize=True)
    Q = 0.25 * es("klcd,ijcd,klab->ijab", oovv, t, t)
    Q += P_ij(es("klcd,ikac,jlbd->ijab", oovv, t, t))
    Q -= 0.5 * P_ij(es("klcd,ikdc,ljab->ijab", oovv, t, t))
    Q -= 0.5 * P_ab(es("klcd,lkac,ijdb->ijab", oovv, t, t))
    return Q


def guess_density(mol):
    return scf.hf.init_guess_by_minao(mol)


def guess_orbitals(mol, S, h, eri):
    d = guess_density(mol)
    F = h + np.einsum("mnls, ls->mn", eri, d) - 0.5 * np.einsum("mlns, ls->mn", eri, d)
    e, C = scipy.linalg.eigh(F, S)
    return C
