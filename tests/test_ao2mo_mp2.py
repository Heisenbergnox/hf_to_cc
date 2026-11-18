"""AO -> MO transformation, spin-orbital integrals and closed-shell MP2 against PySCF."""
import numpy as np
import pytest
from pyscf import ao2mo as pyscf_ao2mo
from pyscf import mp

from hf_to_cc import H2O, ao2mo, integrals, mp2, scf_diis, spin_orbitals


@pytest.fixture(scope="module")
def h2o_mo():
    mol, S, h, eri, E_nuc, N = integrals(H2O)
    E_hf, eps, C, n_iter = scf_diis(S, h, eri, E_nuc, N)
    mo, h_mo = ao2mo(C, eri, h)
    return mol, h, C, eps, N, mo, h_mo


def test_ao2mo_matches_pyscf(h2o_mo):
    mol, h, C, eps, N, mo, h_mo = h2o_mo
    n = C.shape[1]
    assert np.allclose(mo, pyscf_ao2mo.restore(1, pyscf_ao2mo.full(mol, C), n), atol=1e-10)
    assert np.allclose(h_mo, C.T @ h @ C)


def test_spin_orbital_integrals(h2o_mo):
    mol, h, C, eps, N, mo, h_mo = h2o_mo
    hso, g = spin_orbitals(mo, h_mo)
    n = len(h_mo)
    assert g.shape == (2 * n,) * 4

    assert np.allclose(hso[0::2, 0::2], h_mo) and np.allclose(hso[1::2, 1::2], h_mo)
    assert np.allclose(hso[0::2, 1::2], 0)

    assert np.allclose(g, -g.transpose(1, 0, 2, 3)) and np.allclose(g, -g.transpose(0, 1, 3, 2))

    e = np.repeat(eps, 2)
    o, v = slice(0, N), slice(N, None)
    D = e[o, None, None, None] + e[None, o, None, None] - e[None, None, v, None] - e[None, None, None, v]
    assert 0.25 * np.sum(g[o, o, v, v] ** 2 / D) == pytest.approx(mp2(N, mo, eps), abs=1e-12)


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g", "cc-pvdz"])
def test_mp2_matches_pyscf(basis, pyscf_rhf):
    mol, S, h, eri, E_nuc, N = integrals(H2O, basis)
    E_hf, eps, C, n_iter = scf_diis(S, h, eri, E_nuc, N)
    mo, h_mo = ao2mo(C, eri, h)
    e_corr = mp.MP2(pyscf_rhf(H2O, basis)).kernel()[0]
    assert mp2(N, mo, eps) == pytest.approx(e_corr, abs=1e-9)
