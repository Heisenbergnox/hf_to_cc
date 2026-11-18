"""RHF: solve, rhf and scf_diis against PySCF."""
import numpy as np
import pytest

from hf_to_cc import H2O, integrals, rhf, scf_diis, solve


H2O_x2 = "O 0 0 0.119262; H 0 1.526478 -0.954094; H 0 -1.526478 -0.954094"


def test_solve_generalized_eigenproblem():
    mol, S, h, eri, E_nuc, N = integrals(H2O)
    e, C = solve(h, S)
    assert np.all(np.diff(e) > 0)
    assert np.allclose(h @ C, S @ C @ np.diag(e), atol=1e-10)
    assert np.allclose(C.T @ S @ C, np.eye(len(e)), atol=1e-10)


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_rhf_matches_pyscf(basis, pyscf_rhf):
    mol, S, h, eri, E_nuc, N = integrals(H2O, basis)
    E_hf, eps, C, it = rhf(S, h, eri, E_nuc, N)
    mf = pyscf_rhf(H2O, basis)
    assert E_hf == pytest.approx(mf.e_tot, abs=1e-9)
    assert np.allclose(eps, mf.mo_energy, atol=1e-6)


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g", "cc-pvdz"])
def test_scf_diis_matches_pyscf(basis, pyscf_rhf):
    mol, S, h, eri, E_nuc, N = integrals(H2O, basis)
    E, eps, C, n_iter = scf_diis(S, h, eri, E_nuc, N)
    mf = pyscf_rhf(H2O, basis)
    assert E == pytest.approx(mf.e_tot, abs=1e-9)
    assert np.allclose(eps, mf.mo_energy, atol=1e-6)


def test_reference_energy():
    mol, S, h, eri, E_nuc, N = integrals(H2O)
    assert scf_diis(S, h, eri, E_nuc, N)[0] == pytest.approx(-74.9644048240, abs=1e-9)


def test_diis_converges_where_plain_scf_does_not(pyscf_rhf):
    mol, S, h, eri, E_nuc, N = integrals(H2O_x2)
    with pytest.raises(RuntimeError):
        rhf(S, h, eri, E_nuc, N)
    E, eps, C, n_iter = scf_diis(S, h, eri, E_nuc, N)
    assert E == pytest.approx(pyscf_rhf(H2O_x2).e_tot, abs=1e-8)
    assert E == pytest.approx(-74.4704439437, abs=1e-9)
