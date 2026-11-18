"""UHF against PySCF: open-shell H2O+ and CO+, and stretched H2 with broken spin symmetry."""
import os

import numpy as np
import pytest
from pyscf import gto, scf

from hf_to_cc import H2O, guess_orbitals, integrals, rhf, uhf, uhf_diis


def pyscf_uhf(atom, charge=0, spin=0, dm0=None):
    mf = scf.UHF(gto.M(atom=atom, basis="sto-3g", charge=charge, spin=spin, verbose=0))
    mf.conv_tol = 1e-12
    mf.kernel(dm0)
    assert mf.converged
    return mf


def test_uhf_h2o_cation_matches_pyscf():
    mol, S, h, eri, E_nuc, N = integrals(H2O)
    E, Ca, Cb, n_iter = uhf(S, h, eri, E_nuc, 5, 4)
    assert E == pytest.approx(pyscf_uhf(H2O, charge=1, spin=1).e_tot, abs=1e-8)
    assert E == pytest.approx(-74.6592788252, abs=1e-9)


def test_uhf_spin_swap_and_closed_shell_limit():
    mol, S, h, eri, E_nuc, N = integrals(H2O)
    E54 = uhf(S, h, eri, E_nuc, 5, 4)[0]
    E45 = uhf(S, h, eri, E_nuc, 4, 5)[0]
    assert E45 == pytest.approx(E54, abs=1e-9)
    E55 = uhf(S, h, eri, E_nuc, 5, 5)[0]
    assert E55 == pytest.approx(rhf(S, h, eri, E_nuc, N)[0], abs=1e-9)


def test_uhf_stretched_h2_broken_symmetry():
    atom = "H 0 0 0; H 0 0 2.0"
    mol, S, h, eri, E_nuc, N = integrals(atom)
    E_rhf, eps, C, it = rhf(S, h, eri, E_nuc, N)

    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    Ca, Cb = C.copy(), C.copy()
    Ca[:, 0], Ca[:, 1] = c * C[:, 0] + s * C[:, 1], -s * C[:, 0] + c * C[:, 1]
    Cb[:, 0], Cb[:, 1] = c * C[:, 0] - s * C[:, 1], s * C[:, 0] + c * C[:, 1]
    dm0 = np.array([Ca[:, :1] @ Ca[:, :1].T, Cb[:, :1] @ Cb[:, :1].T])
    E, Ca, Cb, n_iter = uhf(S, h, eri, E_nuc, 1, 1, Ca, Cb)
    assert E < E_rhf - 0.1
    assert E == pytest.approx(pyscf_uhf(atom, dm0=dm0).e_tot, abs=1e-8)


def test_uhf_diis_h2o_cation_matches_pyscf():
    mol, S, h, eri, E_nuc, N = integrals(H2O)
    E, Ca, Cb, n_iter = uhf_diis(S, h, eri, E_nuc, 5, 4)
    assert E == pytest.approx(pyscf_uhf(H2O, charge=1, spin=1).e_tot, abs=1e-8)
    assert E == pytest.approx(uhf(S, h, eri, E_nuc, 5, 4)[0], abs=1e-9)


def test_uhf_diis_converges_where_plain_uhf_does_not():
    atom = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "molecules", "co.xyz")
    mol, S, h, eri, E_nuc, N = integrals(atom)
    with pytest.raises(RuntimeError):
        uhf(S, h, eri, E_nuc, N // 2, N // 2 - 1)
    E, Ca, Cb, n_iter = uhf_diis(S, h, eri, E_nuc, N // 2, N // 2 - 1)
    assert E == pytest.approx(pyscf_uhf(atom, charge=1, spin=1).e_tot, abs=1e-8)


def test_uhf_diis_starting_orbitals_select_the_solution():
    atom = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "molecules", "co2.xyz")
    mol, S, h, eri, E_nuc, N = integrals(atom)
    na, nb = N // 2, N // 2 - 1
    E_core = uhf_diis(S, h, eri, E_nuc, na, nb)[0]
    C0 = guess_orbitals(mol, S, h, eri)
    E_guess = uhf_diis(S, h, eri, E_nuc, na, nb, C0, C0.copy())[0]
    assert E_guess == pytest.approx(pyscf_uhf(atom, charge=1, spin=1).e_tot, abs=1e-8)
    assert E_core > E_guess + 0.3
