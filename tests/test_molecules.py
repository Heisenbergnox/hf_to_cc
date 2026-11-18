"""RHF (DIIS from the atomic-density guess) and closed-shell MP2 against PySCF for the 15 molecules in molecules/ (STO-3G)."""
import os

import pytest
from pyscf import mp

from hf_to_cc import ao2mo, guess_density, integrals, mp2, scf_diis

MOLECULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "molecules")
NAMES = ["h2", "lih", "hf", "h2o", "nh3", "ch4", "co", "n2", "co2", "c2h2", "c2h4", "hcn", "h2co", "ch3oh", "c6h6"]


@pytest.mark.parametrize("name", NAMES)
def test_rhf_and_mp2_match_pyscf(name, pyscf_rhf):
    atom = os.path.join(MOLECULES, name + ".xyz")
    mol, S, h, eri, E_nuc, N = integrals(atom)
    E_hf, eps, C, n_iter = scf_diis(S, h, eri, E_nuc, N, d_x=guess_density(mol))
    mf = pyscf_rhf(atom)
    assert E_hf == pytest.approx(mf.e_tot, abs=1e-9)
    mo, h_mo = ao2mo(C, eri, h)
    assert mp2(N, mo, eps) == pytest.approx(mp.MP2(mf).kernel()[0], abs=1e-8)
