"""Spin-orbital CCD against reference values, and size consistency."""
import numpy as np
import pytest

from hf_to_cc import H2O, ao2mo, ccd, ccd_quad, integrals, rhf, spin_orbitals

E_CCD_STO3G = -0.0506409699
E_CCD_631G = -0.1357119496
H2 = "H 0 0 0; H 0 0 0.74"
H2_PAIR = "H 0 0 0; H 0 0 0.74; H 50 0 0; H 50 0 0.74"


def ccd_energy(atom, basis="sto-3g"):
    mol, S, h, eri, E_nuc, N = integrals(atom, basis)
    E_hf, eps, C, it = rhf(S, h, eri, E_nuc, N)
    mo, h_mo = ao2mo(C, eri, h)
    hso, g = spin_orbitals(mo, h_mo)
    e = np.repeat(eps, 2)
    return ccd(g, e, N, lambda t: ccd_quad(t, g, N))


def test_ccd_h2o_sto3g():
    E, n_iter = ccd_energy(H2O)
    assert E == pytest.approx(E_CCD_STO3G, abs=1e-9)


def test_ccd_h2o_631g():
    E, n_iter = ccd_energy(H2O, "6-31g")
    assert E == pytest.approx(E_CCD_631G, abs=1e-9)


def test_ccd_size_consistent():
    E_one = ccd_energy(H2)[0]
    E_two = ccd_energy(H2_PAIR)[0]
    assert E_two == pytest.approx(2 * E_one, abs=1e-9)
