"""RHF, MP2 and CCD (STO-3G) for the 15 molecules in molecules/, SCF started from the atomic-density guess. Takes about two minutes, mostly CCD for benzene."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from hf_to_cc import ao2mo, ccd, ccd_quad, guess_density, integrals, mp2, scf_diis, spin_orbitals  # noqa: E402

NAMES = ["H2", "LiH", "HF", "H2O", "NH3", "CH4", "CO", "N2", "CO2", "C2H2", "C2H4", "HCN", "H2CO", "CH3OH", "C6H6"]

print(f"{'':6}{'basis fns':>10}{'RHF total (Eh)':>18}{'MP2 corr (Eh)':>16}{'CCD corr (Eh)':>16}")
for name in NAMES:
    mol, S, h, eri, E_nuc, N = integrals(os.path.join(HERE, "..", "molecules", name.lower() + ".xyz"))
    E_hf, eps, C, n_scf = scf_diis(S, h, eri, E_nuc, N, d_x=guess_density(mol))
    mo, h_mo = ao2mo(C, eri, h)
    E_mp2 = mp2(N, mo, eps)
    hso, g = spin_orbitals(mo, h_mo)
    try:
        with np.errstate(all="ignore"):
            E_ccd, n_ccd = ccd(g, np.repeat(eps, 2), N, lambda t: ccd_quad(t, g, N))
        ccd_text = f"{E_ccd:16.10f}   ({n_ccd} iterations)"
    except RuntimeError:
        ccd_text = f"{'no convergence':>16}"
    print(f"{name:6}{C.shape[0]:10d}{E_hf:18.10f}{E_mp2:16.10f}{ccd_text}")
