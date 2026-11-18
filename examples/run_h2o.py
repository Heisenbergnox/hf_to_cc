"""H2O: RHF, MP2 and CCD energies with hf_to_cc (STO-3G and 6-31G)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hf_to_cc import H2O, ao2mo, ccd, ccd_quad, integrals, mp2, scf_diis, spin_orbitals  # noqa: E402

for basis in ("sto-3g", "6-31g"):
    mol, S, h, eri, E_nuc, N = integrals(H2O, basis)
    E_hf, eps, C, n_scf = scf_diis(S, h, eri, E_nuc, N)
    mo, h_mo = ao2mo(C, eri, h)
    E_mp2 = mp2(N, mo, eps)
    hso, g = spin_orbitals(mo, h_mo)
    E_ccd, n_ccd = ccd(g, np.repeat(eps, 2), N, lambda t: ccd_quad(t, g, N))

    print(f"H2O / {basis.upper()} ({C.shape[0]} basis functions)")
    print(f"  RHF  total energy  {E_hf:16.10f} Eh   ({n_scf} SCF iterations with DIIS)")
    print(f"  MP2  correlation   {E_mp2:16.10f} Eh   total {E_hf + E_mp2:.10f} Eh")
    print(f"  CCD  correlation   {E_ccd:16.10f} Eh   total {E_hf + E_ccd:.10f} Eh   ({n_ccd} iterations)")
