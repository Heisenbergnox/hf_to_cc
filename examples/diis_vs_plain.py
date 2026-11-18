"""H2O / STO-3G: plain SCF and DIIS, both started from the atomic-density guess, at the equilibrium geometry and with both O-H bonds stretched to 1.87 Angstrom."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hf_to_cc import H2O, guess_density, integrals, rhf, scf_diis  # noqa: E402

H2O_x2 = "O 0 0 0.119262; H 0 1.526478 -0.954094; H 0 -1.526478 -0.954094"

for name, atom in (("equilibrium geometry", H2O), ("both O-H bonds stretched to 1.87 Angstrom", H2O_x2)):
    mol, S, h, eri, E_nuc, N = integrals(atom)
    d0 = guess_density(mol)
    print(f"H2O, {name}")
    try:
        E, eps, C, n_iter = rhf(S, h, eri, E_nuc, N, d_x=d0)
        print(f"  plain SCF  {E:16.10f} Eh   {n_iter:3d} iterations")
    except RuntimeError:
        print("  plain SCF  no convergence in 100 iterations")
    E, eps, C, n_iter = scf_diis(S, h, eri, E_nuc, N, d_x=d0)
    print(f"  DIIS       {E:16.10f} Eh   {n_iter:3d} iterations")
