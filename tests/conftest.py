"""Make the package importable without installing it, and cache the PySCF reference calculations."""
import os
import sys

import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def pyscf_rhf():
    cache = {}

    def run(atom, basis="sto-3g"):
        if (atom, basis) not in cache:
            mf = scf.RHF(gto.M(atom=atom, basis=basis, verbose=0))
            mf.conv_tol = 1e-12
            mf.kernel()
            assert mf.converged
            cache[atom, basis] = mf
        return cache[atom, basis]

    return run
