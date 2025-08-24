"""Groups of Slip Systems Related by Crystal Symmetry"""
import numpy as np


class SlipGroup:
    """Group of symmetrically equivalent slip systems

    This class takes a single generic slip system and uses the
    crystal symmetries to find all equivalent slip systems.

    Parameters
    ----------
    n: array (3)
       slip normal
    d: array (3)
       slip directiion
    csym:
       crystal symmetry group

    Attributes
    ----------
    schmid: array(n, 3, 3)
       array of symmetrically equivalent Schmid tensors
    """

    def __init__(self, n, d, csym):
        self.n = n
        self.d = d
        self.csym = csym
        self._schmid = self._generate_ss()

    def __len__(self):
        return len(self._schmid)

    @property
    def schmid(self):
        """Array of symmetrically equivalent Schmid tensors"""
        return self._schmid

    def _generate_ss(self):
        """Generate array of unique Schmid Tensors"""
        nsym = self.csym.nsymm
        rmats = self.csym.rmats
        n = rmats @ self.n
        d = rmats @ self.d
        ss = np.einsum("ij,ik->ijk", d, n)
        ssu = self._unique_ss(ss)
        ssu_nrm = np.sqrt((ssu * ssu).sum((1, 2)))
        schmid = ssu/ssu_nrm.reshape(nss := len(ssu), 1, 1)
        return schmid.reshape(nss, 3, 3)

    def _unique_ss(self, ss):
        """Find slip systems unique up to a sign"""
        EPS = 1e-8
        nss = len(ss)
        include = np.ones(nss, dtype=bool)
        for i in range(nss):
            if include[i]:
                for j in range(i+1, nss):
                    if include[j]:
                        # Check slip system for equality up to sign.
                        same = (
                            np.allclose(ss[i], ss[j]) or
                            np.allclose(ss[i], -ss[j])
                        )
                        if same:
                            include[j] = False

        return ss[include]
