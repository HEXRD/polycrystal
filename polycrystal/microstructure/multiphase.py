"""Generic multiphase microstructure"""
import numpy as np

from . import CgoMicrostructure



class Multiphase(CgoMicrostructure):
    """Multiphase microstructure instantiated

    Instantiate multiphase microstructure from list of single phase
    microstructures and a phase ID function of positions.


    Parameters
    ----------

    mslist: list
        list of single phase microstructure instances
    phaseID_fun: function
        function of positions array x returning phase ID
    """
    def __init__(self, mslist, phaseID_fun):

        self.mslist = mslist
        self.phaseID_fun = phaseID_fun
        self._orientation_list = np.vstack(
            [ms.orientation_list for ms in self.mslist]
        )

        g_lens = [ms.num_grains for ms in self.mslist]
        offs = [0]
        pmaps = []
        for p, ms in enumerate(self.mslist):
            offset = offs[-1]
            ng = ms.num_grains
            pmaps.append(np.tile(p, ng))
            offs.append(ng + offset)

        self._grain_offsets = offs
        self._phase_a = np.hstack(pmaps)

        _msg = "phase array has wrong shape"
        assert (len(self._phase_a) == self.num_grains), _msg

    def __contains__(self, x):
        return np.ones(len(x), dtype=bool)

    @property
    def num_grains(self):
        return len(self.orientation_list)

    @property
    def num_phases(self):
        return len(self.mslist)

    def grain(self, x):
        gid = np.zeros(len(x), dtype=np.int32)
        pid = self.phaseID_fun(x)
        for p in range(self.num_phases):
            ms = self.mslist[p]
            inds = (pid == p)
            gid[inds] = ms.grain(x[inds]) + self._grain_offsets[p]

        return gid

    def phase(self, g):
        return self._phase_a[g]

    @property
    def orientation_list(self):
        return self._orientation_list

    def grain_orientation(self, g):
        """Return orientation of grain g"""
        return self.orientation_list[g]
