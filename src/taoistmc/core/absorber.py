import numpy as np

from .utils import UtilityFunctions

class AbsorberSampler:
    """
    A class to sample absorbers based on HI column density distributions.
    """

    def __init__(self, NHIs, zs, CGM=False):
        """
        Initialize the AbsorberSampler.

        Parameters:
        - NHIs: HI column density bins (log spacing)
        - zs: redshifts for sampling
        - DH_IGM: IGM HI distribution factors
        - DH_CGM: CGM HI distribution factors
        - CGM: flag to denote CGM HI distribution (default: False)
        """
        self.integrator = UtilityFunctions()
        self.NHIs = NHIs
        self.zs = zs
        self.dz = zs[1]-zs[0]
        self.DH_IGM, self.DH_CGM = self.integrator.do_Hint(self.NHIs)
        self.CGM = CGM

    def _sample_absorber(self, dX):
        """
        Sample absorbers using Poisson distribution.

        Returns:
        - numpy.ndarray: Randomly sampled absorbers in each NHI bin
        """
        Ns = np.array((10. ** 9.305) * self.DH_IGM * dX[1])
        
        if self.CGM:
            t = np.where(self.NHIs[:-1] >= 13.0)[0]
            Ns[t] = (10. ** 6.716) * self.DH_CGM[t] * dX[0]
        else:
            t = np.where(self.NHIs[:-1] >= 15.2)[0]
            Ns[t] = (10. ** 7.542) * self.DH_IGM[t] * dX[0]
        
        return np.random.poisson(lam=Ns * 0.82, size=(1, len(Ns)))
    
    def sample_sightline(self):
        absorption_systems = np.zeros((len(self.zs), len(self.NHIs)-1))
        for i,z in enumerate(self.zs):
            dX = self.integrator.do_Zint(z, self.dz)
            absorption_systems[i] = self._sample_absorber(dX)
        return absorption_systems