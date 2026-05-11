import numpy as np
from numba import njit

from .sampler import Sampler

class OpticalDepthCalculator:
    """
    A class to compute optical depth for LyC and Lyman series absorption.
    """

    def __init__(self, wav):
        """
        Initialize the OpticalDepthCalculator.

        Parameters:
        - wav: wavelength array (angstroms)
        - z: redshift
        """
        self.wav = wav
        self.doppler_sampler = None

        sig_T = 6.625e-25  # cm^2
        c = 2.998e10  # cm/s
        self.a1 = c * np.sqrt((3. * np.pi * sig_T) / 8.)

    @staticmethod
    @njit
    def voigt_approx(lam,lami,b,gamma):
        c = 2.998e18 # angst/s
        ldl = (b/c)*lami
        a = ((lami*lami)*gamma)/(4.*np.pi*c*ldl)
        t_vp = np.where(np.abs(lam-lami) <= (1.812*(b/1.e13)))
        t_vp = t_vp[0]
        
        x = (lam[t_vp]-lami)/ldl

        A1 = np.exp((-1.)*x*x)
        A2= a*(2./np.sqrt(np.pi))

        K1 = (1./(2.*x*x))
        K2 = ((4.*x*x)+3.)*((x*x)+1.)*A1
        K3 = (1./(x*x))*((2.*x*x)+3.)*(np.sinh(x*x))

        Kx = K1*(K2-K3)

        xo = np.zeros(len(lam))
        xo[t_vp] = A1*(1.-(A2*Kx))

        return xo

    @staticmethod
    @njit
    def doppler_dist(n_samples:int=1, b_sigma:float=23.0):
        """
        Inverse Transform Sampling for the Hui & Gnedin b-parameter distribution.
        Faster than a look-up table and perfectly continuous.
        """
        # 1. Sample from a standard exponential distribution
        # If P(b) ~ b^-5 * exp(-(bs/b)^4), then (bs/b)^4 is Exponentially distributed.
        u = np.random.exponential(1.0, size=n_samples)
        
        # 2. Transform back to b
        b_values = b_sigma / (u**0.25)
        
        return np.clip(b_values, 5.0, 100.0)

    def tau_HI_LyC(self, z, NHI):
        """
        Compute optical depth for LyC photons.

        Parameters:
        - NHI: hydrogen column density

        Returns:
        - numpy.ndarray: optical depth values
        """
        l_lc = 911.8 * (1. + z)
        x = self.wav / l_lc
        tau = NHI * (6.3e-18) * (x ** 3)
        
        t = (self.wav / l_lc > 1.)
        t = np.where(t == True)
        tau[t[0]] = 0.
        
        return tau

    def tau_HI_LAF(self, z, NHI, b, LAF_table):
        """
        Compute optical depth for Lyman series absorption.

        Parameters:
        - NHI: hydrogen column density
        - LAF_table: Lyman series data table

        Returns:
        - numpy.ndarray: optical depth values
        """
        tau = np.zeros(len(self.wav))
        lam = self.wav / (1. + z)
        for i in range(len(LAF_table[:, 0])):
            fi = LAF_table[i, 1]
            li = LAF_table[i, 0]  # angstrom
            gamma = LAF_table[i, 2]
 
            A2 = (fi * li) / (np.sqrt(np.pi) * b)
            A3 = self.voigt_approx(lam, li, b, gamma)

            tm_tau = self.a1 * A2 * A3
            bad = np.where(np.isfinite(tm_tau) == False)
            tm_tau[bad[0]] = 0.
            tau += tm_tau

        tau[np.where(lam <= 911.8)[0]] = 0.
        return tau * NHI
    
    def make_tau(self, sl, LAF_table):
        tau = np.zeros(len(self.wav))
        dopps = self.doppler_dist(len(sl))
        for b,item in zip(dopps,sl):
            HI = 10.**(item.get("logNHI"))
            z = item.get('z')
            tau += self.tau_HI_LAF(z, HI, b*1e13, LAF_table)
            tau += self.tau_HI_LyC(z,HI)
        return tau