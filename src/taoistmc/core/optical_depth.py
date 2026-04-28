import numpy as np
from .sampler import Sampler

class OpticalDepthCalculator:
    """
    A class to compute optical depth for LyC and Lyman series absorption.
    """

    def __init__(self, wav, z):
        """
        Initialize the OpticalDepthCalculator.

        Parameters:
        - wav: wavelength array (angstroms)
        - z: redshift
        """
        self.wav = wav
        self.z = z
        self.doppler_sampler = None

    def tau_HI_LyC(self, NHI):
        """
        Compute optical depth for LyC photons.

        Parameters:
        - NHI: hydrogen column density

        Returns:
        - numpy.ndarray: optical depth values
        """
        l_lc = 911.8 * (1. + self.z)
        x = self.wav / l_lc
        tau = NHI * (6.3e-18) * (x ** 3)
        
        t = (self.wav / l_lc > 1.)
        t = np.where(t == True)
        tau[t[0]] = 0.
        
        return tau

    def tau_HI_LAF(self, NHI, LAF_table):
        """
        Compute optical depth for Lyman series absorption.

        Parameters:
        - NHI: hydrogen column density
        - LAF_table: Lyman series data table

        Returns:
        - numpy.ndarray: optical depth values
        """
        me, ce, c = 9.1094e-31, 1.6022e-19, 2.99792e18
        sig_T = 6.625e-25  # cm^2
        c = 2.998e10  # cm/s

        tau = np.zeros(len(self.wav))
        lam = self.wav / (1. + self.z)

        if self.doppler_sampler is None:
            bx = np.arange(1, 1000, .1)
            by = self.doppler_dist(bx)
            self.doppler_sampler = Sampler(bx, by)
        
        b = self.doppler_sampler.sample(1)[0] * 1.e13  # angstrom/s
        for i in range(len(LAF_table[:, 0])):
            fi = LAF_table[i, 1]
            li = LAF_table[i, 0]  # angstrom
            gamma = LAF_table[i, 2]

            A1 = c * np.sqrt((3. * np.pi * sig_T) / 8.)
            A2 = (fi * li) / (np.sqrt(np.pi) * b)
            A3 = self.voigt_approx(lam, li, b, gamma)

            tm_tau = 4.0 * A1 * A2 * A3
            bad = np.where(np.isfinite(tm_tau) == False)
            tm_tau[bad[0]] = 0.
            tau += tm_tau

        tau[np.where(lam <= 911.8)[0]] = 0.
        return tau * NHI