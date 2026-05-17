import numpy as np
from numba import njit

class OpticalDepthCalculator:
    """
    A class to compute optical depth for LyC and Lyman series absorption
    using structured array inputs from SightlineSampler.
    """

    def __init__(self, wav):
        """
        Parameters:
        - wav: wavelength array (angstroms)
        """
        self.wav = wav
        
        # Physical Constants
        sig_T = 6.625e-25  # cm^2
        c = 2.998e10       # cm/s
        self.a1 = c * np.sqrt((3. * np.pi * sig_T) / 8.)

    @staticmethod
    @njit
    def voigt_approx(lam, lami, b, gamma):
        c = 2.998e18 # angst/s
        ldl = (b/c) * lami
        # a is the damping parameter
        a = ((lami*lami) * gamma) / (4. * np.pi * c * ldl)
        
        # Profile calculation window
        t_vp = np.where(np.abs(lam - lami) <= (1.812 * (b / 1.e13)))[0]
        if len(t_vp) == 0:
            return np.zeros(len(lam))
            
        x = (lam[t_vp] - lami) / ldl

        A1 = np.exp(-x*x)
        A2 = a * (2. / np.sqrt(np.pi))

        # Tepper-Garcia 2006 / 2007 approximation terms
        K1 = (1. / (2. * x*x))
        K2 = ((4. * x*x) + 3.) * ((x*x) + 1.) * A1
        K3 = (1. / (x*x)) * ((2. * x*x) + 3.) * (np.sinh(x*x))
        Kx = K1 * (K2 - K3)

        xo = np.zeros(len(lam))
        xo[t_vp] = A1 * (1. - (A2 * Kx))
        return xo

    @staticmethod
    @njit
    def doppler_dist(n_samples: int, b_sigma: float = 23.0):
        """Hui & Gnedin b-parameter distribution."""
        u = np.random.exponential(1.0, size=n_samples)
        b_values = b_sigma / (u**0.25)
        return np.clip(b_values, 5.0, 100.0)

    def tau_HI_LyC(self, z, NHI):
        """Vectorized LyC optical depth."""
        l_lc = 911.8 * (1. + z)
        x = self.wav / l_lc
        
        # Cross section scales as (lambda/lambda_limit)^3
        tau = NHI * (6.3e-18) * (x ** 3)
        
        # Only absorption at wavelengths shorter than the limit
        tau[self.wav > l_lc] = 0.
        return tau

    def tau_HI_LAF(self, z, NHI, b, LAF_table):
        """Lyman series optical depth."""
        tau = np.zeros(len(self.wav))
        lam_rest = self.wav / (1. + z)
        
        for i in range(len(LAF_table)):
            li = LAF_table[i, 0]     # Rest wavelength
            fi = LAF_table[i, 1]     # Oscillator strength
            gamma = LAF_table[i, 2]  # Damping constant

            A2 = (fi * li) / (np.sqrt(np.pi) * b)
            A3 = self.voigt_approx(lam_rest, li, b, gamma)

            tm_tau = self.a1 * A2 * A3
            # Replace NaNs (from x=0 in Voigt) with 0
            tm_tau[~np.isfinite(tm_tau)] = 0.
            tau += tm_tau

        # Cut off at Lyman limit
        tau[lam_rest <= 911.8] = 0.
        return tau * NHI
    
    def make_tau(self, sightline, LAF_table):
        """
        Main entry point using structured array from SightlineSampler.
        """
        total_tau = np.zeros(len(self.wav))
        
        # 1. Sample b-parameters for the whole sightline at once
        n_abs = len(sightline)
        if n_abs == 0:
            return total_tau
            
        dopps = self.doppler_dist(n_abs)
        
        # 2. Iterate through the structured array
        # This is much faster than dictionary lookups
        for i in range(n_abs):
            z = sightline['z'][i]
            NHI = 10.**sightline['logNHI'][i]
            b = dopps[i] * 1e13 # Consistent with your internal scaling
            
            # Add Lyman series
            total_tau += self.tau_HI_LAF(z, NHI, b, LAF_table)
            
            # Add Lyman Continuum
            total_tau += self.tau_HI_LyC(z, NHI)
            
        return total_tau