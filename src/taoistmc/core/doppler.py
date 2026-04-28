import numpy as np

class DopplerBroadening:
    """
    A class to handle Doppler broadening calculations.
    """

    def __init__(self):
        """
        Initialize the DopplerBroadening class.
        """
        pass

    def doppler_dist(self, b):
        """
        Doppler parameter distribution function.

        Parameters:
        - b: doppler broadening

        Returns:
        - numpy.ndarray: distribution values
        """
        bs = 23.
        A1 = (4. * bs ** 4) / (b ** 5)
        A2 = np.exp((-1.) * A1 * b / 4.)
        return A1 * A2

    def voigt_approx(self, lam, lami, b, gamma):
        """
        Voigt profile approximation.

        Parameters:
        - lam: wavelength array
        - lami: central wavelength of current Lyman line
        - b: doppler broadening in angstrom/s
        - gamma: damping parameter of current Lyman line

        Returns:
        - numpy.ndarray: Voigt profile values
        """
        c = 2.998e18  # angstrom/s
        ldl = (b / c) * lami
        a = ((lami ** 2) * gamma) / (4. * np.pi * c * ldl)
        t_vp = np.where(np.abs(lam - lami) <= (1.812 * (b / 1.e13)))
        t_vp = t_vp[0]

        x = (lam[t_vp] - lami) / ldl

        A1 = np.exp((-1.) * x ** 2)
        A2 = a * (2. / np.sqrt(np.pi))

        K1 = (1. / (2. * x ** 2))
        K2 = ((4. * x ** 2) + 3.) * (x ** 2 + 1.) * A1
        K3 = (1. / (x ** 2)) * ((2. * x ** 2) + 3.) * (np.sinh(x ** 2))

        Kx = K1 * (K2 - K3)

        xo = np.zeros(len(lam))
        xo[t_vp] = A1 * (1. - (A2 * Kx))

        return xo