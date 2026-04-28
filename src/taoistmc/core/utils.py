import numpy as np

class UtilityFunctions:
    """
    A class for utility functions used in the package.
    """

    def __init__(self):
        """
        Initialize the UtilityFunctions class.
        """
        pass

    def do_Zint(self, z, dz):
        """
        Compute integral of (1+z) and (1+z)**2.5.

        Parameters:
        - z: current redshift
        - dz: size of redshift slice

        Returns:
        - list: multiplicative factors
        """
        z1, z2 = z, z + dz
        o1 = (((z2 * z2) / 2.) + z2) - (((z1 * z1) / 2.) + z1)
        o2 = (0.285714 * (z2 ** 3.5)) - (0.285714 * (z1 ** 3.5))
        return [o1, o2]

    def do_Hint(self, NHI):
        """
        Compute HI distribution factors.

        Parameters:
        - NHI: HI column density bins

        Returns:
        - tuple: (outIGM, outCGM) distribution factors
        """
        bl, bh, bc = 1.635, 1.463, 1.381

        outIGM = np.zeros(len(NHI) - 1)
        outCGM = np.zeros(len(NHI) - 1)
        for i in range(len(outIGM) - 1):
            H1, H2 = 10. ** (NHI[i]), 10. ** (NHI[i + 1])
            if NHI[i] < 15.2:
                outIGM[i] = ((H2 ** (1. - bl)) - (H1 ** (1. - bl))) / (1. - bl)
            else:
                outIGM[i] = ((H2 ** (1. - bh)) - (H1 ** (1. - bh))) / (1. - bh)

            if NHI[i] < 13.0:
                outCGM[i] = ((H2 ** (1. - bl)) - (H1 ** (1. - bl))) / (1. - bl)
            else:
                outCGM[i] = ((H2 ** (1. - bc)) - (H1 ** (1. - bc))) / (1. - bc)

        return outIGM, outCGM