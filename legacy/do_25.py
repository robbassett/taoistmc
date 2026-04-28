import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate as integ
import warnings

import TAOIST_MC as tao

# THIS CODE CREATES 25 SIMULATED IGM_tau CURVES
# AT z=3.05 BASED ON APPENDIX B OF STEIDEL ET AL. 2018.
# AS WRITTEN, USES IGM+CGM MODEL, IGM ONLY CAN BE
# PRODUCED BY CHANGING "do_CGM" to False IN THE LINE:
# Nab = tao.get_fzs(zs,zem,dz,NHIs,do_CGM=True)
if __name__ == '__main__':
    # IGNORE OVERFLOW WARNINGS IN TAOIST-MC
    warnings.filterwarnings("ignore")
    
    # Define redshift
    zem =3.05
    # Define redshift bin size
    dz = 5e-5
    # Create redshift array
    zs = np.arange(0,zem+.01,dz)
    # Create log(NHI) array
    NHIs = np.arange(12.,21.1,.1)
    # Create wavelength array
    wav = np.arange(600.*(1.+zem),1500.*(1.+zem),5.)
    # Define number of samples
    nex = 25

    F  = plt.figure(figsize=(5,5),dpi=150)
    ax = F.add_subplot(111)
    taua= np.zeros((nex,len(wav)))
    for i in range(nex):

        Nab = tao.get_fzs(zs,zem,dz,NHIs,wav,do_CGM=True)
        tm_tau = tao.make_tau(zs,Nab,NHIs[:-1],wav)
        taua[i,:] = tm_tau

    taum = np.mean(np.exp((-1.)*taua),axis=0)

    ax.plot(wav/(1.+zem),taum,'g-',lw=1)
    [ax.axvline(x=_x,c='r',ls='--',lw=.3) for _x in [911.75,1216.]]
    ax.set_ylim(-.1,1.1)
    ax.set_xlim(799,1249)
    ax.set_xlabel(r'$\lambda_{rest}$',fontsize=18)
    ax.set_ylabel(r'$T_{IGM}$',fontsize=18)
    plt.show()
