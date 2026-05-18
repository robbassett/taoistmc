import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate as integ
import cdf_sampler as cds
import sys

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Retrieves the number of absorption systems in
# each bin of HI column density in a given redshift
# slice. This is based on the HI column density
# distributions in Steidel et al. 2018, Appendix B,
# Figure B1. Values calculated are passed into the
# Poisson sampler of the numpy.random package to give
# the randomly sampled absorbers in the current
# redshift bin. This function is called from within
# the function get_fzs().
# - - - - - - - - - - - - - - - - - - - - - - - - -
# NHIs = HI column density bins, log spacing
# dz   = integral of (1+z)**gamma from the start
#        to the end of the redshift bin. See Steidel
#        et al. 2018, Appendix B for gamma definitions.
# dN   = linear spacing of logarithmic NHI bins.
# CGM  = flag to denote if CGM HI distribution to be
#        used (where (z_em - z) <= 0.0023*(1.+z_em))
# - - - - - - - - - - - - - - - - - - - - - - - - -
# NOTE: values of dz, dN, and CGM are determined from
#       within the function get_fzs()
# - - - - - - - - - - - - - - - - - - - - - - - - -
def one_Nabs(NHIs,dz,dN,DH_IGM,DH_CGM,CGM=False):

    lH = 10.**(NHIs)
    # f(NHI,z) for log column densities below 15.2 for non-CGM and 13.0 for CGM
    Ns = np.array((10.**9.305)*DH_IGM*dz[1])
    if CGM:
        # f(NHI,z) for CGM systems, log column density greater than 13.0
        t  = np.where(NHIs[:-1] >= 13.0)[0]
        Ns[t] = (10.**6.716)*DH_CGM[t]*dz[0]
    if not CGM:
        # f(NHI,z) for non-CGM, log column density greater than 15.2 
        t  = np.where(NHIs[:-1] >= 15.2)[0]
        Ns[t] = (10.**7.542)*DH_IGM[t]*dz[0]

    # RANDOMLY SAMPLED (POISSON) VALUES ARE RETURNED
    return np.random.poisson(lam=Ns*0.82,size=(1,len(Ns)))
    
# - - - - - - - - - - - - - - - - - - - - - - - - -
# Computes integral of (1+z) and (1+z)**2.5. Gives
# multiplicative factors to convert f(NHI,z) to a
# poisson lambda value for random sampling.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# z  = current redshift
# dz = size of redshift slice
# - - - - - - - - - - - - - - - - - - - - - - - - -
def do_Zint(z,dz):
    z1,z2 = z,z+dz
    o1 = (((z2*z2)/2.)+z2)-(((z1*z1)/2.)+z1)
    o2 = (0.285714*(z2**3.5))-(0.285714*(z1**3.5))
    return [o1,o2]

def do_Hint(NHI):
    bl,bh,bc = 1.635,1.463,1.381
    
    outIGM = np.zeros(len(NHI)-1)
    outCGM = np.zeros(len(NHI)-1)
    for i in range(len(outIGM)-1):
        H1,H2 = 10.**(NHI[i]),10.**(NHI[i+1])
        if NHI[i] < 15.2:
            outIGM[i] = ((H2**(1.-bl))-(H1**(1.-bl)))/(1.-bl)
        else:
            outIGM[i] = ((H2**(1.-bh))-(H1**(1.-bh)))/(1.-bh)
            
        if NHI[i] < 13.0:
            outCGM[i] = ((H2**(1.-bl))-(H1**(1.-bl)))/(1.-bl)
        else:
            outCGM[i] = ((H2**(1.-bc))-(H1**(1.-bc)))/(1.-bc)

    return outIGM,outCGM

            
# - - - - - - - - - - - - - - - - - - - - - - - - -
# Get the randomly sampled (poisson) absorption systems
# in each NHI bin in each redshift bin. Returns a 2D
# numpy array with dimensions (n(redshifts),n(NHIs))
# - - - - - - - - - - - - - - - - - - - - - - - - -
# zs   = array of redshifts corresponding to bins
# zem  = redshift of current source (redshift to which
#        the current IGM transmission curve is calculated).
# dz   = size of redshift bins in terms of z
# NHIs = array of log(HI column density)
# - - - - - - - - - - - - - - - - - - - - - - - - -
def get_fzs(zs,zem,dz,NHIs,wav,do_CGM=True):
    # Create empty output array
    fzs = np.zeros((len(zs),len(NHIs)-1))
    # Calculate linear size of logarithmically spaced NHI bins
    dHI = np.array([10.**(NHIs[i+1])-10.**(NHIs[i]) for i in range(len(NHIs)-1)])
    DH1,DH2 = do_Hint(NHIs)
    # Loop over redshifts
    for i,z in enumerate(zs):
        if (1.+z)*1216. >= wav[0]:
            # Calculate integral of (1+z)^gamma across current redshift bin
            DX = do_Zint(z,dz)

        
            # Switch to turn on CGM distribution
            if zem-z <= 0.0023*(1.+zem) and do_CGM:
                fzs[i] = one_Nabs(NHIs,DX,dHI,DH1,DH2,CGM=True)
            else:
                fzs[i] = one_Nabs(NHIs,DX,dHI,DH1,DH2,CGM=False)
            
    return fzs

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Computes the optical depth for LyC photons for
# an absorber of column density NHI at redshift z.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# NHI = hydrogen column density
# lam = wavelength array
# z   = redshift of absorber
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns array with length len(lam) of optical
# depth (tau) values at each wavelength. Convert to
# transmission with "np.exp((-1.)*tau)"
# - - - - - - - - - - - - - - - - - - - - - - - - -
def tau_HI_LyC(NHI,lam,z):
    l_lc = 911.8*(1.+z)
    
    x = lam/l_lc
    tau = NHI*(6.3e-18)*(x*x*x)

    t = (lam/l_lc > 1.)
    t = np.where(t == True)
    tau[t[0]] = 0.

    return tau

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Voigt profile approximation from Tepper-Garcia 2006.
# Used to compute the Ly-alpha forest transmission
# with "tau_HI_LAF" below.
# - - - - - - - - - - - - - - - - - - - - - - - - -
# lam   = wavelength array
# lami  = central wavelength of current Lyman line
# b     = doppler broadening in angst/s
# gamma = damping parameter of current Lyman line
#         taken from VPFIT table
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns array with voigt profile for current
# lyman line with length equal to len(lam)
# - - - - - - - - - - - - - - - - - - - - - - - - -
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

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Doppler parameter distribution function taken from
# Inoue & Iwata 2008, eq 6. Used to randomly sample
# a doppler broadening for a given absorber. Sampling
# is done using inverse cdf sampling with
# cdf_sampler.py
# - - - - - - - - - - - - - - - - - - - - - - - - -
# b = doppler broadening
# - - - - - - - - - - - - - - - - - - - - - - - - -
def doppler_dist(b):
    bs = 23.
    A1 = (4.*bs*bs*bs*bs)/(b*b*b*b*b)
    A2 = np.exp((-1.)*A1*b/4.)
    return A1*A2

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Computes the Lyman line forest cross-sections as
# a function of wavelength following Inoue & Iwata
# 2008 eq 10. To convert to an optical depth spectrum,
# multiply by the column density of the current
# absorber as in "make_tau".
# - - - - - - - - - - - - - - - - - - - - - - - - -
# wav = wavelength array (angstroms)
# z   = redshift
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns cross section spectrum for Lyman lines
# - - - - - - - - - - - - - - - - - - - - - - - - -
def tau_HI_LAF(wav,z):
    me,ce,c = 9.1094e-31,1.6022e-19,2.99792e18
    sig_T = 6.625e-25       #cm^2
    c     = 2.998e10        #cm/s

    tau = np.zeros(len(wav))
    lam = wav/(1.+z)

    bx = np.arange(1,1000,.1)
    by = doppler_dist(bx)
    bcds  =  cds.cdf_sampler(bx,by)
    bcds.sample_n(1)
    
    b  = bcds.sample[0]*1.e13  #angstrom/s
    for i in range(len(LAF_table[:,0])):
        fi    = LAF_table[i,1]  
        li    = LAF_table[i,0]  #angstrom
        gamma = LAF_table[i,2]

        A1 = c*np.sqrt((3.*np.pi*sig_T)/8.)
        A2 = (fi*li)/(np.sqrt(np.pi)*b)
        A3 = voigt_approx(lam,li,b,gamma)

        tm_tau = 4.0*A1*A2*A3
        bad = np.where(np.isfinite(tm_tau) == False)
        tm_tau[bad[0]] = 0.
        tau+=tm_tau

    
    tau[np.where(lam <= 911.8)[0]] = 0.
        
    return tau

# - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the optical depth spectrum for a given
# sightline. Conver to transmission using
# "np.exp((-1.)*tau)".
# - - - - - - - - - - - - - - - - - - - - - - - - -
# zs   = redshift array
# dz   = redshift bin size
# fzs  = number of absorber in each redshift bin
# NHIs = HI column density bins
# wav  = wavelength array
# - - - - - - - - - - - - - - - - - - - - - - - - -
# returns the optical depth spectrum including both
# LyC and Lyman series line absorption
# - - - - - - - - - - - - - - - - - - - - - - - - -
def make_tau(zs,fzs,lNHIs,wav):

    HIm = 10.**(lNHIs)
    zem = np.max(zs)
    tau = np.zeros(len(wav))
    for i in range(len(zs)):
        if np.max(fzs[i]) != 0.:
            t = np.where(fzs[i] > 0.)[0]

            for j in t:
                cdt = HIm[j]
                tau+=tau_HI_LyC(cdt,wav,zs[i])
                tau+=cdt*tau_HI_LAF(wav,zs[i])

    return tau


# - - - - - - - - - - - - - - - - - - - - - - - - -
# When imported as a module, load the Lyman series
# table as a global variable. This table is used to
# compute Lyman series line absorption and is called
# from the function "tau_HI_LAF", which is itself
# called from within "make_tau". The file is located
# in the TAOIST-MC folder, which should be added to
# your python path. 
# - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ != '__main__':

    flag = 0
    for p in sys.path:
        try:
            LAF_table = np.loadtxt(f'{p}/Lyman_series.dat',float)
            print('\n')
            print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
            print('Lyman series data loaded from:')
            print(f'{p}/Lyman_series.dat')
            print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
            print('\n')
            print('....mocking the IGM....')
            print('\n')
            flag = 1
        except:
            pass

    if flag == 0:
        print('\n')
        print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Lyman series data not found.  please add the TAOIST_MC')
        print('folder to your python path and try again.')
        print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('\n')
        sys.exit()
