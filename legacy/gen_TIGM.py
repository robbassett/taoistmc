import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import glob
import os

from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate as integ

import cdf_sampler as cds
import TAOIST_MC as tao

def colorFader(c1,c2,mix):
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def colorFade3(c1,c2,c3,mix):
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    c3=np.array(mpl.colors.to_rgb(c3))
    if mix < 0.: out = mpl.colors.to_hex(c1)
    if mix >= 0. and mix < 0.5: out = mpl.colors.to_hex(2.*(0.5-mix)*c1 + 2.*mix*c2)
    if mix == 0.5: out = mpl.colors.to_hex(c2)
    if mix > 0.5 and mix <= 1.: out = mpl.colors.to_hex(2.*(1.-mix)*c2 + 2.*(mix-1.)*c3)
    if mix > 1.: out = mpl.color.to_hex(c3)
    return out

if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    do_plot = True
    
    # INPUT PARAMETERS
    size_ver    = 'A1'
    n_sightline = 25
    zems        = np.array([2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9])
    dz          = 5.e-5
    NHIs        = np.arange(12.,21.1,.1)
    wav_out     = np.arange(600.,2000.,2.)
    add_CGM     = True

    # SETUP FILE STRUCTURE
    dirs = glob.glob('*')
    if 'taus' not in dirs:
        os.mkdir('taus')
        [os.mkdir(f'taus/{_z}') for _z in zems]
    dirs = glob.glob('taus/*')
    
    if do_plot:
        F  = plt.figure(figsize=(9,5),dpi=150)
        ax = F.add_subplot(111)
    for i,zem in enumerate(zems):
        # DOUBLE CHECK FILE LOCATION EXISTS
        if f'taus/{zem}' not in dirs:
            os.mkdir(f'tau/{zem}')

        print(f'{n_sightline} Sightlines being computed')
        print(f'at z = {zem}.......')
        zs  = np.arange(0,zem+dz,dz)
        wav = np.arange(580.*(1.+zem),1250.*(1.+zem),2.2)
        taus  = np.zeros((n_sightline+1,len(wav)))
        taus[0] = wav

        for j in range(n_sightline):

            Nab    = tao.get_fzs(zs,zem,dz,NHIs,wav)
            tm_tau = tao.make_tau(zs,Nab,NHIs[:-1],wav)

            taus[j+1]  = tm_tau

        if do_plot:
            c = colorFade3('gold','darkcyan','k',float(i)/float(len(zems)-1))
            ax.plot(taus[0],np.mean(np.exp((-1.)*taus[1:]),axis=0),'-',c=c)
            
        print('\n')
        np.save(f'./taus/{zem}/taus_{n_sightline}{size_ver}.npy',taus)

    ax.set_xlabel(r'$\lambda_{obs}$',fontsize=20)
    ax.set_ylabel(r'$T_{IGM}$',fontsize=20)
    plt.savefig('taus/mean_Tigm_seq.png')
