import numpy as np
import scipy.interpolate as interp

# new code
class cdf_sampler(object):

    # __init__ computed the normalisation
    # factor then constructs the cumulative
    # distribution function (cdf). 
    def __init__(self,x,y):
        self.x_input = x
        self.y_input = y
        self.sample  = 'None'

        pdf_fnorm = np.sum(y)
        self.cdf = np.cumsum(y/pdf_fnorm)

    # sample_n in produces a random sample
    # of n with a distribution matched to the
    # input array, y.
    def sample_n(self,n):

        self.sample = np.zeros(n)
        for i in range(n):
            tm = np.random.uniform()
            tt = np.where(np.abs(tm-self.cdf) == np.min(np.abs(tm-self.cdf)))
            if len(tt[0]) > 1:
                bi = np.random.binomial(1,.5)
                bi = int((-1.)*bi)
                self.sample[i] = self.x_input[tt[0][bi]]
                
            else:
                self.sample[i] = self.x_input[tt[0]]

            
# Subclass histogram oversampler creates a
# cdf_sampler that takes as on input the outputs
# of numpy.histogram() (i.e. n and fbin_edges). It
# first creates a new array that has "os_factor"
# sub-bins per bin in the histogram, and each sub-bin
# is assigned the value of the given bin. These are
# then use to create a cdf_sampler, thus allowing
# one to select os_factor values between the bin
# edges.
#
# Also included is the option of "spline", which
# performs a cubic spline fit to the histogram using
# scipy.interpolate.interp1d then passes the resulting
# spline into cdf_sampler. This removes the sharpness
# of the bin edges, but be aware that the agreement
# of the final sample and the input histogram may not
# be as good as spline=False (default). **Consider
# carefully your usage case before employing a spline
# fit**
class histogram_oversampler(cdf_sampler):

    def __init__(self,bin_edges,n,os_factor,spline=False):
        self.spl = 'None'
        
        dx= bin_edges[1]-bin_edges[0]
        
        x = np.arange(bin_edges[0],bin_edges[-1],dx/float(os_factor))
        y = np.zeros(len(x))
        
        for i in range(len(bin_edges)-1):
            tm = (x >= bin_edges[i])&(x <= bin_edges[i+1])
            tm = np.where(tm == True)
            y[tm[0]] = n[i]

        if spline == True:
            xspl = bin_edges[:-1]+(dx/2.)
            xspl = np.concatenate((np.array([xspl[0]-(dx/1.5)]),xspl))
            xspl = np.concatenate((xspl,np.array([xspl[-1]+(dx/1.5)])))
            nspl = np.concatenate((np.array([0]),n))
            nspl = np.concatenate((nspl,np.array([0])))
            f = interp.interp1d(xspl,nspl,kind='cubic')
            y = f(x)
            tt= np.where(y < 0.)
            if len(tt[0]) > 0: y[tt[0]] = 0.
            self.spl = np.array([x,y])
        
        cdf_sampler.__init__(self,x,y)
