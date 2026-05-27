from numba import cuda
import math

@cuda.jit(device=True)
def _voigt_device(lam, lami, b, gamma):
    """Internal device-only Voigt profile."""
    c_angst = 2.998e18 
    ldl = (b / c_angst) * lami
    a = ((lami * lami) * gamma) / (4. * math.pi * c_angst * ldl)
    
    # Profile window check
    if abs(lam - lami) > (1.812 * (b / 1.e13)):
        return 0.0
        
    x = (lam - lami) / ldl
    x2 = x * x
    a1_val = math.exp(-x2)
    
    # Tepper-Garcia approximation terms
    k1 = 1. / (2. * x2)
    
    # Avoid overflow in sinh(x2)
    if x2 > 50.0:
        kx = k1 * ( (4.*x2 + 3.)*(x2 + 1.)*a1_val - ( (2.*x2 + 3.)/x2 * 0.5 ) )
    else:
        k2 = ((4. * x2) + 3.) * ((x2 + 1.)) * a1_val
        k3 = (1. / x2) * ((2. * x2) + 3.) * (math.sinh(x2))
        kx = k1 * (k2 - k3)
    
    a2_val = a * (2. / math.sqrt(math.pi))
    return a1_val * (1. - (a2_val * kx))

@cuda.jit
def gpu_tau_kernel(wav, z_abs, nhi_abs, b_abs, laf_table, a1_const, output_tau):
    """
    Parallelized kernel that ACCUMULATES tau into the output_tau array.
    This allows for processing sightlines in batches to avoid GPU timeouts.
    """
    idx = cuda.grid(1)
    if idx < wav.size:
        curr_wav = wav[idx]
        total_tau = 0.0
        
        # Loop through every absorber in the current batch
        for i in range(z_abs.size):
            z = z_abs[i]
            nhi = nhi_abs[i]
            b = b_abs[i]
            
            lam_rest = curr_wav / (1.0 + z)
            
            # 1. Lyman Series (LAF)
            if lam_rest > 911.8:
                for j in range(laf_table.shape[0]):
                    li, fi, gamma = laf_table[j, 0], laf_table[j, 1], laf_table[j, 2]
                    a2_laf = (fi * li) / (math.sqrt(math.pi) * b)
                    a3_laf = _voigt_device(lam_rest, li, b, gamma)
                    
                    term = a1_const * a2_laf * a3_laf
                    if math.isfinite(term):
                        total_tau += term * nhi
            
            # 2. Lyman Continuum (LyC)
            l_lc = 911.8 * (1.0 + z)
            if curr_wav <= l_lc:
                total_tau += nhi * (6.3e-18) * ((curr_wav / l_lc) ** 3)
                
        output_tau[idx] = total_tau
