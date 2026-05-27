import numpy as np
from numba import njit
try:
    from numba import cuda
    from .gpu_kernels import gpu_tau_kernel
    GPU_AVAILABLE = cuda.is_available()
except ImportError as e:
    GPU_AVAILABLE = True

@njit
def voigt_approx_kernel(lam, lami, b, gamma):
    c = 2.998e18 # angst/s
    ldl = (b/c) * lami
    a = ((lami*lami) * gamma) / (4. * np.pi * c * ldl)
    
    t_vp = np.where(np.abs(lam - lami) <= (1.812 * (b / 1.e13)))[0]
    if len(t_vp) == 0:
        return np.zeros(len(lam))
        
    x = (lam[t_vp] - lami) / ldl
    A1 = np.exp(-x*x)
    A2 = a * (2. / np.sqrt(np.pi))

    K1 = (1. / (2. * x*x))
    K2 = ((4. * x*x) + 3.) * ((x*x) + 1.) * A1
    K3 = (1. / (x*x)) * ((2. * x*x) + 3.) * (np.sinh(x*x))
    Kx = K1 * (K2 - K3)

    xo = np.zeros(len(lam))
    xo[t_vp] = A1 * (1. - (A2 * Kx))
    return xo

@njit
def tau_HI_LyC_kernel(wav, z, NHI):
    """Standalone LyC cross-section calculation."""
    l_lc = 911.8 * (1. + z)
    x = wav / l_lc
    tau = NHI * (6.3e-18) * (x ** 3)
    tau[wav > l_lc] = 0.
    return tau

@njit
def _tau_hi_laf_kernel(wav, laf_table, z, nhi, b, a1):
    tau = np.zeros(len(wav))
    lam_rest = wav / (1. + z)
    
    for i in range(len(laf_table)):
        li = laf_table[i, 0]
        fi = laf_table[i, 1]
        gamma = laf_table[i, 2]

        A2 = (fi * li) / (np.sqrt(np.pi) * b)
        # Call the standalone kernel directly
        A3 = voigt_approx_kernel(lam_rest, li, b, gamma)

        tm_tau = a1 * A2 * A3
        
        # Fast finite check
        for j in range(len(tm_tau)):
            if not np.isfinite(tm_tau[j]):
                tm_tau[j] = 0.
        tau += tm_tau

    tau[lam_rest <= 911.8] = 0.
    return tau * nhi

@njit
def _make_tau_loop(wav, sightline_z, sightline_logNHI, dopps, laf_table, a1):
    total_tau = np.zeros(len(wav))
    for i in range(len(sightline_z)):
        z = sightline_z[i]
        nhi = 10.**sightline_logNHI[i]
        b = dopps[i] * 1e13
        
        # Direct calls to kernels
        total_tau += _tau_hi_laf_kernel(wav, laf_table, z, nhi, b, a1)
        total_tau += tau_HI_LyC_kernel(wav, z, nhi)
    return total_tau

class OpticalDepthCalculator:
    def __init__(self, wav, LAF_table, use_gpu=False):
        self.wav = wav
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.laf_table = LAF_table
        
        # Pre-transfer wavelength to GPU if using GPU mode
        if self.use_gpu:
            cuda.select_device(0) 
            print(f"✅ GPU Context initialized on: {cuda.get_current_device().name}")
            
            # Pre-transfer data
            self.d_wav = cuda.to_device(self.wav)
            self.d_laf = cuda.to_device(np.ascontiguousarray(LAF_table, dtype=np.float64))
            

        # Constants
        sig_T = 6.625e-25
        c = 2.998e10
        self.a1 = c * np.sqrt((3. * np.pi * sig_T) / 8.)

    @staticmethod
    @njit
    def doppler_dist(n_samples: int, b_sigma: float = 23.0):
        u = np.random.exponential(1.0, size=n_samples)
        b_values = b_sigma / (u**0.25)
        return np.clip(b_values, 5.0, 100.0)
    
    def make_tau(self, sightline):
        if self.use_gpu:
            return self._make_tau_gpu(sightline)
        return self._make_tau_cpu(sightline)

    def _make_tau_gpu(self, sightline):
        n_abs = len(sightline)
        if n_abs == 0:
            return np.zeros_like(self.wav)

        # 1. Prepare data (ensure contiguous)
        z_all = np.ascontiguousarray(sightline['z'], dtype=np.float64)
        nhi_all = np.ascontiguousarray(10.**sightline['logNHI'], dtype=np.float64)
        dopps_all = np.ascontiguousarray(self.doppler_dist(n_abs) * 1e13, dtype=np.float64)
        
        # 2. Update GPU buffers with new data
        # 'cuda.to_device' creates a new buffer. 
        # To reuse the same GPU memory, use d_array.copy_to_device(host_array)
        d_z = cuda.to_device(z_all)
        d_nhi = cuda.to_device(nhi_all)
        d_b = cuda.to_device(dopps_all)
        
        # 3. Create a clean output array for this sightline
        d_out = cuda.device_array_like(self.d_wav)

        # 4. Configure Grid
        tpb = 256
        bpg = (self.wav.size + tpb - 1) // tpb

        # 5. Launch with correct variables
        gpu_tau_kernel[bpg, tpb](
            self.d_wav, d_z, d_nhi, d_b, self.d_laf, self.a1, d_out
        )
        
        cuda.synchronize()
        return d_out.copy_to_host()

    def _make_tau_cpu(self, sightline):
        n_abs = len(sightline)
        if n_abs == 0:
            return np.zeros(len(self.wav))
            
        dopps = self.doppler_dist(n_abs)
        
        # Simply feed the raw data into the master loop
        return _make_tau_loop(
            self.wav, 
            sightline['z'], 
            sightline['logNHI'], 
            dopps, 
            self.laf_table, 
            self.a1
        )