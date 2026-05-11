import numpy as np

from taoistmc.config import SightlineConfig, PowerLawSegment

class SightlineSampler:
    def __init__(self, z_em: float, config: SightlineConfig):
        self.z_em = z_em
        self.config = config
        self.c_km_s = 299792.458
        self.z_bins = np.arange(0, self.z_em + config.dz, config.dz)
    
    def _get_expected_n(self, segment: PowerLawSegment, z_low: float, z_high: float) -> float:
        z_mid = (z_low + z_high) / 2.0
        A = 10**segment.log_A
        dz = z_high - z_low
        
        # 1. NHI Integral: Constant across both methods
        n_term = ((10**segment.log_N_max)**(1 - segment.beta) - 
                (10**segment.log_N_min)**(1 - segment.beta)) / (1 - segment.beta)
        
        # 2. COSMOLOGY CHECK
        # dX/dz = (1+z)^2 / E(z)
        e_z = np.sqrt(self.config.omega_m * (1 + z_mid)**3 + self.config.omega_lambda)
        dx_dz = ((1 + z_mid)**2) / e_z
        
        return A * n_term * dx_dz * dz

    def _sample_column_densities(self, segment: PowerLawSegment, n: int):
        """Vectorized Inverse Transform Sampling for column densities."""
        if n <= 0: return np.array([])
        u = np.random.uniform(0, 1, size=n)
        
        n_low_1b = (10**segment.log_N_min)**(1 - segment.beta)
        n_high_1b = (10**segment.log_N_max)**(1 - segment.beta)
        
        val_n = (u * (n_high_1b - n_low_1b) + n_low_1b)**(1 / (1 - segment.beta))
        return np.log10(val_n)

    def generate_sightline(self):
        dz_cgm = (self.config.cgm_influence_km_s / self.c_km_s) * (1 + self.z_em)
        z_cgm_limit = self.z_em - dz_cgm

        sightline = []
        for i in range(len(self.z_bins) - 1):
            z_l, z_h = self.z_bins[i], self.z_bins[i+1]
            z_mid = (z_l + z_h) / 2.0
            
            is_cgm_zone = self.config.use_cgm and z_mid >= z_cgm_limit
            active_segments = self.config.cgm_segments if is_cgm_zone else self.config.igm_segments
            
            for seg in active_segments:
                mu = self._get_expected_n(seg, z_l, z_h)
                n_absorbers = np.random.poisson(lam=mu)
                
                if n_absorbers > 0:
                    log_Ns = self._sample_column_densities(seg, n_absorbers)
                    for val in log_Ns:
                        sightline.append({"z": z_mid, "logNHI": val, "is_cgm": is_cgm_zone})
                        
        return sightline
