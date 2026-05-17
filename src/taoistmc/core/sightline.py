import numpy as np
from numba import njit

from taoistmc.config import SightlineConfig, PowerLawSegment

@njit
def sample_absorbers_numba(z_bins, hi_bins, log_A, beta, gamma):
    """
    JITed kernel returning flat arrays for z and logNHI.
    """
    z_out = []
    nhi_out = []
    
    A_lin = 10.0**log_A
    b_term = 1.0 - beta
    g_term = gamma + 1.0
    
    z_ints = (((1.0 + z_bins[1:])**g_term) - ((1.0 + z_bins[:-1])**g_term)) / g_term
    h_ints = ((10.0**(hi_bins[1:] * b_term)) - (10.0**(hi_bins[:-1] * b_term))) / b_term
    
    for i in range(len(z_ints)):
        z_mid = (z_bins[i] + z_bins[i+1]) / 2.0
        for j in range(len(h_ints)):
            mu = A_lin * h_ints[j] * z_ints[i]
            n_abs = np.random.poisson(mu)
            
            if n_abs > 0:
                n_low_1b = (10.0**hi_bins[j])**b_term
                n_high_1b = (10.0**hi_bins[j+1])**b_term
                
                for _ in range(n_abs):
                    u = np.random.random()
                    val_n = (u * (n_high_1b - n_low_1b) + n_low_1b)**(1.0 / b_term)
                    z_out.append(z_mid)
                    nhi_out.append(np.log10(val_n))
                    
    return np.array(z_out), np.array(nhi_out)

class SightlineSampler:
    def __init__(self, z_em: float, config: SightlineConfig):
        self.z_em = z_em
        self.config = config
        self.c_km_s = 299792.458
        self.z_bins = np.arange(0, self.z_em + config.dz, config.dz)

    def generate_sightline(self):
        dz_cgm = (self.config.cgm_influence_km_s / self.c_km_s) * (1 + self.z_em)
        z_cgm_limit = self.z_em - dz_cgm

        # Lists to collect arrays from different segments
        all_zs = []
        all_nhis = []
        all_is_cgm = []

        for is_cgm_zone in [False, True]:
            if is_cgm_zone and not self.config.use_cgm: continue
            
            # Segment and Z-bin logic
            if is_cgm_zone:
                mask = self.z_bins >= z_cgm_limit
                if np.sum(mask) < 2: continue 
                curr_z_bins = self.z_bins[mask]
                segments = self.config.cgm_segments
            else:
                mask = self.z_bins <= z_cgm_limit
                curr_z_bins = np.append(self.z_bins[mask], z_cgm_limit)
                segments = self.config.igm_segments

            for seg in segments:
                hi_bins = np.arange(seg.log_N_min, seg.log_N_max + self.config.dhi, self.config.dhi)
                
                zs, nhis = sample_absorbers_numba(
                    curr_z_bins, hi_bins, seg.log_A, seg.beta, seg.gamma
                )
                
                if len(zs) > 0:
                    all_zs.append(zs)
                    all_nhis.append(nhis)
                    # Create boolean array for the CGM flag
                    all_is_cgm.append(np.full(len(zs), is_cgm_zone, dtype=bool))

        if not all_zs:
            return np.array([], dtype=[('z', 'f8'), ('logNHI', 'f8'), ('is_cgm', '?')])

        # Concatenate all fragments into a single structured array
        final_zs = np.concatenate(all_zs)
        final_nhis = np.concatenate(all_nhis)
        final_cgm = np.concatenate(all_is_cgm)

        # Build the structured array
        sightline = np.zeros(len(final_zs), dtype=[
            ('z', 'f8'), 
            ('logNHI', 'f8'), 
            ('is_cgm', '?')
        ])
        
        sightline['z'] = final_zs
        sightline['logNHI'] = final_nhis
        sightline['is_cgm'] = final_cgm

        return sightline
                