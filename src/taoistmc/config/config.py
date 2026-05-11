from pydantic import BaseModel
from typing import List, Optional

class PowerLawSegment(BaseModel):
    """Represents a single segment of the N_HI distribution as defined in Table B1."""
    log_N_min: float
    log_N_max: float
    beta: float  # Slope of f(NHI, X)
    log_A: float # Normalization of f(NHI, X)
    gamma: float # Redshift evolution exponent

class SightlineConfig(BaseModel):
    dz: float = 5e-5
    use_cgm: bool = True
    cgm_influence_km_s: float = 700.0
    
    # Cosmological Parameters (Standard flat Lambda-CDM)
    omega_m: float = 0.3
    omega_lambda: float = 0.7
    
    # Distribution Segments
    igm_segments: List[PowerLawSegment]
    cgm_segments: Optional[List[PowerLawSegment]] = None

class TaoistConfig(BaseModel):
    n_jobs: int = -1
    rest_wav_min: int = 600
    rest_wav_max: int = 1500
    delta_wav: float = 5.0

    sightline_config: SightlineConfig