from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

class PowerLawSegment(BaseModel):
    """
    Represents a single segment of f(N_HI, X) as defined by
    Carswell et al. (1984) using the f(N_HI, z) equation for
    N_abs found in Steidel et al (2018), eq B3. The full range
    of HI column densities considered in your taoistmc run can
    be made up of any number of segments in a piecewise manner.
    """
    log_N_min: float = Field(..., description="log of the minimum HI column density for this segment")
    log_N_max: float = Field(..., description="log of the maximum HI column density for this segment")
    beta: float = Field(..., description="Exponent associated with N_HI - negative powerlaw slope")
    log_A: float = Field(..., description="The normalisation factor")
    gamma: float = Field(..., description="The redshift evolution term - (1+z)^gamma")

    @model_validator(mode='after')
    def check_log_N(self):
        """Ensure log_N_min is less than log_N_max"""
        if self.log_N_min >= self.log_N_max:
            raise ValueError("log_N_min must be less than log_N_max")
        return self

class SightlineConfig(BaseModel):
    """
    Configuration for the sightline sampling module
    """
    dz: float = Field(5e-5, description="The size of redshift bins to be used")
    dhi: float = Field(0.25, description="The size of the logNHI bins to be used")
    use_cgm: bool = Field(True, description="Whether or not a different set of PowerLawSegment should be used for CGM")
    cgm_influence_km_s: float = Field(700.0, description="The velocity range to be considered 'CGM' when CGM models are in use")
    
    # Distribution Segments
    igm_segments: List[PowerLawSegment] = Field(..., description="Set of PowerLawSegments to use for Poisson sampling of HI absorbers - note this must be a single segment or a continuous piecewise segment.")
    cgm_segments: Optional[List[PowerLawSegment]] = Field(None, description="Set of PowerLawSegments to use for Poisson sampling of HI absorbers in CGM regions - note this must be a single segment or a continuous piecewise segment.")

    @staticmethod
    def _check_segments(segments: List[PowerLawSegment]) -> bool:
        if len(segments) == 1:
            return True
        for seg1, seg2 in zip(segments[:-1],segments[1:]):
            if seg1.log_N_max != seg2.log_N_min:
                return False
        return True

    @model_validator(mode='after')
    def check_segments_continuous(self):
        """Ensure igm_segments and cgm_segments are continuous"""
        if not self._check_segments(self.igm_segments):
            raise ValueError("igm_segments must be continous, no gaps")
        if self.use_cgm:
            if self.cgm_segments is None:
                raise ValueError("cgm_segments must be specified when use_cgm is True")
            if not self._check_segments(self.cgm_segments):
                raise ValueError("cgm_segments must be continous, no gaps")
        return self
    
    @model_validator(mode='after')
    def check_cgm_segments(self):
        """Ensure CGM segments are defined if use_cgm is True"""
        if self.use_cgm and self.cgm_segments is None:
            raise ValueError("If use_cgm is True, you must specify CGM segments")
        return self

class TaoistConfig(BaseModel):
    n_jobs: int = Field(-1, description="Number of parallel jobs to run, -1 will run maximum permitted on your system")
    use_gpu: bool = Field(False, description="Use a GPU or not")
    rest_wav_min: int = Field(600, description="Minimum rest wavelength for output transimission curves")
    rest_wav_max: int = Field(1500, description="Maximum rest wavelength for output transmission curves")
    delta_wav: float = Field(1.25, description="Rest wavelength resolution for output transmission curve")         

    sightline_config: SightlineConfig

    verbose: bool = False
    save: bool = True
    output_dir: str = Field("taoist_runs", description="Output directory for runs")