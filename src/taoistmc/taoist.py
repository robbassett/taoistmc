from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import yaml

from taoistmc.core.sightline import SightlineSampler
from taoistmc.core.optical_depth import OpticalDepthCalculator
from taoistmc.config import TaoistConfig

DATA_DIR = Path(__file__).parent / 'data'
LAF_TABLE = np.loadtxt(DATA_DIR / 'lyman_series.dat')

class TaoistMc:
    """
    The main class for generating IGM transmission curves
    """
    def __init__(self, config: TaoistConfig | None = None):
        self.config = config
        self.z_em = None
        self.wav = None
        self.sightline = None
        self.optical_depth = None

    def _set_zem(self, z_em):
        """
        Set the source redshift and initiate the wavelength array,
        sightline sampler, and optical depth calculator
        """
        self.z_em = z_em
        self.wav = np.arange(
            self.config.rest_wav_min*(1.+z_em),
            self.config.rest_wav_max*(1.+z_em),
            self.config.delta_wav
        )

        self.sightline = SightlineSampler(z_em, self.config.sightline_config)
        self.optical_depth = OpticalDepthCalculator(self.wav)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> TaoistMc:
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = TaoistConfig(**config_dict)
        return TaoistMc(config)

    def _single_sightline(self, i: int | None = None) -> np.array:
        """
        generate a single sightline and calculate the IGM transmission
        """
        if i is not None:
            np.random.seed(i)

        sightline = self.sightline.generate_sightline()
        tau = self.optical_depth.make_tau(sightline, LAF_TABLE)
        return tau
    
    def run(self, z_em, n_sightlines: int) -> np.array:
        """
        Generate n_sightlines sightlines in parallel for a source
        redshift of z_em
        """
        self._set_zem(z_em)

        vb = 10 if self.config.verbose else 0
        output = Parallel(
            n_jobs = self.config.n_jobs,
            verbose=vb
        )(
            delayed(self._single_sightline)(i)
            for i in range(n_sightlines)
        )

        return np.array(output)
    