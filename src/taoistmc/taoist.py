from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

from taoistmc.core.sightline import SightlineSampler
from taoistmc.core.optical_depth import OpticalDepthCalculator
from taoistmc.config import TaoistConfig

DATA_DIR = Path(__file__).parent / 'data'
LAF_TABLE = np.loadtxt(DATA_DIR / 'lyman_series.dat')

class TaoistMc:

    def __init__(self, z_em: float, config: TaoistConfig | None = None):
        self.config = config
        self.z_em = z_em
        self.wav = np.arange(
            self.config.rest_wav_min*(1.+z_em),
            self.config.rest_wav_max*(1.+z_em),
            self.config.delta_wav
        )

        self.sightline = SightlineSampler(z_em, config.sightline_config)
        self.optical_depth = OpticalDepthCalculator(self.wav)

    def _single_sightline(self, i: int):
        np.random.seed(i)

        sightline = self.sightline.generate_sightline()
        tau = self.optical_depth.make_tau(sightline, LAF_TABLE)
        return tau
    
    def run(self, n_sightlines: int):
        output = Parallel(
            n_jobs = self.config.n_jobs,
            verbose=10
        )(
            delayed(self._single_sightline)(i)
            for i in range(n_sightlines)
        )

        return np.array(output)
    