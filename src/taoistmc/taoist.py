from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import yaml
import time
import json

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
        self.loaded_results = None
        self.results = None

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
        self.optical_depth = OpticalDepthCalculator(self.wav, LAF_TABLE, self.config.use_gpu)

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = TaoistConfig(**config_dict)
        return cls(config)

    def _single_sightline(self) -> np.array:
        """
        generate a single sightline and calculate the IGM transmission
        """
        sightline = self.sightline.generate_sightline()
        tau = self.optical_depth.make_tau(sightline)
        return {"sightline": sightline, "tau": tau}
    
    def run(self, z_em, n_sightlines: int) -> np.array:
        """
        Generate n_sightlines sightlines in parallel for a source
        redshift of z_em
        """
        if self.config.verbose:
            print(f"--- Mocking the IGM: z_em = {z_em}, n = {n_sightlines} ---")

        self._set_zem(z_em)
        self.loaded_results = None

        existing_count = self.load_redshift(z_em)

        # 3. Calculate how many more we actually need
        n_to_generate = n_sightlines - existing_count
    
        if n_to_generate <= 0:
            if self.config.verbose:
                print(f"--- Found {existing_count} existing sightlines. None needed. ---")
            
            return np.array([_.get("tau") for _ in self.loaded_results.get("sightlines")])
        
        if self.config.verbose:
            print(f"--- Found {existing_count} existing. Generating {n_to_generate} more. ---")

        vb = 10 if self.config.verbose else 0
        output = Parallel(
            n_jobs = self.config.n_jobs,
            verbose=vb,
            backend="threading" if self.optical_depth.use_gpu else "loky"
        )(
            delayed(self._single_sightline)()
            for _ in range(n_to_generate)
        )

        self.results = {
            "z_em":z_em,
            "sightlines":output
        }

        # Save prior to concatenating results
        if self.config.save:
            self._save(n_to_generate)

        if self.loaded_results is not None:
            self.results['sightlines'] += self.loaded_results['sightlines']

        return np.array([_.get("tau") for _ in output])
    
    def _save(self, n_generated):
        """
        Saves the internal self.results using pathlib for modern path handling.
        """
        if not self.config.save or self.results is None:
            return
        
        z_em = self.results.get("z_em","")

        # Convert output_dir to a Path object
        out_dir = Path(self.config.output_dir) / f'z{str(z_em).replace(".","p")}'
        
        # Create the directory (parents=True handles nested paths, 
        # exist_ok prevents errors if it already exists)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the / operator for clean path joining
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"taoist_zem{z_em:.3f}_n{n_generated}_{timestamp}.npz"
        fpath = out_dir / filename

        config_json = self.config.model_dump_json()

        # Prepare the archive data
        save_kwargs = {
            "z_em": np.array([z_em]),
            "config_json": np.array([config_json]),
        }

        for i, res in enumerate(self.results["sightlines"]):
            save_kwargs[f"sl_{i}"] = res["sightline"]
            save_kwargs[f"tau_{i}"] = res["tau"]

        # np.savez_compressed accepts Path objects directly
        np.savez_compressed(fpath, **save_kwargs)

        if self.config.verbose:
            print(f"--- Saved {len(self.results['sightlines'])} sightlines to {fpath} ---")

    def load_results(self, filepath: str | Path):
        """
        Loads a compressed .npz file and reconstructs the TaoistMc environment.
        """
        fpath = Path(filepath)
        if not fpath.exists():
            raise FileNotFoundError(f"No simulation file found at {fpath}")

        # allow_pickle is required because we are loading string arrays for the JSON
        with np.load(fpath, allow_pickle=True) as data:
            # 1. Identify and sort sightline keys to maintain parallel order
            sl_keys = sorted([k for k in data.files if k.startswith('sl_')], 
                            key=lambda x: int(x.split('_')[1]))
            
            sightlines = []
            for key in sl_keys:
                idx = key.split('_')[1]
                sightlines.append({
                    "sightline": data[key],
                    "tau": data[f"tau_{idx}"]
                })

            # 2. Reconstruct the TaoistConfig object if metadata exists
            config = None
            if "config_json" in data:
                # model_validate_json is the Pydantic V2 way to hydrate the model
                config = TaoistConfig.model_validate_json(data["config_json"][0])

            # 3. Assemble the master results dictionary
            if self.loaded_results is None:
                self.loaded_results = {
                    "z_em": float(data["z_em"][0]),
                    "sightlines": sightlines,
                    "config": config
                }
            else:
                self.loaded_results['sightlines'] += sightlines

    def load_redshift(self, z_em: float) -> int:
        """
        Finds and loads all valid files for a given redshift into self.loaded_results.
        Returns the total count of valid sightlines found.
        """
        z_str = f"z{str(z_em).replace('.', 'p')}"
        target_dir = Path(self.config.output_dir) / z_str
        
        if not target_dir.exists():
            return 0

        # Define physics fields to ignore for comparison
        physics_mask = {'save', 'output_dir', 'verbose', 'n_jobs'}
        current_physics = self.config.model_dump(exclude=physics_mask)

        valid_count = 0
        for fpath in target_dir.glob("*.npz"):
            # We peek at the config first to see if it's a match
            with np.load(fpath, allow_pickle=True) as data:
                if "config_json" in data:
                    saved_cfg = TaoistConfig.model_validate_json(data["config_json"][0])
                    if saved_cfg.model_dump(exclude=physics_mask) == current_physics:
                        # It's a match! Use the existing method to load it
                        self.load_results(fpath)
                        
        # Return count of sightlines currently held for this redshift
        if self.loaded_results and 'sightlines' in self.loaded_results:
            return len(self.loaded_results['sightlines'])
        return 0