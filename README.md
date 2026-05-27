![Da Way](docs/tmc.png)

# TAOIST-MC: TrAnsmission Of IoniSing lightT - Monte Carlo

[![Tests](https://github.com/robbassett/TAOIST_MC/actions/workflows/tests.yml/badge.svg)](https://github.com/robbassett/TAOIST_MC/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/robbassett/TAOIST_MC/branch/main/graph/badge.svg)](https://codecov.io/gh/robbassett/TAOIST_MC)

**Simulated IGM UV Transmission for Lyman Continuum Studies**

---

## Overview

TAOIST-MC generates realistic intergalactic medium (IGM) transmission curves for studies of Lyman continuum (LyC) emission from distant galaxies. Using hydrogen absorption system statistics from the literature (redshift and column density distribution functions), it simulates the transmission of ionizing radiation through neutral hydrogen in the IGM for sources at specified redshifts.

The primary outputs are ensembles of IGM transmission functions T(λ) at UV wavelengths. These can be coupled with model spectra from population synthesis codes (e.g., BPASS) to predict the observed ionizing flux from high-redshift sources.

### Key Improvements (v0.3+)

- **Modular OOP Design** — Clean separation of sightline sampling, optical depth calculation, and configuration
- **Parallel Processing** — Multi-core CPU acceleration via `joblib`
- **GPU Support** — CUDA-accelerated optical depth kernels via Numba (10×+ speedup)
- **YAML Configuration** — Physics parameters defined via validated Pydantic models
- **Automatic Caching** — Sightlines are saved and reused across runs with matching physics configurations
- **CLI Interface** — `taoistmc init` and `taoistmc run` commands for streamlined workflows

---

## Installation

```bash
pip install .
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, Numba, Pydantic, Typer, PyYAML, joblib.

> **GPU Support:** GPU acceleration requires a CUDA-capable NVIDIA GPU and the appropriate CUDA toolkit. If unavailable, the package gracefully falls back to CPU execution.

---

## Quick Start (Python API)

```python
import taoistmc as tmc
import numpy as np
import matplotlib.pyplot as plt

from taoistmc.config import PowerLawSegment, SightlineConfig, TaoistConfig

if __name__ == "__main__":

    igm_low = PowerLawSegment(log_N_min=12.0, log_N_max=15.2, beta=1.635, log_A=9.305, gamma=2.5)
    igm_high = PowerLawSegment(log_N_min=15.2, log_N_max=21.0, beta=1.463, log_A=7.542, gamma=1.0)
    cgm_seg = PowerLawSegment(log_N_min=13.0, log_N_max=21.0, beta=1.381, log_A=6.716, gamma=1.0)

    config = SightlineConfig(
        igm_segments=[igm_low, igm_high],
        cgm_segments=[cgm_seg]
    )

    full_config = TaoistConfig(
        sightline_config=config,
        delta_wav=0.25
    )

    F = plt.figure()
    ax = F.add_subplot(111)
    for z in (1.5, 2.5, 3.5):
        tao = tmc.TaoistMc(full_config)
        output = tao.run(z, 200)

        taum = np.mean(np.exp(-output), axis=0)
        ax.plot(tao.wav / (1. + z), taum, lw=1)
    [ax.axvline(x=_x, c='r', ls='--', lw=.3) for _x in [911.75, 1216.]]
    ax.set_ylim(-.1, 1.1)
    ax.set_xlim(799, 1249)
    ax.set_xlabel(r'$\lambda_{rest}$', fontsize=18)
    ax.set_ylabel(r'$T_{IGM}$', fontsize=18)
    plt.show()
```

---

## Command-Line Interface

TAOIST-MC provides a Typer-based CLI:

```bash
# Generate a starter config (Steidel et al. 2018 parameters)
taoistmc init

# Run 50 sightlines at z=2.4
taoistmc run -n 50 2.4

# Verbose output, custom config path
taoistmc run --verbose --config my_config.yaml 3.1
```

- `taoistmc init` — Copies `starter_config.yaml` to the current directory
- `taoistmc run` — Executes a simulation, automatically reusing cached sightlines when physics parameters match

---

## Configuration

### PowerLawSegment

Defines a piecewise segment of the column density distribution function f(N_HI, X) following the form used in Steidel et al. (2018), eq. B3:

```python
PowerLawSegment(
    log_N_min=12.0,   # Minimum log column density
    log_N_max=15.2,   # Maximum log column density
    beta=1.635,       # Power-law slope (negative exponent)
    log_A=9.305,      # Normalization
    gamma=2.5         # Redshift evolution: (1+z)^gamma
)
```

Multiple segments can be combined for a continuous, piecewise f(N) model. Segments must be contiguous (no gaps in log N).

### SightlineConfig

Controls absorber sampling:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dz` | 5e-5 | Redshift bin size for integration |
| `dhi` | 0.25 | log N_HI bin size |
| `use_cgm` | True | Enable separate CGM absorber population |
| `cgm_influence_km_s` | 700.0 | Velocity window (km/s) around source considered "CGM" |
| `igm_segments` | (required) | List of PowerLawSegments for IGM |
| `cgm_segments` | (required if use_cgm) | List of PowerLawSegments for CGM |

### TaoistConfig

Top-level simulation settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_jobs` | -1 | Parallel jobs (-1 = all cores) |
| `use_gpu` | False | Enable CUDA acceleration |
| `rest_wav_min` | 600 | Minimum rest-frame wavelength (Å) |
| `rest_wav_max` | 1500 | Maximum rest-frame wavelength (Å) |
| `delta_wav` | 1.25 | Wavelength sampling resolution (Å) |
| `sightline_config` | (required) | Nested SightlineConfig |
| `verbose` | False | Print progress messages |
| `save` | True | Save results to disk |
| `output_dir` | "taoist_runs" | Base directory for saved runs |

---

## Performance & Acceleration

- **CPU Parallelism:** Sightlines are generated in parallel using `joblib`. Set `n_jobs` to control worker count.
- **GPU Acceleration:** When `use_gpu=True` and a CUDA device is available, the optical depth calculation runs on the GPU via Numba CUDA kernels. This can yield >10× speedup for large sightline counts.
- **Caching:** Results are saved as compressed `.npz` files under `taoist_runs/zXpX/`. Subsequent runs with identical physics parameters automatically load and append to existing sightlines.

---

## Scientific Background

This code implements the methodology described in:

> **Bassett et al. 2021**, "The Detection of Ionizing Radiation from Star-Forming Galaxies at z ~ 3–4", *MNRAS* (arXiv:2101.00727)

The transmission curves account for both Lyman-series absorption and Lyman-continuum opacity from the Lyman-α forest (LAF) and CGM absorbers. Future versions may include helium ionization physics.

For more details, see Section 2 of the paper and the legacy documentation in `legacy/README.md`.

---

## Output Format

When `save=True`, each run creates files like:

```
taoist_runs/z2p400/taoist_zem2.400_n50_20260527_221530.npz
```

These archives contain:
- `z_em`: Source redshift
- `config_json`: Serialized TaoistConfig for reproducibility checks
- `sl_i`, `tau_i`: Sightline absorber arrays and optical depth vectors

The `TaoistMc.load_results()` and `load_redshift()` methods can reconstruct a simulation state from these files.

---

## Contact & Citation

**Author:** Robert Bassett  
**Email:** rbassett.astro@gmail.com

If you use TAOIST-MC in your research, please cite Bassett et al. 2021 and link to this repository.

---