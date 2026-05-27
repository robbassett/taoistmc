"""Tests for OpticalDepthCalculator (CPU path)."""
import numpy as np
from taoistmc.core.optical_depth import OpticalDepthCalculator
from taoistmc.taoist import LAF_TABLE


def test_cpu_tau_shape(sample_taoist_config):
    wav = np.arange(900, 1200, 1.0)
    calc = OpticalDepthCalculator(wav, LAF_TABLE, use_gpu=False)

    # Create a minimal sightline
    dtype = [("z", "f8"), ("logNHI", "f8"), ("is_cgm", "?")]
    sightline = np.zeros(3, dtype=dtype)
    sightline["z"] = [1.0, 1.5, 2.0]
    sightline["logNHI"] = [13.0, 14.0, 15.0]
    sightline["is_cgm"] = [False, False, True]

    tau = calc.make_tau(sightline)
    assert tau.shape == wav.shape
    assert np.all(tau >= 0)


def test_empty_sightline_returns_zeros(sample_taoist_config):
    wav = np.arange(900, 1100, 1.0)
    calc = OpticalDepthCalculator(wav, LAF_TABLE, use_gpu=False)
    dtype = [("z", "f8"), ("logNHI", "f8"), ("is_cgm", "?")]
    empty = np.array([], dtype=dtype)
    tau = calc.make_tau(empty)
    assert np.allclose(tau, 0)