"""Integration tests for the main TaoistMc class."""
import tempfile
from pathlib import Path
import numpy as np
from taoistmc import TaoistMc


def test_run_returns_correct_shape(sample_taoist_config):
    mc = TaoistMc(sample_taoist_config)
    taus = mc.run(z_em=2.0, n_sightlines=5)
    assert taus.shape[0] == 5
    assert taus.shape[1] == len(mc.wav)


def test_save_and_load_roundtrip(sample_taoist_config, tmp_path):
    mc = TaoistMc(sample_taoist_config)
    mc.config.save = True
    mc.config.output_dir = str(tmp_path)

    taus = mc.run(z_em=1.5, n_sightlines=3)
    assert len(taus) == 3

    # Find the saved file
    z_dir = tmp_path / "z1p5"
    files = list(z_dir.glob("*.npz"))
    assert len(files) == 1

    # Load it back
    mc2 = TaoistMc(sample_taoist_config)
    mc2.load_results(files[0])
    assert "sightlines" in mc2.loaded_results
    assert len(mc2.loaded_results["sightlines"]) == 3


def test_caching_avoids_regeneration(sample_taoist_config, tmp_path):
    mc = TaoistMc(sample_taoist_config)
    mc.config.save = True
    mc.config.output_dir = str(tmp_path)

    # First run
    taus1 = mc.run(z_em=2.2, n_sightlines=4)
    assert len(taus1) == 4

    # Second run requesting fewer should return cached only
    mc2 = TaoistMc(sample_taoist_config)
    mc2.config.save = True
    mc2.config.output_dir = str(tmp_path)
    taus2 = mc2.run(z_em=2.2, n_sightlines=2)
    assert len(taus2) >= 4  # should have loaded the previous ones