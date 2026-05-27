"""Tests for SightlineSampler."""
import numpy as np
from taoistmc.core.sightline import SightlineSampler


def test_generate_sightline_basic(sample_sightline_config):
    sampler = SightlineSampler(z_em=2.5, config=sample_sightline_config)
    sightline = sampler.generate_sightline()

    assert sightline.dtype.names == ("z", "logNHI", "is_cgm")
    assert len(sightline) > 0
    assert np.all(sightline["z"] >= 0)
    assert np.all(sightline["z"] <= 2.5 + 1e-6)


def test_generate_sightline_no_cgm(sample_powerlaw_segments):
    from taoistmc.config import SightlineConfig
    cfg = SightlineConfig(
        igm_segments=sample_powerlaw_segments["igm"],
        cgm_segments=None,
        use_cgm=False
    )
    sampler = SightlineSampler(z_em=1.8, config=cfg)
    sightline = sampler.generate_sightline()
    assert np.all(sightline["is_cgm"] == False)