"""Shared pytest fixtures for taoistmc tests."""
import numpy as np
import pytest
from taoistmc.config import PowerLawSegment, SightlineConfig, TaoistConfig


@pytest.fixture
def sample_powerlaw_segments():
    """Return a set of valid PowerLawSegment objects for IGM and CGM."""
    igm_low = PowerLawSegment(
        log_N_min=12.0, log_N_max=15.2,
        beta=1.635, log_A=9.305, gamma=2.5
    )
    igm_high = PowerLawSegment(
        log_N_min=15.2, log_N_max=21.0,
        beta=1.463, log_A=7.342, gamma=1.0
    )
    cgm_low = PowerLawSegment(
        log_N_min=12.0, log_N_max=13.0,
        beta=1.635, log_A=9.305, gamma=2.5
    )
    cgm_high = PowerLawSegment(
        log_N_min=13.0, log_N_max=21.0,
        beta=1.381, log_A=6.716, gamma=1.0
    )
    return {
        "igm": [igm_low, igm_high],
        "cgm": [cgm_low, cgm_high]
    }


@pytest.fixture
def sample_sightline_config(sample_powerlaw_segments):
    """Return a valid SightlineConfig."""
    return SightlineConfig(
        igm_segments=sample_powerlaw_segments["igm"],
        cgm_segments=sample_powerlaw_segments["cgm"],
        use_cgm=True
    )


@pytest.fixture
def sample_taoist_config(sample_sightline_config):
    """Return a valid TaoistConfig with minimal settings for fast tests."""
    return TaoistConfig(
        sightline_config=sample_sightline_config,
        delta_wav=1.0,
        rest_wav_min=900,
        rest_wav_max=1200,
        n_jobs=1,
        verbose=False,
        save=False
    )


@pytest.fixture
def empty_sightline():
    """Return an empty structured array matching sightline dtype."""
    dtype = [("z", "f8"), ("logNHI", "f8"), ("is_cgm", "?")]
    return np.array([], dtype=dtype)