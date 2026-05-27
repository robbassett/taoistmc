"""Tests for Pydantic configuration models."""
import pytest
from pydantic import ValidationError
from taoistmc.config import PowerLawSegment, SightlineConfig, TaoistConfig


class TestPowerLawSegment:
    def test_valid_segment(self):
        seg = PowerLawSegment(
            log_N_min=12.0, log_N_max=15.0,
            beta=1.5, log_A=9.0, gamma=2.0
        )
        assert seg.log_N_min == 12.0
        assert seg.log_N_max == 15.0

    def test_invalid_range_raises(self):
        with pytest.raises(ValidationError):
            PowerLawSegment(
                log_N_min=15.0, log_N_max=12.0,  # reversed
                beta=1.5, log_A=9.0, gamma=2.0
            )


class TestSightlineConfig:
    def test_continuous_segments(self, sample_powerlaw_segments):
        cfg = SightlineConfig(
            igm_segments=sample_powerlaw_segments["igm"],
            cgm_segments=sample_powerlaw_segments["cgm"]
        )
        assert cfg.use_cgm is True

    def test_discontinuous_segments_raises(self, sample_powerlaw_segments):
        bad = sample_powerlaw_segments["igm"][0]
        bad2 = PowerLawSegment(
            log_N_min=16.0, log_N_max=20.0,
            beta=1.4, log_A=7.0, gamma=1.0
        )
        with pytest.raises(ValidationError):
            SightlineConfig(igm_segments=[bad, bad2], cgm_segments=None, use_cgm=False)

    def test_missing_cgm_segments_raises(self, sample_powerlaw_segments):
        with pytest.raises(ValidationError):
            SightlineConfig(
                igm_segments=sample_powerlaw_segments["igm"],
                cgm_segments=None,
                use_cgm=True
            )


class TestTaoistConfig:
    def test_minimal_config(self, sample_sightline_config):
        cfg = TaoistConfig(sightline_config=sample_sightline_config)
        assert cfg.delta_wav == 1.25  # default
        assert cfg.n_jobs == -1