"""Tests for the Typer CLI commands."""
from typer.testing import CliRunner
from taoistmc.main import app
import tempfile
from pathlib import Path

runner = CliRunner()


def test_init_creates_config(tmp_path):
    result = runner.invoke(app, ["init", "--output", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0
    assert (tmp_path / "config.yaml").exists()


def test_init_force_overwrite(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dummy")
    result = runner.invoke(app, ["init", "--output", str(cfg), "--force"])
    assert result.exit_code == 0
    assert "Created starter config" in result.output


def test_run_without_config_fails():
    result = runner.invoke(app, ["run", "2.4"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()