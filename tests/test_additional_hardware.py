from __future__ import annotations

from pathlib import Path

from checkrcq_eval.additional_hardware.cli import main
from checkrcq_eval.additional_hardware.config import load_supplemental_hardware_config, resolve_config_path


def test_load_resume_spotcheck_config() -> None:
    path = resolve_config_path("resume_spotcheck.yaml")
    config = load_supplemental_hardware_config(path)
    assert config.experiment_name == "resume_spotcheck"
    assert config.results_subdir == "resume_spotcheck"
    assert len(config.cases) == 4
    assert all(case.family == "resume_spotcheck" for case in config.cases)


def test_supplemental_cli_dry_run(capsys) -> None:
    exit_code = main(
        [
            "run",
            "--config",
            "additional_hardware_experiments/configs/migration_matrix.yaml",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "supplemental_hardware/migration_matrix" in captured.out
    assert "processed_csv:" in captured.out

