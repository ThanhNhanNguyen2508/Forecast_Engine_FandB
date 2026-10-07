"""Checkout-local contracts: no outer engine paths, credentials or historical outputs."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from shelfcash_forecast.registry.loader import load_artifacts
from shelfcash_pipeline import context
from shelfcash_pipeline.config_runner import (
    RunConfiguration, configuration_root, configured_options, load_configuration, main,
)
from shelfcash_pipeline.what_if_runner import load_what_if_configuration, WhatIfConfiguration
from shelfcash_preprocess.config import project_root

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    return json.loads((ROOT / "shelfcash.config.json").read_text(encoding="utf-8-sig"))


def portable_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    for name in ("shelfcash_pipeline", "shelfcash_forecast", "shelfcash_preprocess", "scripts", "tests", ".git", "demo/input", "demo/artifacts"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_repository_defaults_and_preprocess_root(monkeypatch):
    monkeypatch.delenv("SHELFCASH_ENGINE_ROOT", raising=False)
    assert context.REPOSITORY_LAYOUT
    assert context.ENGINE_ROOT == ROOT
    assert project_root() == ROOT
    assert context.DEFAULT_INPUT == ROOT / "demo/input"
    assert context.DEFAULT_ARTIFACTS == ROOT / "demo/artifacts"
    assert context.default_output_path("m6") == ROOT / "codex_tests/runs/pipeline_until_m6"
    assert configuration_root(ROOT / "configs/example.json") == ROOT


def test_explicit_legacy_configuration_root(tmp_path):
    (tmp_path / "source_code/shelfcash_pipeline").mkdir(parents=True)
    assert configuration_root(tmp_path / "shelfcash.config.json") == tmp_path


def test_all_fixed_assets_and_model_loader_are_complete():
    manifest = json.loads((ROOT / "demo/ASSETS_MANIFEST.json").read_text(encoding="utf-8-sig"))
    assert len(manifest["files"]) == 27
    for row in manifest["files"]:
        path = ROOT / row["path"]
        assert path.is_relative_to(ROOT / "demo")
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    artifacts = load_artifacts(ROOT / "demo/artifacts")
    assert artifacts.warnings == ()
    assert len(artifacts.model_bundle.models) == 3


@pytest.mark.parametrize("stage", context.MILESTONES)
def test_all_stops_resolve_inside_checkout_without_precreating_output(tmp_path, raw, stage):
    root = portable_root(tmp_path)
    config = RunConfiguration.model_validate(raw)
    options = configured_options(config, stage, engine_root=root)
    assert options.workspace_root == root
    assert options.input_path == root / "demo/input"
    assert options.artifacts_path == root / "demo/artifacts"
    assert options.output_dir == root / "outputs" / ("pipeline_until_" + stage)
    assert not options.output_dir.exists()
    assert options.planning_config.is_relative_to(root / ".runtime/configs")
    assert json.loads(options.planning_config.read_text(encoding="utf-8")) == config.planning.model_dump(mode="json")


@pytest.mark.parametrize("protected", ["shelfcash_pipeline", "shelfcash_forecast", "shelfcash_preprocess", "scripts", "tests", ".git", "demo/input", "demo/artifacts"])
def test_output_cannot_replace_checkout_source_or_assets(tmp_path, raw, protected):
    root = portable_root(tmp_path)
    options = configured_options(RunConfiguration.model_validate(raw), "m1", engine_root=root)
    from dataclasses import replace
    with pytest.raises(ValueError, match="outside source/input/artifacts"):
        context.reserve_output(replace(options, output_dir=root / protected / "bad-output"))


def test_owned_output_is_allowed_and_foreign_output_is_preserved(tmp_path, raw):
    root = portable_root(tmp_path)
    options = configured_options(RunConfiguration.model_validate(raw), "m1", engine_root=root)
    output = context.reserve_output(options)
    assert (output / context.OWNER_FILE).is_file()
    context.release_output(output)
    foreign = root / "outputs/foreign"
    foreign.mkdir()
    marker = foreign / "keep.txt"
    marker.write_text("user content", encoding="utf-8")
    from dataclasses import replace
    with pytest.raises(FileExistsError, match="not owned"):
        context.reserve_output(replace(options, output_dir=foreign))
    assert marker.read_text(encoding="utf-8") == "user content"


def test_budget_only_edit_and_stale_behavior_rejection(raw):
    previous = copy.deepcopy(raw)
    raw["planning"]["budget"] = 10000000
    config = RunConfiguration.model_validate(raw)
    assert config.planning.scenario_profile.configuration_binding_hash == previous["planning"]["scenario_profile"]["configuration_binding_hash"]
    raw["planning"]["cost_policy"]["holding_cost_rate_per_day"] = 0.002
    with pytest.raises(ValueError, match="STALE_PROFILE_CONFIGURATION_BINDING"):
        RunConfiguration.model_validate(raw)


@pytest.mark.parametrize("section,field,value", [
    ("pipeline", "scenario_count", True), ("pipeline", "seed", True),
    ("pipeline", "cutoff_date", "12/08/2026"), ("pipeline", "output_prefix", "../outside"),
    ("pipeline", "unknown", 1), ("planning", "budget", -1),
    ("planning", "budget", True), ("pipeline", "execution_mode", "production"),
])
def test_invalid_settings_rejected(raw, section, field, value):
    raw[section][field] = value
    with pytest.raises(ValueError):
        RunConfiguration.model_validate(raw)


def test_validate_only_accepts_bom_and_writes_nothing(tmp_path, raw):
    config = tmp_path / "settings.json"
    config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8-sig")
    assert load_configuration(config).pipeline.cutoff_date == date(2026, 8, 12)
    assert main(["--config", str(config), "--validate-only"]) == 0
    assert list(tmp_path.iterdir()) == [config]


@pytest.mark.parametrize("what_if", [False, True])
def test_powershell_wrapper_from_other_directory_preserves_environment(tmp_path, what_if):
    if os.name != "nt":
        # The documented one-command wrapper targets Windows; Python APIs are portable.
        return
    ps = shutil.which("powershell.exe")
    assert ps is not None
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"
    script = "$env:PYTHONPATH='kept-path'; $env:SHELFCASH_ENGINE_ROOT='kept-root'; & " + quote(ROOT / "run.ps1") + " -Python " + quote(sys.executable) + " -ValidateOnly"
    if what_if:
        script += " -WhatIf"
    script += "; if($env:PYTHONPATH -ne 'kept-path' -or $env:SHELFCASH_ENGINE_ROOT -ne 'kept-root'){throw 'environment changed'}"
    result = subprocess.run([ps, "-NoProfile", "-Command", script], cwd=tmp_path,
                            capture_output=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "VALID"' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_what_if_example_has_explicit_demo_scope_and_budget():
    config = load_what_if_configuration(ROOT / "configs/what_if_budget.example.json")
    assert config.modifications[0].budget == 5000000
    assert config.label == "DEMO_ONLY_NOT_FOR_OPERATION"


def test_powershell_entrypoints_parse_without_running_setup():
    if os.name != "nt":
        return
    paths = [ROOT / "run.ps1", ROOT / "scripts/setup.ps1", ROOT / "shelfcash_pipeline/run.ps1",
             ROOT / "scripts/setup_preprocess.ps1", ROOT / "scripts/check_preprocess_api.ps1"]
    arguments = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    script = "foreach($Path in @(" + arguments + ")) {$Tokens=$null; $Errors=$null; [System.Management.Automation.Language.Parser]::ParseFile($Path,[ref]$Tokens,[ref]$Errors) | Out-Null; if($Errors.Count){throw ($Errors | Out-String)}}"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True,
                            encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mutation", [
    {"label": "PRODUCTION"}, {"unknown": 1}, {"modifications": []},
    {"modifications": [{"modification_type": "BUDGET", "budget": -1}]},
    {"modifications": [{"modification_type": "DEMAND_SCALE", "selector": {}, "multiplier": 1.1}]},
])
def test_what_if_invalid_config_rejected(mutation):
    raw = json.loads((ROOT / "configs/what_if_budget.example.json").read_text(encoding="utf-8-sig"))
    raw.update(mutation)
    with pytest.raises(ValueError):
        WhatIfConfiguration.model_validate(raw)
