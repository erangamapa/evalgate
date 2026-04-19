import json
from pathlib import Path

from evalgate.config import GateConfig
from evalgate.layers.l2_regression import BaselineMetrics
from evalgate.layers.l5_online import build_artifacts, run_l5
from tests.conftest import config_dict


def _config(golden_path: Path) -> GateConfig:
    return GateConfig.model_validate(
        config_dict(
            golden_path,
            l5={'enabled': True, 'canary': {'stages_pct': [10, 100], 'hold_minutes': 15}},
        )
    )


def test_artifacts_derive_drift_from_baseline(golden_path: Path) -> None:
    config = _config(golden_path)
    baseline = BaselineMetrics(
        task='t', dataset_sha256='s', rows=1, metrics={'event_type': {'f1': 0.9}}
    )
    artifacts = build_artifacts('t', config.layers.l5, config.layers.l2, baseline)
    drift = artifacts['drift.json']['metrics']
    assert drift['event_type.f1'] == {'baseline': 0.9, 'warn_below': 0.87, 'block_below': 0.82}
    assert drift['certainty.f1']['baseline'] == 0.7
    assert artifacts['canary.json']['stages'][0] == {'traffic_pct': 10, 'hold_minutes': 15}
    assert artifacts['shadow.json']['compare_fields'] == ['certainty', 'event_type']


def test_run_l5_writes_json_files(golden_path: Path, tmp_path: Path) -> None:
    config = _config(golden_path)
    result = run_l5('t', config.layers.l5, config.layers.l2, None, tmp_path / 'online')
    assert result.status == 'pass'
    assert result.warnings
    names = sorted(p.name for p in (tmp_path / 'online').iterdir())
    assert names == ['canary.json', 'drift.json', 'shadow.json']
    assert json.loads((tmp_path / 'online' / 'drift.json').read_text())['schema_version'] == 1
