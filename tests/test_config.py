from pathlib import Path

import pytest
import yaml

from evalgate.config import GateConfig
from evalgate.errors import ConfigError
from tests.conftest import config_dict

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_example_config_is_valid() -> None:
    config = GateConfig.load(ROOT / 'examples' / 'evalgate.yaml')
    assert config.task == 'tweet-classifier'
    assert config.layers.l4.enabled
    assert config.layers.l2.field_average == {'entity': 'micro'}


def test_load_reads_yaml(tmp_path: Path, golden_path: Path) -> None:
    target = tmp_path / 'evalgate.yaml'
    target.write_text(yaml.safe_dump(config_dict(golden_path)), encoding='utf-8')
    config = GateConfig.load(target)
    assert config.fail_policy == 'block'
    assert config.layers.l3.enabled is False


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        GateConfig.load(tmp_path / 'nope.yaml')


def test_unknown_keys_are_rejected(golden_path: Path) -> None:
    raw = config_dict(golden_path)
    raw['surprise'] = 1
    with pytest.raises(ValueError):
        GateConfig.model_validate(raw)


def test_threshold_on_unknown_field_is_rejected(golden_path: Path) -> None:
    raw = config_dict(golden_path)
    raw['layers']['l2']['thresholds'] = {'colour': {'f1': 0.5}}
    with pytest.raises(ValueError, match='not a label field'):
        GateConfig.model_validate(raw)


def test_enabled_l3_needs_scenarios(golden_path: Path) -> None:
    raw = config_dict(golden_path, l3={'enabled': True})
    with pytest.raises(ValueError, match='scenarios_dir'):
        GateConfig.model_validate(raw)


def test_enabled_l4_needs_judges_and_calibration(golden_path: Path) -> None:
    raw = config_dict(golden_path, l4={'enabled': True})
    with pytest.raises(ValueError, match='judges'):
        GateConfig.model_validate(raw)


def test_canary_stages_must_increase(golden_path: Path) -> None:
    raw = config_dict(golden_path, l5={'enabled': True, 'canary': {'stages_pct': [50, 5]}})
    with pytest.raises(ValueError, match='increasing'):
        GateConfig.model_validate(raw)


def test_api_candidate_needs_provider(golden_path: Path) -> None:
    raw = config_dict(golden_path)
    raw['candidate'] = {'kind': 'api', 'model': 'm'}
    with pytest.raises(ValueError, match='provider'):
        GateConfig.model_validate(raw)


def _l4(golden_path, **extra):
    return config_dict(
        golden_path,
        l4={
            'enabled': True,
            'judges': [{'provider': 'anthropic', 'model': 'm'}],
            'calibration_path': 'cal.jsonl',
            **extra,
        },
    )


def test_required_l4_rejects_a_no_signal_min_score(golden_path) -> None:
    with pytest.raises(ValueError, match='above 0.5'):
        GateConfig.model_validate(_l4(golden_path, min_score=0.5))
    with pytest.raises(ValueError, match='above 0.5'):
        GateConfig.model_validate(_l4(golden_path, min_score=0.45))
    assert GateConfig.model_validate(_l4(golden_path, min_score=0.55))


def test_advisory_l4_may_use_a_loose_min_score(golden_path) -> None:
    assert GateConfig.model_validate(_l4(golden_path, min_score=0.4, required=False))


def test_l1_cannot_be_disabled_or_made_advisory(golden_path) -> None:
    with pytest.raises(ValueError, match='cannot be disabled'):
        GateConfig.model_validate(config_dict(golden_path, l1={'enabled': False}))
    with pytest.raises(ValueError, match='cannot be disabled'):
        GateConfig.model_validate(config_dict(golden_path, l1={'required': False}))
