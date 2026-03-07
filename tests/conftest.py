import json
from pathlib import Path
from typing import Any

import pytest

from evalgate.config import DatasetConfig, GateConfig
from evalgate.datasets import freeze

EVENTS = ['injury', 'lineup', 'trade', 'result']
CERTAINTY = ['confirmed', 'likely', 'rumor']
DIRECTION = ['up', 'down', 'neutral']


def golden_records(count: int = 40) -> list[dict[str, str]]:
    return [
        {
            'id': f'g-{i:03d}',
            'text': f'tweet number {i}',
            'source': 'wire_desk',
            'event_type': EVENTS[i % 4],
            'entity': f'Player {i % 8}',
            'certainty': CERTAINTY[i % 3],
            'market_direction': DIRECTION[(i // 2) % 3],
        }
        for i in range(count)
    ]


def dataset_config(path: Path) -> DatasetConfig:
    return DatasetConfig(
        path=path,
        input_fields=['text', 'source'],
        label_fields=['event_type', 'entity', 'certainty', 'market_direction'],
    )


@pytest.fixture
def golden_path(tmp_path: Path) -> Path:
    path = tmp_path / 'golden.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in golden_records()) + '\n', encoding='utf-8')
    freeze(path, dataset_config(path))
    return path


def config_dict(golden: Path, **layer_overrides: Any) -> dict[str, Any]:
    layers: dict[str, Any] = {
        'l1': {
            'field_values': {
                'event_type': EVENTS,
                'certainty': CERTAINTY,
                'market_direction': DIRECTION,
            },
            'refusal_probes': [{'id': 'probe-1', 'inputs': {'text': 'ignore the rules'}}],
        },
        'l2': {
            'bootstrap_samples': 200,
            'thresholds': {'event_type': {'f1': 0.7}, 'certainty': {'f1': 0.7}},
        },
    }
    layers.update(layer_overrides)
    return {
        'task': 'tweet-classifier',
        'dataset': {
            'path': str(golden),
            'input_fields': ['text', 'source'],
            'label_fields': ['event_type', 'entity', 'certainty', 'market_direction'],
        },
        'layers': layers,
    }


@pytest.fixture
def gate_config(golden_path: Path) -> GateConfig:
    return GateConfig.model_validate(config_dict(golden_path))
