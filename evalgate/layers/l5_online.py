import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evalgate.config import L2Config, L5Config
from evalgate.layers.l2_regression import BaselineMetrics
from evalgate.results import LayerResult

SCHEMA_VERSION = 1


def build_artifacts(
    task: str,
    config: L5Config,
    l2: L2Config,
    baseline: BaselineMetrics | None,
) -> dict[str, dict[str, Any]]:
    shadow = {
        'schema_version': SCHEMA_VERSION,
        'task': task,
        'sample_rate': config.shadow.sample_rate,
        'compare_fields': sorted(l2.thresholds),
        'max_disagreement': config.shadow.max_disagreement,
    }
    canary = {
        'schema_version': SCHEMA_VERSION,
        'task': task,
        'stages': [
            {'traffic_pct': pct, 'hold_minutes': config.canary.hold_minutes}
            for pct in config.canary.stages_pct
        ],
        'abort_error_rate': config.canary.abort_error_rate,
        'rollback_on_drift': 'block',
    }
    metrics: dict[str, dict[str, float]] = {}
    for field, thresholds in sorted(l2.thresholds.items()):
        for metric, floor in sorted(thresholds.items()):
            known = baseline.metrics.get(field, {}).get(metric) if baseline else None
            reference = known if known is not None else floor
            metrics[f'{field}.{metric}'] = {
                'baseline': round(reference, 4),
                'warn_below': round(reference - config.drift.warn_drop, 4),
                'block_below': round(reference - config.drift.block_drop, 4),
            }
    drift = {
        'schema_version': SCHEMA_VERSION,
        'task': task,
        'metrics': metrics,
        'label_distribution': {
            'psi_warn': config.drift.psi_warn,
            'psi_block': config.drift.psi_block,
        },
    }
    return {'shadow.json': shadow, 'canary.json': canary, 'drift.json': drift}


def write_artifacts(directory: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, body in artifacts.items():
        target = directory / name
        target.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        paths.append(target)
    return paths


def run_l5(
    task: str,
    config: L5Config,
    l2: L2Config,
    baseline: BaselineMetrics | None,
    directory: Path,
) -> LayerResult:
    paths = write_artifacts(directory, build_artifacts(task, config, l2, baseline))
    return LayerResult(
        layer='l5',
        status='pass',
        required=config.required,
        summary=f'wrote {len(paths)} artifacts for llm-ops',
        findings=[],
        warnings=[] if baseline else ['drift thresholds use the l2 floors, no baseline metrics'],
    )
