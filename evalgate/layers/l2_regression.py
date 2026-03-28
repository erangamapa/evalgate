import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from evalgate.config import Average, L2Config, Metric
from evalgate.datasets import GoldenRow
from evalgate.errors import DatasetError
from evalgate.results import LayerResult, MetricRow

MISSING = '__missing__'
METRICS: tuple[Metric, ...] = ('precision', 'recall', 'f1')


class BaselineMetrics(BaseModel):
    model_config = ConfigDict(extra='forbid')

    task: str
    dataset_sha256: str
    rows: int
    metrics: dict[str, dict[str, float]]


def load_baseline(path: Path) -> BaselineMetrics:
    if not path.exists():
        raise DatasetError(f'baseline metrics {path} do not exist')
    return BaselineMetrics.model_validate_json(path.read_text(encoding='utf-8'))


@dataclass(frozen=True)
class FieldScores:
    point: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _scores(
    truth: np.ndarray, pred: np.ndarray, classes: Sequence[int], average: Average
) -> dict[str, np.ndarray]:
    """Metrics for a batch of samples, both inputs have shape (samples, rows)."""
    if average == 'micro':
        accuracy = (truth == pred).mean(axis=1)
        return {'precision': accuracy, 'recall': accuracy, 'f1': accuracy}

    precision, recall, f1, present = [], [], [], []
    for cls in classes:
        is_true, is_pred = truth == cls, pred == cls
        tp = (is_true & is_pred).sum(axis=1).astype(float)
        fp = (~is_true & is_pred).sum(axis=1).astype(float)
        fn = (is_true & ~is_pred).sum(axis=1).astype(float)
        p, r = _safe_divide(tp, tp + fp), _safe_divide(tp, tp + fn)
        precision.append(p)
        recall.append(r)
        f1.append(_safe_divide(2 * p * r, p + r))
        present.append((is_true.sum(axis=1) > 0).astype(float))
    weight = np.stack(present)
    denominator = weight.sum(axis=0)
    return {
        name: (np.stack(values) * weight).sum(axis=0) / denominator
        for name, values in (('precision', precision), ('recall', recall), ('f1', f1))
    }


def bootstrap_indices(rows: int, samples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, rows, size=(samples, rows))


def score_field(
    truth: Sequence[str],
    pred: Sequence[str],
    *,
    average: Average,
    indices: np.ndarray,
    confidence: float,
) -> FieldScores:
    vocabulary = {label: i for i, label in enumerate(sorted({*truth, *pred}))}
    t = np.array([vocabulary[v] for v in truth])
    p = np.array([vocabulary[v] for v in pred])
    classes = sorted({int(v) for v in t})

    point = _scores(t[None, :], p[None, :], classes, average)
    resampled = _scores(t[indices], p[indices], classes, average)
    tail = (1 - confidence) / 2
    return FieldScores(
        point={k: float(v[0]) for k, v in point.items()},
        lower={k: float(np.quantile(v, tail)) for k, v in resampled.items()},
        upper={k: float(np.quantile(v, 1 - tail)) for k, v in resampled.items()},
    )


def point_metrics(
    rows: Sequence[GoldenRow],
    predictions: Mapping[str, Mapping[str, Any]],
    fields: Sequence[str],
    field_average: Mapping[str, Average] | None = None,
) -> dict[str, dict[str, float]]:
    """Point estimates only, used to freeze a baseline file."""
    average = field_average or {}
    result = {}
    for field in fields:
        truth = [row.labels[field] for row in rows]
        pred = [str(predictions.get(row.id, {}).get(field, MISSING)) for row in rows]
        scores = score_field(
            truth,
            pred,
            average=average.get(field, 'macro'),
            indices=np.zeros((1, len(rows)), dtype=int),
            confidence=0.95,
        )
        result[field] = scores.point
    return result


def run_l2(
    config: L2Config,
    rows: Sequence[GoldenRow],
    predictions: Mapping[str, Mapping[str, Any]],
    baseline: BaselineMetrics | None,
    dataset_sha256: str,
) -> LayerResult:
    indices = bootstrap_indices(len(rows), config.bootstrap_samples, config.seed)
    metric_rows: list[MetricRow] = []
    findings: list[str] = []
    warnings: list[str] = []

    if baseline is None:
        warnings.append('no baseline metrics, only absolute thresholds apply')
    elif baseline.dataset_sha256 != dataset_sha256:
        warnings.append('baseline was computed on a different golden file, refresh it')

    for field, thresholds in config.thresholds.items():
        truth = [row.labels[field] for row in rows]
        pred = [str(predictions.get(row.id, {}).get(field, MISSING)) for row in rows]
        scores = score_field(
            truth,
            pred,
            average=config.field_average.get(field, 'macro'),
            indices=indices,
            confidence=config.confidence,
        )
        for metric, absolute in thresholds.items():
            threshold = absolute
            if baseline is not None and field in baseline.metrics:
                floor = baseline.metrics[field].get(metric, 0.0) - config.max_regression
                threshold = max(absolute, floor)
            lower = scores.lower[metric]
            passed = lower >= threshold
            metric_rows.append(
                MetricRow(
                    name=f'{field}.{metric}',
                    value=scores.point[metric],
                    lower=lower,
                    upper=scores.upper[metric],
                    threshold=threshold,
                    passed=passed,
                )
            )
            if not passed:
                findings.append(
                    f'{field}.{metric} lower bound {lower:.3f} is below {threshold:.3f}, '
                    f'point estimate {scores.point[metric]:.3f}'
                )

    failed = len(findings)
    return LayerResult(
        layer='l2',
        status='fail' if failed else 'pass',
        required=config.required,
        summary=(
            f'{len(metric_rows)} metrics on {len(rows)} rows, '
            f'{int(config.confidence * 100)} percent lower bound, {failed} below threshold'
        ),
        metrics=metric_rows,
        findings=findings,
        warnings=warnings,
    )


def baseline_payload(
    task: str, dataset_sha256: str, rows: int, metrics: Mapping[str, Mapping[str, float]]
) -> str:
    rounded = {f: {m: round(v, 4) for m, v in s.items()} for f, s in metrics.items()}
    body = {'task': task, 'dataset_sha256': dataset_sha256, 'rows': rows, 'metrics': rounded}
    return json.dumps(body, indent=2, sort_keys=True) + '\n'
