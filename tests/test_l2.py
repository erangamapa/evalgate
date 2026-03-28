import numpy as np
import pytest

from evalgate.config import L2Config
from evalgate.datasets import GoldenRow
from evalgate.layers.l2_regression import (
    BaselineMetrics,
    bootstrap_indices,
    point_metrics,
    run_l2,
    score_field,
)


def _score(truth, pred, average='macro', samples=500, seed=1, confidence=0.95):
    indices = bootstrap_indices(len(truth), samples, seed)
    return score_field(truth, pred, average=average, indices=indices, confidence=confidence)


def _rows(labels):
    return [GoldenRow(id=f'r{i}', inputs={}, labels={'f': v}) for i, v in enumerate(labels)]


def test_macro_metrics_on_known_confusion() -> None:
    truth = ['a', 'a', 'a', 'b', 'b', 'c']
    pred = ['a', 'a', 'b', 'b', 'c', 'c']
    point = _score(truth, pred).point
    # class a precision 1, recall 2/3. class b precision 1/2, recall 1/2. class c 1/2 and 1.
    assert point['precision'] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert point['recall'] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert point['f1'] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_micro_average_is_accuracy() -> None:
    point = _score(['a', 'b', 'a', 'b'], ['a', 'b', 'b', 'b'], average='micro').point
    assert point == {'precision': 0.75, 'recall': 0.75, 'f1': 0.75}


def test_perfect_predictions_have_a_degenerate_interval() -> None:
    scores = _score(['a', 'b'] * 10, ['a', 'b'] * 10)
    assert scores.lower['f1'] == scores.upper['f1'] == 1.0


def test_interval_brackets_the_point_estimate() -> None:
    truth = ['a', 'b'] * 25
    pred = list(truth)
    for i in range(0, 50, 5):
        pred[i] = 'b' if truth[i] == 'a' else 'a'
    scores = _score(truth, pred, average='micro')
    assert scores.point['f1'] == pytest.approx(0.8)
    assert 0.6 < scores.lower['f1'] < 0.8 < scores.upper['f1'] < 0.95


def test_bootstrap_is_seeded() -> None:
    truth = ['a', 'b', 'c'] * 10
    pred = ['a', 'b', 'a'] * 10
    first = _score(truth, pred, seed=5)
    again = _score(truth, pred, seed=5)
    other = _score(truth, pred, seed=6)
    assert first == again
    assert first.lower != other.lower
    assert np.array_equal(bootstrap_indices(30, 10, 5), bootstrap_indices(30, 10, 5))


def test_missing_prediction_counts_as_wrong() -> None:
    rows = _rows(['a', 'b'])
    metrics = point_metrics(rows, {'r0': {'f': 'a'}}, ['f'], {'f': 'micro'})
    assert metrics['f']['f1'] == 0.5


def _config(**overrides) -> L2Config:
    base = {
        'bootstrap_samples': 400,
        'thresholds': {'f': {'f1': 0.85}},
        'field_average': {'f': 'micro'},
    }
    return L2Config.model_validate({**base, **overrides})


def test_lucky_run_is_blocked_by_the_lower_bound() -> None:
    labels = ['a', 'b'] * 10
    rows = _rows(labels)
    predictions = {r.id: {'f': r.labels['f']} for r in rows}
    predictions['r0'] = {'f': 'b'}
    predictions['r1'] = {'f': 'a'}
    result = run_l2(_config(), rows, predictions, None, 'sha')
    metric = result.metrics[0]
    assert metric.value == 0.9
    assert metric.value >= metric.threshold
    assert metric.lower < metric.threshold
    assert result.status == 'fail'
    assert 'lower bound' in result.findings[0]


def test_same_quality_on_more_rows_passes() -> None:
    rows = _rows(['a', 'b'] * 200)
    predictions = {r.id: {'f': r.labels['f']} for r in rows}
    for i in range(0, 80, 2):
        predictions[f'r{i}'] = {'f': 'b'}
    result = run_l2(_config(thresholds={'f': {'f1': 0.8}}), rows, predictions, None, 'sha')
    assert result.metrics[0].value == pytest.approx(0.9)
    assert result.status == 'pass'


def test_baseline_regression_raises_the_effective_threshold() -> None:
    rows = _rows(['a', 'b'] * 200)
    predictions = {r.id: {'f': r.labels['f']} for r in rows}
    for i in range(0, 80, 2):
        predictions[f'r{i}'] = {'f': 'b'}
    baseline = BaselineMetrics(
        task='t', dataset_sha256='sha', rows=400, metrics={'f': {'f1': 0.99}}
    )
    result = run_l2(
        _config(thresholds={'f': {'f1': 0.5}}, max_regression=0.02),
        rows,
        predictions,
        baseline,
        'sha',
    )
    assert result.metrics[0].threshold == pytest.approx(0.97)
    assert result.status == 'fail'


def test_stale_baseline_warns() -> None:
    rows = _rows(['a', 'b'] * 50)
    predictions = {r.id: {'f': r.labels['f']} for r in rows}
    baseline = BaselineMetrics(task='t', dataset_sha256='old', rows=100, metrics={})
    result = run_l2(_config(), rows, predictions, baseline, 'new')
    assert result.status == 'pass'
    assert any('different golden file' in w for w in result.warnings)
