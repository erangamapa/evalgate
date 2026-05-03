from collections.abc import Callable, Sequence
from pathlib import Path

from evalgate.config import GateConfig
from evalgate.datasets import GoldenSet, load_golden
from evalgate.errors import CandidateUnavailable, EvalgateError
from evalgate.layers import l1_checks, l2_regression, l3_trajectories, l4_judge, l5_online
from evalgate.layers.l4_judge import Judge
from evalgate.report import render_html, render_markdown
from evalgate.results import LayerName, LayerResult, RunReport
from evalgate.runners.candidates import (
    Candidate,
    Complete,
    Output,
    build_candidate,
    provider_complete,
    read_predictions,
    run_candidate,
    write_predictions,
)

PREDICTIONS_FILE = 'predictions.jsonl'
TRANSCRIPTS_DIR = 'transcripts'
REPORT_DIR = 'report'


def _failed(layer: LayerName, required: bool, error: Exception) -> LayerResult:
    return LayerResult(
        layer=layer,
        status='fail',
        required=required,
        summary='layer could not run',
        findings=[str(error)],
    )


def _skipped(layer: LayerName, reason: str) -> LayerResult:
    return LayerResult(layer=layer, status='skipped', required=False, summary=reason)


def _payloads(config: GateConfig, golden: GoldenSet) -> list[dict[str, str]]:
    payloads = [{'id': row.id, **row.inputs} for row in golden.rows]
    payloads += [{'id': p.id, **p.inputs} for p in config.layers.l1.refusal_probes]
    return payloads


def _outputs(
    config: GateConfig, golden: GoldenSet, directory: Path, candidate: Candidate | None
) -> dict[str, Output]:
    if candidate is None:
        return read_predictions(directory / PREDICTIONS_FILE)
    predictions = run_candidate(candidate, _payloads(config, golden))
    write_predictions(directory / PREDICTIONS_FILE, predictions)
    return predictions


def _resolve_judges(
    config: GateConfig, judges: Sequence[Judge] | None, factory: Callable[[str, str], Complete]
) -> Sequence[Judge]:
    if judges is not None:
        return judges
    return l4_judge.build_judges(config.layers.l4, factory)


def evaluate(
    config: GateConfig,
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    candidate: Candidate | None = None,
    baseline: Candidate | None = None,
    judges: Sequence[Judge] | None = None,
    complete_factory: Callable[[str, str], Complete] = provider_complete,
) -> RunReport:
    layers = config.layers
    results: list[LayerResult] = []

    golden: GoldenSet | None = None
    golden_error: EvalgateError | None = None
    try:
        golden = load_golden(config.dataset)
    except EvalgateError as error:
        golden_error = error

    predictions: dict[str, Output] | None = None
    prediction_error: EvalgateError | None = None
    if golden is not None:
        try:
            if candidate is None:
                candidate = build_candidate(config.candidate, golden.rows)
            predictions = _outputs(config, golden, candidate_dir, candidate)
        except EvalgateError as error:
            prediction_error = error

    baseline_metrics = None
    baseline_path = baseline_dir / layers.l2.baseline_file
    try:
        baseline_metrics = l2_regression.load_baseline(baseline_path)
    except EvalgateError:
        baseline_metrics = None

    if golden_error is not None:
        results.append(_failed('l1', True, golden_error))
    elif prediction_error is not None:
        results.append(_failed('l1', layers.l1.required, prediction_error))
    elif golden is not None and predictions is not None:
        try:
            results.append(
                l1_checks.run_l1(layers.l1, golden.rows, config.dataset.label_fields, predictions)
            )
        except EvalgateError as error:
            results.append(_failed('l1', layers.l1.required, error))

    if layers.l2.enabled:
        if golden_error is not None:
            results.append(_failed('l2', layers.l2.required, golden_error))
        elif prediction_error is not None:
            results.append(_failed('l2', layers.l2.required, prediction_error))
        elif golden is not None and predictions is not None:
            results.append(
                l2_regression.run_l2(
                    layers.l2, golden.rows, predictions, baseline_metrics, golden.sha256
                )
            )

    if layers.l3.enabled:
        try:
            results.append(l3_trajectories.run_l3(layers.l3, candidate_dir / TRANSCRIPTS_DIR))
        except EvalgateError as error:
            results.append(_failed('l3', layers.l3.required, error))

    if layers.l4.enabled:
        results.append(
            _run_l4(config, golden, predictions, baseline, baseline_dir, judges, complete_factory)
        )

    if layers.l5.enabled:
        try:
            results.append(
                l5_online.run_l5(
                    config.task,
                    layers.l5,
                    layers.l2,
                    baseline_metrics,
                    candidate_dir / REPORT_DIR / 'online',
                )
            )
        except OSError as error:
            results.append(_failed('l5', layers.l5.required, error))

    results.sort(key=lambda r: r.layer)
    blocking = any(r.status == 'fail' and r.required for r in results)
    decision = 'block' if blocking else 'pass'
    return RunReport(task=config.task, decision=decision, policy=config.fail_policy, layers=results)


def _run_l4(
    config: GateConfig,
    golden: GoldenSet | None,
    predictions: dict[str, Output] | None,
    baseline: Candidate | None,
    baseline_dir: Path,
    judges: Sequence[Judge] | None,
    complete_factory: Callable[[str, str], Complete],
) -> LayerResult:
    l4 = config.layers.l4
    if golden is None or predictions is None:
        return _skipped('l4', 'candidate outputs unavailable')
    try:
        active = _resolve_judges(config, judges, complete_factory)
    except CandidateUnavailable as error:
        return _skipped('l4', f'no judge keys, {error}')
    try:
        assert l4.calibration_path is not None
        calibration = l4_judge.load_calibration(l4.calibration_path)
        if baseline is not None:
            baseline_outputs = run_candidate(
                baseline, [{'id': r.id, **r.inputs} for r in golden.rows]
            )
        else:
            baseline_outputs = read_predictions(baseline_dir / PREDICTIONS_FILE)
        items = l4_judge.build_items(golden.rows, predictions, baseline_outputs, l4.max_items)
        return l4_judge.run_l4(l4, active, items, calibration)
    except EvalgateError as error:
        return _failed('l4', l4.required, error)


def write_report(report: RunReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'report.md').write_text(render_markdown(report), encoding='utf-8')
    (directory / 'report.html').write_text(render_html(report), encoding='utf-8')
    (directory / 'report.json').write_text(
        report.model_dump_json(indent=2) + '\n', encoding='utf-8'
    )


def run(
    config: GateConfig,
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    candidate: Candidate | None = None,
    baseline: Candidate | None = None,
    judges: Sequence[Judge] | None = None,
    complete_factory: Callable[[str, str], Complete] = provider_complete,
) -> int:
    """Run the pyramid and write the report. Returns 0 to pass and 1 to block."""
    report = evaluate(
        config,
        baseline_dir,
        candidate_dir,
        candidate=candidate,
        baseline=baseline,
        judges=judges,
        complete_factory=complete_factory,
    )
    write_report(report, candidate_dir / REPORT_DIR)
    if config.fail_policy == 'warn_only':
        return 0
    return 0 if report.decision == 'pass' else 1
