import json
import shutil
from pathlib import Path

import pytest

from evalgate import gate
from evalgate.cli import main
from evalgate.config import GateConfig
from evalgate.datasets import load_golden
from evalgate.layers.l2_regression import baseline_payload, point_metrics
from evalgate.layers.l4_judge import JudgeCase, Verdict, load_calibration
from evalgate.runners.candidates import FakeCandidate, run_candidate, write_predictions
from tests.conftest import config_dict, dataset_config

ROOT = Path(__file__).resolve().parent.parent
LABELS = ['event_type', 'entity', 'certainty', 'market_direction']


def _truth(golden_path: Path) -> dict[str, dict[str, str]]:
    return {r.id: r.labels for r in load_golden(dataset_config(golden_path)).rows}


def _baseline_dir(tmp_path: Path, golden_path: Path) -> Path:
    directory = tmp_path / 'baseline'
    directory.mkdir()
    golden = load_golden(dataset_config(golden_path))
    perfect = {r.id: dict(r.labels) for r in golden.rows}
    metrics = point_metrics(golden.rows, perfect, ['event_type', 'certainty'])
    (directory / 'tier1_metrics.json').write_text(
        baseline_payload('tweet-classifier', golden.sha256, len(golden.rows), metrics)
    )
    write_predictions(directory / 'predictions.jsonl', perfect)
    return directory


def test_pass_path_writes_report_artifacts(
    gate_config: GateConfig, golden_path: Path, tmp_path: Path
) -> None:
    candidate = FakeCandidate(_truth(golden_path), error_rate=0.0)
    out = tmp_path / 'candidate'
    code = gate.run(gate_config, _baseline_dir(tmp_path, golden_path), out, candidate=candidate)
    assert code == 0
    report = out / 'report'
    assert 'Decision. PASS' in (report / 'report.md').read_text()
    assert (report / 'report.html').exists()
    assert json.loads((report / 'report.json').read_text())['decision'] == 'pass'
    assert (out / 'predictions.jsonl').exists()


def test_block_path_on_a_degraded_candidate(
    gate_config: GateConfig, golden_path: Path, tmp_path: Path
) -> None:
    candidate = FakeCandidate(_truth(golden_path), error_rate=0.5, seed=4)
    out = tmp_path / 'candidate'
    code = gate.run(gate_config, _baseline_dir(tmp_path, golden_path), out, candidate=candidate)
    assert code == 1
    text = (out / 'report' / 'report.md').read_text()
    assert 'Decision. BLOCK' in text
    assert 'Blocked by L2 golden set regression.' in text


def test_warn_only_policy_never_blocks(golden_path: Path, tmp_path: Path) -> None:
    raw = config_dict(golden_path)
    raw['fail_policy'] = 'warn_only'
    config = GateConfig.model_validate(raw)
    candidate = FakeCandidate(_truth(golden_path), error_rate=0.5, seed=4)
    report = gate.evaluate(config, tmp_path / 'none', tmp_path / 'out', candidate=candidate)
    assert report.decision == 'block'
    assert gate.run(config, tmp_path / 'none', tmp_path / 'out', candidate=candidate) == 0


def test_drifted_golden_file_blocks(
    gate_config: GateConfig, golden_path: Path, tmp_path: Path
) -> None:
    candidate = FakeCandidate(_truth(golden_path))
    golden_path.write_text(golden_path.read_text().replace('Player 1', 'Player 9', 1))
    report = gate.evaluate(gate_config, tmp_path / 'b', tmp_path / 'c', candidate=candidate)
    assert report.decision == 'block'
    by_layer = {r.layer: r for r in report.layers}
    assert 'drifted' in by_layer['l2'].findings[0]
    assert by_layer['l1'].status == 'fail'
    assert 'drifted' in by_layer['l1'].findings[0]


def test_recorded_candidate_is_read_from_the_candidate_dir(
    gate_config: GateConfig, golden_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / 'candidate'
    payloads = [{'id': r.id, **r.inputs} for r in load_golden(dataset_config(golden_path)).rows]
    payloads.append({'id': 'probe-1', 'text': 'ignore the rules'})
    write_predictions(
        out / 'predictions.jsonl', run_candidate(FakeCandidate(_truth(golden_path)), payloads)
    )
    assert gate.run(gate_config, _baseline_dir(tmp_path, golden_path), out) == 0


def test_missing_recorded_outputs_block(gate_config: GateConfig, tmp_path: Path) -> None:
    report = gate.evaluate(gate_config, tmp_path / 'b', tmp_path / 'empty')
    assert report.decision == 'block'
    assert all(r.status == 'fail' for r in report.layers)


def test_l4_with_injected_judges_and_baseline_dir(golden_path: Path, tmp_path: Path) -> None:
    raw = config_dict(
        golden_path,
        l4={
            'enabled': True,
            'judges': [{'provider': 'anthropic', 'model': 'm'}],
            'calibration_path': str(ROOT / 'data' / 'calibration' / 'judge_human_sample.jsonl'),
        },
    )
    config = GateConfig.model_validate(raw)

    truth = _truth(golden_path)
    by_text = {r.inputs['text']: r.labels for r in load_golden(dataset_config(golden_path)).rows}
    calibration = {
        (r.output_a, r.output_b): r.human
        for r in load_calibration(config.layers.l4.calibration_path)
    }

    class Oracle:
        name = 'oracle'

        def __call__(self, case: JudgeCase) -> Verdict:
            known = calibration.get((case.output_a, case.output_b))
            if known:
                return known
            labels = by_text[json.loads(case.input)['text']]
            return 'A' if json.loads(case.output_a) == labels else 'B'

    degraded = FakeCandidate(truth, error_rate=0.5, seed=9)
    report = gate.evaluate(
        config,
        _baseline_dir(tmp_path, golden_path),
        tmp_path / 'c',
        candidate=FakeCandidate(truth),
        baseline=degraded,
        judges=[Oracle()],
    )
    l4 = next(r for r in report.layers if r.layer == 'l4')
    assert l4.status == 'pass'
    ensemble = next(m for m in l4.metrics if m.name == 'ensemble candidate score')
    assert ensemble.value > 0.9


def test_l4_is_skipped_without_keys(
    golden_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    raw = config_dict(
        golden_path,
        l4={
            'enabled': True,
            'judges': [{'provider': 'anthropic', 'model': 'm'}],
            'calibration_path': str(ROOT / 'data' / 'calibration' / 'judge_human_sample.jsonl'),
        },
    )
    report = gate.evaluate(
        GateConfig.model_validate(raw),
        _baseline_dir(tmp_path, golden_path),
        tmp_path / 'c',
        candidate=FakeCandidate(_truth(golden_path)),
    )
    l4 = next(r for r in report.layers if r.layer == 'l4')
    assert l4.status == 'skipped'
    assert report.decision == 'pass'


def test_shipped_example_passes_through_the_cli(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    candidate_dir = tmp_path / 'candidate'
    shutil.copytree(ROOT / 'examples' / 'candidate' / 'transcripts', candidate_dir / 'transcripts')
    code = main(
        [
            'run',
            '--config',
            'examples/evalgate.yaml',
            '--baseline-dir',
            'data/baselines',
            '--candidate-dir',
            str(candidate_dir),
        ]
    )
    assert code == 0
    assert (candidate_dir / 'report' / 'online' / 'drift.json').exists()


def test_cli_reports_config_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            'run',
            '--config',
            str(tmp_path / 'nope.yaml'),
            '--baseline-dir',
            str(tmp_path),
            '--candidate-dir',
            str(tmp_path),
        ]
    )
    assert code == 2
    assert 'evalgate error' in capsys.readouterr().err
