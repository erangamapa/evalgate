import copy
import json
from pathlib import Path

import pytest

from evalgate.config import L3Config
from evalgate.errors import DatasetError
from evalgate.layers.l3_trajectories import (
    Transcript,
    load_scenario,
    replay,
    run_l3,
)

ROOT = Path(__file__).resolve().parent.parent
SCENARIO = ROOT / 'examples' / 'scenarios' / 'dm_bet_flow.yaml'
TRANSCRIPT = ROOT / 'examples' / 'candidate' / 'transcripts' / 'dm_bet_flow.json'


def _transcript(mutate=None) -> Transcript:
    raw = json.loads(TRANSCRIPT.read_text())
    if mutate:
        mutate(raw)
    return Transcript.model_validate(raw)


def _drop_step(step_id):
    def mutate(raw):
        raw['steps'] = [s for s in raw['steps'] if s['id'] != step_id]

    return mutate


def test_recorded_transcript_passes() -> None:
    assert replay(load_scenario(SCENARIO), _transcript()) == []


def test_missing_gate_stop_blocks() -> None:
    violations = replay(load_scenario(SCENARIO), _transcript(_drop_step('confirmation_gate')))
    assert 'confirmation_gate step is missing from the transcript' in violations
    assert any('without the confirmation_gate gate' in v for v in violations)


def test_gate_step_without_an_interrupt_blocks() -> None:
    def mutate(raw):
        gate = next(s for s in raw['steps'] if s['id'] == 'confirmation_gate')
        gate['events'] = [{'type': 'message', 'text': 'sure thing'}]

    violations = replay(load_scenario(SCENARIO), _transcript(mutate))
    assert 'confirmation_gate must stop at an interrupt gate and did not' in violations


def test_placing_before_the_gate_is_a_forbidden_call() -> None:
    def mutate(raw):
        proposal = raw['steps'][0]
        proposal['events'].append({'type': 'tool_call', 'name': 'place_bet', 'arguments': {}})

    violations = replay(load_scenario(SCENARIO), _transcript(mutate))
    assert 'proposal called forbidden tool place_bet' in violations


def test_diverging_tool_result_is_flagged() -> None:
    def mutate(raw):
        raw['steps'][0]['events'][1]['result'] = {'state': 'confirmed'}

    violations = replay(load_scenario(SCENARIO), _transcript(mutate))
    assert any('differs from the scripted result' in v for v in violations)


def test_missing_escalation_blocks() -> None:
    def mutate(raw):
        raw['steps'][-1]['events'] = []

    assert 'handoff must escalate and did not' in replay(
        load_scenario(SCENARIO), _transcript(mutate)
    )


def test_out_of_order_steps_are_flagged() -> None:
    def mutate(raw):
        raw['steps'].reverse()

    assert 'steps are out of order' in replay(load_scenario(SCENARIO), _transcript(mutate))


def test_stop_expectation(tmp_path: Path) -> None:
    scenario = tmp_path / 's.yaml'
    scenario.write_text('id: s\nsteps:\n  - id: halt\n    expect:\n      stop: true\n')
    loaded = load_scenario(scenario)
    bad = Transcript.model_validate({'scenario': 's', 'steps': [{'id': 'halt', 'events': []}]})
    good = Transcript.model_validate(
        {'scenario': 's', 'steps': [{'id': 'halt', 'events': [{'type': 'stop'}]}]}
    )
    assert replay(loaded, good) == []
    assert replay(loaded, bad) == ['halt must stop and did not']


def test_scenario_gate_reference_must_come_first(tmp_path: Path) -> None:
    path = tmp_path / 's.yaml'
    path.write_text(
        'id: s\nsteps:\n  - id: a\n    expect:\n      requires_gate: b\n'
        '  - id: b\n    expect:\n      interrupt: true\n'
    )
    with pytest.raises(DatasetError):
        load_scenario(path)


def test_run_l3_over_a_directory(tmp_path: Path) -> None:
    config = L3Config(enabled=True, scenarios_dir=SCENARIO.parent)
    passing = run_l3(config, TRANSCRIPT.parent)
    assert passing.status == 'pass'

    empty = run_l3(config, tmp_path)
    assert empty.status == 'fail'
    assert 'dm_bet_flow has no recorded transcript' in empty.findings

    raw = json.loads(TRANSCRIPT.read_text())
    raw['steps'] = copy.deepcopy(raw['steps'][:1])
    (tmp_path / 'dm_bet_flow.json').write_text(json.dumps(raw))
    assert run_l3(config, tmp_path).status == 'fail'


def test_requires_gate_without_a_tool_call_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / 's.yaml'
    path.write_text(
        'id: s\nsteps:\n  - id: gate\n    expect:\n      interrupt: true\n'
        '  - id: after\n    expect:\n      requires_gate: gate\n'
    )
    with pytest.raises(DatasetError, match='without a tool_call'):
        load_scenario(path)
