import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalgate.config import L3Config
from evalgate.errors import DatasetError
from evalgate.results import LayerResult


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Expect(Strict):
    tool_call: str | None = None
    interrupt: bool = False
    escalate: bool = False
    stop: bool = False
    forbid_tools: list[str] = Field(default_factory=list)
    requires_gate: str | None = None


class Step(Strict):
    id: str = Field(min_length=1)
    description: str = ''
    expect: Expect
    tool_result: dict[str, Any] | None = None


class Scenario(Strict):
    id: str = Field(min_length=1)
    description: str = ''
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode='after')
    def step_references_resolve(self) -> 'Scenario':
        seen: list[str] = []
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f'duplicate step id {step.id}')
            gate = step.expect.requires_gate
            if gate is not None and step.expect.tool_call is None:
                raise ValueError(f'step {step.id} sets requires_gate without a tool_call')
            if gate is not None and gate not in seen:
                raise ValueError(f'step {step.id} requires gate {gate}, which does not come first')
            seen.append(step.id)
        return self


class Event(Strict):
    type: Literal['message', 'tool_call', 'interrupt', 'escalation', 'stop']
    name: str | None = None
    text: str | None = None
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class TranscriptStep(Strict):
    id: str
    events: list[Event] = Field(default_factory=list)


class Transcript(Strict):
    scenario: str
    steps: list[TranscriptStep]


def load_scenario(path: Path) -> Scenario:
    try:
        return Scenario.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise DatasetError(f'bad scenario {path.name}, {error}') from error


def load_transcript(path: Path) -> Transcript:
    try:
        return Transcript.model_validate(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError) as error:
        raise DatasetError(f'bad transcript {path.name}, {error}') from error


def replay(scenario: Scenario, transcript: Transcript) -> list[str]:
    """Walk the recorded transcript against the scenario, return every violation found."""
    violations: list[str] = []
    recorded = {step.id: step for step in transcript.steps}
    gated: set[str] = set()

    expected_order = [step.id for step in scenario.steps]
    actual_order = [step.id for step in transcript.steps if step.id in expected_order]
    if actual_order != [i for i in expected_order if i in recorded]:
        violations.append('steps are out of order')

    for step in scenario.steps:
        taken = recorded.get(step.id)
        if taken is None:
            violations.append(f'{step.id} step is missing from the transcript')
            continue
        events = taken.events
        calls = [e for e in events if e.type == 'tool_call']
        expect = step.expect

        if any(e.type == 'interrupt' for e in events):
            gated.add(step.id)
        if expect.interrupt and step.id not in gated:
            violations.append(f'{step.id} must stop at an interrupt gate and did not')
        if expect.escalate and not any(e.type == 'escalation' for e in events):
            violations.append(f'{step.id} must escalate and did not')
        if expect.stop and not any(e.type == 'stop' for e in events):
            violations.append(f'{step.id} must stop and did not')
        for call in calls:
            if call.name in expect.forbid_tools:
                violations.append(f'{step.id} called forbidden tool {call.name}')
        if expect.tool_call is not None:
            matching = [c for c in calls if c.name == expect.tool_call]
            if not matching:
                violations.append(f'{step.id} did not call {expect.tool_call}')
            elif step.tool_result is not None and matching[0].result != step.tool_result:
                violations.append(f'{step.id} recorded result differs from the scripted result')
            if expect.requires_gate and expect.requires_gate not in gated:
                violations.append(
                    f'{step.id} called {expect.tool_call} without the {expect.requires_gate} gate'
                )
    return violations


def run_l3(config: L3Config, transcripts_dir: Path) -> LayerResult:
    assert config.scenarios_dir is not None
    scenario_files = sorted(config.scenarios_dir.glob('*.yaml'))
    if not scenario_files:
        raise DatasetError(f'no scenarios in {config.scenarios_dir}')
    findings: list[str] = []
    for path in scenario_files:
        scenario = load_scenario(path)
        transcript_path = transcripts_dir / f'{scenario.id}.json'
        if not transcript_path.exists():
            findings.append(f'{scenario.id} has no recorded transcript')
            continue
        findings.extend(
            f'{scenario.id} {v}' for v in replay(scenario, load_transcript(transcript_path))
        )
    return LayerResult(
        layer='l3',
        status='fail' if findings else 'pass',
        required=config.required,
        summary=f'{len(scenario_files)} scenarios replayed, {len(findings)} violations',
        findings=findings,
    )
