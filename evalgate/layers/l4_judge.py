import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from jinja2 import Template
from pydantic import BaseModel, ConfigDict
from scipy.stats import binomtest

from evalgate.config import L4Config
from evalgate.datasets import GoldenRow
from evalgate.errors import DatasetError, JudgeError
from evalgate.results import LayerResult, MetricRow
from evalgate.runners.candidates import Complete

Verdict = Literal['A', 'B', 'tie']
CATEGORIES: tuple[Verdict, ...] = ('A', 'B', 'tie')

SYSTEM_PROMPT = (
    'You compare two candidate outputs for the same input. '
    'Reply with one json object, {"verdict": "A"}, {"verdict": "B"} or {"verdict": "tie"}.'
)

USER_TEMPLATE = Template(
    'Criteria\n{{ criteria }}\n\nInput\n{{ input }}\n\nOutput A\n{{ output_a }}\n\n'
    'Output B\n{{ output_b }}\n'
)


@dataclass(frozen=True)
class JudgeCase:
    criteria: str
    input: str
    output_a: str
    output_b: str


class Judge(Protocol):
    name: str

    def __call__(self, case: JudgeCase) -> Verdict: ...


@dataclass(frozen=True)
class JudgeItem:
    id: str
    input: str
    candidate: str
    baseline: str


class CalibrationRow(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    input: str
    output_a: str
    output_b: str
    human: Verdict


def parse_verdict(text: str) -> Verdict:
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        try:
            value = json.loads(text[start : end + 1]).get('verdict')
        except (json.JSONDecodeError, AttributeError):
            value = None
        if value in CATEGORIES:
            return value
    match = re.match(r'\W*(A|B|tie)\b', text.strip(), flags=re.IGNORECASE)
    if match:
        word = match.group(1)
        return 'tie' if word.lower() == 'tie' else word.upper()  # type: ignore[return-value]
    raise JudgeError(f'cannot read a verdict from {text[:60]!r}')


class CompletionJudge:
    def __init__(self, name: str, complete: Complete) -> None:
        self.name = name
        self._complete = complete

    def __call__(self, case: JudgeCase) -> Verdict:
        user = USER_TEMPLATE.render(
            criteria=case.criteria,
            input=case.input,
            output_a=case.output_a,
            output_b=case.output_b,
        )
        return parse_verdict(self._complete(SYSTEM_PROMPT, user))


def render_output(output: Mapping[str, Any]) -> str:
    return json.dumps(output, sort_keys=True)


def _credit(verdict: Verdict, candidate_slot: Literal['A', 'B']) -> float:
    if verdict == 'tie':
        return 0.5
    return 1.0 if verdict == candidate_slot else 0.0


def pairwise_score(judge: Judge, criteria: str, item: JudgeItem) -> float:
    """Candidate credit averaged over both orders, so a position biased judge lands on 0.5."""
    forward = judge(JudgeCase(criteria, item.input, item.candidate, item.baseline))
    swapped = judge(JudgeCase(criteria, item.input, item.baseline, item.candidate))
    return (_credit(forward, 'A') + _credit(swapped, 'B')) / 2


@dataclass(frozen=True)
class JudgeSummary:
    name: str
    items: int
    wins: int
    losses: int
    ties: int
    score: float
    p_worse: float


def summarize(judge: Judge, criteria: str, items: Sequence[JudgeItem]) -> JudgeSummary:
    scores = [pairwise_score(judge, criteria, item) for item in items]
    wins = sum(s > 0.5 for s in scores)
    losses = sum(s < 0.5 for s in scores)
    decisive = wins + losses
    p_worse = (
        float(binomtest(losses, decisive, 0.5, alternative='greater').pvalue) if decisive else 1.0
    )
    return JudgeSummary(
        name=judge.name,
        items=len(items),
        wins=wins,
        losses=losses,
        ties=len(scores) - decisive,
        score=sum(scores) / len(scores),
        p_worse=p_worse,
    )


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError('kappa needs two equal non empty label lists')
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    expected = sum((first.count(c) / total) * (second.count(c) / total) for c in CATEGORIES)
    if expected == 1:
        return 0.0
    return (observed - expected) / (1 - expected)


def load_calibration(path: Path) -> list[CalibrationRow]:
    if not path.exists():
        raise DatasetError(f'calibration sample {path} does not exist')
    lines = path.read_text(encoding='utf-8').splitlines()
    return [CalibrationRow.model_validate_json(line) for line in lines if line.strip()]


def calibrate(judge: Judge, criteria: str, sample: Sequence[CalibrationRow]) -> float:
    verdicts = [judge(JudgeCase(criteria, r.input, r.output_a, r.output_b)) for r in sample]
    return cohen_kappa(verdicts, [r.human for r in sample])


def build_items(
    rows: Sequence[GoldenRow],
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    limit: int | None,
) -> list[JudgeItem]:
    items = []
    for row in rows:
        if row.id in candidate and row.id in baseline:
            items.append(
                JudgeItem(
                    id=row.id,
                    input=json.dumps(row.inputs, sort_keys=True),
                    candidate=render_output(candidate[row.id]),
                    baseline=render_output(baseline[row.id]),
                )
            )
    return items[:limit] if limit else items


def build_judges(config: L4Config, factory: Callable[[str, str], Complete]) -> list[Judge]:
    return [
        CompletionJudge(f'{s.provider}:{s.model}', factory(s.provider, s.model))
        for s in config.judges
    ]


def run_l4(
    config: L4Config,
    judges: Sequence[Judge],
    items: Sequence[JudgeItem],
    calibration: Sequence[CalibrationRow],
) -> LayerResult:
    """Only judges that clear the kappa floor vote. The rest stay in the report as advisory.

    The sign test counts and p_worse are listed for every judge on every run, but they never
    decide the verdict. The ensemble score and the decisive fraction do.
    """
    if not items:
        raise JudgeError('no rows have both a candidate and a baseline output')
    metrics: list[MetricRow] = []
    warnings: list[str] = []
    voting: list[JudgeSummary] = []
    breakdown: list[str] = []
    for judge in judges:
        summary = summarize(judge, config.criteria, items)
        kappa = calibrate(judge, config.criteria, calibration)
        trusted = kappa >= config.min_kappa
        label = summary.name if trusted else f'{summary.name} (advisory)'
        decisive = (summary.wins + summary.losses) / summary.items
        metrics += [
            MetricRow(
                name=f'{label} candidate score',
                value=summary.score,
                threshold=config.min_score,
                passed=summary.score >= config.min_score,
            ),
            MetricRow(
                name=f'{label} kappa',
                value=kappa,
                threshold=config.min_kappa,
                passed=trusted,
            ),
            MetricRow(
                name=f'{label} decisive fraction',
                value=decisive,
                threshold=config.min_decisive,
                passed=decisive >= config.min_decisive,
            ),
            MetricRow(name=f'{label} wins', value=summary.wins, passed=True),
            MetricRow(name=f'{label} losses', value=summary.losses, passed=True),
            MetricRow(name=f'{label} ties', value=summary.ties, passed=True),
            MetricRow(name=f'{label} sign test p worse', value=summary.p_worse, passed=True),
        ]
        breakdown.append(
            f'{label} wins {summary.wins} losses {summary.losses} ties {summary.ties}, '
            f'sign test p {summary.p_worse:.3f} that the candidate is worse'
        )
        if trusted:
            voting.append(summary)
        else:
            warnings.append(
                f'{summary.name} kappa {kappa:.2f} against the human sample is below '
                f'{config.min_kappa:.2f}, excluded from the ensemble and advisory only'
            )

    findings: list[str] = []
    if not voting:
        findings.append(f'no judge reached kappa {config.min_kappa:.2f} on the human sample')
        ensemble = decisive_mean = 0.0
    else:
        ensemble = sum(s.score for s in voting) / len(voting)
        decisive_mean = sum((s.wins + s.losses) / s.items for s in voting) / len(voting)
        metrics.append(
            MetricRow(
                name='ensemble candidate score',
                value=ensemble,
                threshold=config.min_score,
                passed=ensemble >= config.min_score,
            )
        )
        metrics.append(
            MetricRow(
                name='ensemble decisive fraction',
                value=decisive_mean,
                threshold=config.min_decisive,
                passed=decisive_mean >= config.min_decisive,
            )
        )
        if decisive_mean < config.min_decisive:
            findings.append(
                f'only {decisive_mean:.2f} of pairs were decisive in both orders, '
                f'below {config.min_decisive:.2f}, the judges carry no signal'
            )
        if ensemble < config.min_score:
            findings.append(f'ensemble score {ensemble:.3f} is below {config.min_score:.3f}')
    if findings:
        findings += breakdown
    else:
        warnings += breakdown
    return LayerResult(
        layer='l4',
        status='fail' if findings else 'pass',
        required=config.required,
        summary=(
            f'{len(voting)} of {len(judges)} judges voted on {len(items)} pairs, '
            f'both orders, ensemble {ensemble:.3f}'
        ),
        metrics=metrics,
        findings=findings,
        warnings=warnings,
    )
