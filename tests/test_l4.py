import json
from pathlib import Path

import pytest

from evalgate.config import L4Config
from evalgate.errors import JudgeError
from evalgate.layers.l4_judge import (
    CompletionJudge,
    JudgeCase,
    JudgeItem,
    Verdict,
    calibrate,
    cohen_kappa,
    load_calibration,
    pairwise_score,
    parse_verdict,
    run_l4,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = ROOT / 'data' / 'calibration' / 'judge_human_sample.jsonl'


class ScriptedJudge:
    def __init__(self, name: str, decide) -> None:
        self.name = name
        self._decide = decide

    def __call__(self, case: JudgeCase) -> Verdict:
        return self._decide(case)


def prefers(text: str):
    def decide(case: JudgeCase) -> Verdict:
        if text in case.output_a and text not in case.output_b:
            return 'A'
        if text in case.output_b and text not in case.output_a:
            return 'B'
        return 'tie'

    return decide


def always_a(case: JudgeCase) -> Verdict:
    return 'A'


def items(count: int = 10) -> list[JudgeItem]:
    return [JudgeItem(f'i{n}', 'in', 'candidate good', 'baseline bad') for n in range(count)]


def config(**overrides) -> L4Config:
    raw = {
        'enabled': True,
        'judges': [{'provider': 'anthropic', 'model': 'm'}],
        'calibration_path': str(CALIBRATION),
        'min_score': 0.55,
        'min_kappa': 0.6,
    }
    return L4Config.model_validate({**raw, **overrides})


def test_position_swap_evens_a_biased_judge() -> None:
    biased = ScriptedJudge('biased', always_a)
    assert pairwise_score(biased, 'c', items(1)[0]) == 0.5
    summary = summarize(biased, 'c', items())
    assert summary.score == 0.5
    assert summary.ties == 10
    assert summary.p_worse == 1.0


def test_consistent_judge_scores_a_clear_winner() -> None:
    judge = ScriptedJudge('good', prefers('good'))
    summary = summarize(judge, 'c', items())
    assert summary.score == 1.0
    assert summary.wins == 10


def test_consistent_judge_flags_a_clear_loser_with_a_small_p_value() -> None:
    judge = ScriptedJudge('good', prefers('good'))
    swapped = [JudgeItem(f'i{n}', 'in', 'x bad', 'y good') for n in range(12)]
    summary = summarize(judge, 'c', swapped)
    assert summary.score == 0.0
    assert summary.losses == 12
    assert summary.p_worse < 0.001


def test_kappa_on_known_labels() -> None:
    assert cohen_kappa(['A', 'B', 'tie', 'A'], ['A', 'B', 'tie', 'A']) == 1.0
    # observed 0.5, expected 0.5, so kappa is zero
    assert cohen_kappa(['A', 'A', 'B', 'B'], ['A', 'B', 'A', 'B']) == 0.0
    # degenerate marginals carry no evidence of agreement, so they never read as trust
    assert cohen_kappa(['A', 'A'], ['A', 'A']) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa(['A'], [])


def test_calibration_sample_has_forty_rows_with_all_verdicts() -> None:
    sample = load_calibration(CALIBRATION)
    assert len(sample) == 40
    assert len({r.input for r in sample}) == 40
    assert {r.human for r in sample} == {'A', 'B', 'tie'}


def human_judge() -> ScriptedJudge:
    lookup = {(r.output_a, r.output_b): r.human for r in load_calibration(CALIBRATION)}
    return ScriptedJudge('human-like', lambda case: lookup[(case.output_a, case.output_b)])


def test_judge_matching_humans_gets_kappa_one() -> None:
    assert calibrate(human_judge(), 'c', load_calibration(CALIBRATION)) == 1.0


def test_biased_judge_has_low_kappa_on_the_sample() -> None:
    kappa = calibrate(ScriptedJudge('b', always_a), 'c', load_calibration(CALIBRATION))
    assert kappa == 0.0


def calibrated(name: str, decide) -> ScriptedJudge:
    """Agrees with the humans on the calibration sample, uses decide everywhere else."""
    lookup = {(r.output_a, r.output_b): r.human for r in load_calibration(CALIBRATION)}
    return ScriptedJudge(
        name, lambda case: lookup.get((case.output_a, case.output_b)) or decide(case)
    )


def test_low_kappa_judge_is_excluded_and_marked_advisory() -> None:
    sample = load_calibration(CALIBRATION)
    trusted = calibrated('trusted', prefers('good'))
    biased = ScriptedJudge('biased', prefers('bad'))
    result = run_l4(config(), [trusted, biased], items(), sample)
    ensemble = next(m for m in result.metrics if m.name == 'ensemble candidate score')
    assert result.status == 'pass'
    assert ensemble.value == 1.0
    assert any(
        m.name == 'biased (advisory) candidate score' and m.value == 0.0 for m in result.metrics
    )
    assert any('biased kappa' in w and 'excluded' in w for w in result.warnings)


def test_untrusted_judge_would_have_dragged_the_ensemble_under_the_line() -> None:
    sample = load_calibration(CALIBRATION)
    trusted = calibrated('trusted', prefers('good'))
    biased = ScriptedJudge('biased', prefers('bad'))
    assert config().min_score > (1.0 + 0.0) / 2
    assert run_l4(config(), [trusted, biased], items(), sample).status == 'pass'


def test_layer_fails_when_no_judge_clears_calibration() -> None:
    sample = load_calibration(CALIBRATION)
    judges = [ScriptedJudge('one', prefers('good')), ScriptedJudge('two', always_a)]
    result = run_l4(config(), judges, items(), sample)
    assert result.status == 'fail'
    assert 'no judge reached kappa' in result.findings[0]


def test_fully_biased_judge_cannot_pass_a_required_layer() -> None:
    sample = load_calibration(CALIBRATION)
    judge = calibrated('biased', always_a)
    result = run_l4(config(), [judge], items(), sample)
    assert result.status == 'fail'
    assert any('carry no signal' in f for f in result.findings)
    assert any(
        m.name == 'ensemble candidate score' and m.value == 0.5 and not m.passed
        for m in result.metrics
    )


def test_decisive_floor_is_configurable() -> None:
    sample = load_calibration(CALIBRATION)
    mixed = [JudgeItem(f'g{n}', 'in', 'candidate good', 'baseline bad') for n in range(2)]
    mixed += [JudgeItem(f'n{n}', 'in', 'same', 'same') for n in range(8)]
    judge = calibrated('j', prefers('good'))
    assert run_l4(config(min_decisive=0.2), [judge], mixed, sample).status == 'pass'
    assert run_l4(config(min_decisive=0.5), [judge], mixed, sample).status == 'fail'


def test_sign_test_detail_is_listed_even_when_the_layer_passes() -> None:
    sample = load_calibration(CALIBRATION)
    result = run_l4(config(), [calibrated('j', prefers('good'))], items(), sample)
    assert result.status == 'pass'
    names = {m.name: m.value for m in result.metrics}
    assert names['j wins'] == 10
    assert names['j losses'] == 0
    assert names['j ties'] == 0
    assert names['j sign test p worse'] == 1.0
    assert any('sign test p' in w for w in result.warnings)


def test_run_l4_fails_when_the_ensemble_prefers_the_baseline() -> None:
    sample = load_calibration(CALIBRATION)
    judges = [calibrated('one', prefers('bad')), calibrated('two', prefers('bad'))]
    result = run_l4(config(), judges, items(), sample)
    assert result.status == 'fail'
    assert result.findings[0].startswith('ensemble score 0.000')
    assert any('sign test p' in f for f in result.findings)


def test_run_l4_with_no_pairs_is_an_error() -> None:
    with pytest.raises(JudgeError):
        run_l4(config(), [ScriptedJudge('j', always_a)], [], load_calibration(CALIBRATION))


@pytest.mark.parametrize(
    ('reply', 'expected'),
    [
        ('{"verdict": "A"}', 'A'),
        ('Sure. {"verdict": "tie"} done', 'tie'),
        ('B, because it is closer', 'B'),
        ('Tie.', 'tie'),
    ],
)
def test_parse_verdict(reply: str, expected: str) -> None:
    assert parse_verdict(reply) == expected


def test_parse_verdict_rejects_noise() -> None:
    with pytest.raises(JudgeError):
        parse_verdict('I cannot decide')


def test_completion_judge_renders_prompt_and_parses_reply() -> None:
    seen: list[tuple[str, str]] = []

    def complete(system: str, user: str) -> str:
        seen.append((system, user))
        return json.dumps({'verdict': 'B'})

    judge = CompletionJudge('fake', complete)
    verdict = judge(JudgeCase('be right', 'the input', 'first', 'second'))
    assert verdict == 'B'
    assert 'be right' in seen[0][1]
    assert 'Output A\nfirst' in seen[0][1]
    assert 'Output B\nsecond' in seen[0][1]
