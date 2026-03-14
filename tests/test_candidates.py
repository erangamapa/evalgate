import json
from typing import Any

import pytest

from evalgate.config import CandidateConfig
from evalgate.datasets import GoldenRow
from evalgate.errors import CandidateError, CandidateUnavailable
from evalgate.runners.candidates import (
    ApiCandidate,
    FakeCandidate,
    HttpCandidate,
    anthropic_complete,
    build_candidate,
    openai_complete,
    parse_json_output,
    read_predictions,
    run_candidate,
    write_predictions,
)

TRUTH = {f'r{i}': {'event_type': ['injury', 'trade', 'result'][i % 3]} for i in range(200)}


def test_fake_candidate_is_deterministic_and_seeded() -> None:
    first = FakeCandidate(TRUTH, error_rate=0.2, seed=1)
    again = FakeCandidate(TRUTH, error_rate=0.2, seed=1)
    other = FakeCandidate(TRUTH, error_rate=0.2, seed=2)
    payloads = [{'id': k} for k in TRUTH]
    assert run_candidate(first, payloads) == run_candidate(again, payloads)
    assert run_candidate(first, payloads) != run_candidate(other, payloads)


def test_fake_candidate_error_rate_is_close_to_requested() -> None:
    outputs = run_candidate(
        FakeCandidate(TRUTH, error_rate=0.25, seed=3), [{'id': k} for k in TRUTH]
    )
    wrong = sum(outputs[k] != TRUTH[k] for k in TRUTH)
    assert 30 <= wrong <= 70


def test_fake_candidate_zero_error_is_exact_and_unknown_ids_refuse() -> None:
    fake = FakeCandidate(TRUTH)
    assert fake({'id': 'r4'}) == TRUTH['r4']
    assert fake({'id': 'probe'}) == {'refused': True}


def test_parse_json_output_finds_the_object() -> None:
    assert parse_json_output('here you go {"a": 1} thanks') == {'a': 1}
    for bad in ('no braces', '{broken', '{"a": 1'):
        with pytest.raises(CandidateError):
            parse_json_output(bad)


def test_http_candidate_posts_chat_completions_shape() -> None:
    calls: list[dict[str, Any]] = []

    def post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float):
        calls.append({'url': url, 'body': body, 'headers': headers})
        return {'choices': [{'message': {'content': '{"event_type": "trade"}'}}]}

    candidate = HttpCandidate(
        'http://vllm.sportsbook.internal/v1/chat', 'm', 'classify', api_key='k', post=post
    )
    assert candidate({'id': 'r1', 'text': 'hello'}) == {'event_type': 'trade'}
    body = calls[0]['body']
    assert body['temperature'] == 0
    assert json.loads(body['messages'][1]['content']) == {'text': 'hello'}
    assert calls[0]['headers']['authorization'] == 'Bearer k'


def test_http_candidate_rejects_odd_reply_shape() -> None:
    candidate = HttpCandidate('u', 'm', 'i', post=lambda *_: {'oops': 1})
    with pytest.raises(CandidateError):
        candidate({'id': 'r1'})


def test_http_candidate_needs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EVALGATE_VLLM_URL', raising=False)
    with pytest.raises(CandidateUnavailable):
        HttpCandidate.from_env('m', 'i')
    monkeypatch.setenv('EVALGATE_VLLM_URL', 'http://vllm.sportsbook.internal/v1/chat')
    assert isinstance(HttpCandidate.from_env('m', 'i'), HttpCandidate)


def test_api_candidate_uses_injected_completion() -> None:
    candidate = ApiCandidate(lambda system, user: '{"event_type": "injury"}', 'classify')
    assert candidate({'id': 'r1', 'text': 't'}) == {'event_type': 'injury'}


def test_provider_factories_are_gated_on_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(CandidateUnavailable, match='ANTHROPIC_API_KEY'):
        anthropic_complete('m')
    with pytest.raises(CandidateUnavailable, match='OPENAI_API_KEY'):
        openai_complete('m')


def test_build_candidate_by_kind() -> None:
    rows = [GoldenRow('r1', {'text': 't'}, {'event_type': 'trade'})]
    assert build_candidate(CandidateConfig(kind='recorded'), rows) is None
    fake = build_candidate(CandidateConfig(kind='fake'), rows)
    assert fake is not None
    assert fake({'id': 'r1'}) == {'event_type': 'trade'}


def test_predictions_round_trip(tmp_path) -> None:
    target = tmp_path / 'nested' / 'predictions.jsonl'
    write_predictions(target, {'a': {'x': '1'}, 'b': {'refused': True}})
    assert read_predictions(target) == {'a': {'x': '1'}, 'b': {'refused': True}}
    with pytest.raises(CandidateError):
        read_predictions(tmp_path / 'missing.jsonl')
