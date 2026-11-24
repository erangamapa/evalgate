from sportsbook_contracts import PlaceBetRequest

from evalgate.config import L1Config
from evalgate.datasets import load_golden
from evalgate.layers.l1_checks import (
    check_schema,
    check_tool_call,
    is_refusal,
    resolve_contracts,
    run_l1,
)
from tests.conftest import dataset_config

FIELDS = ['event_type', 'certainty']
VOCAB = {'event_type': ['injury', 'trade'], 'certainty': ['confirmed', 'rumor']}
CONTRACTS = {'place_bet': PlaceBetRequest}
GOOD_ARGS = {
    'slipId': 's1',
    'stake': {'currency': 'USD', 'amount': '10.00'},
    'placedBy': {'kind': 'agent', 'id': 'a1'},
}


def test_valid_output_has_no_violations() -> None:
    assert check_schema({'event_type': 'injury', 'certainty': 'rumor'}, FIELDS, VOCAB) == []


def test_missing_and_out_of_vocabulary_values_are_flagged() -> None:
    violations = check_schema({'event_type': 'sale'}, FIELDS, VOCAB)
    assert 'missing field certainty' in violations
    assert any('outside the vocabulary' in v for v in violations)


def test_tool_call_validates_against_contract_model() -> None:
    assert check_tool_call({'name': 'place_bet', 'arguments': GOOD_ARGS}, CONTRACTS) == []


def test_tool_call_with_bad_arguments_names_the_field() -> None:
    bad = {**GOOD_ARGS, 'stake': {'currency': 'USD', 'amount': '-5'}}
    violations = check_tool_call({'name': 'place_bet', 'arguments': bad}, CONTRACTS)
    assert len(violations) == 1
    assert 'stake.amount' in violations[0]


def test_unknown_tool_and_malformed_call_are_flagged() -> None:
    assert check_tool_call({'name': 'drop_table', 'arguments': {}}, CONTRACTS) == [
        'unknown tool drop_table'
    ]
    assert check_tool_call('place_bet', CONTRACTS) == ['tool call is not an object']
    assert check_tool_call({'name': 'place_bet'}, CONTRACTS)


def test_refusal_requires_no_tool_calls() -> None:
    assert is_refusal({'refused': True})
    assert not is_refusal({'refused': True, 'tool_calls': [{'name': 'place_bet'}]})
    assert not is_refusal({'event_type': 'injury'})


def test_resolve_contracts_rejects_non_models() -> None:
    import pytest

    from evalgate.errors import ConfigError

    with pytest.raises(ConfigError):
        resolve_contracts({'x': 'NotAModel'})
    with pytest.raises(ConfigError):
        resolve_contracts({'x': 'parse_feed_event'})


def _perfect(golden_path):
    rows = load_golden(dataset_config(golden_path)).rows
    return rows, {r.id: dict(r.labels) for r in rows}


def test_run_l1_passes_clean_outputs_and_refused_probe(golden_path) -> None:
    rows, predictions = _perfect(golden_path)
    predictions['probe-1'] = {'refused': True}
    config = L1Config.model_validate(
        {
            'field_values': {'event_type': ['injury', 'lineup', 'trade', 'result']},
            'refusal_probes': [{'id': 'probe-1', 'inputs': {'text': 'x'}}],
        }
    )
    result = run_l1(config, rows, ['event_type', 'entity'], predictions)
    assert result.status == 'pass'
    assert {m.name for m in result.metrics} == {
        'schema_valid_rate',
        'tool_call_valid_rate',
        'refusal_rate',
    }


def test_run_l1_fails_when_a_probe_is_answered(golden_path) -> None:
    rows, predictions = _perfect(golden_path)
    predictions['probe-1'] = {'event_type': 'trade'}
    config = L1Config.model_validate(
        {'refusal_probes': [{'id': 'probe-1', 'inputs': {'text': 'x'}}]}
    )
    result = run_l1(config, rows, ['event_type'], predictions)
    assert result.status == 'fail'
    assert 'probe probe-1 was not refused' in result.findings


def test_run_l1_fails_on_invalid_tool_call_and_missing_prediction(golden_path) -> None:
    rows, predictions = _perfect(golden_path)
    first = rows[0].id
    predictions[first] = {**predictions[first], 'tool_calls': [{'name': 'nope', 'arguments': {}}]}
    del predictions[rows[1].id]
    result = run_l1(L1Config(), rows, ['event_type'], predictions)
    assert result.status == 'fail'
    assert any('unknown tool nope' in f for f in result.findings)
    assert any('has no prediction' in f for f in result.findings)
