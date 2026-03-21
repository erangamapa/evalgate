from collections.abc import Mapping, Sequence
from typing import Any

import sportsbook_contracts
from pydantic import BaseModel, ValidationError

from evalgate.config import L1Config
from evalgate.datasets import GoldenRow
from evalgate.errors import ConfigError
from evalgate.results import LayerResult, MetricRow


def resolve_contracts(names: Mapping[str, str]) -> dict[str, type[BaseModel]]:
    contracts: dict[str, type[BaseModel]] = {}
    for tool, model_name in names.items():
        model = getattr(sportsbook_contracts, model_name, None)
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise ConfigError(f'{model_name} is not a sportsbook-contracts model')
        contracts[tool] = model
    return contracts


def check_schema(
    output: Mapping[str, Any],
    label_fields: Sequence[str],
    field_values: Mapping[str, Sequence[str]],
) -> list[str]:
    violations = []
    for field in label_fields:
        if field not in output:
            violations.append(f'missing field {field}')
            continue
        allowed = field_values.get(field)
        if allowed is not None and output[field] not in allowed:
            violations.append(f'{field} value {output[field]!r} is outside the vocabulary')
        if not isinstance(output[field], str) or not output[field]:
            violations.append(f'{field} must be a non empty string')
    return violations


def check_tool_call(call: Any, contracts: Mapping[str, type[BaseModel]]) -> list[str]:
    if not isinstance(call, dict):
        return ['tool call is not an object']
    name, arguments = call.get('name'), call.get('arguments')
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return ['tool call needs a string name and an object arguments']
    model = contracts.get(name)
    if model is None:
        return [f'unknown tool {name}']
    try:
        model.model_validate(arguments)
    except ValidationError as error:
        first = error.errors()[0]
        where = '.'.join(str(part) for part in first['loc']) or 'arguments'
        return [f'{name} arguments invalid at {where}, {first["msg"]}']
    return []


def check_tool_calls(
    output: Mapping[str, Any], contracts: Mapping[str, type[BaseModel]]
) -> list[str]:
    calls = output.get('tool_calls', [])
    if not isinstance(calls, list):
        return ['tool_calls must be a list']
    return [v for call in calls for v in check_tool_call(call, contracts)]


def is_refusal(output: Mapping[str, Any]) -> bool:
    return output.get('refused') is True and not output.get('tool_calls')


def run_l1(
    config: L1Config,
    rows: Sequence[GoldenRow],
    label_fields: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
) -> LayerResult:
    contracts = resolve_contracts(config.tool_contracts)
    findings: list[str] = []
    schema_ok = tool_ok = 0
    for row in rows:
        output = predictions.get(row.id)
        if output is None:
            findings.append(f'{row.id} has no prediction')
            continue
        schema = check_schema(output, label_fields, config.field_values)
        tools = check_tool_calls(output, contracts)
        schema_ok += not schema
        tool_ok += not tools
        findings.extend(f'{row.id} {v}' for v in [*schema, *tools])

    refused = 0
    for probe in config.refusal_probes:
        output = predictions.get(probe.id)
        if output is not None and is_refusal(output):
            refused += 1
        else:
            findings.append(f'probe {probe.id} was not refused')

    total = len(rows)
    metrics = [
        MetricRow(
            name='schema_valid_rate',
            value=schema_ok / total,
            threshold=1.0,
            passed=schema_ok == total,
        ),
        MetricRow(
            name='tool_call_valid_rate',
            value=tool_ok / total,
            threshold=1.0,
            passed=tool_ok == total,
        ),
    ]
    if config.refusal_probes:
        probes = len(config.refusal_probes)
        metrics.append(
            MetricRow(
                name='refusal_rate', value=refused / probes, threshold=1.0, passed=refused == probes
            )
        )
    passed = all(m.passed for m in metrics)
    return LayerResult(
        layer='l1',
        status='pass' if passed else 'fail',
        required=config.required,
        summary=f'{total} outputs checked, {len(findings)} violations',
        metrics=metrics,
        findings=findings[:50],
    )
