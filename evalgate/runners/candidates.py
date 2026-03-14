import hashlib
import json
import os
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from evalgate.config import CandidateConfig
from evalgate.datasets import GoldenRow
from evalgate.errors import CandidateError, CandidateUnavailable

Output = dict[str, Any]
Complete = Callable[[str, str], str]
PostJson = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class Candidate(Protocol):
    def __call__(self, payload: Mapping[str, Any]) -> Output: ...


def parse_json_output(text: str) -> Output:
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise CandidateError('model reply has no json object')
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise CandidateError(f'model reply is not valid json, {error}') from error
    if not isinstance(value, dict):
        raise CandidateError('model reply json is not an object')
    return value


def _user_message(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != 'id'}
    return json.dumps(body, sort_keys=True)


class FakeCandidate:
    """Deterministic stand-in. Each label field is corrupted by a seeded hash roll."""

    def __init__(
        self,
        truth: Mapping[str, Mapping[str, str]],
        *,
        error_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._truth = truth
        self._error_rate = error_rate
        self._seed = seed
        self._vocabulary: dict[str, list[str]] = {}
        for labels in truth.values():
            for field, value in labels.items():
                self._vocabulary.setdefault(field, [])
                if value not in self._vocabulary[field]:
                    self._vocabulary[field].append(value)
        for values in self._vocabulary.values():
            values.sort()

    def _roll(self, key: str) -> float:
        digest = hashlib.sha256(f'{self._seed}:{key}'.encode()).digest()
        return int.from_bytes(digest[:8], 'big') / 2**64

    def _corrupt(self, field: str, value: str) -> str:
        vocabulary = self._vocabulary[field]
        if len(vocabulary) < 2:
            return f'not_{value}'
        return vocabulary[(vocabulary.index(value) + 1) % len(vocabulary)]

    def __call__(self, payload: Mapping[str, Any]) -> Output:
        row_id = str(payload['id'])
        labels = self._truth.get(row_id)
        if labels is None:
            return {'refused': True}
        return {
            field: self._corrupt(field, value)
            if self._roll(f'{row_id}:{field}') < self._error_rate
            else value
            for field, value in labels.items()
        }


class HttpCandidate:
    """Chat completions endpoint such as vllm. Gated on EVALGATE_VLLM_URL."""

    def __init__(
        self,
        url: str,
        model: str,
        instructions: str,
        *,
        api_key: str | None = None,
        post: PostJson | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._url = url
        self._model = model
        self._instructions = instructions
        self._headers = {'content-type': 'application/json'}
        if api_key:
            self._headers['authorization'] = f'Bearer {api_key}'
        self._post = post or _urllib_post
        self._timeout = timeout

    @classmethod
    def from_env(cls, model: str, instructions: str) -> 'HttpCandidate':
        url = os.environ.get('EVALGATE_VLLM_URL')
        if not url:
            raise CandidateUnavailable('EVALGATE_VLLM_URL is not set')
        return cls(url, model, instructions, api_key=os.environ.get('EVALGATE_VLLM_API_KEY'))

    def __call__(self, payload: Mapping[str, Any]) -> Output:
        body = {
            'model': self._model,
            'temperature': 0,
            'messages': [
                {'role': 'system', 'content': self._instructions},
                {'role': 'user', 'content': _user_message(payload)},
            ],
        }
        reply = self._post(self._url, body, self._headers, self._timeout)
        try:
            content = reply['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as error:
            raise CandidateError('unexpected chat completions reply shape') from error
        return parse_json_output(content)


def _urllib_post(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method='POST'
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


class ApiCandidate:
    """Paid model behind a complete(system, user) function. Gated on provider keys."""

    def __init__(self, complete: Complete, instructions: str) -> None:
        self._complete = complete
        self._instructions = instructions

    def __call__(self, payload: Mapping[str, Any]) -> Output:
        return parse_json_output(self._complete(self._instructions, _user_message(payload)))


def _require_key(variable: str) -> str:
    key = os.environ.get(variable)
    if not key:
        raise CandidateUnavailable(f'{variable} is not set')
    return key


def anthropic_complete(model: str, max_tokens: int = 512) -> Complete:
    api_key = _require_key('ANTHROPIC_API_KEY')
    try:
        import anthropic
    except ImportError as error:
        raise CandidateUnavailable(
            "install the judge extra, pip install 'evalgate[judge]'"
        ) from error
    client = anthropic.Anthropic(api_key=api_key)

    def complete(system: str, user: str) -> str:
        reply = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{'role': 'user', 'content': user}],
        )
        return ''.join(block.text for block in reply.content if block.type == 'text')

    return complete


def openai_complete(model: str) -> Complete:
    api_key = _require_key('OPENAI_API_KEY')
    try:
        import openai
    except ImportError as error:
        raise CandidateUnavailable(
            "install the judge extra, pip install 'evalgate[judge]'"
        ) from error
    client = openai.OpenAI(api_key=api_key)

    def complete(system: str, user: str) -> str:
        reply = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        )
        return reply.choices[0].message.content or ''

    return complete


def provider_complete(provider: str, model: str) -> Complete:
    if provider == 'anthropic':
        return anthropic_complete(model)
    if provider == 'openai':
        return openai_complete(model)
    raise CandidateError(f'unknown provider {provider}')


def build_candidate(spec: CandidateConfig, rows: Sequence[GoldenRow]) -> Candidate | None:
    """Return None for recorded candidates, their outputs are read from the candidate dir."""
    if spec.kind == 'recorded':
        return None
    if spec.kind == 'fake':
        truth = {row.id: row.labels for row in rows}
        return FakeCandidate(truth, error_rate=spec.error_rate, seed=spec.seed)
    assert spec.model is not None
    if spec.kind == 'http':
        return HttpCandidate.from_env(spec.model, spec.instructions)
    assert spec.provider is not None
    return ApiCandidate(provider_complete(spec.provider, spec.model), spec.instructions)


def run_candidate(candidate: Candidate, payloads: Sequence[Mapping[str, Any]]) -> dict[str, Output]:
    return {str(payload['id']): candidate(payload) for payload in payloads}


def write_predictions(path: Path, predictions: Mapping[str, Output]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({'id': key, 'output': value}, sort_keys=True)
        for key, value in predictions.items()
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def read_predictions(path: Path) -> dict[str, Output]:
    if not path.exists():
        raise CandidateError(f'{path} does not exist')
    predictions: dict[str, Output] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            record = json.loads(line)
            predictions[str(record['id'])] = record['output']
    return predictions
