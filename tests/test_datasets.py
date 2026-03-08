import json
from pathlib import Path

import pytest

from evalgate.datasets import GoldenManifest, freeze, load_golden, manifest_path
from evalgate.errors import DatasetError, GoldenDriftError
from tests.conftest import dataset_config

ROOT = Path(__file__).resolve().parent.parent


def test_load_returns_rows_and_digest(golden_path: Path) -> None:
    golden = load_golden(dataset_config(golden_path))
    assert len(golden.rows) == 40
    assert golden.rows[0].labels['event_type'] == 'injury'
    assert golden.rows[0].inputs == {'text': 'tweet number 0', 'source': 'wire_desk'}
    manifest = GoldenManifest.model_validate_json(manifest_path(golden_path).read_text())
    assert golden.sha256 == manifest.sha256


def test_drifted_file_is_refused(golden_path: Path) -> None:
    with golden_path.open('a', encoding='utf-8') as handle:
        handle.write('\n')
    with pytest.raises(GoldenDriftError, match='drifted'):
        load_golden(dataset_config(golden_path))


def test_edited_label_is_refused(golden_path: Path) -> None:
    text = golden_path.read_text().replace('"injury"', '"trade"', 1)
    golden_path.write_text(text)
    with pytest.raises(GoldenDriftError):
        load_golden(dataset_config(golden_path))


def test_missing_manifest_is_refused(golden_path: Path) -> None:
    manifest_path(golden_path).unlink()
    with pytest.raises(GoldenDriftError, match='no manifest'):
        load_golden(dataset_config(golden_path))


def test_freeze_cannot_overwrite_a_different_hash(golden_path: Path) -> None:
    golden_path.write_text(golden_path.read_text() + json.dumps({'id': 'x'}) + '\n')
    with pytest.raises(DatasetError):
        freeze(golden_path, dataset_config(golden_path))


def test_freeze_is_idempotent(golden_path: Path) -> None:
    first = freeze(golden_path, dataset_config(golden_path))
    assert freeze(golden_path, dataset_config(golden_path)) == first


def test_shipped_sample_matches_its_manifest() -> None:
    path = ROOT / 'data' / 'golden' / 'tweets_golden_sample.jsonl'
    golden = load_golden(dataset_config(path))
    assert len(golden.rows) == 30
