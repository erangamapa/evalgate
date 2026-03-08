import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evalgate.config import DatasetConfig
from evalgate.errors import DatasetError, GoldenDriftError


class GoldenManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sha256: str
    rows: int


@dataclass(frozen=True)
class GoldenRow:
    id: str
    inputs: dict[str, str]
    labels: dict[str, str]


@dataclass(frozen=True)
class GoldenSet:
    rows: tuple[GoldenRow, ...]
    sha256: str


def manifest_path(path: Path) -> Path:
    return path.with_name(path.name + '.manifest.json')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(path: Path, config: DatasetConfig) -> tuple[GoldenRow, ...]:
    rows: list[GoldenRow] = []
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [
                f
                for f in (config.id_field, *config.input_fields, *config.label_fields)
                if f not in record
            ]
            if missing:
                raise DatasetError(f'{path.name} line {number} is missing {", ".join(missing)}')
            rows.append(
                GoldenRow(
                    id=str(record[config.id_field]),
                    inputs={f: record[f] for f in config.input_fields},
                    labels={f: record[f] for f in config.label_fields},
                )
            )
    ids = [row.id for row in rows]
    if len(set(ids)) != len(ids):
        raise DatasetError(f'{path.name} has duplicate ids')
    return tuple(rows)


def freeze(path: Path, config: DatasetConfig) -> GoldenManifest:
    """Write the manifest for a golden file. A frozen set can never be silently replaced."""
    rows = _parse(path, config)
    manifest = GoldenManifest(sha256=_digest(path), rows=len(rows))
    target = manifest_path(path)
    if target.exists():
        existing = GoldenManifest.model_validate_json(target.read_text(encoding='utf-8'))
        if existing != manifest:
            raise GoldenDriftError(f'{path.name} is already frozen with a different hash')
        return existing
    target.write_text(json.dumps(manifest.model_dump(), indent=2) + '\n', encoding='utf-8')
    return manifest


def load_golden(config: DatasetConfig) -> GoldenSet:
    path = config.path
    target = manifest_path(path)
    if not path.exists():
        raise DatasetError(f'{path} does not exist')
    if not target.exists():
        raise GoldenDriftError(f'{path.name} has no manifest, freeze it first')
    manifest = GoldenManifest.model_validate_json(target.read_text(encoding='utf-8'))
    actual = _digest(path)
    if actual != manifest.sha256:
        raise GoldenDriftError(
            f'{path.name} drifted, expected {manifest.sha256[:12]} got {actual[:12]}'
        )
    rows = _parse(path, config)
    if len(rows) != manifest.rows:
        raise GoldenDriftError(f'{path.name} row count changed')
    return GoldenSet(rows=rows, sha256=actual)
