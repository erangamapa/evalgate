from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalgate.errors import ConfigError

Metric = Literal['precision', 'recall', 'f1']
Average = Literal['macro', 'micro']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class DatasetConfig(Strict):
    path: Path
    id_field: str = 'id'
    input_fields: list[str] = Field(min_length=1)
    label_fields: list[str] = Field(min_length=1)


class CandidateConfig(Strict):
    kind: Literal['recorded', 'fake', 'http', 'api'] = 'recorded'
    error_rate: float = Field(default=0.0, ge=0, le=1)
    seed: int = 0
    provider: Literal['anthropic', 'openai'] | None = None
    model: str | None = None
    instructions: str = 'Classify the input. Reply with one JSON object.'

    @model_validator(mode='after')
    def kind_has_what_it_needs(self) -> 'CandidateConfig':
        if self.kind == 'http' and not self.model:
            raise ValueError('an http candidate needs a model name')
        if self.kind == 'api' and not (self.provider and self.model):
            raise ValueError('an api candidate needs a provider and a model')
        return self


class LayerToggle(Strict):
    enabled: bool = True
    required: bool = True


class RefusalProbe(Strict):
    id: str = Field(min_length=1)
    inputs: dict[str, str]


class L1Config(LayerToggle):
    @model_validator(mode='after')
    def cannot_be_relaxed(self) -> 'L1Config':
        if not (self.enabled and self.required):
            raise ValueError('l1 always runs and always blocks, it cannot be disabled or advisory')
        return self

    field_values: dict[str, list[str]] = Field(default_factory=dict)
    tool_contracts: dict[str, str] = Field(default_factory=dict)
    refusal_probes: list[RefusalProbe] = Field(default_factory=list)


class L2Config(LayerToggle):
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    bootstrap_samples: int = Field(default=1000, ge=100)
    seed: int = 17
    thresholds: dict[str, dict[Metric, float]] = Field(min_length=1)
    field_average: dict[str, Average] = Field(default_factory=dict)
    max_regression: float = Field(default=0.1, ge=0, le=1)
    baseline_file: str = 'tier1_metrics.json'


class L3Config(LayerToggle):
    enabled: bool = False
    scenarios_dir: Path | None = None

    @model_validator(mode='after')
    def enabled_needs_scenarios(self) -> 'L3Config':
        if self.enabled and self.scenarios_dir is None:
            raise ValueError('l3 needs scenarios_dir when enabled')
        return self


class JudgeSpec(Strict):
    provider: Literal['anthropic', 'openai']
    model: str = Field(min_length=1)


class L4Config(LayerToggle):
    enabled: bool = False
    judges: list[JudgeSpec] = Field(default_factory=list)
    calibration_path: Path | None = None
    criteria: str = 'Prefer the output that is correct and complete for the input.'
    min_kappa: float = Field(default=0.6, ge=-1, le=1)
    min_score: float = Field(default=0.55, ge=0, le=1)
    min_decisive: float = Field(default=0.2, ge=0, le=1)
    max_items: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def enabled_needs_judges(self) -> 'L4Config':
        if self.enabled and not (self.judges and self.calibration_path):
            raise ValueError('l4 needs judges and calibration_path when enabled')
        if self.enabled and self.required and self.min_score <= 0.5:
            raise ValueError('a required l4 needs min_score above 0.5, 0.5 means no signal')
        return self


class ShadowConfig(Strict):
    sample_rate: float = Field(default=0.05, gt=0, le=1)
    max_disagreement: float = Field(default=0.1, gt=0, le=1)


class CanaryConfig(Strict):
    stages_pct: list[int] = Field(default_factory=lambda: [5, 25, 50, 100])
    hold_minutes: int = Field(default=30, ge=1)
    abort_error_rate: float = Field(default=0.02, gt=0, le=1)

    @model_validator(mode='after')
    def stages_increase(self) -> 'CanaryConfig':
        stages = self.stages_pct
        if not stages or stages != sorted(set(stages)) or stages[0] <= 0 or stages[-1] > 100:
            raise ValueError('stages_pct must be strictly increasing within 1 to 100')
        return self


class DriftConfig(Strict):
    warn_drop: float = Field(default=0.03, gt=0, le=1)
    block_drop: float = Field(default=0.08, gt=0, le=1)
    psi_warn: float = Field(default=0.1, gt=0)
    psi_block: float = Field(default=0.25, gt=0)

    @model_validator(mode='after')
    def block_is_stricter(self) -> 'DriftConfig':
        if self.block_drop <= self.warn_drop or self.psi_block <= self.psi_warn:
            raise ValueError('block thresholds must be larger than warn thresholds')
        return self


class L5Config(LayerToggle):
    enabled: bool = False
    required: bool = False
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


class LayersConfig(Strict):
    l1: L1Config = Field(default_factory=L1Config)
    l2: L2Config
    l3: L3Config = Field(default_factory=L3Config)
    l4: L4Config = Field(default_factory=L4Config)
    l5: L5Config = Field(default_factory=L5Config)


class GateConfig(Strict):
    task: str = Field(min_length=1)
    fail_policy: Literal['block', 'warn_only'] = 'block'
    dataset: DatasetConfig
    candidate: CandidateConfig = Field(default_factory=CandidateConfig)
    layers: LayersConfig

    @model_validator(mode='after')
    def thresholds_reference_known_fields(self) -> 'GateConfig':
        known = set(self.dataset.label_fields)
        l2 = self.layers.l2
        for name in [*l2.thresholds, *l2.field_average]:
            if name not in known:
                raise ValueError(f'l2 references {name}, which is not a label field')
        for name in self.layers.l1.field_values:
            if name not in known:
                raise ValueError(f'l1 references {name}, which is not a label field')
        return self

    @classmethod
    def load(cls, path: Path) -> 'GateConfig':
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f'cannot read {path}, {error}') from error
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise ConfigError(f'invalid config {path}\n{error}') from error
