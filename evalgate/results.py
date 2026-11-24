from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LayerName = Literal['l1', 'l2', 'l3', 'l4', 'l5']
LayerStatus = Literal['pass', 'fail', 'skipped']

LAYER_TITLES: dict[str, str] = {
    'l1': 'L1 deterministic checks',
    'l2': 'L2 golden set regression',
    'l3': 'L3 trajectory replays',
    'l4': 'L4 pairwise judge',
    'l5': 'L5 online configs',
}


class MetricRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    lower: float | None = None
    upper: float | None = None
    threshold: float | None = None
    passed: bool


class LayerResult(BaseModel):
    layer: LayerName
    status: LayerStatus
    summary: str
    required: bool = True
    metrics: list[MetricRow] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def title(self) -> str:
        return LAYER_TITLES[self.layer]


class RunReport(BaseModel):
    task: str
    decision: Literal['pass', 'block']
    policy: str
    layers: list[LayerResult]

    @property
    def blocked_by(self) -> list[LayerResult]:
        return [r for r in self.layers if r.status == 'fail' and r.required]
