# evalgate

I run this as the release gate for the nondeterministic parts of the platform. Unit and integration tests do that job for deterministic code. For models, prompts and adapters I need something else. A model change is a release. It runs a pyramid of checks, cheapest first, and a regression blocks the merge.

## The pyramid

L1 deterministic checks. Output schema, tool call structure and refusal cases. Tool call arguments are validated against the sportsbook-contracts models, so I never redefine a type here. This layer always runs, it cannot be disabled or made advisory, and it has no tolerance. One bad output fails it. A missing or drifted golden set blocks here too.

L2 golden set regression. Frozen datasets, per field precision, recall and f1, with seeded bootstrap confidence intervals. The golden file must match its sha256 manifest, same format model-foundry freezes. A drifted file is refused.

L3 trajectory replays. A scenario is a yaml file with steps. Each step says which tool call is expected, where the agent must stop at an interrupt gate, and when it must escalate or halt. The replayer walks a recorded transcript against it and checks the scripted tool results too. The example scenario is a bet over direct message. Proposal, confirmation gate, placement, handoff. Placement without the gate fails.

L4 pairwise judge. Paid models judge the candidate against the baseline, both orders, ties allowed. Each judge is calibrated against a committed human label sample with Cohen's kappa.

L5 online configs. Shadow, canary and drift thresholds written as json for llm-ops to consume. Drift floors come from the baseline metrics.

## Why the ci lower bound decides

A metric on a small golden set is a noisy number. A candidate can land on 0.90 by luck and a real 0.80 model can do the same on another day. So the point estimate never decides. I bootstrap the rows with a fixed seed, take the lower bound of the confidence interval, and compare that to the threshold. A lucky run on thin data has a wide interval and its lower bound stays under the line. To pass cheaply you need more rows or a better model.

The effective threshold is the larger of the configured floor and the baseline value minus the allowed regression. So a candidate cannot hide a slide behind a generous absolute floor.

## Why position swap

LLM judges favour whichever answer comes first, or sometimes second. Every pair goes to the judge twice with the order swapped. The candidate gets full credit only if it wins in both orders. A judge that always says A scores every pair at exactly 0.5, which is what an uninformative judge should score. So a required L4 must set `min_score` above 0.5, the config refuses anything else, and the layer also fails when the decisive fraction of pairs (the ones with a winner in both orders) drops below `min_decisive`. A judge with no signal cannot pass.

Every judge gets a sign test on the decisive pairs. The wins, losses, ties and p value are listed for every judge on every run, passing or not. They stay advisory. The ensemble score and the decisive fraction decide.

Kappa against the human sample tells me if a judge can be trusted at all. A judge below the floor is left out of the ensemble and its rows are marked advisory in the report. If no judge clears the floor the layer fails.

The committed calibration sample is synthetic demo data with invented players and teams. Production uses real human labels.

## Running it

```
python3.11 -m venv .venv
.venv/bin/pip install -e ../sportsbook-contracts/packages/py
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/python -m evalgate run \
  --config examples/evalgate.yaml \
  --baseline-dir data/baselines \
  --candidate-dir examples/candidate
```

Exit code 0 means pass and 1 means block. Exit code 2 means the config or setup is broken. The report lands in `report/` under the candidate directory as markdown, a self contained html page and json.

Directory layout for a model repo.

```
evalgate/                     baseline and candidate artifacts
  predictions.jsonl           one line per id, {"id": ..., "output": {...}}
  tier1_metrics.json          baseline metrics only
  transcripts/<scenario>.json recorded agent transcripts for L3
```

The candidate in the config picks how outputs are produced. `recorded` reads `predictions.jsonl` from the candidate directory. `fake` is the deterministic stand-in used in tests and demos. `http` calls a chat completions endpoint such as vllm. `api` calls a paid model.

## What runs with keys and what runs without

Without any key, L1, L2, L3 and L5 run in full, and so does L4 when judges are injected, which is how the tests run it. From the command line L4 reports skipped when the judge keys are absent, and a skipped layer never blocks.

With keys, L4 calls the real judges and the `api` and `http` candidates become available.

```
pip install -e '.[judge]'
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export EVALGATE_VLLM_URL=...
```

`.env.example` lists the variables. The anthropic and openai SDKs are imported inside the factory functions, never at module load.

## Wiring the action into a model repo

`action/action.yml` is a composite github action. It checks out the baseline and candidate refs, installs the package, runs the gate and uploads the report. The job fails on a block.

```yaml
jobs:
  evalgate:
    runs-on: ubuntu-latest
    steps:
      - uses: erangamapa/evalgate/action@main
        with:
          config: evalgate.yaml
          baseline-ref: ${{ github.event.pull_request.base.sha }}
          candidate-ref: ${{ github.event.pull_request.head.sha }}
```

model-foundry calls it on every adapter change. The `artifacts-dir` input says where each ref keeps its predictions and transcripts.

## Layer policy

L1 is always on and always blocking. Every other layer has `enabled` and `required`. A failing required layer blocks. A failing advisory layer is reported and does not. Setting `fail_policy` to `warn_only` writes the report and always exits 0, which I use while a new threshold beds in.

## Layout

```
evalgate/config.py        typed config from evalgate.yaml
evalgate/datasets.py      golden loader and manifest check
evalgate/runners/         candidates, fake, http and api
evalgate/layers/          one module per layer
evalgate/report.py        markdown and html
evalgate/gate.py          the entry point
data/                     golden sample, judge calibration sample, baseline metrics
examples/                 config, scenario and a recorded transcript
```

The baseline file in `data/baselines` is frozen from a fake candidate on the 30 row sample. Refresh it with `point_metrics` and `baseline_payload` from `evalgate.layers.l2_regression` when the golden set changes. L2 warns when the baseline hash does not match the golden file.

## License

MIT.
