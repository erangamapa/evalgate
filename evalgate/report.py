from jinja2 import Environment

from evalgate.results import LayerResult, MetricRow, RunReport


def _fmt(value: float | None) -> str:
    return '-' if value is None else f'{value:.3f}'


def _row(metric: MetricRow) -> str:
    verdict = 'ok' if metric.passed else 'below'
    cells = [
        metric.name,
        _fmt(metric.value),
        _fmt(metric.lower),
        _fmt(metric.upper),
        _fmt(metric.threshold),
        verdict,
    ]
    return '| ' + ' | '.join(cells) + ' |'


def _layer_markdown(result: LayerResult) -> list[str]:
    advisory = '' if result.required else ' (advisory)'
    lines = [f'## {result.title}', '', f'Status. {result.status}{advisory}', '', result.summary, '']
    if result.metrics:
        lines += [
            '| metric | value | ci low | ci high | threshold | result |',
            '| --- | --- | --- | --- | --- | --- |',
            *[_row(m) for m in result.metrics],
            '',
        ]
    if result.findings:
        lines += ['Findings', '', *[f'- {f}' for f in result.findings], '']
    if result.warnings:
        lines += ['Warnings', '', *[f'- {w}' for w in result.warnings], '']
    return lines


def render_markdown(report: RunReport) -> str:
    lines = [f'# evalgate report for {report.task}', '', f'Decision. {report.decision.upper()}', '']
    blocked = report.blocked_by
    if blocked:
        names = ', '.join(r.title for r in blocked)
        lines += [f'Blocked by {names}.', '']
    lines += [f'Fail policy. {report.policy}', '']
    for result in report.layers:
        lines += _layer_markdown(result)
    return '\n'.join(lines).rstrip() + '\n'


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>evalgate report for {{ report.task }}</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#1b1f23}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2rem}
.decision{padding:.6rem 1rem;border-radius:4px;font-weight:600}
.pass{background:#e3f4e6;color:#145a1f}.block,.fail{background:#fbe4e4;color:#8a1c1c}
.skipped{background:#eee;color:#444}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #d0d4d9;padding:.3rem .5rem;text-align:left}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{padding:.1rem .5rem;border-radius:3px;font-size:.85rem}
</style></head><body>
<h1>evalgate report for {{ report.task }}</h1>
<p class="decision {{ report.decision }}">Decision. {{ report.decision | upper }}</p>
{% if report.blocked_by %}<p>Blocked by {{ report.blocked_by | map(attribute='title') | join(', ') }}.</p>{% endif %}
<p>Fail policy. {{ report.policy }}</p>
{% for r in report.layers %}
<h2>{{ r.title }} <span class="badge {{ r.status }}">{{ r.status }}{{ '' if r.required else ', advisory' }}</span></h2>
<p>{{ r.summary }}</p>
{% if r.metrics %}<table><tr><th>metric</th><th>value</th><th>ci low</th><th>ci high</th><th>threshold</th><th>result</th></tr>
{% for m in r.metrics %}<tr><td>{{ m.name }}</td><td class="num">{{ fmt(m.value) }}</td><td class="num">{{ fmt(m.lower) }}</td><td class="num">{{ fmt(m.upper) }}</td><td class="num">{{ fmt(m.threshold) }}</td><td>{{ 'ok' if m.passed else 'below' }}</td></tr>
{% endfor %}</table>{% endif %}
{% if r.findings %}<p>Findings</p><ul>{% for f in r.findings %}<li>{{ f }}</li>{% endfor %}</ul>{% endif %}
{% if r.warnings %}<p>Warnings</p><ul>{% for w in r.warnings %}<li>{{ w }}</li>{% endfor %}</ul>{% endif %}
{% endfor %}
</body></html>
"""

_ENV = Environment(autoescape=True)
_ENV.globals['fmt'] = _fmt
_TEMPLATE = _ENV.from_string(_HTML)


def render_html(report: RunReport) -> str:
    return _TEMPLATE.render(report=report)
