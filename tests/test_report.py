from evalgate.report import render_html, render_markdown
from evalgate.results import LayerResult, MetricRow, RunReport


def _report(decision: str = 'block') -> RunReport:
    layers = [
        LayerResult(
            layer='l1',
            status='pass',
            summary='40 outputs checked, 0 violations',
            metrics=[MetricRow(name='schema_valid_rate', value=1.0, threshold=1.0, passed=True)],
        ),
        LayerResult(
            layer='l2',
            status='fail',
            summary='2 metrics on 40 rows',
            metrics=[
                MetricRow(
                    name='event_type.f1',
                    value=0.9,
                    lower=0.71,
                    upper=0.98,
                    threshold=0.8,
                    passed=False,
                )
            ],
            findings=['event_type.f1 lower bound 0.710 is below 0.800, point estimate 0.900'],
            warnings=['baseline was computed on a different golden file, refresh it'],
        ),
        LayerResult(layer='l4', status='skipped', required=False, summary='no judge keys'),
    ]
    return RunReport(task='tweet-classifier', decision=decision, policy='block', layers=layers)


def test_markdown_renders_tables_and_findings() -> None:
    text = render_markdown(_report())
    assert text.startswith('# evalgate report for tweet-classifier')
    assert '| event_type.f1 | 0.900 | 0.710 | 0.980 | 0.800 | below |' in text
    assert '- event_type.f1 lower bound 0.710 is below 0.800' in text
    assert 'Warnings' in text
    assert 'Status. skipped (advisory)' in text


def test_block_decision_names_the_failing_layer() -> None:
    text = render_markdown(_report())
    assert 'Decision. BLOCK' in text
    assert 'Blocked by L2 golden set regression.' in text


def test_pass_decision_has_no_blocked_line() -> None:
    report = RunReport(
        task='t',
        decision='pass',
        policy='block',
        layers=[LayerResult(layer='l1', status='pass', summary='ok')],
    )
    text = render_markdown(report)
    assert 'Decision. PASS' in text
    assert 'Blocked by' not in text


def test_advisory_failure_does_not_count_as_blocking() -> None:
    report = _report('pass')
    report.layers[1].required = False
    assert report.blocked_by == []


def test_html_is_self_contained_and_escapes() -> None:
    report = _report()
    report.layers[1].findings.append('<script>alert(1)</script>')
    page = render_html(report)
    assert page.startswith('<!doctype html>')
    assert 'Decision. BLOCK' in page
    assert 'L2 golden set regression' in page
    assert '&lt;script&gt;' in page
    assert '<script>' not in page
    assert 'http://' not in page and 'https://' not in page
