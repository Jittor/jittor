#!/usr/bin/env python3
"""Render a Jittor-versus-PyTorch CUDA table from bench_examples.py records.

Reads any number of JSON records produced by ``tools/bench_examples.py`` and
prints (or writes) one markdown table per example variant.  A pair is only
compared when both a jittor and a torch record exist for the same variant.
"""
import argparse
import json
import pathlib
import statistics


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def metrics(record):
    timer = record.get('timer') or {}
    steady = timer.get('steady') or {}
    if record.get('example') == 'pippy' and record.get('pippy_average_batch_seconds'):
        latency = 1000.0 * record['pippy_average_batch_seconds']
        return dict(ms=latency, p90=None, first=record.get('pippy_first_pass_seconds'),
                    throughput=record.get('pippy_batches_per_second'),
                    throughput_unit='batch/s')
    return dict(ms=steady.get('median_ms'), p90=steady.get('p90_ms'),
                first=timer.get('first_step_seconds'),
                throughput=steady.get('samples_per_second'), throughput_unit='samples/s')


def median_of(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def aggregate(records):
    """Median of each metric across repeated executions of one configuration."""
    rows = [metrics(record) for record in records]
    return dict(ms=median_of([row['ms'] for row in rows]),
                p90=median_of([row['p90'] for row in rows]),
                first=median_of([row['first'] for row in rows]),
                throughput=median_of([row['throughput'] for row in rows]),
                unit=(rows[0]['throughput_unit'] if rows else None),
                e2e=median_of([record['subprocess_wall_seconds'] for record in records]))


def cell(value, digits=2):
    return '-' if value is None else ('%.*f' % (digits, value))


def rate(value, unit):
    return '-' if value is None else '%.1f %s' % (value, unit)


def render(records):
    groups = {}
    for record in records:
        key = (record['example'], record.get('variant') or 'default')
        groups.setdefault(key, {}).setdefault(record['backend'], []).append(record)
    lines = ['| Example | Variant | Jittor ms/step | PyTorch ms/step | Jittor speed vs PyTorch | '
             'Jittor throughput | PyTorch throughput | Jittor first step (s) | PyTorch first step (s) | '
             'Jittor E2E (s) | PyTorch E2E (s) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for (example, variant) in sorted(groups):
        by_backend = groups[(example, variant)]
        jm = aggregate(by_backend['jittor']) if by_backend.get('jittor') else None
        tm = aggregate(by_backend['torch']) if by_backend.get('torch') else None
        ratio = (tm['ms'] / jm['ms']) if jm and tm and jm.get('ms') and tm.get('ms') else None
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            example, variant, cell(jm['ms']) if jm else '-', cell(tm['ms']) if tm else '-',
            ('%.2fx' % ratio) if ratio else '-',
            rate(jm['throughput'], jm['unit']) if jm else '-',
            rate(tm['throughput'], tm['unit']) if tm else '-',
            cell(jm['first']) if jm else '-', cell(tm['first']) if tm else '-',
            cell(jm['e2e']) if jm else '-', cell(tm['e2e']) if tm else '-'))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='JSON record from bench_examples.py')
    parser.add_argument('--output', help='Markdown file; defaults to stdout')
    args = parser.parse_args()
    text = render([load(path) for path in args.run])
    if args.output:
        pathlib.Path(args.output).write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
