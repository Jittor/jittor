"""Check a running local vLLM completion server using only the Python stdlib.

Run this independently against each backend. This is a bounded HTTP concurrency
check, not a long-duration load test. SSE chunks can contain multiple tokens;
reported inter-chunk timings must not be described as per-token latency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time
import traceback
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROMPTS = [
    'The capital of France is',
    '1 + 1 =',
    'Write a short story about a cat.',
    'Explain why the sky is blue.',
]


def stream_request(base_url, model, prompt, max_tokens, timeout):
    payload = dict(model=model, prompt=prompt, temperature=0,
                   max_tokens=max_tokens, ignore_eos=True, stream=True,
                   return_token_ids=True,
                   stream_options={'include_usage': True})
    request = Request(base_url.rstrip('/') + '/v1/completions',
                      data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json',
                               'Accept': 'text/event-stream'})
    start = time.perf_counter()
    row = dict(prompt=prompt, token_ids=[], text='', chunks=[], usage=None,
               finish_reason=None, done=False)
    data_lines = []

    def consume_event():
        data = '\n'.join(data_lines)
        data_lines.clear()
        if data == '[DONE]':
            row['done'] = True
            return
        if not data:
            return
        event = json.loads(data)
        if event.get('error'):
            raise RuntimeError('SSE error: ' + json.dumps(event['error']))
        if event.get('usage') is not None:
            row['usage'] = event['usage']
        for choice in event.get('choices', []):
            assert choice['index'] == 0, 'unexpected completion choice'
            delta = choice.get('token_ids') or []
            assert all(isinstance(t, int) for t in delta), 'noninteger token ID'
            text_delta = choice.get('text', '')
            if delta:
                row['chunks'].append(dict(seconds=time.perf_counter() - start,
                                          tokens=len(delta), text=text_delta))
            row['token_ids'].extend(delta)
            row['text'] += text_delta
            if choice.get('finish_reason') is not None:
                row['finish_reason'] = choice['finish_reason']

    try:
        with urlopen(request, timeout=timeout) as response:
            row['http_status'] = response.status
            assert response.status == 200
            assert 'text/event-stream' in response.headers.get('Content-Type', '')
            for raw in response:
                if time.perf_counter() - start > timeout:
                    raise TimeoutError('request exceeded total timeout')
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    consume_event()
                    if row['done']:
                        break
                elif line.startswith('data:'):
                    data_lines.append(line[5:].lstrip(' '))
            if data_lines and not row['done']:
                consume_event()
    except HTTPError as exc:
        raise RuntimeError('HTTP %s: %s' % (exc.code, exc.read().decode())) from exc
    row['seconds'] = time.perf_counter() - start
    assert row['done'], 'stream ended without [DONE]'
    assert row['finish_reason'] == 'length', row['finish_reason']
    assert len(row['token_ids']) == max_tokens, 'wrong output token count'
    assert row['usage'] is not None, 'missing final usage'
    assert row['usage']['completion_tokens'] == max_tokens, 'usage disagrees with tokens'
    assert row['chunks'], 'no token-bearing chunk'
    row['ttft_seconds'] = row['chunks'][0]['seconds']
    row['inter_chunk_seconds'] = [right['seconds'] - left['seconds']
                                  for left, right in zip(row['chunks'], row['chunks'][1:])]
    return row


def summarize(rows, seconds):
    def stats(values):
        values = sorted(values)
        return dict(median=statistics.median(values),
                    p95=values[min(len(values) - 1, int(.95 * len(values)))])
    intervals = [t for row in rows for t in row['inter_chunk_seconds']]
    return dict(requests=len(rows), output_tokens=sum(len(r['token_ids']) for r in rows),
                seconds=seconds, requests_per_second=len(rows) / seconds,
                output_tokens_per_second=sum(len(r['token_ids']) for r in rows) / seconds,
                request_seconds=stats([r['seconds'] for r in rows]),
                ttft_seconds=stats([r['ttft_seconds'] for r in rows]),
                inter_chunk_seconds=stats(intervals) if intervals else None,
                max_tokens_per_chunk=max(c['tokens'] for r in rows for c in r['chunks']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=('oracle', 'jittor'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests-per-concurrency', type=int, default=20)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    assert args.requests_per_concurrency >= len(PROMPTS)
    assert args.max_tokens > 0 and all(c > 0 for c in args.concurrency)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('refusing to overwrite ' + str(path))
    report = dict(status='running', backend=args.backend, options=vars(args),
                  phases=[], timing_scope='client observed; first token-bearing SSE '
                  'chunk and gaps between token-bearing chunks, not per-token ITL')

    def save():
        path.write_text(json.dumps(report, indent=2))

    def run(label, concurrency, count, reference=None):
        rows = []
        start = time.perf_counter()
        phase = dict(label=label, concurrency=concurrency, results=rows)
        report['phases'].append(phase)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(stream_request, args.base_url, args.model,
                                   PROMPTS[i % len(PROMPTS)], args.max_tokens,
                                   args.timeout): i for i in range(count)}
            for future in as_completed(futures):
                i = futures[future]
                row = future.result()
                row['request_index'] = i
                if reference is not None:
                    assert row['token_ids'] == reference[i % len(PROMPTS)], (
                        'greedy tokens changed for prompt %d' % (i % len(PROMPTS)))
                rows.append(row)
                save()
        rows.sort(key=lambda row: row['request_index'])
        phase['summary'] = summarize(rows, time.perf_counter() - start)
        phase['status'] = 'completed'
        save()
        print(label, json.dumps(phase['summary']), flush=True)
        return rows

    try:
        save()
        with urlopen(args.base_url.rstrip('/') + '/v1/models', timeout=args.timeout) as response:
            report['models'] = json.load(response)
        assert args.model in [m['id'] for m in report['models']['data']]
        smoke = run('smoke', 1, len(PROMPTS))
        reference = [r['token_ids'] for r in smoke]
        for concurrency in args.concurrency:
            run('concurrency_%d' % concurrency, concurrency,
                args.requests_per_concurrency, reference)
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['exception'] = '%s: %s' % (type(exc).__name__, exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
