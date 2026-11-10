"""Compare offline model outputs with NumPy, outside the source checkout."""
import argparse
import json
from pathlib import Path

import numpy as np

from _paths import lab_output_path, resolve_source_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--max-relative-l2', type=float, default=0.02)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', help='Source checkout; defaults to this script checkout')
    parser.add_argument('--allow-attention-difference', action='store_true',
                        help='Allow explicit eager/SDPA metadata differences; all other settings and outputs are checked')
    args = parser.parse_args()
    try:
        source_root = resolve_source_root(args.source_root)
        output = lab_output_path(args.output, source_root)
    except ValueError as error:
        parser.error(str(error))
    left_path, right_path = Path(args.left).resolve(), Path(args.right).resolve()
    left, right = [json.loads(path.read_text()) for path in (left_path, right_path)]
    if not args.allow_attention_difference and left.get('attention') != right.get('attention'):
        raise ValueError('different comparison condition: attention (use --allow-attention-difference explicitly)')
    for field in ('case', 'model', 'device', 'tiny', 'profile', 'dtype',
                  'parameters', 'transformers', 'diffusers', 'spec'):
        settings=[report[field] for report in (left,right)]
        if field=='spec' and args.allow_attention_difference:
            settings=[dict(value) for value in settings]
            for value in settings:
                value.pop('attention',None)
        if settings[0] != settings[1]:
            raise ValueError('different comparison condition: ' + field)
    for side, report in (('left', left), ('right', right)):
        if report.get('backend') == 'jittor' and report.get('fallback_count') != 0:
            raise ValueError(side + ' Jittor fallback count must be zero')
    fields = {}
    passed = True
    with np.load(str(left_path) + '.npz') as l, np.load(str(right_path) + '.npz') as r:
        if set(l.files) != set(r.files):
            raise ValueError('different output fields')
        for key in l.files:
            x = l[key].astype(np.float64)
            y = r[key].astype(np.float64)
            if x.shape != y.shape:
                raise ValueError('different output shape: ' + key)
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            relative = float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-12))
            maximum = float(np.abs(x-y).max())
            cosine = float(np.vdot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-12))
            fields[key] = dict(shape=list(x.shape), finite=finite, relative_l2=relative,
                               max_abs=maximum, cosine_similarity=cosine)
            passed = passed and finite and relative <= args.max_relative_l2
    result = dict(passed=passed, left=str(left_path), right=str(right_path),
                  threshold=args.max_relative_l2, fields=fields,
                  attention_difference_allowed=args.allow_attention_difference,
                  attention_implementations=[report.get('attention') for report in (left,right)])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
