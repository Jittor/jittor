"""Split a one-file git diff into per-task patches, by what each hunk touches.

Usage:  python3 split_hunks.py all.patch out_prefix
        -> out_prefix.<task>.patch, one per RULES entry that matched

RULES below is an example (the 2026-09-03 5.10/5.13 split); edit it for your
own tasks. The load-bearing part is the assertion that every hunk matches
*exactly one* rule -- a hunk that matches none or several is what silently
lands in the wrong commit, so this refuses to guess.

See SKILL.md section 19 for the surrounding `git apply --cached` recipe.
"""
import re, sys, io

patch_path, out_prefix = sys.argv[1], sys.argv[2]
text = io.open(patch_path, encoding="utf-8").read()
lines = text.split("\n")
head, i = [], 0
while i < len(lines) and not lines[i].startswith("@@"):
    head.append(lines[i]); i += 1
hunks, cur = [], None
for line in lines[i:]:
    if line.startswith("@@"):
        if cur: hunks.append(cur)
        cur = [line]
    elif cur is not None:
        cur.append(line)
if cur: hunks.append(cur)

RULES = [
    ("510", ("randperm", "topk", "repeat_interleave", "out_row", "offsets[mid]",
             "offsets_p[mid]", "int64_t inner")),
    ("unique", ("input_flatten", "keys_out", "cub::DeviceRadixSort", "indice.dtype",
                "duplicate-dropping", "One implementation, every dtype")),
    ("isnan", ("_classify", "isnan(", "_isnan_acl", "_isinf_acl", "_isfinite_acl",
               "isposinf", "isneginf")),
    ("cumsum", ("_cumsum_dim", "_scan_2d", "_Cumsum", "cumulative sum in dim",
                "numpy host callback")),
]

buckets = {name: [] for name, _ in RULES}
for h in hunks:
    body = "\n".join(h)
    hits = [name for name, keys in RULES if any(k in body for k in keys)]
    if len(hits) != 1:
        raise SystemExit("hunk matches %r, cannot classify:\n%s" % (hits, body[:800]))
    buckets[hits[0]].append(h)

for name, hs in buckets.items():
    if not hs:
        print("(no hunks for %s)" % name); continue
    out = "\n".join(head + [l for h in hs for l in h])
    if not out.endswith("\n"): out += "\n"
    io.open("%s.%s.patch" % (out_prefix, name), "w", encoding="utf-8").write(out)
    print("%s: %d hunks" % (name, len(hs)))
