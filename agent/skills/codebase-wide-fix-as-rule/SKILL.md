---
name: codebase-wide-fix-as-rule
description: How to land a fix that touches hundreds of call sites so it survives other agents' rebases - encode it as a machine-checkable structure test rather than a finished inventory, prove a rule (or an exact-list structure test rewritten as a rule) has teeth with counterexamples, and detect and repair the silent revert when someone resolves a rebase conflict by taking their own pre-fix side. Use when a structure test goes from red to green because its assertion was rewritten, when a task is phrased "change all N occurrences of X", when a sweeping change must be exempted for some subtree, or when a fix you already pushed appears to be missing from the tree.
---

# A sweeping fix is only as durable as the rule that re-checks it

A task like "compat/ 有 144 个 `except: pass`，全改掉" looks like an inventory job.
It is not. In a repo with a dozen agents rebasing onto each other, **the edit is the
cheap half and the rule is the valuable half**, for one reason:

> Reverted code does not raise, does not fail, and does not appear in the
> reverting commit's diff. It is simply *not there any more* — and the agent who
> wrote it has already reported done.

Nothing else catches that. Not review (the diff looks like the author's own work),
not the test suite (the reverted code was a *quality* property, not a behaviour),
not the author (they moved on). A rule test catches it on the next run, in one line.

## 1. Write the rule, not the list

Encode the property over the tree, so it fails on **any** new violation, including
one that arrives by reversion:

```python
def test_no_handler_body_is_only_pass(self):
    offenders = [where(p, h) for p, h in _handlers()
                 if len(h.body) == 1 and isinstance(h.body[0], ast.Pass)]
    self.assertEqual(offenders, [], "record it with diagnostics.swallowed(...)")
```

Requirements that make such a rule trustworthy:

- **State the exemptions and why**, as their own test. A subtree you had to skip is
  a decision; unstated, the next sweep rediscovers it the hard way. Pin it:
  `test_the_deployed_stubs_are_excluded_for_a_stated_reason`.
- **Assert the rule is looking at something.** A rule over an empty set passes for
  the wrong reason — a bad glob silently disables it forever.
  `self.assertGreater(len(handlers), 200)`.
- **Prove it fails on the old form.** Re-introduce one violation, watch it go red,
  put it back. A rule never seen red is a rule you are guessing about.

## 2. Turning an exact-list structure test into a rule: prove it has teeth

The same discipline applies in reverse when a `tests/structure` assertion that
pins an exact list (a byte manifest, "65 calls to `checkRet`", a frozen
allowlist) is rewritten as a rule. Every such rewrite turns a red test green,
and **"the rule is right" and "the rule is empty" look identical: both are
green.** The acceptance is not "it passed" but "it still fails on a
counterexample". Without that step a rewrite silently deletes a gate.

Build at least two counterexamples per rewritten rule; both must go red:

1. **What the test originally existed to catch** -- re-create that violation (a
   definition inside a facade, a child process that does not pin `PYTHONPATH`,
   a packaged resource missing from the list).
2. **The boundary the rewrite introduced** -- wherever the rule is wider than the
   old assertion, put a violation exactly there. Example: replacing
   `assertIn("from .runtime import enable", source)` with "every name in
   `__all__` comes from a re-export" calls for a counterexample that adds a name
   to `__all__` that is neither imported nor aliased.

Edit the real file, run the real nodeid, restore -- no mocks or temp copies,
because the shape of the real file is what is under test:

```bash
probe () {  # probe <label> <nodeid>; a collection error is not "teeth"
  JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cpu nvcc_path="" PYTHONPATH=python \
    python -m pytest "$2" -q -p no:cacheprovider 2>&1 | grep -qE "^1 failed" \
    && echo "  [has teeth] $1" || echo "  [EMPTY RULE or wrong failure] $1"
}
echo 'def defined_here(): return 1' >> <file the rule constrains>
probe "definition inside the facade" "tests/structure/<file>::<Class>::<test>"
git checkout -- <file the rule constrains>
git status --short   # must list only the test you are changing
```

**Commit the real change first, then run the counterexamples.** `git checkout --`
restores HEAD, not "the file as it was before the counterexample"; if the file
also carries your own uncommitted edit, restoring silently deletes it while the
script still prints "has teeth". If you must probe before committing, copy the
file aside (`cp <file> "$TMPDIR/<file>.keep"`) and copy it back. Never use
`git stash` (see `git-worktree-shared-state`).

Put both halves in the commit message: why the old assertion stopped holding
(which change altered the asserted shape) and the counterexample list with its
result ("6 counterexamples, all red").

**A growing exemption set is the smell of an exact list**: if every legitimate
edit adds an exemption, ask what the test was meant to prevent and keep only
that. The one legitimate exemption table is a **closed classification**, used
when the question itself is "which of the N sites that write some process-wide
state are not done yet" (grep answers "what matched", not "what is left"):

1. **Closed set**: every site the scanner finds must be in the table, and every
   table entry must still exist in the tree; counterexamples in both directions.
2. **Each entry carries one of a few categories** that say *why it is not in
   scope* (e.g. `ledger`, `runtime`, `pre-ledger`, `deployed-payload`,
   `pending`), not merely "allowed".
3. **`pending` is its own dict whose values name the obstacle**, so finishing an
   item shortens that dict in the same diff:

   ```python
   PENDING = {"…/external_backend.py": "restores sys.path/sys.modules from a whole-table "
              "snapshot and would drop concurrent writers' entries"}

   def test_pending_names_exactly_the_unfinished_files():
       assert {p for (p, _o, _k), c in CLASSIFIED.items() if c == "pending"} == set(PENDING)
   ```

Key the table by the enclosing `def`, not by line number, and give the scanner
verbs per owner type (a `dict` has no `insert`; counting `modules.insert(0, x)`
on a plain list as a `sys.modules` write is a real false positive).

## 3. Detecting the silent revert

Count the fix's own marker per file, now versus at your commit. Cheap and exact:

```bash
for f in <files the sweep touched>; do
  echo "$f: now=$(grep -c 'swallowed(' $f) at_fix=$(git show <your-sha>:$f | grep -c 'swallowed(')"
done
```

A `24 -> 0` is a reversion, not a refactor. Confirm the offender was *aware* of
your commit — if so, this was a rebase-conflict resolution that took their side
wholesale:

```bash
git merge-base --is-ancestor <your-sha> <their-sha> && echo "they built on top of it and still dropped it"
git log --oneline -3 -- <file>          # who last touched it
```

## 4. Repairing it: three-way merge, never a revert

Do **not** revert the other agent's commit — their work is newer and real. Replay
your change onto their content:

```bash
git show <base>:path/file  > f.base    # the version BOTH of you started from
git show <your-sha>:path/file > f.yours # your version (fix, no their-work)
cp path/file f.merged                  # current HEAD (their work, no fix)
git merge-file -L current -L base -L fix f.merged f.base f.yours
```

`<base>` is your commit's parent when they branched from at-or-before it. In
practice this merges clean, because the two changes touch different lines
(handlers vs. logic).

**Verify the merge at the right grain** — "tests pass" is not enough here:

```bash
grep -c 'swallowed(' f.merged                    # your fix is back, full count
for pat in <their distinctive identifiers>; do   # their work survived
  echo "$pat: cur=$(grep -c $pat f.cur) merged=$(grep -c $pat f.merged)"; done
diff f.cur f.merged | grep '^[<>]' | grep -v 'except\|swallowed\|EXPECTED\|import'
#   ^ must be empty-ish: the ONLY removed lines should be the old handler bodies
```

Then apply the rule to **their new code too**, not just to the regression — a
sweep that only restores its own lines leaves the newest violations standing.

## 5. Commit it separately, and say what happened

One commit, its message naming the clobbering sha, the marker counts (`24 -> 0`),
the merge base used, and the verification above. The next person to see the rule
go red needs to find this, not re-derive it.

## Related

- `git-worktree-shared-state` — why `git stash` is banned here (shared stack).
  To park work for a clean-tree experiment use
  `git diff > mydir/x.patch` + `git checkout -- <files>` + `git apply` to restore.
  That same trick is how you **prove a failure is pre-existing**: revert to the
  pristine tree, run the failing test, restore.
- `verifying-a-gate-actually-ran` — which suites to run, in which separate
  invocations, and how to tell a real red from cache, disk or concurrency noise.
