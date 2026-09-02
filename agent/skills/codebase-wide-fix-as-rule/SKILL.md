---
name: codebase-wide-fix-as-rule
description: How to land a fix that touches hundreds of call sites so it survives other agents' rebases - encode it as a machine-checkable structure test rather than a finished inventory, and how to detect and repair the silent revert when someone resolves a rebase conflict by taking their own pre-fix side. Use when a task is phrased "change all N occurrences of X", when a sweeping change must be exempted for some subtree, or when a fix you already pushed appears to be missing from the tree.
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

## 2. Detecting the silent revert

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

## 3. Repairing it: three-way merge, never a revert

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

## 4. Commit it separately, and say what happened

One commit, its message naming the clobbering sha, the marker counts (`24 -> 0`),
the merge base used, and the verification above. The next person to see the rule
go red needs to find this, not re-derive it.

## Related

- `git-worktree-shared-state` — why `git stash` is banned here (shared stack).
  To park work for a clean-tree experiment use
  `git diff > mydir/x.patch` + `git checkout -- <files>` + `git apply` to restore.
  That same trick is how you **prove a failure is pre-existing**: revert to the
  pristine tree, run the failing test, restore.
- `jittor-refactor-gates` — which suites to run and in which separate invocations.
