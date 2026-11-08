"""Resolve evidence paths against the measured Git tree, never today's disk."""

from pathlib import Path
import subprocess


RELOCATIONS = (
    ('tools/ecosystem/flash_attention/', 'agent/skills/flash-attention-torch-compat/scripts/'),
    ('tools/ecosystem/trellis/', 'agent/skills/trellis-torch-compat/scripts/'),
    ('tools/ecosystem/vllm_omni/', 'agent/skills/vllm-omni-torch-compat/scripts/'),
    ('tools/ecosystem/', 'agent/skills/torch-compat-repo-runbook/scripts/'),
)


def source_path(root, current, baseline=None):
    """Keep the original name in historical hashes; current captures use tools/."""
    if baseline is None:
        return current
    legacy = next((old + current[len(new):] for new, old in RELOCATIONS
                   if current.startswith(new)), current)
    if legacy == current:
        return current
    names = subprocess.check_output(
        ['git', '-C', str(Path(root)), 'ls-tree', '-r', '--name-only', baseline,
         '--', current, legacy], text=True).splitlines()
    if len(names) > 1:
        raise ValueError('Ambiguous measured tool ownership: ' + current)
    # Optional historical helpers may not exist yet; callers decide whether
    # absence is allowed. They must never substitute current file bytes.
    return names[0] if names else current
