"""Fresh UNet evidence binds the complete installed package, never a pipeline."""

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def component_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    tools = {name: importlib.import_module(name) for name in (
        'registry_inventory', 'component_measure', 'collect_component_scorecard')}
    assert all(Path(module.__file__).resolve().parent == SCRIPTS for module in tools.values())
    return tools


def test_full_source_binding_includes_modules_outside_the_registry(tmp_path, monkeypatch, component_tools):
    registry = component_tools['registry_inventory']
    contents = {
        'src/diffusers/__init__.py': '__version__ = "0.35.1"\n',
        'src/diffusers/models/__init__.py':
            '_import_structure = {}\n_import_structure["unet"] = ["UNet2DModel"]\n',
        'src/diffusers/models/unet.py': 'class UNet2DModel: pass\n',
        'src/diffusers/utils/helper.py': 'raise RuntimeError("Do not import upstream")\n',
        'src/diffusers/pipelines/__init__.py':
            '_import_structure = {}\n_import_structure["ddpm"] = ["DDPMPipeline"]\n',
        'src/diffusers/pipelines/auto_pipeline.py': 'AUTO_TEXT2IMAGE_PIPELINES_MAPPING = {"ddpm": Class}\n',
    }
    for name, content in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def git(root, *args):
        if args[0] == 'rev-parse':
            return 'a' * 40
        if args[0] == 'status':
            return ''
        if args[0] == 'remote':
            return 'https://github.com/huggingface/diffusers.git'
        assert args[0] == 'ls-files'
        return '\n'.join(sorted(contents))

    monkeypatch.setattr(registry, 'git', git)
    source = registry.inventory('diffusers', tmp_path)
    assert {row['path'] for row in source['package_sources']} == set(contents)
    helper = 'src/diffusers/utils/helper.py'
    assert helper not in {row['path'] for row in source['sources']}
    (tmp_path / helper).write_text('raise RuntimeError("Different helper")\n')
    changed = registry.inventory('diffusers', tmp_path)
    assert source['axes'] == changed['axes']
    assert source['package_sources'] != changed['package_sources']


def test_modified_installed_nested_module_is_rejected(tmp_path, component_tools):
    package = tmp_path / 'diffusers'
    package.mkdir()
    origin = package / '__init__.py'
    origin.write_text('__version__ = "0.35.1"\n')
    nested = package / 'models/unet.py'
    nested.parent.mkdir()
    nested.write_text('class UNet2DModel: pass\n')
    source = {'repo': 'diffusers', 'upstream': {'version': '0.35.1'},
              'package_sources': [dict(path='src/diffusers/' + p.relative_to(package).as_posix(),
                                       sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                  for p in (origin, nested)]}
    module = SimpleNamespace(__file__=str(origin), __version__='0.35.1')
    measurement = component_tools['component_measure']
    # A complete source-only fixture cannot establish the newly required
    # official distribution/receipt authority. Keep that refusal explicit.
    with pytest.raises(ValueError, match='fixed official-wheel source descriptor'):
        measurement.installed_distribution(source, module)
    nested.write_text('class ChangedModel: pass\n')
    with pytest.raises(ValueError, match='differs from clean upstream'):
        measurement.installed_distribution(source, module)


def test_source_claim_cannot_disagree_with_complete_installed_hashes(component_tools):
    source = {'repo': 'diffusers', 'upstream': {'version': '0.35.1', 'commit': 'a' * 40},
              'package_sources': [{'path': 'src/diffusers/models/unet.py', 'sha256': 'b' * 64}]}
    value = {'jittor_commit': 'c' * 40, 'upstream': source['upstream'], 'hf_hub_offline': True,
             'downstream_distribution': {'repo': 'diffusers', 'version': '0.35.1',
                 'source_sha256': {'src/diffusers/models/unet.py': 'b' * 64},
                 'installed_files_sha256': {'models/unet.py': 'd' * 64}},
             'measurement_source_sha256': {'runner.py': 'e' * 64}}
    with pytest.raises(ValueError, match='disagree with source binding'):
        component_tools['collect_component_scorecard'].validate_binding(value, source, 'c' * 40)


@pytest.mark.parametrize('count', [143, 144, 145])
def test_unet_parameter_names_come_from_the_complete_measured_snapshot(tmp_path, count, component_tools):
    collector = component_tools['collect_component_scorecard']
    spec = collector.SPECS['diffusers']
    weights = tmp_path / 'weights.npz'
    expected = sorted('actual_parameter_%d' % i for i in range(count))
    np.savez(weights, **{name: np.zeros(1, dtype='float32') for name in expected})
    if count == 144:
        assert collector.parameter_names(spec, weights) == expected
    else:
        with pytest.raises(ValueError, match='Incomplete component parameter snapshot'):
            collector.parameter_names(spec, weights)
    assert spec['axis'] == 'model_api'
    assert spec['members'] == ['UNet2DModel']
    assert spec['level'] == 3 and spec['inputs'] == ['sample']
