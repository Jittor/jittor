"""Bounded generic Torch data protocol publication, not a Torch source loader.

Only package-owned, hash-admitted Python resources below are importable. No
oracle site-packages path, Native torch._C, arbitrary sys.path or meta_path hook
is used. This module is not a third-party torchdata state implementation.
"""
import hashlib
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

from ..transaction import current_transaction
from .fidelity import Fidelity, register_fidelity


_SOURCE_SHA256 = {
    "LICENSE.pytorch": "a8a2c6b67cf8a597ffaa5c82380a6bd1b82629c0417e00673d8013d662eb500a",
    "NOTICE.txt": "bd3ebb9d8f2dee83e62351ac77957c59dcbebc7342d1697de3fbe602f034b27c",
    "torch/utils/_import_utils.py": "259fbe1e90bb4d57d31e18ac0da9b117a58d1a3a8408b799a0f1247db79b01d0",
    "torch/utils/data/__init__.py": "3c9029293d62ce640c5d2f81d76dfff544aabbc9c3f820e822316f66431dab43",
    "torch/utils/data/_utils/__init__.py": "b5e3527d87e6397cf3f2b8ced32a45f085979cc3885fb8c7a6fa032be54f5c90",
    "torch/utils/data/_utils/collate.py": "6d0a69df509263103d0d6c1793d999fb6cf9cdbd7d0623659e6c36d95f7861a1",
    "torch/utils/data/_utils/fetch.py": "66b969970a44e99c916f33385ac9e74c01219b19dfe14a7601120bd89402f179",
    "torch/utils/data/_utils/pin_memory.py": "c15d4ed164c968ef0783fdc76f8568e4aeb5268875c8979b3bc4d60c9cc8be43",
    "torch/utils/data/_utils/signal_handling.py": "afa54562501c96ae07ed2ae5ae48a9a8fd14255ecec6906fea8a26082a0791db",
    "torch/utils/data/_utils/worker.py": "01c6299e2ed85b2973a26d0c822dce3966c6b8f91faedc2e300150b6314de486",
    "torch/utils/data/dataloader.py": "657394a61b32e2f9b81f928343968bfdf58bdde8b8b279fb38d60b332adc618e",
    "torch/utils/data/datapipes/__init__.py": "0374476cf4a400f1b3732bbef291793c87a885ce398fc166ebd8a13cfd4524fb",
    "torch/utils/data/datapipes/_decorator.py": "83ed12a13c8d2a9cb61a320445ae4f6a85e11521c950e8cf0b2d3431df4fd803",
    "torch/utils/data/datapipes/_hook_iterator.py": "9a6f00c57a2313a36507750864a45b40eb6b6dd1fd6e064cd13fdcf702bff344",
    "torch/utils/data/datapipes/_typing.py": "a816cddcd0187ec871f40d8287a240ab410a3b076367f0985048d94047de15c9",
    "torch/utils/data/datapipes/datapipe.py": "7a7cc09d129046d1575aa23cf62db6b994f3a186d6f62b84c207a9ea158c86f6",
    "torch/utils/data/datapipes/iter/__init__.py": "5c95cc9fe97d62abe72a4b5b37f0e1071c091fa84c6fe2b534dafc8a97e9b82e",
    "torch/utils/data/datapipes/iter/sharding.py": "8fc1cee52ee1a4a9c7089082831770c3440a8ff6e45d0973edf8520c91b6264e",
    "torch/utils/data/datapipes/utils/__init__.py": "2fdfa7c98ec1d44f05d2f7798180304fc0d555f4060990c2c5d35fb3b3903596",
    "torch/utils/data/datapipes/utils/common.py": "6a1bbcd7a56b522af6cdcfa41671f2cbb708fafea92931d2e881e1bdd994f99a",
    "torch/utils/data/dataset.py": "e3e06ffc754c9472f15b5e7ab99f1f191c6e75f0099b4d73cb75408cb60a5086",
    "torch/utils/data/distributed.py": "354edf0f15f2b2cd8286b2ae5017c0f43de8d2607d35f42a5ac1cd559e939dab",
    "torch/utils/data/graph.py": "a58d6046653cdb2d873bd7bf6833e91d8a3d2dbf003f4b7448b833d3dfd8751d",
    "torch/utils/data/graph_settings.py": "d15606261814e9fbb42e7ab0b07778b43f2700f560288e8dbd33f0d5ad71e8c4",
    "torch/utils/data/sampler.py": "eca02eff92c40d627571d20d8f54ccb5c27b2cd38b46f8ef11b8884f1257a772"
}


def _resource_root():
    return Path(__file__).resolve().parent.parent / 'shim' / 'resources' / 'data_protocol'


def _verify_inventory(root):
    root = root.resolve(strict=True)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob('*')
        if path.is_file() and '__pycache__' not in path.relative_to(root).parts
    }
    if actual != set(_SOURCE_SHA256):
        raise ImportError('Torch data protocol resource inventory differs')
    for relative, expected in _SOURCE_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ImportError('unsafe Torch data protocol resource')
        if root not in path.resolve(strict=True).parents:
            raise ImportError('Torch data protocol resource escaped its owner')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ImportError('Torch data protocol resource hash differs')
    return root


def _module_name(relative):
    parts = Path(relative).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def _mutate_attr(target, name, value):
    transaction = current_transaction()
    if transaction is None:
        setattr(target, name, value)
    else:
        transaction.mutate_attr(target, name, value)


def _profiler_enabled():
    raise NotImplementedError(
        'DataPipe iterator profiling-state queries are not implemented')


def _register_loaded(ctx, root):
    # Normal Python submodule imports must also appear in the existing ownership
    # registry; otherwise torch_namespace_owned correctly rejects the new graph.
    for relative in sorted(_SOURCE_SHA256, key=lambda value: (value.count('/'), value)):
        if not relative.endswith('.py'):
            continue
        name = _module_name(relative)
        module = sys.modules.get(name)
        if module is None:
            continue
        expected = (root / relative).resolve(strict=True)
        if Path(getattr(module, '__file__', '')).resolve() != expected:
            raise RuntimeError('Torch data protocol module has a foreign source: ' + name)
        ctx.registry.publish(name, module)


def _data_owner():
    from .installers import data
    return data


def _isolated_data_getattr(name):
    if name in ('IterDataPipe', 'MapDataPipe', 'DataChunk', 'DFIterDataPipe',
                'graph', 'graph_settings', 'datapipes'):
        raise NotImplementedError(
            'DataPipe/third-party imports require real Torch namespace activation, '
            'not an isolated ModuleRegistry publication')
    raise AttributeError('isolated torch.utils.data has no attribute %r' % name)


def _install_isolated(ctx):
    """Publish aliases only; do not impersonate a Python import installation.

    The same generic DataLoader/Dataset/Sampler/iterator/fetch owners are used.
    No resource is executed, no global import map is modified, no torchdata
    module/state class is registered. Actual third-party importing remains a
    real-activation-only contract. This is not a second loading algorithm.
    """
    owner = _data_owner()
    registry = ctx.registry
    utils = registry.ensure('torch.utils', package=True)
    if getattr(ctx.target_namespace, 'utils', None) is not utils:
        _mutate_attr(ctx.target_namespace, 'utils', utils)
    bindings = {
        'torch.utils.data': dict(owner._DATA_APIS),
        'torch.utils.data.dataset': {
            name: owner._DATA_APIS[name] for name in
            ('Dataset', 'IterableDataset', 'TensorDataset', 'ConcatDataset', 'Subset')},
        'torch.utils.data.sampler': {
            name: owner._DATA_APIS[name] for name in
            ('Sampler', 'SequentialSampler', 'RandomSampler',
             'SubsetRandomSampler', 'BatchSampler', 'DistributedSampler')},
        'torch.utils.data.distributed': {'DistributedSampler': owner._DistributedSampler},
        'torch.utils.data.dataloader': {
            'DataLoader': owner._DataLoader,
            '_DatasetKind': owner._DatasetKind,
            '_InfiniteConstantSampler': owner._InfiniteConstantSampler,
            '_BaseDataLoaderIter': owner._BaseDataLoaderIter,
            '_SingleProcessDataLoaderIter': owner._SingleProcessDataLoaderIter,
            'default_collate': owner._default_collate,
            'default_convert': owner._default_convert,
            'get_worker_info': owner._get_worker_info},
        'torch.utils.data._utils': {},
        'torch.utils.data._utils.fetch': {'_MapDatasetFetcher': owner._MapDatasetFetcher},
        'torch.utils.data._utils.collate': {'default_collate': owner._default_collate},
    }
    bindings['torch.utils.data']['__getattr__'] = _isolated_data_getattr
    # Parent-first registry publication preserves the existing ownership graph.
    # No __file__, loader or successful third-party state is manufactured.
    for name in sorted(bindings, key=lambda value: (value.count('.'), value)):
        module = registry.ensure(name, package=name in ('torch.utils.data', 'torch.utils.data._utils'))
        for attr, implementation in bindings[name].items():
            if attr in vars(module) and vars(module)[attr] is not implementation:
                raise RuntimeError('isolated Torch data alias has a foreign owner: ' + name + '.' + attr)
            if attr not in vars(module):
                _mutate_attr(module, attr, implementation)
    return registry.get('torch.utils.data')


def install(ctx):
    registry = ctx.registry
    if registry._modules is not sys.modules:
        return _install_isolated(ctx)
    if sys.modules.get('torch') is not ctx.target_namespace:
        raise RuntimeError('Torch data protocol requires its actual independent Torch root')
    root = _verify_inventory(_resource_root())
    current = registry.get('torch.utils.data')
    if current is not None:
        if Path(getattr(current, '__file__', '')).resolve() != root / 'torch/utils/data/__init__.py':
            raise RuntimeError('cannot replace a foreign torch.utils.data module')
        _register_loaded(ctx, root)
        return current

    # Only this package-owned resource path is added. No installed Native Torch
    # source is discoverable through it. All resources were verified above.
    utils = registry.ensure('torch.utils', package=True)
    owned_path = str(root / 'torch/utils')
    paths = list(getattr(utils, '__path__', ()))
    if owned_path not in paths:
        _mutate_attr(utils, '__path__', paths + [owned_path])
    spec = importlib.machinery.ModuleSpec('torch.utils', loader=None, is_package=True)
    spec.submodule_search_locations = list(utils.__path__)
    _mutate_attr(utils, '__spec__', spec)

    # Existing profiling support must not be replaced; missing state queries are
    # explicitly unsupported, never a successful fixed False.
    autograd = registry.get('torch.autograd')
    if autograd is None:
        raise RuntimeError('autograd owner must be installed before data protocol')
    if '_profiler_enabled' not in vars(autograd):
        _mutate_attr(autograd, '_profiler_enabled', _profiler_enabled)
        register_fidelity(
            'torch.autograd._profiler_enabled', _profiler_enabled,
            Fidelity.UNIMPLEMENTED,
            'DataPipe execution profiling queries explicitly fail; map loading does not call them.',
        )

    name = 'torch.utils.data'
    source = root / 'torch/utils/data/__init__.py'
    spec = importlib.util.spec_from_file_location(
        name, str(source), submodule_search_locations=[str(source.parent)])
    if spec is None or spec.loader is None:
        raise ImportError('Torch data protocol has no package source loader')
    module = importlib.util.module_from_spec(spec)
    _mutate_attr(utils, 'data', module)
    registry.publish(name, module)
    # Errors propagate unchanged; the existing outer installation transaction
    # owns namespace rollback. No fallback package or suppressed import exists.
    spec.loader.exec_module(module)

    for relative in sorted(_SOURCE_SHA256):
        if not relative.endswith('.py'):
            continue
        # Only optional DataPipe protocols use upstream Python3.9+ generic syntax.
        # Do not break plain DataLoader activation on the advertised older Python.
        if sys.version_info < (3, 9) and (
                relative == 'torch/utils/_import_utils.py' or '/datapipes/' in relative
                or relative.endswith(('/graph.py', '/graph_settings.py'))):
            continue
        importlib.import_module(_module_name(relative))
    _register_loaded(ctx, root)

    for api in ('default_convert', 'get_worker_info'):
        implementation = getattr(module, api)
        register_fidelity(
            'torch.utils.data.' + api, implementation, Fidelity.APPROXIMATE,
            'Real generic Python protocol; target-device downstream validation remains required.',
        )
    unsupported = {
        'signal_handling': (
            '_error_if_any_worker_fails', '_remove_worker_pids', '_set_worker_pids',
            '_set_worker_signal_handlers', '_set_SIGCHLD_handler'),
        'pin_memory': ('pin_memory', '_pin_memory_loop'),
        'worker': ('_generate_state', '_worker_loop'),
    }
    for child, names in unsupported.items():
        child_module = getattr(module._utils, child)
        for attr in names:
            register_fidelity(
                child_module.__name__ + '.' + attr, getattr(child_module, attr),
                Fidelity.UNIMPLEMENTED,
                'Worker processes, signals and pinned DataLoader execution explicitly raise NotImplementedError.',
            )
    for name in ('_sharding_worker_init_fn', '_MultiProcessingDataLoaderIter'):
        register_fidelity(
            'torch.utils.data.dataloader.' + name, getattr(module.dataloader, name),
            Fidelity.UNIMPLEMENTED,
            'DataLoader worker-process execution explicitly raises NotImplementedError.',
        )
    return module
