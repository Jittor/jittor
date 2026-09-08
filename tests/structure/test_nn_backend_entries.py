"""Neural-network CUDA implementations have one physical backend owner."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NN = ROOT / "python/jittor/nn"
CUDA_NN = ROOT / "backends/cuda/kernels/nn"
BACKEND_MODULES = (
    "batch_norm_training_cuda", "channel_bias_cuda", "full_reduce_cuda",
    "group_norm_cuda", "layer_norm_cuda", "layer_norm_training_cuda",
    "modulated_layer_norm_cuda", "rms_norm_training_cuda", "softmax_cuda",
)
ROOT_MODULES = (
    "rms_norm_cuda", "rope_cuda", "swiglu_cuda", "kv_cache_cuda", "packed_qkv_cuda",
)
ALIASES = {
    **{"jittor.nn.backends." + name: "jittor.backends.cuda.kernels.nn." + name
       for name in BACKEND_MODULES},
    **{"jittor.nn." + name: "jittor.backends.cuda.kernels.nn." + name
       for name in ROOT_MODULES},
    "jittor.nn._cuda_inference": "jittor.backends.cuda.kernels.nn._inference",
    "jittor.nn.kv_cache_acl": "jittor.backends.acl.kernels.kv_cache",
}


def test_nn_backend_directory_contains_only_composition_and_hooks():
    assert {path.name for path in (NN / "backends").glob("*.py")} == {
        "__init__.py", "cudnn.py", "hooks.py",
    }
    assert list(NN.glob("*_cuda.py")) == []
    assert not (NN / "_cuda_inference.py").exists()
    assert not (NN / "kv_cache_acl.py").exists()
    for name in BACKEND_MODULES + ROOT_MODULES + ("_inference",):
        path = CUDA_NN / (name + ".py")
        assert path.is_file(), path
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 7))
    assert (ROOT / "backends/acl/kernels/kv_cache.py").is_file()


def test_legacy_nn_aliases_declare_the_canonical_backend_owner():
    tree = ast.parse((ROOT / "python/jittor/_runtime/import_aliases.py").read_text(encoding="utf-8"))
    published = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "ALIASES"
                             for target in node.targets))
    for legacy, canonical in ALIASES.items():
        assert published[legacy] == canonical


def test_legacy_nn_imports_preserve_identity_and_canonical_origin():
    import importlib

    for legacy, canonical in ALIASES.items():
        old_module = importlib.import_module(legacy)
        owner = importlib.import_module(canonical)
        assert old_module is owner
        assert owner.__name__ == canonical
        relative = canonical[len("jittor."):].replace(".", "/") + ".py"
        assert Path(owner.__file__).resolve() == (ROOT / relative).resolve()
