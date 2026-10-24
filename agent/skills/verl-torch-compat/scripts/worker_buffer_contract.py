"""Host-safe evidence contract for controlled tiny Qwen3 nonpersistent buffers.

NumPy supplies only an independent mathematical expectation and validation;
it never creates/replaces a model buffer. Both registered RoPE buffer names
must be observed, even when they alias the same tensor.
"""

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def buffer_shapes():
    return {"model.rotary_emb.inv_freq": (4,), "model.rotary_emb.original_inv_freq": (4,)}


def canonical_buffer(name):
    require(isinstance(name, str) and name and all(name.split(".")), "invalid buffer name")
    normalized = ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")
    require(normalized in buffer_shapes(), "unexpected controlled buffer: " + name)
    return normalized


def rope_config(config):
    """Read the public config, rejecting a different RoPE recipe."""
    config = config if isinstance(config, dict) else config.to_dict()
    parameters = config.get("rope_parameters") or {}
    require(isinstance(parameters, dict), "unsupported layered RoPE configuration")
    result = {"head_dim": config.get("head_dim"),
              "rope_theta": parameters.get("rope_theta", config.get("rope_theta")),
              "rope_type": parameters.get("rope_type", config.get("rope_type", "default")),
              "partial_rotary_factor": parameters.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0))}
    require(result == {"head_dim": 8, "rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
            "unsupported controlled RoPE configuration")
    require(not config.get("rope_scaling"), "legacy scaled RoPE is outside controlled scope")
    return result


def expected_inv_freq(config):
    config = rope_config(config)
    exponents = np.arange(0, config["head_dim"], 2, dtype=np.float64) / config["head_dim"]
    return np.power(float(config["rope_theta"]), -exponents)


def capture_initial_buffers(module, record, device_assert):
    """Observe every registered buffer after actual residency assertion."""
    buffers = dict(module.named_buffers(remove_duplicate=False))
    canonical = [canonical_buffer(name) for name in buffers]
    require(len(canonical) == len(set(canonical)) and set(canonical) == set(buffer_shapes()),
            "incomplete or colliding controlled buffer inventory")
    persistent_keys = set(module.state_dict())
    record.report["rope_config"] = rope_config(module.config)
    manifest = {}
    record.report["initial_buffers"] = manifest
    for name, buffer in buffers.items():
        device_assert(buffer)
        shape = tuple(buffer.shape)
        require(shape == buffer_shapes()[canonical_buffer(name)], "wrong actual buffer shape")
        require(name not in persistent_keys, "controlled RoPE buffer unexpectedly persistent")
        key = "initial_buffer/" + name
        manifest[name] = {"shape": list(shape), "dtype": str(buffer.dtype), "device": str(buffer.device),
                          "persistent": False, "array_key": key}
        record.tensor(key, buffer)
    record.report["buffer_count"] = len(buffers)


def validate_buffer_manifest(manifest):
    config = manifest.get("rope_config")
    require(isinstance(config, dict) and rope_config(config) == config, "missing or invalid RoPE config")
    buffers = manifest.get("initial_buffers")
    require(isinstance(buffers, dict), "missing initial buffer manifest")
    canonical = [canonical_buffer(name) for name in buffers]
    require(len(canonical) == len(set(canonical)) and set(canonical) == set(buffer_shapes()),
            "incomplete or colliding initial buffer manifest")
    require(type(manifest.get("buffer_count")) is int and manifest["buffer_count"] == 2, "wrong buffer count")
    for name, item in buffers.items():
        require(item == {"shape": [4], "dtype": "torch.float32", "device": "npu:0",
                         "persistent": False, "array_key": "initial_buffer/" + name},
                "invalid initial buffer metadata: " + name)


def validate_buffer_arrays(manifest, arrays):
    expected = expected_inv_freq(manifest["rope_config"])
    for name in buffer_shapes():
        key = "initial_buffer/" + name
        require(key in arrays, "missing initial buffer snapshot: " + name)
        actual = np.asarray(arrays[key])
        require(actual.shape == (4,) and actual.dtype == np.dtype("float32") and np.isfinite(actual).all(),
                "invalid initial buffer shape/dtype/finite: " + name)
        # Fixed, independent of caller's potentially permissive worker tolerance.
        require(np.allclose(actual, expected, atol=1e-10, rtol=1e-7),
                "initial buffer violates independent RoPE formula: " + name)
