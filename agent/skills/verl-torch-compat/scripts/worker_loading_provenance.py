"""Observe the loader already selected by installed adapters; never install one."""

import hashlib
import os
from pathlib import Path
import sys


ENVIRONMENT_KEY = "JITTOR_TRANSFORMERS_MATERIALIZED_INIT"


def observe_loading(framework):
    if framework not in ("torch", "jittor"):
        raise ValueError("unknown worker framework")
    requested = os.environ.get(ENVIRONMENT_KEY, "0")
    if requested not in ("0", "1"):
        raise ValueError("materialized initialization opt-in must be 0 or 1")
    module = sys.modules.get("transformers.modeling_utils")
    owner = getattr(module, "PreTrainedModel", None)
    descriptor = vars(owner).get("get_init_context") if isinstance(owner, type) else None
    function = descriptor.__func__ if isinstance(descriptor, classmethod) else None
    if not callable(function):
        raise ValueError("cannot observe the loaded Transformers initialization entry")
    active = getattr(function, "_jittor_transformers_materialized_init", False) is True
    result = {
        "scope": "initialization strategy only; not an exhaustive adapter inventory",
        "requested_materialized_cpu": requested == "1", "active_materialized_cpu": active,
        "mode": "upstream_default", "adapter_source_sha256": None,
        "supported_versions": [], "meta_support_claimed": False,
    }
    if framework == "torch":
        if active or requested == "1":
            raise ValueError("native Torch oracle must use its unmodified initialization strategy")
        return result
    if active != (requested == "1"):
        raise ValueError("requested initialization strategy and observed adapter disagree")
    if active:
        adapter = sys.modules.get("jittor_adapters.transformers_loading")
        path = getattr(adapter, "__file__", None)
        if not path:
            raise ValueError("active initialization adapter source is unavailable")
        result.update(
            mode="materialized_cpu_constructor",
            adapter_source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            supported_versions=sorted(getattr(adapter, "SUPPORTED_VERSIONS", ())),
        )
        if result["supported_versions"] != ["5.9.0"]:
            raise ValueError("unverified materialized initialization adapter version scope")
    return result


def validate_loading(value, framework):
    """Fail closed on provenance, independently of numeric tolerances."""
    expected = {"scope", "requested_materialized_cpu", "active_materialized_cpu", "mode",
                "adapter_source_sha256", "supported_versions", "meta_support_claimed"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("missing or invalid model-loading provenance")
    if value["scope"] != "initialization strategy only; not an exhaustive adapter inventory":
        raise ValueError("unexpected loading provenance scope")
    if value["meta_support_claimed"] is not False:
        raise ValueError("materialized initialization does not establish meta support")
    if framework not in ("torch", "jittor"):
        raise ValueError("unknown worker framework")
    requested, active = value["requested_materialized_cpu"], value["active_materialized_cpu"]
    if type(requested) is not bool or type(active) is not bool or requested != active:
        raise ValueError("initialization activation mismatch")
    if active:
        digest = value["adapter_source_sha256"]
        if (framework != "jittor" or value["mode"] != "materialized_cpu_constructor"
                or value["supported_versions"] != ["5.9.0"] or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid materialized initialization provenance")
    elif (value["mode"] != "upstream_default" or value["adapter_source_sha256"] is not None
          or value["supported_versions"] != []):
        raise ValueError("inactive loader must not claim adapter provenance")
