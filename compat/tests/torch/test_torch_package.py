"""``torch.package`` stores data resources and refuses source packaging.

Upstream ``torch.package`` is a hermetic code packager.  The shim implements
the data container (pickle/text/binary resources over a zip archive) and raises
a clear ``NotImplementedError`` for the code-packaging half, so these tests pin
both the round-trip behavior and the explicit boundary.
"""
import io
import json
import sys
import weakref
import zipfile

import numpy as np
import pytest

import torch
import torch.package as package


def _roundtrip(buffer):
    buffer.seek(0)
    return package.PackageImporter(buffer)


def test_public_surface_is_published_under_torch():
    assert torch.package is package
    for name in ("PackageExporter", "PackageImporter", "Importer",
                 "OrderedImporter", "sys_importer", "ObjNotFoundError",
                 "ObjMismatchError", "PackagingError", "EmptyMatchError"):
        assert hasattr(package, name), name
    assert package.package_exporter.PackageExporter is package.PackageExporter
    assert package.package_importer.PackageImporter is package.PackageImporter
    assert package.importer.sys_importer is package.sys_importer
    assert isinstance(package.package_importer._package_imported_modules,
                      weakref.WeakValueDictionary)


def test_pickle_text_and_binary_round_trip_through_a_buffer():
    weights = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    buffer = io.BytesIO()
    with package.PackageExporter(buffer) as exporter:
        exporter.save_pickle(
            "model", "weights",
            {"w": weights, "meta": {"name": "tiny", "steps": [1, 2]}},
            dependencies=False)
        exporter.save_pickle("model", "ids", [torch.tensor([1, 2, 3], dtype=torch.int64)])
        exporter.save_text("model", "config.json", '{"hidden": 8}')
        exporter.save_binary("model", "blob.bin", b"\x00\x01\xfe")

    importer = _roundtrip(buffer)
    try:
        loaded = importer.load_pickle("model", "weights")
        np.testing.assert_array_equal(loaded["w"].numpy(), weights.numpy())
        assert loaded["meta"] == {"name": "tiny", "steps": [1, 2]}
        assert type(loaded["w"]) is torch.Tensor

        ids = importer.load_pickle("model", "ids")
        assert isinstance(ids, list) and len(ids) == 1
        assert ids[0].dtype == torch.int64
        np.testing.assert_array_equal(ids[0].numpy(), [1, 2, 3])

        assert importer.load_text("model", "config.json") == '{"hidden": 8}'
        assert json.loads(importer.load_text("model", "config.json")) == {"hidden": 8}
        assert importer.load_binary("model", "blob.bin") == b"\x00\x01\xfe"
        assert importer.python_version() == sys.version
        assert importer.extern_modules == []
    finally:
        importer.close()


def test_pickle_loads_a_module_state_dict(tmp_path):
    model = torch.nn.Linear(4, 2)
    state = model.state_dict()
    archive = tmp_path / "model.ptp"
    with package.PackageExporter(archive) as exporter:
        exporter.save_pickle("model", "state_dict", state)

    with package.PackageImporter(archive) as importer:
        loaded = importer.load_pickle("model", "state_dict", map_location="cpu")
        assert set(loaded) == set(state)
        for name, value in state.items():
            np.testing.assert_array_equal(loaded[name].numpy(), value.numpy())


def test_map_location_is_accepted_and_dtype_is_preserved():
    buffer = io.BytesIO()
    with package.PackageExporter(buffer) as exporter:
        exporter.save_pickle("t", "i64", torch.tensor([2 ** 40 + 1], dtype=torch.int64))
    with _roundtrip(buffer) as importer:
        loaded = importer.load_pickle("t", "i64", map_location="cpu")
        assert loaded.dtype == torch.int64
        np.testing.assert_array_equal(loaded.numpy(), [2 ** 40 + 1])


def test_duplicate_resource_is_refused():
    buffer = io.BytesIO()
    exporter = package.PackageExporter(buffer)
    exporter.save_text("pkg", "resource", "first")
    with pytest.raises(ValueError, match="already written"):
        exporter.save_text("pkg", "resource", "second")
    exporter.close()
    with _roundtrip(buffer) as importer:
        assert importer.load_text("pkg", "resource") == "first"


def test_distinct_names_that_resolve_to_one_record_are_refused():
    exporter = package.PackageExporter(io.BytesIO())
    exporter.save_text("a.b", "c", "first")
    with pytest.raises(ValueError, match="already written"):
        exporter.save_text("a", "b/c", "second")
    exporter.close()


def test_duplicate_record_paths_in_a_manifest_are_refused():
    manifest = {
        "format": "jittor-torch-package-1",
        "entries": [
            {"package": "a.b", "resource": "c", "kind": "text",
             "path": "data/a/b/c"},
            {"package": "a", "resource": "b/c", "kind": "text",
             "path": "data/a/b/c"},
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("version", "jittor-torch-package-1")
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("data/a/b/c", b"data")
    buffer.seek(0)
    with pytest.raises(ImportError, match="records .* twice"):
        package.PackageImporter(buffer)


@pytest.mark.parametrize("package_name", ["", "pkg..sub", "pkg/sub", "/pkg", ".."])
def test_invalid_package_names_are_refused(package_name):
    exporter = package.PackageExporter(io.BytesIO())
    with pytest.raises(ValueError):
        exporter.save_text(package_name, "resource", "text")
    exporter.close()


@pytest.mark.parametrize("resource", ["", "/absolute", "../escape", "a//b", "a/./b", "a\\b"])
def test_invalid_resource_names_are_refused(resource):
    exporter = package.PackageExporter(io.BytesIO())
    with pytest.raises(ValueError):
        exporter.save_text("pkg", resource, "text")
    exporter.close()


def test_missing_resource_raises_key_error():
    buffer = io.BytesIO()
    with package.PackageExporter(buffer) as exporter:
        exporter.save_text("pkg", "present", "text")
    with _roundtrip(buffer) as importer:
        with pytest.raises(KeyError, match="not present"):
            importer.load_text("pkg", "absent")


def test_code_packaging_raises_a_named_not_implemented_error():
    buffer = io.BytesIO()
    with package.PackageExporter(buffer) as exporter:
        exporter.save_text("pkg", "present", "text")
    with _roundtrip(buffer) as importer:
        for method in ("intern", "mock", "extern", "deny", "save_source_file",
                       "save_source_string", "save_module", "add_dependency",
                       "register_extern_hook", "register_mock_hook",
                       "register_intern_hook"):
            with pytest.raises(NotImplementedError, match="source"):
                getattr(exporter, method)("pkg")
        with pytest.raises(NotImplementedError, match="code packaging"):
            importer.import_module("pkg")
        with pytest.raises(NotImplementedError, match="relative imports"):
            importer.__import__("pkg", level=1)
        assert importer.__import__("json") is json
        with pytest.raises(NotImplementedError, match="file_structure"):
            importer.file_structure()


def test_save_pickle_rejects_unsupported_pickle_protocols():
    exporter = package.PackageExporter(io.BytesIO())
    with pytest.raises(AssertionError, match="protocols 3 and 4"):
        exporter.save_pickle("pkg", "obj", {"a": 1}, pickle_protocol=5)
    exporter.close()


def test_an_unclosed_exporter_cannot_be_loaded():
    buffer = io.BytesIO()
    with pytest.raises(RuntimeError, match="abort"):
        with package.PackageExporter(buffer) as exporter:
            exporter.save_text("pkg", "resource", "text")
            raise RuntimeError("abort")
    with pytest.raises(ImportError, match="manifest.json"):
        package.PackageImporter(buffer)


def test_a_foreign_zip_is_rejected_with_a_clear_error(tmp_path):
    archive = tmp_path / "foreign.ptp"
    with zipfile.ZipFile(archive, "w") as foreign:
        foreign.writestr("some/record", b"data")
    with pytest.raises(ImportError, match="manifest.json"):
        package.PackageImporter(archive)

    versioned = tmp_path / "versioned.ptp"
    with zipfile.ZipFile(versioned, "w") as foreign:
        foreign.writestr("manifest.json", b'{"format": "pytorch-torch-package"}')
    with pytest.raises(ImportError, match="declares format"):
        package.PackageImporter(versioned)


def test_a_non_zip_file_is_rejected_with_a_clear_error(tmp_path):
    archive = tmp_path / "not-a-zip.ptp"
    archive.write_bytes(b"not a zip")
    with pytest.raises(ImportError, match="not a zip archive"):
        package.PackageImporter(archive)


def test_saving_after_close_is_refused():
    buffer = io.BytesIO()
    exporter = package.PackageExporter(buffer)
    exporter.close()
    with pytest.raises(ValueError, match="closed"):
        exporter.save_text("pkg", "resource", "text")


def test_importer_id_and_repr_identity_helpers():
    buffer = io.BytesIO()
    with package.PackageExporter(buffer) as exporter:
        exporter.save_binary("pkg", "resource", b"x")
    buffer.seek(0)
    first = package.PackageImporter(buffer)
    buffer2 = io.BytesIO(buffer.getvalue())
    second = package.PackageImporter(buffer2)
    try:
        assert first.id().startswith("<torch_package_") and first.id().endswith(">")
        assert first.id() != second.id()
    finally:
        first.close()
        second.close()


def test_sys_importer_and_ordered_importer_resolve_the_same_object():
    module_name, attribute = package.sys_importer.get_name(package.PackageExporter)
    assert getattr(package.sys_importer.import_module(module_name), attribute) \
        is package.PackageExporter

    ordered = package.OrderedImporter(package.sys_importer)
    assert ordered.import_module("json") is json
    with pytest.raises(package.ObjNotFoundError):
        ordered.get_name(object())
    with pytest.raises(TypeError):
        package.OrderedImporter(object())


def test_fidelity_is_registered_as_approximate():
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of

    for api in ("torch.package.PackageExporter",
                "torch.package.PackageImporter"):
        record = fidelity_of(api)
        assert record.level is Fidelity.APPROXIMATE
        assert "NotImplementedError" in record.detail
        assert "not binary-compatible" in record.detail


def test_archives_are_plain_zip_files_with_a_manifest():
    buffer = io.BytesIO()
    with package.PackageExporter(buffer) as exporter:
        exporter.save_text("pkg.sub", "nested/resource", "text")
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        names = set(archive.namelist())
        assert "version" in names and "manifest.json" in names
        assert "data/pkg/sub/nested/resource" in names
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["format"] == "jittor-torch-package-1"
        assert manifest["entries"][0]["sha256"]
