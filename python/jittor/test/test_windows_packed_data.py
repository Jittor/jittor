import importlib.util
import pathlib
import unittest

_PACKED_DATA_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "jittor_utils" / "packed_data.py"
)
_PACKED_DATA_SPEC = importlib.util.spec_from_file_location(
    "jittor_packed_data_test_helper", _PACKED_DATA_PATH
)
packed_data = importlib.util.module_from_spec(_PACKED_DATA_SPEC)
assert _PACKED_DATA_SPEC.loader is not None
_PACKED_DATA_SPEC.loader.exec_module(packed_data)


class TestWindowsPackedData(unittest.TestCase):
    def test_windows_preprocessed_source_uses_wrappers_for_node_callbacks(self):
        source = """
namespace jittor {
static const char * x8006(void ( * func)(Node * )) {
    if (func == (void( * )(Node * )) & Node :: release_forward_liveness) return "rf";
    if (func == (void( * )(Node * )) & Node :: own_pending_liveness) return "op";
    return "unknown";
}
void Node :: free () {
    x16239 . emplace_back(this, (void( * )(Node * )) & Node :: release_backward_liveness);
}
void Node :: own_both_liveness () {
    x16239 . emplace_back(this, (void( * )(Node * )) & Node :: own_forward_liveness);
}
}
"""
        patched = packed_data.patch_windows_source(source)

        self.assertIn(
            "static void _jt_release_forward_liveness(Node* node)",
            patched,
        )
        self.assertIn(
            "func == _jt_release_forward_liveness",
            patched,
        )
        self.assertIn(
            "func == _jt_own_pending_liveness",
            patched,
        )
        self.assertIn(
            "x16239 . emplace_back(this, _jt_release_backward_liveness);",
            patched,
        )
        self.assertIn(
            "x16239 . emplace_back(this, _jt_own_forward_liveness);",
            patched,
        )


if __name__ == "__main__":
    unittest.main()
