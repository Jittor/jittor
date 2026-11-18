import re
import unittest
from pathlib import Path


class TestTypeStub(unittest.TestCase):
    def test_var_shape_is_property(self):
        stub_path = Path(__file__).resolve().parents[1] / "__init__.pyi"
        content = stub_path.read_text(encoding="utf8")
        self.assertRegex(content, r"@property\s+def shape\(self\)-> Tuple\[int\]:")


if __name__ == "__main__":
    unittest.main()
