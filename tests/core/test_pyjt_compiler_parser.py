# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The pyjt generator's reading of C++ declarations.

``pyjt_compiler.py`` parses the headers with a character scan rather than a C++
parser, and its failure mode is not an error: it emits C++ that compiles and
calls the wrong thing.  Each case below is a *legal* way to write a binding
that the scan used to get wrong, so that the next person who adds one finds out
from a test instead of from a wrong answer at runtime.

The module is loaded straight from its file: importing ``jittor`` would build
the whole C++ core, and none of this needs it.

Note for anyone adding a case: ``compile_src`` starts scanning at offset 16
(``reg.finditer(src, re.S)`` passes the flag where ``pos`` goes), so a source
fragment must be padded before its first ``// @pyjt`` or it is silently not
seen at all.
"""

import importlib.util
import os
import unittest


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAD = "// pad: compile_src starts scanning at offset 16\n"

_generator = None


def generator():
    """The generator module, loaded on first use.

    Deliberately not loaded at import time: collection must not execute
    anything (tests/structure/test_pytest_contract.py enforces that).
    """
    global _generator
    if _generator is None:
        path = os.path.join(REPO_ROOT, "python", "jittor", "pyjt_compiler.py")
        spec = importlib.util.spec_from_file_location(
            "_pyjt_compiler_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _generator = module
    return _generator


def render(declaration, annotation="// @pyjt(f)\n"):
    """Generate the binding for one declaration and return the C++ text."""
    return generator().compile_src(
        PAD + annotation + declaration, "test.h", "test")


class TestSplitArgs(unittest.TestCase):
    """``split_args`` divides a parameter list on the commas between parameters."""

    def test_plain_parameters(self):
        self.assertEqual(
            [a.strip() for a in generator().split_args("int a, int b")],
            ["int a", "int b"])

    def test_comma_inside_a_default_value_call(self):
        # Only angle brackets were counted, so this split inside ``g(1,2)`` and
        # produced three parameters, the last two of them nonsense.
        self.assertEqual(
            [a.strip() for a in generator().split_args("int a=g(1,2), int b")],
            ["int a=g(1,2)", "int b"])

    def test_greater_than_does_not_go_below_zero(self):
        # ``>`` decremented unconditionally, so after one top-level ``>`` the
        # depth was -1 and *every* later comma was ignored: the whole tail of
        # the signature collapsed into a single parameter.
        self.assertEqual(
            [a.strip() for a in generator().split_args("bool d=(0>1), int a, int b")],
            ["bool d=(0>1)", "int a", "int b"])

    def test_shift_operators_are_not_template_brackets(self):
        self.assertEqual(
            [a.strip() for a in generator().split_args("int a=(1<<3), int b=(8>>1)")],
            ["int a=(1<<3)", "int b=(8>>1)"])

    def test_template_argument_commas_are_kept(self):
        self.assertEqual(
            [a.strip() for a in generator().split_args("map<int,int> a, int b")],
            ["map<int,int> a", "int b"])

    def test_nested_template_close(self):
        self.assertEqual(
            [a.strip() for a in generator().split_args("vector<vector<int>> a, int b")],
            ["vector<vector<int>> a", "int b"])

    def test_function_type_parameter(self):
        self.assertEqual(
            [a.strip() for a in
             generator().split_args("std::function<void(int,int)> cb, int x")],
            ["std::function<void(int,int)> cb", "int x"])

    def test_comma_inside_a_string_literal(self):
        self.assertEqual(
            [a.strip() for a in generator().split_args('const char* s=",", int b')],
            ['const char* s=","', "int b"])

    def test_unbalanced_brackets_raise_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            generator().split_args("int a=(1, int b")


class TestDeclarationScan(unittest.TestCase):
    """``find_bc`` finds the parameter list, and only the parameter list."""

    def test_closing_paren_inside_a_string_literal(self):
        # The scan stopped at the ``)`` inside the literal, so the parameter
        # list was read as ``const char* s = "`` and ``int b`` disappeared --
        # a one-parameter binding for a two-parameter function.
        code = render('void f(const char* s=")", int b=0);\n')
        self.assertIn("arg1", code)

    def test_comment_inside_the_parameter_list(self):
        code = render("void f(int a /* , ) */, int b);\n")
        self.assertIn("arg1", code)

    def test_line_comment_inside_the_parameter_list(self):
        code = render("void f(int a, // , ) \n int b);\n")
        self.assertIn("arg1", code)


    def test_doc_comment_above_an_unrelated_declaration(self):
        # The annotation regex allowed its optional doc-comment group to run
        # past its own `*/` (`(.*?)` under re.DOTALL).  A `/** ... */` that was
        # not followed by an annotation therefore expanded to the next `*/`
        # that was, swallowing the annotations in between.  One doc comment
        # above a plain declaration ate the `// @pyjt(Thing)` that opens the
        # class, and every method of that class came out as a free function.
        source = (
            PAD
            + "/** An ordinary doc comment on a plain declaration. */\n"
            + "void unrelated(int a);\n"
            + "\n"
            + "// @pyjt(Thing)\n"
            + "struct Thing {\n"
            + "    /** doc for value */\n"
            + "    // @pyjt(value)\n"
            + "    int value();\n"
            + "};\n"
        )
        code = generator().compile_src(source, "test.h", "test")
        self.assertIn("GET_RAW_PTR(Thing,self)", code)
        self.assertNotIn("((value()))", code)


class TestReturnTypeAndName(unittest.TestCase):
    def test_pointer_bound_to_the_name(self):
        # ``VarHolder *f`` splits into ["VarHolder", "*f"], and "*f" became the
        # function name: the emitted call was ``(*f(arg0))``, dereferencing the
        # returned pointer and converting a *copy* of the pointee.
        code = render("VarHolder *f(int x);\n")
        self.assertIn("to_py_object<VarHolder*>", code)
        self.assertNotIn("(*f(", code)

    def test_pointer_bound_to_the_type_is_unchanged(self):
        code = render("VarHolder* f(int x);\n")
        self.assertIn("to_py_object<VarHolder*>", code)


class TestDefaultValues(unittest.TestCase):
    def test_default_containing_a_second_equals_sign(self):
        # ``arg.split('=')`` returned four pieces and raised "too many values
        # to unpack" -- at build time, from inside the code generator.
        code = render("void f(bool a=(1==1), int b=0);\n")
        self.assertIn("arg1", code)

    def test_default_containing_a_comparison(self):
        code = render("void f(bool a=(1>=1), int b=0);\n")
        self.assertIn("arg1", code)


class TestGeneratedExceptionHandling(unittest.TestCase):
    """Every generated entry point needs a catch-all, not only ``std::exception``."""

    def test_catch_all_is_emitted(self):
        # These functions are called from CPython across an extern "C"
        # boundary: an exception that escapes calls std::terminate and takes
        # the interpreter down with no traceback.  A throw of a pointer or of
        # any non-std type -- pyjt_console.h throws ``new std::runtime_error``,
        # a pointer -- does not match ``catch (const std::exception&)``.
        code = render("void f(int a);\n")
        self.assertIn("catch (...)", code)
        self.assertEqual(
            code.count("catch (...)"),
            code.count("} catch (const std::exception& e)"))


class TestRealHeadersStillParse(unittest.TestCase):
    """The hardening must not change how today's headers are read."""

    def test_all_annotated_headers_generate_balanced_code(self):
        source_root = os.path.join(REPO_ROOT, "python", "jittor", "src")
        seen = 0
        for dirpath, _dirnames, filenames in os.walk(source_root):
            for name in sorted(filenames):
                if not name.endswith(".h"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf8") as handle:
                    text = handle.read()
                if "@pyjt" not in text:
                    continue
                code = generator().compile_src(text, path, name.split('.')[0])
                if not code:
                    continue
                seen += 1
                self.assertEqual(
                    code.count("{"), code.count("}"),
                    "unbalanced braces generated for " + path)
        self.assertGreater(seen, 5, "no annotated headers were generated")


if __name__ == "__main__":
    unittest.main()
