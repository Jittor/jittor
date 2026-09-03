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

A fragment may start with its ``// @pyjt`` annotation on line one.  It could
not until recently: ``compile_src`` scanned from offset 16, so an annotation
in the first 16 characters was never seen.
"""

import importlib.util
import os
import unittest


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        annotation + declaration, "test.h", "test")


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


class TestAnnotationScanStart(unittest.TestCase):
    """The scan starts at the beginning of the file, not 16 bytes in."""

    def test_annotation_on_the_first_line(self):
        # ``compile_src`` called ``reg.finditer(src, re.S)``: re.S is 16, and
        # the second positional argument of ``finditer`` is ``pos``.  Scanning
        # therefore began at offset 16 and an annotation before that was never
        # seen -- ``compile_src`` returned None and the bindings were silently
        # absent.  Every real header opens with a copyright banner, which is
        # the only reason nothing was ever missing.
        code = generator().compile_src(
            "// @pyjt(f)\nvoid f(int a);\n", "test.h", "test")
        self.assertTrue(code, "an annotation on line 1 was not seen at all")
        self.assertIn("arg0", code)

    def test_annotation_after_a_short_prologue(self):
        code = generator().compile_src(
            "#pragma once\n// @pyjt(f)\nvoid f(int a);\n", "test.h", "test")
        self.assertTrue(code)
        self.assertIn("arg0", code)


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
            "/** An ordinary doc comment on a plain declaration. */\n"
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


DEALLOC_CLASS = """
// @pyjt(Thing)
struct Thing {
    // @pyjt(__init__)
    Thing(int a);
    // @pyjt(__dealloc__)
    ~Thing();
};
"""


def balanced_block(text, start_marker):
    """The ``{...}`` that follows ``start_marker``, braces matched."""
    i = text.index(start_marker)
    open_at = text.index("{", i)
    depth = 0
    for j in range(open_at, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:j + 1]
    raise AssertionError("unbalanced block after " + start_marker)


class TestGeneratedDealloc(unittest.TestCase):
    """``tp_dealloc`` has two obligations no other slot has.

    It must free the instance's storage *whatever happens*, and it must not
    change the interpreter's exception state.  CPython calls it from arbitrary
    points -- including while another exception is propagating -- and there is
    no caller to report to.

    The generated body used to meet neither.  ``tp_free`` was emitted only on
    the success path (``~T(); tp_free(self); return;``), so a destructor that
    threw -- ``~VarHolder`` reaches the allocator, whose ``free`` asserts --
    landed in the shared catch, which called ``PyErr_Format`` and returned.
    The instance's memory (and, for a type with an instance dict, the dict)
    was never released, and a ``RuntimeError`` was left set to surface at
    whatever unrelated bytecode ran next.
    """

    def dealloc_body(self):
        code = generator().compile_src(DEALLOC_CLASS, "test.h", "test")
        return balanced_block(code, "tp.tp_dealloc")

    def test_storage_is_freed_on_the_exception_path(self):
        body = self.dealloc_body()
        tail = body[body.index("catch (...)"):]
        tail = tail[len(balanced_block(tail, "catch (...)")):]
        self.assertIn(
            "tp_free", tail,
            "the catch-all falls through to a plain `return`: an instance "
            "whose destructor threw is never freed")

    def test_the_ambient_exception_is_preserved(self):
        body = self.dealloc_body()
        self.assertIn("PyErr_Fetch", body)
        self.assertIn("PyErr_Restore", body)

    def test_a_failed_destructor_is_reported_unraisably(self):
        # Not silently swallowed: a destructor that threw is still a defect,
        # and sys.unraisablehook is where CPython reports what cannot be
        # raised.  It just may not become the exception of an unrelated frame.
        self.assertIn("PyErr_WriteUnraisable", self.dealloc_body())

    def test_other_slots_are_untouched(self):
        # The exception-state parking is specific to deallocation; a normal
        # method must still raise into its caller.
        code = render("void f(int a);\n")
        self.assertNotIn("PyErr_Fetch", code)
        self.assertNotIn("PyErr_WriteUnraisable", code)


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
