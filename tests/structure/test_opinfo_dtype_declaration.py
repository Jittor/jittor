# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Every OpInfo entry must choose the dtypes it is tested on.

``OpInfo.dtypes`` defaults to ``floating_types()``. 193 of 231 entries never
wrote ``dtypes=``, so a default nobody looked at decided what 84% of the
operator database is exercised on -- and it decided "floats only". Nothing
reported that; the entries read as covered.

The narrowing that is *deliberate* is not safe either, but at least it is
visible: ``bitwise_not`` declared ``dtypes=_INT`` (``integral_types()``, which
excludes bool) and its sample generator additionally coerced integers, so no
test could feed it a bool. It returned ``True`` for every bool input, and had
an OpInfo entry the whole time.

The repository already has the rule this file applies, in
``_helpers/gate_scope.EXCLUDED``: something that stops being checked has to say
why. So an entry either declares its dtypes, or it is on the shrinking list
below with the others that have not been revisited yet.

The list may only shrink. Declaring dtypes for an entry and leaving it on the
list is reported as progress, not tolerated silently -- a stale grandfather
list is how an exception outlives its reason.
"""

import json
import sys
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.structure

REPO_ROOT = Path(__file__).resolve().parents[2]
LIST_PATH = REPO_ROOT / "tests" / "structure" / "opinfo_implicit_dtypes.json"

sys.path.insert(0, str(REPO_ROOT / "tests"))
from opinfo.database import op_db  # noqa: E402


def _entry_name(info):
    return getattr(info, "full_name", None) or info.name


def _grandfathered():
    return set(json.loads(LIST_PATH.read_text(encoding="utf-8"))["implicit"])


class TestDtypeDeclaration(unittest.TestCase):

    def setUp(self):
        self.allowed = _grandfathered()
        self.implicit = {_entry_name(info) for info in op_db
                         if not info.dtypes_are_explicit}

    def test_a_new_entry_must_declare_its_dtypes(self):
        undeclared = sorted(self.implicit - self.allowed)
        self.assertEqual(
            undeclared, [],
            "these OpInfo entries inherit dtypes=floating_types() without "
            "saying so. Pass dtypes= with the types the operator supports, or "
            "add the entry to opinfo_implicit_dtypes.json with the others that "
            "have not been revisited: %s" % undeclared)

    def test_the_list_shrinks_and_does_not_go_stale(self):
        # A name that now declares its dtypes, or that has left the database,
        # must not keep occupying the list: an exception that outlives its
        # reason is indistinguishable from one that is still needed.
        present = {_entry_name(info) for info in op_db}
        fixed = sorted((self.allowed - self.implicit) & present)
        gone = sorted(self.allowed - present)
        self.assertEqual(
            fixed, [], "these entries now declare dtypes; remove them from "
                       "opinfo_implicit_dtypes.json: %s" % fixed)
        self.assertEqual(
            gone, [], "these entries are no longer in the database; remove "
                      "them from opinfo_implicit_dtypes.json: %s" % gone)

    def test_the_list_is_sorted_and_deduplicated(self):
        names = json.loads(LIST_PATH.read_text(encoding="utf-8"))["implicit"]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))


class TestTheRuleHasTeeth(unittest.TestCase):
    """The rule must fail on the case it exists to catch."""

    def test_an_undeclared_entry_outside_the_list_is_caught(self):
        allowed = _grandfathered()
        implicit = {_entry_name(info) for info in op_db
                    if not info.dtypes_are_explicit}
        # Simulate a newly added entry that forgot dtypes=.
        implicit = implicit | {"a_brand_new_op"}
        self.assertEqual(sorted(implicit - allowed), ["a_brand_new_op"])

    def test_an_entry_that_declares_its_dtypes_is_not_flagged(self):
        explicit = [info for info in op_db if info.dtypes_are_explicit]
        self.assertTrue(explicit, "no entry declares dtypes; the flag is broken")
        allowed = _grandfathered()
        for info in explicit[:5]:
            self.assertNotIn(_entry_name(info), allowed)


if __name__ == "__main__":
    unittest.main()
