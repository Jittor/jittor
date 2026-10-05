#pragma once

#include <Python.h>
#include "core/common.h"

namespace jittor {

// Caches keyed by a Python type's address.
//
// Holding the type keeps its address from being reused by another type, but
// it also keeps alive every class made at run time -- a Module defined inside
// a function, a Parameter subclass in a test -- and whatever its class body
// closes over, tensors included. A cache that does not hold the type asks
// instead to be told when it is collected: `forget(type)` runs from the type's
// weak-reference callback, before its memory is freed, so the address cannot
// have been reused by then. A type is watched once however many caches ask.
// Static types are never collected and are not watched. GIL held.
void on_type_collected(PyObject* type, void (*forget)(PyObject* type));

} // jittor
