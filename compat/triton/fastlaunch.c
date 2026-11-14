/*
 * Launch a cubin's kernel through cuLaunchKernel without building the
 * ctypes parameter array in Python.  Every argument the kernel declares
 * occupies one 8-byte slot here; `argv` points at the slots.  This is the
 * same layout the Python path produces with `ctypes.c_void_p * n`, moved
 * into one C call: the per-launch cost of that array (a cast per argument)
 * showed up in the bridge's launch time.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int (*cuLaunch_t)(void *, unsigned, unsigned, unsigned,
                          unsigned, unsigned, unsigned, unsigned,
                          void *, void **, void **);

static cuLaunch_t g_launch;
static void *g_lib;

static int load_cuda(void) {
    if (g_launch) {
        return 0;
    }
    const char *names[] = {"libcuda.so.1", "libcuda.so", "nvcuda.dll"};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        g_lib = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
        if (g_lib) {
            break;
        }
    }
    if (!g_lib) {
        return -1;
    }
    g_launch = (cuLaunch_t)dlsym(g_lib, "cuLaunchKernel");
    return g_launch ? 0 : -2;
}

static PyObject *fast_launch(PyObject *self, PyObject *args) {
    unsigned long long fn, stream;
    unsigned gx, gy, gz, bx, by, bz, shared;
    PyObject *values;
    if (!PyArg_ParseTuple(args, "KIIIIIIIKO", &fn,
                          &gx, &gy, &gz, &bx, &by, &bz, &shared,
                          &stream, &values)) {
        return NULL;
    }
    if (load_cuda() != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cuLaunchKernel is not available");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(values, "values must be a sequence");
    if (!seq) {
        return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n > 256) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "too many kernel arguments");
        return NULL;
    }
    uint64_t slots[256];
    void *argv[256];
    for (Py_ssize_t i = 0; i < n; i++) {
        unsigned long long v = PyLong_AsUnsignedLongLong(
            PySequence_Fast_GET_ITEM(seq, i));
        if (v == (unsigned long long)-1 && PyErr_Occurred()) {
            Py_DECREF(seq);
            return NULL;
        }
        slots[i] = (uint64_t)v;
        argv[i] = &slots[i];
    }
    Py_DECREF(seq);
    int rc = g_launch((void *)fn, gx, gy, gz, bx, by, bz, shared,
                      (void *)stream, argv, NULL);
    return PyLong_FromLong(rc);
}

static PyMethodDef methods[] = {
    {"launch", fast_launch, METH_VARARGS,
     "cuLaunchKernel(fn, grid, block, shared, stream, values)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_fastlaunch", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit__fastlaunch(void) {
    return PyModule_Create(&module);
}
