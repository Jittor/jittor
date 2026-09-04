#!/usr/bin/env python3
"""Generate a stub CANN include tree so ACL sources can be syntax-checked.

The ACL backend cannot be compiled on a host without CANN, so an edit to
``python/jittor/extern/acl/**`` normally reaches review with nothing having
parsed it. This emits just enough of the CANN surface for ``g++ -fsyntax-only``
to accept the real sources: opaque handle types, the ``aclnn`` status/enum
vocabulary, and one declaration per ``aclnn`` entry point referenced by the
tree.

The execute entry points are declared with their real ABI
``(void*, uint64_t, aclOpExecutor*, aclrtStream)`` on purpose. That is what
makes the check able to reject a runner whose launcher does not fit
``BaseOpRunner::launch``. The ``GetWorkspaceSize`` queries are declared
variadic because their signatures differ per operator and are not knowable
without the SDK -- so this harness checks that the tail is well-formed, not
that a workspace query is passed the right arguments.
"""

import argparse
import pathlib
import re

# Names matched by the aclnn regex that are vocabulary, not entry points.
NOT_ENTRY_POINTS = {"aclnnStatus"}

ACL_H = """#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>

typedef int aclError;
typedef int aclnnStatus;
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;

#define ACL_SUCCESS 0
#define ACL_ERROR_NONE 0

typedef enum {
    ACL_FLOAT = 0, ACL_FLOAT16 = 1, ACL_INT8 = 2, ACL_INT32 = 3,
    ACL_UINT8 = 4, ACL_INT16 = 6, ACL_UINT16 = 7, ACL_UINT32 = 8,
    ACL_INT64 = 9, ACL_UINT64 = 10, ACL_DOUBLE = 11, ACL_BOOL = 12,
    ACL_STRING = 13, ACL_COMPLEX64 = 16, ACL_BF16 = 27,
} aclDataType;

typedef enum {
    ACL_FORMAT_UNDEFINED = -1, ACL_FORMAT_NCHW = 0, ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2, ACL_FORMAT_NC1HWC0 = 3, ACL_FORMAT_NCDHW = 30,
} aclFormat;

typedef enum {
    ACL_MEMCPY_HOST_TO_HOST = 0, ACL_MEMCPY_HOST_TO_DEVICE = 1,
    ACL_MEMCPY_DEVICE_TO_HOST = 2, ACL_MEMCPY_DEVICE_TO_DEVICE = 3,
} aclrtMemcpyKind;

typedef enum { ACL_MEM_MALLOC_HUGE_FIRST = 0, ACL_MEM_MALLOC_NORMAL_ONLY = 2 } aclrtMemMallocPolicy;

struct aclTensor; struct aclScalar; struct aclIntArray; struct aclFloatArray;
struct aclBoolArray; struct aclTensorList; struct aclScalarList; struct aclOpExecutor;

aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                           const int64_t *stride, int64_t offset, aclFormat format,
                           const int64_t *storageDims, uint64_t storageDimsNum, void *addr);
int aclDestroyTensor(const aclTensor *tensor);
aclScalar *aclCreateScalar(void *value, aclDataType dataType);
int aclDestroyScalar(const aclScalar *scalar);
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
int aclDestroyIntArray(const aclIntArray *array);
aclFloatArray *aclCreateFloatArray(const float *value, uint64_t size);
int aclDestroyFloatArray(const aclFloatArray *array);
aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size);
int aclDestroyBoolArray(const aclBoolArray *array);
aclTensorList *aclCreateTensorList(aclTensor *const *value, uint64_t size);
int aclDestroyTensorList(const aclTensorList *array);
aclScalarList *aclCreateScalarList(aclScalar *const *value, uint64_t size);
int aclDestroyScalarList(const aclScalarList *array);
int aclSetTensorAddr(aclOpExecutor *executor, int index, aclTensor *tensor, void *addr);

const char *aclGetRecentErrMsg();
const char *aclGetErrorMessage(aclError code);
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void *devPtr);
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                          aclrtMemcpyKind kind, aclrtStream stream);
aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtGetDevice(int32_t *deviceId);
aclError aclrtGetDeviceCount(uint32_t *count);
aclError aclrtResetDevice(int32_t deviceId);
aclError aclrtSynchronizeDevice();
aclError aclrtCreateEvent(aclrtEvent *event);
aclError aclrtDestroyEvent(aclrtEvent event);
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
aclError aclrtSynchronizeEvent(aclrtEvent event);
aclError aclrtGetMemInfo(int attr, size_t *free, size_t *total);
aclError aclrtMallocHost(void **hostPtr, size_t size);
aclError aclrtFreeHost(void *hostPtr);
aclError aclrtProcessReport(int32_t timeout);
aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);
aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);
aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclInit(const char *configPath);
aclError aclFinalize();
"""

ENTRY_HEADER = """#pragma once
#include "acl/acl.h"

// Every aclnn entry point referenced by python/jittor/extern/acl. Execute
// entry points get the real four-argument ABI; workspace queries are variadic
// because their per-operator signatures need the SDK.
"""


def collect_entry_points(acl_root):
    # Not just call sites: the registry in acl_jittor.h names both the query
    # and the launcher as plain values, e.g. AclOpFunctions(aclnnAbs...,
    # aclnnAbs), so a "followed by (" pattern would miss half of them.
    mention = re.compile(r"\baclnn[A-Za-z0-9_]*\b")
    names = set()
    for path in sorted(acl_root.rglob("*")):
        if path.suffix not in (".cc", ".h"):
            continue
        for name in mention.findall(path.read_text(encoding="utf-8", errors="ignore")):
            if name not in NOT_ENTRY_POINTS:
                names.add(name)
    return sorted(names)


def collect_error_codes(acl_root):
    """Every ACL_ERROR_* name the tree mentions.

    acl_error_code.cc maps a few hundred of them to strings, so a stub that
    only declares the handful used by the runners cannot check that file. The
    values are arbitrary here but must be distinct, because the consumers
    switch on them.
    """
    mention = re.compile(r"\bACL_ERROR_[A-Z0-9_]+\b")
    names = set()
    for path in sorted(acl_root.rglob("*")):
        if path.suffix not in (".cc", ".h"):
            continue
        names.update(mention.findall(path.read_text(encoding="utf-8", errors="ignore")))
    names.discard("ACL_ERROR_NONE")  # already defined as 0 in the stub acl.h
    return sorted(names)


def collect_sdk_headers(acl_root):
    include = re.compile(r'#include\s*[<"]((?:acl|aclnn|aclnnop)/[^">]+)[>"]')
    headers = set()
    for path in sorted(acl_root.rglob("*")):
        if path.suffix not in (".cc", ".h"):
            continue
        for name in include.findall(path.read_text(encoding="utf-8", errors="ignore")):
            # acl/aclops/* is this repository's own tree, not the SDK.
            if not name.startswith("acl/aclops/"):
                headers.add(name)
    return sorted(headers)


def build(acl_root, out):
    """Write the stub tree and report what it contains."""
    acl_root = pathlib.Path(acl_root).resolve()
    out = pathlib.Path(out).resolve()

    codes = collect_error_codes(acl_root)
    acl_h = ACL_H + "\n".join(
        "#define {} {}".format(name, -index - 1) for index, name in enumerate(codes))

    (out / "acl").mkdir(parents=True, exist_ok=True)
    (out / "acl" / "acl.h").write_text(acl_h + "\n", encoding="utf-8")
    (out / "acl" / "acl_op_compiler.h").write_text(
        '#pragma once\n#include "acl/acl.h"\n', encoding="utf-8")

    entries = collect_entry_points(acl_root)
    lines = [ENTRY_HEADER]
    for name in entries:
        if name.endswith("GetWorkspaceSize"):
            lines.append("aclnnStatus {}(...);".format(name))
        else:
            lines.append("aclnnStatus {}(void *workspace, uint64_t workspaceSize, "
                         "aclOpExecutor *executor, aclrtStream stream);".format(name))
    (out / "acl" / "aclnn_entry_points.h").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")

    shim = '#pragma once\n#include "acl/aclnn_entry_points.h"\n'
    headers = 0
    for header in collect_sdk_headers(acl_root):
        target = out / header
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(shim, encoding="utf-8")
        headers += 1

    return {"out": out, "headers": headers, "entry_points": len(entries),
            "error_codes": len(codes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acl-root", required=True,
                        help="path to python/jittor/extern/acl")
    parser.add_argument("--out", required=True, help="stub include tree to create")
    args = parser.parse_args()

    report = build(args.acl_root, args.out)
    print("stub tree: {out}\nsdk headers stubbed: {headers}\n"
          "aclnn entry points declared: {entry_points}\n"
          "error codes defined: {error_codes}".format(**report))


if __name__ == "__main__":
    main()
