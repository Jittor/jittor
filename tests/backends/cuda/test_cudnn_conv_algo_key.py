# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Two convolutions that need different cuDNN algorithms must get different keys.

The six legacy algorithm caches (forward, backward-data and backward-filter,
each shared by the 2-D and the 3-D op) are keyed by ``ConvAlgoKey`` from
``backends/cuda/libraries/cudnn/include/cudnn_conv_algo_key.h``, whose bytes are
the key.  What goes wrong when a key fails to separate two configurations is not
a crash: the cache hands back the algorithm chosen for the *other* one, and the
convolution runs and produces numbers.  So "different configurations get
different keys" is the property, and it is checked here rather than asserted in
the header's comment.

The header depends on nothing but ``<cstdint>``, ``<cstring>`` and
``<unordered_map>``, so the cases below compile it on its own -- no cuDNN, no
jittor, no GPU.  If this file stops building, the reason is almost certainly a
new ``#include`` in the header.

Three cases, two of which are negative controls, because a collision test that
cannot fail proves nothing:

``test_every_configuration_gets_its_own_key``
    the real key over a matrix of configurations that differ in exactly one
    field each.

``test_the_text_key_this_replaced_is_what_the_matrix_catches``
    the same matrix through the *previous* encoding -- decimal and hexadecimal
    text written into the shared jit key buffer -- which collides, and is the
    reason the matrix is shaped the way it is.  Both defects it exhibits were
    real and shipped:

      * the 2-D backward keys carried neither the operand dtypes, nor the
        tensor strides, nor the output extent, nor the workspace budget, so an
        fp16 convolution took the algorithm measured for the fp32 one of the
        same shape and an NHWC one took the NCHW one's; the 3-D keys carried the
        dtypes but still not the strides;
      * padding, stride and dilation were concatenated *with no separator* into
        variable-length hexadecimal (``jk << int`` is ``JK::hex``), so padding
        ``(1, 17)`` and ``(17, 1)`` both encode as ``111`` and, in 3-D,
        ``(1,1,17)``, ``(1,17,1)`` and ``(17,1,1)`` all encode as ``1111``.

``test_a_constant_hash_is_caught``
    the real header with ``ConvAlgoKeyHash``'s body replaced by ``return 1``,
    which must turn the hash half of the first case red.  Note what this control
    does and does not show.  A constant hash degrades the table to a linear scan
    but keeps it *correct*, because ``ConvAlgoKeyEq`` is a ``memcmp`` -- so the
    byte and placement halves of the first case still pass here, and that is
    asserted too.  The correctness property is the distinctness of the key
    bytes, and the control that breaks it is the one above.

Run::  python -m pytest tests/backends/cuda/test_cudnn_conv_algo_key.py
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "backends/cuda/libraries/cudnn/include/cudnn_conv_algo_key.h"

#: How many configurations ``matrix()`` builds.  Asserted, so that a matrix that
#: silently shrank could not report zero collisions out of nothing.
CONFIG_COUNT = 40

#: Configurations that differ in exactly one field, written once in C++ and
#: shared by all three cases, so the real key, the text key it replaced and the
#: constant-hash variant are all measured against the same matrix.
#:
#: cuDNN's own enumerators are not available here -- that is the point, the
#: header includes no cuDNN -- so dtypes, compute types and math types are plain
#: small integers.  The key only ever compares them.
MATRIX = r"""
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "cudnn_conv_algo_key.h"

using jittor::ConvAlgoKey;
using jittor::conv_algo_key;

// A convolution, in the terms the six call sites have on hand.
struct Config {
    const char* label;
    int pass;
    int spatial_dims;
    int dtype_x, dtype_w, dtype_y;
    int compute_type, math_type;
    int filter_format, groups;
    float workspace_ratio;
    int dim_x[5], stride_x[5];
    int dim_w[5];
    int dim_y[5], stride_y[5];
    int pad[3], conv_stride[3], dilation[3];
};

// NCHW / NCDHW contiguous strides for `dim`, which is in normalised
// (N, C, spatial...) order -- the order the tensor descriptors take.
static void contiguous_nchw(int* stride, const int* dim, int rank) {
    stride[rank-1] = 1;
    for (int i = rank-2; i >= 0; i--) stride[i] = stride[i+1] * dim[i+1];
    for (int i = rank; i < 5; i++) stride[i] = 0;
}

// The same tensor stored channels-last: identical `dim`, different strides.
// This is the pair the 2-D backward text keys could not tell apart, because
// `dimX` is read through the layout string and so is the same for both.
static void contiguous_nhwc(int* stride, const int* dim, int rank) {
    stride[1] = 1;
    stride[rank-1] = dim[1];
    for (int i = rank-2; i >= 2; i--) stride[i] = stride[i+1] * dim[i+1];
    stride[0] = stride[2] * dim[2];
    for (int i = rank; i < 5; i++) stride[i] = 0;
}

static Config base_2d() {
    Config c;
    memset(&c, 0, sizeof(c));
    c.label = "base-2d";
    c.pass = jittor::CONV_ALGO_FWD;
    c.spatial_dims = 2;
    c.dtype_x = c.dtype_w = c.dtype_y = 0;
    c.compute_type = 0;
    c.math_type = 0;
    c.filter_format = 0;
    c.groups = 1;
    c.workspace_ratio = 0.25f;
    int dim_x[5] = {4, 8, 16, 16, 0};
    int dim_w[5] = {8, 8, 3, 3, 0};
    int dim_y[5] = {4, 8, 16, 16, 0};
    for (int i = 0; i < 5; i++) {
        c.dim_x[i] = dim_x[i]; c.dim_w[i] = dim_w[i]; c.dim_y[i] = dim_y[i];
    }
    contiguous_nchw(c.stride_x, c.dim_x, 4);
    contiguous_nchw(c.stride_y, c.dim_y, 4);
    for (int i = 0; i < 2; i++) { c.pad[i] = 1; c.conv_stride[i] = 1; c.dilation[i] = 1; }
    return c;
}

static Config base_3d() {
    Config c = base_2d();
    c.label = "base-3d";
    c.spatial_dims = 3;
    int dim_x[5] = {4, 8, 8, 16, 16};
    int dim_w[5] = {8, 8, 3, 3, 3};
    int dim_y[5] = {4, 8, 8, 16, 16};
    for (int i = 0; i < 5; i++) {
        c.dim_x[i] = dim_x[i]; c.dim_w[i] = dim_w[i]; c.dim_y[i] = dim_y[i];
    }
    contiguous_nchw(c.stride_x, c.dim_x, 5);
    contiguous_nchw(c.stride_y, c.dim_y, 5);
    c.pad[2] = 1; c.conv_stride[2] = 1; c.dilation[2] = 1;
    return c;
}

static std::vector<Config> matrix() {
    std::vector<Config> v;
    Config c;

    v.push_back(base_2d());

    // There is a table per pass, so the pass has to be in the key for the 2-D
    // and 3-D entries of one table not to reach each other.
    c = base_2d(); c.label = "pass=bwd-data";   c.pass = jittor::CONV_ALGO_BWD_DATA;   v.push_back(c);
    c = base_2d(); c.label = "pass=bwd-filter"; c.pass = jittor::CONV_ALGO_BWD_FILTER; v.push_back(c);

    // 2-D and 3-D share all three tables.
    v.push_back(base_3d());
    c = base_3d(); c.label = "3d/pass=bwd-data";   c.pass = jittor::CONV_ALGO_BWD_DATA;   v.push_back(c);
    c = base_3d(); c.label = "3d/pass=bwd-filter"; c.pass = jittor::CONV_ALGO_BWD_FILTER; v.push_back(c);

    // One operand dtype at a time. The 2-D backward text keys had none of these.
    c = base_2d(); c.label = "dtype_x=half"; c.dtype_x = 2; v.push_back(c);
    c = base_2d(); c.label = "dtype_w=half"; c.dtype_w = 2; v.push_back(c);
    c = base_2d(); c.label = "dtype_y=half"; c.dtype_y = 2; v.push_back(c);

    // Numerics: which algorithms are admissible depends on both.
    c = base_2d(); c.label = "compute=double"; c.compute_type = 1; v.push_back(c);
    c = base_2d(); c.label = "math=tensor-op"; c.math_type = 1;    v.push_back(c);

    // Filter layout, group count, and the workspace budget a cached algorithm
    // has to be re-chosen under.
    c = base_2d(); c.label = "filter=nhwc";  c.filter_format = 1;      v.push_back(c);
    c = base_2d(); c.label = "groups=2";     c.groups = 2;             v.push_back(c);
    c = base_2d(); c.label = "workspace=.5"; c.workspace_ratio = 0.5f; v.push_back(c);

    // One extent at a time, on each of the three tensors.
    c = base_2d(); c.label = "x-n=5"; c.dim_x[0] = 5; c.dim_y[0] = 5;
        contiguous_nchw(c.stride_x, c.dim_x, 4);
        contiguous_nchw(c.stride_y, c.dim_y, 4); v.push_back(c);
    c = base_2d(); c.label = "x-h=17"; c.dim_x[2] = 17;
        contiguous_nchw(c.stride_x, c.dim_x, 4); v.push_back(c);
    c = base_2d(); c.label = "x-w=17"; c.dim_x[3] = 17;
        contiguous_nchw(c.stride_x, c.dim_x, 4); v.push_back(c);
    c = base_2d(); c.label = "w-kh=5"; c.dim_w[2] = 5; v.push_back(c);
    c = base_2d(); c.label = "w-kw=5"; c.dim_w[3] = 5; v.push_back(c);
    c = base_2d(); c.label = "y-h=17"; c.dim_y[2] = 17;
        contiguous_nchw(c.stride_y, c.dim_y, 4); v.push_back(c);
    c = base_3d(); c.label = "3d/x-d=9"; c.dim_x[2] = 9;
        contiguous_nchw(c.stride_x, c.dim_x, 5); v.push_back(c);
    c = base_3d(); c.label = "3d/w-kd=5"; c.dim_w[2] = 5; v.push_back(c);

    // Same extents, channels-last storage: only the strides differ.
    c = base_2d(); c.label = "x-nhwc"; contiguous_nhwc(c.stride_x, c.dim_x, 4); v.push_back(c);
    c = base_2d(); c.label = "y-nhwc"; contiguous_nhwc(c.stride_y, c.dim_y, 4); v.push_back(c);
    c = base_3d(); c.label = "3d/x-ndhwc"; contiguous_nhwc(c.stride_x, c.dim_x, 5); v.push_back(c);

    // One convolution parameter at a time.
    c = base_2d(); c.label = "pad-h=2";      c.pad[0] = 2;         v.push_back(c);
    c = base_2d(); c.label = "pad-w=2";      c.pad[1] = 2;         v.push_back(c);
    c = base_2d(); c.label = "stride-h=2";   c.conv_stride[0] = 2; v.push_back(c);
    c = base_2d(); c.label = "stride-w=2";   c.conv_stride[1] = 2; v.push_back(c);
    c = base_2d(); c.label = "dilation-h=2"; c.dilation[0] = 2;    v.push_back(c);
    c = base_2d(); c.label = "dilation-w=2"; c.dilation[1] = 2;    v.push_back(c);

    // The pairs the separator-less concatenation could not tell apart: 17 is
    // 0x11, so `jk << 1 << 17` and `jk << 17 << 1` both wrote "111".
    c = base_2d(); c.label = "pad=(1,17)";      c.pad[0] = 1;  c.pad[1] = 17; v.push_back(c);
    c = base_2d(); c.label = "pad=(17,1)";      c.pad[0] = 17; c.pad[1] = 1;  v.push_back(c);
    c = base_2d(); c.label = "stride=(1,17)";   c.conv_stride[0] = 1;  c.conv_stride[1] = 17; v.push_back(c);
    c = base_2d(); c.label = "stride=(17,1)";   c.conv_stride[0] = 17; c.conv_stride[1] = 1;  v.push_back(c);
    c = base_2d(); c.label = "dilation=(1,17)"; c.dilation[0] = 1;  c.dilation[1] = 17; v.push_back(c);
    c = base_2d(); c.label = "dilation=(17,1)"; c.dilation[0] = 17; c.dilation[1] = 1;  v.push_back(c);

    // And the three-way version of it in 3-D.
    c = base_3d(); c.label = "3d/pad=(1,1,17)"; c.pad[0]=1;  c.pad[1]=1;  c.pad[2]=17; v.push_back(c);
    c = base_3d(); c.label = "3d/pad=(1,17,1)"; c.pad[0]=1;  c.pad[1]=17; c.pad[2]=1;  v.push_back(c);
    c = base_3d(); c.label = "3d/pad=(17,1,1)"; c.pad[0]=17; c.pad[1]=1;  c.pad[2]=1;  v.push_back(c);

    return v;
}

static ConvAlgoKey key_of(const Config& c) {
    return conv_algo_key(
        c.pass, c.spatial_dims,
        c.dtype_x, c.dtype_w, c.dtype_y,
        c.compute_type, c.math_type,
        c.filter_format, c.groups, c.workspace_ratio,
        c.dim_x, c.stride_x, c.dim_w, c.dim_y, c.stride_y,
        c.pad, c.conv_stride, c.dilation);
}
"""

#: The real key: every configuration in the matrix must get its own.
#:
#: `bytes` and `hash` are different properties.  `bytes` is the correctness one
#: -- `ConvAlgoKeyEq` is a `memcmp`, so two configurations with equal bytes *are*
#: one cache entry.  `hash` is the one a constant hash breaks.
CURRENT = MATRIX + r"""
int main() {
    std::vector<Config> configs = matrix();
    jittor::ConvAlgoKeyHash hash;
    jittor::ConvAlgoKeyEq equal;
    int byte_collisions = 0, hash_collisions = 0;
    for (size_t i = 0; i < configs.size(); i++)
        for (size_t j = i+1; j < configs.size(); j++) {
            ConvAlgoKey a = key_of(configs[i]), b = key_of(configs[j]);
            if (equal(a, b)) {
                byte_collisions++;
                printf("BYTES %s == %s\n", configs[i].label, configs[j].label);
            }
            if (hash(a) == hash(b)) {
                hash_collisions++;
                printf("HASH %s == %s\n", configs[i].label, configs[j].label);
            }
        }
    // A key must also find itself. The memset in `conv_algo_key` is what makes
    // this hold; without it an uninitialised gap would make the same
    // convolution miss its own entry.
    int unstable = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        ConvAlgoKey a = key_of(configs[i]), b = key_of(configs[i]);
        if (!equal(a, b) || hash(a) != hash(b)) {
            unstable++;
            printf("UNSTABLE %s\n", configs[i].label);
        }
    }
    // And the table must keep them apart, which is what the call sites rely on.
    jittor::ConvAlgoCache<int> cache;
    for (size_t i = 0; i < configs.size(); i++) cache[key_of(configs[i])] = (int)i;
    int misplaced = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        auto iter = cache.find(key_of(configs[i]));
        if (iter == cache.end() || iter->second != (int)i) misplaced++;
    }
    printf("configs=%zu bytes=%d hash=%d unstable=%d entries=%zu misplaced=%d\n",
           configs.size(), byte_collisions, hash_collisions, unstable,
           cache.size(), misplaced);
    return (byte_collisions || hash_collisions || unstable || misplaced) ? 1 : 0;
}
"""

#: The encoding this replaced, over the same matrix.
#:
#: `jk << int` is `JK::hex`: lowercase, variable length, no leading zeros, and
#: nothing between consecutive values.  `put_hex` reproduces it exactly, so the
#: collisions counted here are the ones the shipped code had rather than an
#: approximation of them.
PREDECESSOR = MATRIX + r"""
// python/jittor/src/jit_key.h, `operator<<(JK&, const JK::hex&)`:
// `nbits = 64 - lzcnt(a); nbits = a ? nbits-1 : 0;` then one nibble per step
// down from `nbits/4`, each taken `% 16`.
static void put_hex(std::string& s, unsigned long long a) {
    unsigned lz = 0;
    while (lz < 64 && !((a >> (63 - lz)) & 1ull)) lz++;
    unsigned nbits = a ? (64 - lz) - 1 : 0;
    for (int i = (int)(nbits / 4); i >= 0; i--) {
        unsigned d = (unsigned)((a >> (i * 4)) % 16);
        s += (char)(d < 10 ? d + '0' : d - 10 + 'a');
    }
}

// backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc as of e5e353644.
// Note what is absent: the dtypes, both stride vectors, the output extent, the
// compute type and the workspace ratio. And that pad, stride and dilation run
// together with nothing between their components.
static std::string legacy_2d_backward(const Config& c) {
    std::string s;
    for (int i = 0; i < 4; i++) { put_hex(s, c.dim_x[i]); s += ","; }
    for (int i = 0; i < 4; i++) { put_hex(s, c.dim_w[i]); s += ","; }
    put_hex(s, c.pad[0]); put_hex(s, c.pad[1]); s += ",";
    put_hex(s, c.conv_stride[0]); put_hex(s, c.conv_stride[1]); s += ",";
    put_hex(s, c.dilation[0]); put_hex(s, c.dilation[1]); s += ",";
    put_hex(s, c.groups); s += ".";
    s += "math="; put_hex(s, c.math_type); s += ".";
    return s;
}

// backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc as of e5e353644. This one does
// carry the dtypes, the output extent and the budget; the strides are still
// missing and the three padding values still run together.
static std::string legacy_3d(const Config& c) {
    const char* tag = c.pass == jittor::CONV_ALGO_FWD ? "conv3d.fwd;"
                    : c.pass == jittor::CONV_ALGO_BWD_DATA ? "conv3d.bwdx;"
                    : "conv3d.bwdw;";
    std::string s = tag;
    s += "x="; put_hex(s, c.dtype_x); s += ":";
    for (int i = 0; i < 5; i++) { put_hex(s, c.dim_x[i]); if (i < 4) s += ","; }
    s += ";w="; put_hex(s, c.dtype_w); s += ":";
    for (int i = 0; i < 5; i++) { put_hex(s, c.dim_w[i]); if (i < 4) s += ","; }
    s += ";y="; put_hex(s, c.dtype_y); s += ":";
    for (int i = 0; i < 5; i++) { put_hex(s, c.dim_y[i]); if (i < 4) s += ","; }
    s += ";conv=";
    put_hex(s, c.pad[0]); put_hex(s, c.pad[1]); put_hex(s, c.pad[2]); s += ",";
    put_hex(s, c.conv_stride[0]); put_hex(s, c.conv_stride[1]); put_hex(s, c.conv_stride[2]); s += ",";
    put_hex(s, c.dilation[0]); put_hex(s, c.dilation[1]); put_hex(s, c.dilation[2]); s += ",";
    put_hex(s, c.groups); s += ";";
    s += "compute="; put_hex(s, c.compute_type); s += ":";
    s += "math="; put_hex(s, c.math_type); s += ":";
    // `jk << float` promoted to the float64 overload: itof(0x<hex of bits>).
    unsigned long long bits = 0;
    double as_double = c.workspace_ratio;
    memcpy(&bits, &as_double, sizeof(bits));
    s += "workspace_ratio=itof(0x"; put_hex(s, bits); s += ").";
    return s;
}

static std::string legacy_key(const Config& c) {
    return c.spatial_dims == 3 ? legacy_3d(c) : legacy_2d_backward(c);
}

int main() {
    std::vector<Config> configs = matrix();
    int collisions = 0;
    for (size_t i = 0; i < configs.size(); i++)
        for (size_t j = i+1; j < configs.size(); j++) {
            // Only pairs that one table would have held. There was a table per
            // pass -- fwd_algo_cache, bwdx_algo_cache, bwdw_algo_cache -- so
            // the pass was implicit in which table was consulted rather than
            // written into the key; and the 2-D and the 3-D encodings are
            // different shapes, the 3-D one carrying a `conv3d.` tag.
            if (configs[i].spatial_dims != configs[j].spatial_dims) continue;
            if (configs[i].pass != configs[j].pass) continue;
            if (legacy_key(configs[i]) == legacy_key(configs[j])) {
                collisions++;
                printf("COLLIDES %s == %s  [%s]\n", configs[i].label,
                       configs[j].label, legacy_key(configs[i]).c_str());
            }
        }
    printf("configs=%zu collisions=%d\n", configs.size(), collisions);
    return collisions ? 1 : 0;
}
"""


def _compile_and_run(source, header_text=None):
    """Build `source` against the header and run it.

    `header_text` overrides the header the case compiles against, which is how
    the constant-hash control gets at `ConvAlgoKeyHash` itself rather than at a
    copy of it in the case.
    """
    compiler = os.environ.get("CXX", "g++")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        include = HEADER.parent
        if header_text is not None:
            (scratch / HEADER.name).write_text(header_text, encoding="utf-8")
            include = scratch
        source_path = scratch / "case.cc"
        source_path.write_text(source, encoding="utf-8")
        binary = scratch / "case"
        build = subprocess.run(
            [compiler, "-std=c++14", "-g", "-O0", "-I" + str(include),
             str(source_path), "-o", str(binary)],
            capture_output=True, text=True, timeout=300)
        if build.returncode != 0:
            return None, build.stdout + build.stderr
        run = subprocess.run([str(binary)], capture_output=True, text=True,
                             timeout=300)
        return run, run.stdout + run.stderr


def _header_with_constant_hash():
    """The real header with `ConvAlgoKeyHash`'s result replaced by a constant."""
    text = HEADER.read_text(encoding="utf-8")
    marker = "return (size_t)h;"
    assert text.count(marker) == 1, (
        "cannot find ConvAlgoKeyHash's return in the header; this control would "
        "otherwise compile the unmodified hash and pass for the wrong reason")
    return text.replace(marker, "return (size_t)1;")


class TestCudnnConvAlgoKey(unittest.TestCase):
    def test_every_configuration_gets_its_own_key(self):
        run, output = _compile_and_run(CURRENT)
        self.assertIsNotNone(run, "could not build the case:\n" + output)
        self.assertIn("configs=%d" % CONFIG_COUNT, output)
        self.assertIn("entries=%d" % CONFIG_COUNT, output)
        self.assertIn("bytes=0", output)
        self.assertIn("hash=0", output)
        self.assertIn("unstable=0", output)
        self.assertIn("misplaced=0", output)
        self.assertEqual(run.returncode, 0, output)

    def test_the_text_key_this_replaced_is_what_the_matrix_catches(self):
        """The case above would pass on a key that separated nothing."""
        run, output = _compile_and_run(PREDECESSOR)
        self.assertIsNotNone(run, "could not build the case:\n" + output)
        self.assertIn("configs=%d" % CONFIG_COUNT, output)
        self.assertNotIn("collisions=0", output)
        self.assertNotEqual(run.returncode, 0, output)
        # The dropped fields, named. Each of these is one convolution taking
        # another's measured algorithm.
        for expected in (
                "COLLIDES base-2d == dtype_x=half",
                "COLLIDES base-2d == dtype_w=half",
                "COLLIDES base-2d == dtype_y=half",
                "COLLIDES base-2d == compute=double",
                "COLLIDES base-2d == filter=nhwc",
                "COLLIDES base-2d == workspace=.5",
                "COLLIDES base-2d == y-h=17",
                "COLLIDES base-2d == x-nhwc",
                "COLLIDES base-2d == y-nhwc",
                "COLLIDES base-3d == 3d/x-ndhwc",
                # and the separator-less concatenation, in 2-D and in 3-D:
                "COLLIDES pad=(1,17) == pad=(17,1)",
                "COLLIDES stride=(1,17) == stride=(17,1)",
                "COLLIDES dilation=(1,17) == dilation=(17,1)",
                "COLLIDES 3d/pad=(1,1,17) == 3d/pad=(1,17,1)",
                "COLLIDES 3d/pad=(1,1,17) == 3d/pad=(17,1,1)",
                "COLLIDES 3d/pad=(1,17,1) == 3d/pad=(17,1,1)"):
            self.assertIn(expected, output)

    def test_a_constant_hash_is_caught(self):
        """The hash half of the first case must not be vacuous either."""
        run, output = _compile_and_run(
            CURRENT, header_text=_header_with_constant_hash())
        self.assertIsNotNone(run, "could not build the case:\n" + output)
        self.assertNotIn("hash=0", output)
        self.assertNotEqual(run.returncode, 0, output)
        # Only the hash half. A constant hash keeps the table correct, because
        # the equality is a memcmp; it is the bytes that carry the identity.
        self.assertIn("bytes=0", output)
        self.assertIn("misplaced=0", output)


if __name__ == "__main__":
    unittest.main()
