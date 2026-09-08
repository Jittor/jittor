"""No accelerator op builds a cache key out of the shared jit key buffer.

``get_jk()`` returns one buffer per thread, and by the time an op's ``jit_run``
is entered the executor has already filled it with the jit key of the kernel
that is running.  Six cuDNN convolution sites, one cuFFT site and one cuTT site
used to ``jk.clear()`` it and write their own plan or algorithm cache key into
it, then take a ``std::string`` off it -- so every convolution paid a string
construction and scribbled over the executor's buffer, and the cache key's
identity was a text encoding rather than the configuration itself.

Those keys are POD structs now, hashed and compared as bytes.  This file is the
part of that which a later edit cannot undo quietly: it asserts on the shape of
the *execution* half of every op source under ``backends/cuda/kernels``, where
"execution half" is everything from the ``#else // JIT`` that closes
``#ifndef JIT``.  ``jit_prepare`` -- whose whole job is to build the jit key --
sits in the other half and is untouched by any of this.

Two things guard against this file passing for the wrong reason:

  * both halves of every named file are asserted to be non-empty and to contain
    the function they should, so an absence check cannot succeed because the
    split returned nothing;
  * ``jit_prepare`` is asserted to still use ``jk``, which is the positive
    control for the same token the execution half is checked against.  A test
    that looked for a token nothing anywhere used would pass on any tree.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "backends/cuda/kernels"

#: The marker that closes ``#ifndef JIT`` in every op source.
JIT_MARKER = "#else // JIT"

#: Any use of the shared buffer or its type, as a whole word so that
#: identifiers like ``jk_put_str_with_len`` or a ``jkl`` local still match and
#: an unrelated ``jack`` does not.
JIT_KEY_USE = re.compile(r"\bjk\b|\bJK\b|\bget_jk\b|\bJitKey\b")

#: The op sources whose execution path used to reuse the shared jit key buffer,
#: each with the POD key its execution path must now consult instead.
#:
#: Named one at a time rather than discovered, so that a rename, a move or a
#: deletion shows up as a failure here instead of quietly shrinking the set of
#: files this file has an opinion about.
CONVERTED = {
    "cufft/cufft_fft_op.cc": "CufftPlanKey",
    "cutt/cutt_transpose_op.cc": "CuttPlanKey",
    "cudnn/cudnn_conv_op.cc": "conv_algo_key(",
    "cudnn/cudnn_conv_backward_x_op.cc": "conv_algo_key(",
    "cudnn/cudnn_conv_backward_w_op.cc": "conv_algo_key(",
    "cudnn/cudnn_conv3d_op.cc": "conv_algo_key(",
    "cudnn/cudnn_conv3d_backward_x_op.cc": "conv_algo_key(",
    "cudnn/cudnn_conv3d_backward_w_op.cc": "conv_algo_key(",
}

#: The six cuDNN algorithm caches, and the pass each one's key must declare.
#: A key that did not name its pass would let the 2-D and the 3-D entries of one
#: table reach each other, which is what the old ``"conv3d.fwd;"`` prefix was
#: for.
CUDNN_PASSES = {
    "cudnn/cudnn_conv_op.cc": ("fwd_algo_cache", "CONV_ALGO_FWD", 2),
    "cudnn/cudnn_conv_backward_x_op.cc": ("bwdx_algo_cache", "CONV_ALGO_BWD_DATA", 2),
    "cudnn/cudnn_conv_backward_w_op.cc": ("bwdw_algo_cache", "CONV_ALGO_BWD_FILTER", 2),
    "cudnn/cudnn_conv3d_op.cc": ("fwd_algo_cache", "CONV_ALGO_FWD", 3),
    "cudnn/cudnn_conv3d_backward_x_op.cc": ("bwdx_algo_cache", "CONV_ALGO_BWD_DATA", 3),
    "cudnn/cudnn_conv3d_backward_w_op.cc": ("bwdw_algo_cache", "CONV_ALGO_BWD_FILTER", 3),
}


def _halves(path):
    """The source split into (before ``#else // JIT``, from it onwards)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    index = text.find(JIT_MARKER)
    assert index > 0, "%s has no %r; the split below would be meaningless" % (
        path.relative_to(ROOT), JIT_MARKER)
    return text[:index], text[index:]


def _sources_with_a_jit_half():
    for path in sorted(KERNELS.rglob("*.cc")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if JIT_MARKER in text:
            yield path


def _strip_comments(text):
    """Comments do not execute, and several of these files explain in a comment
    exactly what they no longer do -- including quoting the ``jk <<`` lines that
    were removed."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


@pytest.mark.parametrize("relative", sorted(CONVERTED))
def test_converted_op_executes_without_touching_the_jit_key_buffer(relative):
    path = KERNELS / relative
    prepare_half, jit_half = _halves(path)

    # Both halves are real. Without this, the absence assertion below would
    # also hold for a split that returned an empty string.
    assert "jit_prepare" in prepare_half, relative
    assert "jit_run" in jit_half, relative

    # The positive control: `jit_prepare` builds the jit key and still says so,
    # with the same token the execution half is checked against.
    assert JIT_KEY_USE.search(prepare_half), (
        "%s: jit_prepare no longer mentions the jit key, so the check below is "
        "looking for a token this tree does not use anywhere" % relative)

    found = JIT_KEY_USE.findall(_strip_comments(jit_half))
    assert not found, (
        "%s builds something out of the shared jit key buffer while it executes "
        "(%r). The executor has already filled that buffer with the key of the "
        "kernel running right now, and taking a string off it is a string "
        "construction per call. Use a POD key instead -- see "
        "backends/cuda/libraries/cudnn/include/cudnn_conv_algo_key.h."
        % (relative, sorted(set(found))))

    # And it consults a POD key in place of the text.
    assert CONVERTED[relative] in jit_half, (
        "%s: expected its execution path to build %s"
        % (relative, CONVERTED[relative]))


@pytest.mark.parametrize("relative", sorted(CUDNN_PASSES))
def test_each_cudnn_algo_cache_names_its_pass_and_rank(relative):
    table, pass_name, spatial_dims = CUDNN_PASSES[relative]
    _, jit_half = _halves(KERNELS / relative)
    stripped = _strip_comments(jit_half)

    # The table is keyed by the POD struct, not by a string.
    assert "ConvAlgoCache<" in stripped, relative
    assert "unordered_map<string" not in stripped, relative
    assert table in stripped, relative

    # The key declares which pass and which rank it stands for. There is one
    # table per pass shared by the 2-D and the 3-D op, so a key missing either
    # would be one convolution reading the other's measured algorithm.
    key_call = stripped[stripped.find("conv_algo_key("):]
    assert key_call, relative
    assert pass_name in key_call, relative
    assert re.search(r"%s,\s*%d," % (pass_name, spatial_dims), key_call), (
        "%s: expected conv_algo_key(%s, %d, ...)"
        % (relative, pass_name, spatial_dims))

    # Lookup and store must use the same key object rather than rebuilding it.
    assert "%s.find(algo_key)" % table in stripped, relative
    assert "%s[algo_key]" % table in stripped, relative


def test_no_cuda_kernel_touches_the_jit_key_buffer_while_it_executes():
    """The same property over every op source, not only the converted ones."""
    sources = list(_sources_with_a_jit_half())
    # Non-vacuity: this swept 27 sources when it was written. A collapse to a
    # handful would otherwise read exactly like a pass.
    assert len(sources) >= 20, (
        "only found %d op sources with a JIT half under %s"
        % (len(sources), KERNELS.relative_to(ROOT)))

    offenders = {}
    for path in sources:
        _, jit_half = _halves(path)
        found = JIT_KEY_USE.findall(_strip_comments(jit_half))
        if found:
            offenders[str(path.relative_to(ROOT))] = sorted(set(found))
    assert not offenders, (
        "these op sources use the shared jit key buffer on their execution "
        "path: %r" % offenders)
