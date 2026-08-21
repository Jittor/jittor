# JIT 浮点常量与小数 padding 修复

- Status: fixed; KI-COMPILER-002 withdrawn
- Date: 2026-08-21
- Baseline: `8b34f33a`
- Owner: JIT compiler and operator-codegen maintainers
- Review when: JIT-key floating serialization, generated-kernel language
  standard, reindex overflow constants, or CPU/CUDA compiler flags change

## Finding

Before the change, the strict expected failure for
`nn.pad(..., mode="constant", value=0.7)` reproduced on CPU. The generated
reindex source contained `0x1.6666666666666p-1`, and the first compiler command
inside `asm_tuner.py` failed with `error: exponent has no digits` under
`g++ -std=c++14`.

The failure occurred while compiling C++ to the intermediate `.post.s`, before
`pass_asm` inspected any `@begin replace` directive. The old ledger attribution
to assembly rewriting was therefore incorrect. `parse_jit_keys` decoded raw
`itof(0x...)` bits with `std::hexfloat`, producing a hexadecimal floating
literal that is not a portable C++14 source token.

The existing C++ test also did not exercise this decoder for zero: it used
`f=itof(0x0)`, where `=` selects the integer-hex path. Real floating JIT values
use `:`, so the sample was converted to integer zero before `convert_itof` could
run.

## Change

The raw JIT key remains bit-exact and unchanged. During macro parsing, finite
doubles are now rendered with the classic locale and `max_digits10` decimal
precision. Integral values and signed zero retain a decimal point, while the
existing infinity and NaN expressions remain valid C++14.

The C++ regression now reaches the real floating decoder and covers positive and
negative fractions, negative zero, infinity, and NaN. The Python regression is
no longer an expected failure: it checks `0.7`, `-0.7`, and a value constructed
with `float.fromhex(...)` against NumPy. Pad OpInfo samples again use positive
and negative fractional fills instead of the old integral-value workaround.

## Verification

Tests used Python 3.12.13 and GCC 12.3. CPU and CUDA state was isolated below
`$JITTOR_LAB_ROOT/_state/`, with the first core and operator builds serialized.
Real-device checks used CUDA 12.2, cuDNN 8.9.7, and an `sm_89` GPU. Generated
CUDA source contained `#define JIT_cuda 1`, an `__global__` kernel, and the
decimal `0.69999999999999996` and `-0.69999999999999996` constants.

| Gate | Result |
| --- | --- |
| Pre-change CPU strict expected failure | reproduced: `1 xfailed` |
| C++ JIT-key regression | `1 passed` |
| CPU fractional/negative/hex-origin pad regression | `1 passed`, 3 subtests passed |
| Pad OpInfo CPU forward, gradcheck, and gradgradcheck | `3 passed`, 678 deselected |
| Pad OpInfo real-CUDA forward | `1 passed`, 226 deselected |
| Torch-compatible pad module, CPU + real CUDA | `23 passed` |
| asm tuner plus isolated allocator timing rerun | `2 passed` |
| Repository layout | passed |
| `tests/structure` | `211 passed`, 2 skipped, 1503 subtests passed |

The full JIT-test plus asm-tuner run had 24 passes and one unrelated timing
failure: `sfrl_allocator_time` measured 623.288 microseconds against a 600
microsecond threshold. Its immediate isolated rerun passed. The full kernel-trap
module had seven passes and 50 passing subtests, plus two unrelated native-mode
contract failures: `.long()` returned `int32`, and `float32 + int64` promoted to
`float64`. Neither failure executes the changed constant-decoding path.

## Maintained commands

With an isolated environment configured according to
`agent/manuals/environment.md`:

```bash
python -m pytest -q \
  tests/compiler/test_jit_tests.py::TestJitTests::test_jit_key
python -m pytest -q tests/compiler/test_kernel_traps.py \
  -k constant_pad_fractional_fill_cpu_codegen
JITTOR_TEST_DEVICES=cpu python -m pytest -q tests/ops/test_ops.py \
  -k pad_constant
python -m pytest -q tests/compat/torch/test_torch_compat_pad.py
```

## Limits

This change is a correctness fix and makes no compile-time or runtime performance
claim. CUDA was validated on one NVIDIA configuration; NPU and ROCm were not
available. The two native-mode kernel-trap failures and the timing-sensitive
allocator threshold remain separate unit-test cleanup work.
