from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/development/async-error-diagnostics.md"


def test_async_error_contract_states_ring_and_cuda_requirements():
    text = GUIDE.read_text(encoding="utf-8")
    for token in (
        "TraceData",
        "每线程、有界的发射记录环",
        "算子 id / 名字",
        "Python 文件与行号",
        "流标识",
        "免分配",
        "not-found",
        "CUDA 探测",
        "CUDA_VISIBLE_DEVICES=0",
        "test_async_error_location.py",
    ):
        assert token in text
