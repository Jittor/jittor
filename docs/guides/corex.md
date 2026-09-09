# 天数 Corex：探测与后续验证

本文记录 Corex 后端的**探测契约**。当前代码可以在不导入编译器、不改动 Jittor 全局
状态的前提下检查一个 Corex 安装。完整的编译与性能验证需要受支持的 Corex 设备，
属于单独的门禁。

## 配置安装路径

把 `COREX_HOME` 指向厂商 SDK 根目录。探测代码只查找 `$COREX_HOME/bin/clang++`，
**不创建目录、不加载库、不修改编译选项**。

```bash
export COREX_HOME=/opt/corex
PYTHONPATH="$PWD/python" python - <<'PY'
from jittor.extern.corex.corex_compiler import discover
result = discover()
print(result)
raise SystemExit(0 if result.available else 1)
PY
```

做离线契约测试时，建一个只含 `bin/clang++` 的临时目录即可（假的可执行文件就够）。
返回的结果对象会报告解析出的 home、编译器路径、可用性和一个稳定的原因字符串。

## 上机交接

**仓库目前没有可信的 Corex 硬件基线。** 在受支持的天数智芯机器上，先记录确切的
加速器型号、驱动、SDK 和 clang 版本，再运行 Jittor。使用独立的 `JITTOR_HOME` 与
`TMPDIR`，然后跑设备可见性探测和一个融合 kernel。结果必须显示 Corex 编译器路径，
且**没有 CPU 回退**。

```bash
export COREX_HOME=/path/to/corex
export JITTOR_HOME=/tmp/jittor-corex-home
export TMPDIR=/tmp/jittor-corex-tmp
export CUDA_VISIBLE_DEVICES=<分配到的设备>
PYTHONPATH="$PWD/python" python -m pytest -q \
  tests/backends/corex/test_corex_discovery.py
```

这条命令**只是探测契约**。将来的硬件门禁还必须跑相应的 CUDA 兼容数值探测，并记录
厂商的设备与编译器版本。**不要把本机的离线结果当作 Corex 硬件验证。**
