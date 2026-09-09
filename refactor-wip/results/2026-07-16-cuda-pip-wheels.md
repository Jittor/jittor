# CUDA 12 component wheel 验证

状态：✅ 已完成（2026-07-16）

## 结论

`jittor[cuda12]` 的固定 CUDA 12.2/cuDNN 8 component stack 已接入 Jittor
编译器和外部库加载链路。基础安装不拉 NVIDIA 包；错版本或不完整 stack 不会
与 JTCUDA 混用。

CUDA 11/12 官方 `nvidia-cuda-nvcc` wheel 不含 `nvcc` 驱动，因此系统/JTCUDA
编译器回退必须保留。首版不支持 cuDNN 9、CUDA 11 或 CUDA 13。

## 验证

- `python3 python/jittor/test/test_cuda_wheel.py`：6/6 OK。
- JTCUDA 12.2 冷启动回归：核心和五个外部库加载成功，CUDA 求和 `14.0`。
- 模拟官方 versioned-only component 布局，`LD_LIBRARY_PATH` 为空：冷启动 OK。
- GPU：JIT、matmul、cuDNN conv、cuRAND、cuSPARSE、NCCL dlopen、NVTX OK。
- `readelf -d`：`DT_NEEDED` 和 RUNPATH 均指向 component wheel ABI/目录。
- `setup.py egg_info`：基础 requirements 无 NVIDIA；`cuda12` extra 固定版本并
  限定 Linux x86_64。

完整用法、版本矩阵和边界见：
`/home/zy/projects/doc/2026-07-16-jittor-cuda-pip-wheels.md`。
