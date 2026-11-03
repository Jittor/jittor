---
name: cuda-negative-path-verification
description: 测试 CUDA 的编译、加载、rendezvous 或资源错误时，确认真正触发预期失败且完整清理进程。
---

# 验证 CUDA 失败路径

## 操作

- 先用同配置正向对照证明 CUDA 和输入可用，再在本任务隔离子进程中注入单一故障。
  不设置 `nvcc_path` 可能触发自动发现，不能当作禁用 CUDA 的证据。
- 对编译错误，记录生成源码、完整命令和诊断；断言错误对应预期算子/阶段，
  不能任意非零退出都算命中（导入失败、设备不可见属于另一问题）。
- 使用 `tests/_helpers/child_process.py` 或维护中的 launcher，明确超时、退出码、stderr
  和进程清理。崩溃、挂起、正确异常要分别判断。
- peer 缺失用例必须实际触发 lazy 初始化；仅 `import jittor` 可能不进入 NCCL rendezvous。
  按当前维护测试调用 `compile_extern.setup_nccl()`，并使用独立 rendezvous 文件/端口。
- 验证成功路径依然运行；不能以永远报错的实现“通过”负向测试。结束后确认本任务 worker 退出，
  不靠删除正在使用的缓存解除等待。

记录实际执行的设备和用例。环境缺失导致 skip 时，只能记未验证。
