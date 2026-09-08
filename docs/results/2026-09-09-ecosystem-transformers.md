# Transformers / ms-swift / verl 生态验证

状态：环境验证与阻塞定位，不能关闭生态任务。

`transformers==4.56.2` 在开发环境可导入；`ms-swift` 的 `swift==4.5.2` 只在
Jittor Python 3.11 环境存在，独立 PyTorch oracle 环境没有该包；`verl` 两侧均未安装。
因此 ms-swift 与 verl 尚无独立 Torch 前反向对拍证据。

GPT-2 使用独立 PyTorch 2.12.1+cu126 oracle，在临时隔离不匹配的 torchvision 后可构建。
Jittor shim forward 可运行，但 backward 在复杂 residual/attention/tied lm_head 图触发
`backward liveness release without a matching owner`。单层 Linear/平方/求和 backward
正常。根因已定位为 stale liveness queue：节点由 `free()` 完成传播并标记
`queued_for_free` 后，残留队列再次调用 release。三个 release 入口现在在该标志下
安全返回，非 stale 节点仍保留原 underflow 检查。真实 runner 修复后返回
`RC=0`、`29 tensors`、`fallback_count=0`，日志为 `/tmp/gpt2-fixed.log`；不能通过
关闭检查或改变测试期望掩盖。

生态测试已修复 oracle 环境隔离与可选 torchvision 处理，避免把 Jittor facade 当成 PyTorch
参考实现。独立 PyTorch 3.12 隔离目录已安装 `diffusers==0.35.1`、
`ms-swift==3.8.0`、`verl==0.9.0`：diffusers 可导入；ms-swift 仍缺
`modelscope`，verl 仍缺 `ray`，所以两者尚未可运行。当前没有速度结论；
vLLM/TRELLIS 需匹配版本或上游源码环境后重新验证。
